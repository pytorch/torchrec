#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import copy
from collections import OrderedDict
from dataclasses import dataclass
from typing import (
    List,
    Dict,
    TypeVar,
    Optional,
    Type,
    Any,
    Mapping,
    Union,
    Iterator,
    Tuple,
    Set,
)

import torch
from torch import nn
from torch.nn.modules.module import _IncompatibleKeys
from torchrec.distributed.embedding_sharding import (
    EmbeddingSharding,
    SparseFeaturesListAwaitable,
    SparseFeaturesIndicesAwaitable,
)
from torchrec.distributed.embedding_types import (
    SparseFeatures,
    BaseEmbeddingSharder,
    EmbeddingComputeKernel,
    ShardingType,
    SparseFeaturesList,
)
from torchrec.distributed.sharding.dp_sequence_sharding import (
    DpSequenceEmbeddingSharding,
)
from torchrec.distributed.sharding.rw_sequence_sharding import (
    RwSequenceEmbeddingSharding,
)
from torchrec.distributed.sharding.rw_sharding import RwSparseFeaturesDist
from torchrec.distributed.sharding.sequence_sharding import SequenceShardingContext
from torchrec.distributed.sharding.tw_sequence_sharding import (
    TwSequenceEmbeddingSharding,
)
from torchrec.distributed.types import (
    Awaitable,
    LazyAwaitable,
    ParameterSharding,
    ShardedModule,
    ShardedModuleContext,
    ShardedTensor,
    ShardingEnv,
)
from torchrec.distributed.utils import append_prefix
from torchrec.distributed.utils import filter_state_dict
from torchrec.modules.embedding_configs import EmbeddingTableConfig, PoolingType
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.optim.fused import FusedOptimizerModule
from torchrec.optim.keyed import KeyedOptimizer, CombinedOptimizer
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, JaggedTensor

try:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
except OSError:
    pass


def create_embedding_sharding(
    sharding_type: str,
    embedding_configs: List[
        Tuple[EmbeddingTableConfig, ParameterSharding, torch.Tensor]
    ],
    env: ShardingEnv,
    device: Optional[torch.device] = None,
) -> EmbeddingSharding[SparseFeatures, torch.Tensor]:
    pg = env.process_group
    if pg is not None:
        if sharding_type == ShardingType.TABLE_WISE.value:
            return TwSequenceEmbeddingSharding(embedding_configs, env, device)
        elif sharding_type == ShardingType.ROW_WISE.value:
            return RwSequenceEmbeddingSharding(embedding_configs, pg, device)
        elif sharding_type == ShardingType.DATA_PARALLEL.value:
            return DpSequenceEmbeddingSharding(embedding_configs, env, device)
        else:
            raise ValueError(f"Sharding not supported {sharding_type}")
    else:
        if sharding_type == ShardingType.DATA_PARALLEL.value:
            return DpSequenceEmbeddingSharding(embedding_configs, env, device)
        else:
            raise ValueError(f"Sharding not supported {sharding_type}")


def _create_embedding_configs_by_sharding(
    module: EmbeddingCollection,
    table_name_to_parameter_sharding: Dict[str, ParameterSharding],
) -> Dict[str, List[Tuple[EmbeddingTableConfig, ParameterSharding, torch.Tensor]]]:
    sharding_type_to_embedding_configs: Dict[
        str, List[Tuple[EmbeddingTableConfig, ParameterSharding, torch.Tensor]]
    ] = {}
    state_dict = module.state_dict()
    for (
        config,
        embedding_names,
    ) in zip(module.embedding_configs, module.embedding_names_by_table):
        table_name = config.name
        assert table_name in table_name_to_parameter_sharding

        parameter_sharding = table_name_to_parameter_sharding[table_name]
        if parameter_sharding.compute_kernel not in [
            kernel.value for kernel in EmbeddingComputeKernel
        ]:
            raise ValueError(
                f"Compute kernel not supported {parameter_sharding.compute_kernel}"
            )

        param_name = "embeddings." + config.name + ".weight"
        assert param_name in state_dict
        param = state_dict[param_name]

        if parameter_sharding.sharding_type not in sharding_type_to_embedding_configs:
            sharding_type_to_embedding_configs[parameter_sharding.sharding_type] = []
        sharding_type_to_embedding_configs[parameter_sharding.sharding_type].append(
            (
                EmbeddingTableConfig(
                    num_embeddings=config.num_embeddings,
                    embedding_dim=config.embedding_dim,
                    name=config.name,
                    data_type=config.data_type,
                    feature_names=copy.deepcopy(config.feature_names),
                    pooling=PoolingType.NONE,
                    is_weighted=False,
                    has_feature_processor=False,
                    embedding_names=embedding_names,
                    weight_init_max=config.weight_init_max,
                    weight_init_min=config.weight_init_min,
                ),
                parameter_sharding,
                param,
            )
        )
    return sharding_type_to_embedding_configs


def _construct_jagged_tensors(
    embeddings: torch.Tensor,
    features: KeyedJaggedTensor,
    embedding_names: List[str],
) -> Dict[str, JaggedTensor]:
    ret: Dict[str, JaggedTensor] = {}
    offset_per_key = features.offset_per_key()
    lengths = features.lengths().view(-1, features.stride())
    for i, key in enumerate(features.keys()):
        start = offset_per_key[i]
        end = offset_per_key[i + 1]
        i_lengths = lengths[i]
        ret[key] = JaggedTensor(
            lengths=i_lengths,
            values=embeddings[start:end, :],
        )
    return ret


@dataclass
class EmbeddingCollectionContext(ShardedModuleContext):
    sharding_contexts: List[SequenceShardingContext]

    def record_stream(self, stream: torch.cuda.streams.Stream) -> None:
        for ctx in self.sharding_contexts:
            ctx.record_stream(stream)


class EmbeddingCollectionAwaitable(LazyAwaitable[Dict[str, JaggedTensor]]):
    def __init__(
        self,
        awaitables_per_sharding: List[Awaitable[torch.Tensor]],
        features_per_sharding: List[KeyedJaggedTensor],
        embedding_names_per_sharding: List[str],
    ) -> None:
        super().__init__()
        self._awaitables_per_sharding = awaitables_per_sharding
        self._features_per_sharding = features_per_sharding
        self._embedding_names_per_sharding = embedding_names_per_sharding

    def _wait_impl(self) -> Dict[str, JaggedTensor]:
        jt_dict: Dict[str, JaggedTensor] = {}
        for w, f, e in zip(
            self._awaitables_per_sharding,
            self._features_per_sharding,
            self._embedding_names_per_sharding,
        ):
            jt_dict.update(_construct_jagged_tensors(w.wait(), f, e))
        return jt_dict


class ShardedEmbeddingCollection(
    ShardedModule[
        SparseFeaturesList,
        List[torch.Tensor],
        Dict[str, torch.Tensor],
    ],
    FusedOptimizerModule,
):
    """
    Sharded implementation of `EmbeddingCollection`.
    This is part of the public API to allow for manual data dist pipelining.
    """

    def __init__(
        self,
        module: EmbeddingCollection,
        table_name_to_parameter_sharding: Dict[str, ParameterSharding],
        env: ShardingEnv,
        fused_params: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        sharding_type_to_embedding_configs = _create_embedding_configs_by_sharding(
            module, table_name_to_parameter_sharding
        )
        self._sharding_type_to_sharding: Dict[
            str, EmbeddingSharding[SparseFeatures, torch.Tensor]
        ] = {
            sharding_type: create_embedding_sharding(
                sharding_type, embedding_confings, env, device
            )
            for sharding_type, embedding_confings in sharding_type_to_embedding_configs.items()
        }

        self._device = device
        self._input_dists: nn.ModuleList = nn.ModuleList()
        self._lookups: nn.ModuleList = nn.ModuleList()
        self._create_lookups(fused_params)
        self._output_dists: nn.ModuleList = nn.ModuleList()
        self._create_output_dist()

        self._feature_splits: List[int] = []
        self._features_order: List[int] = []

        self._has_uninitialized_input_dist: bool = True

        # Get all fused optimizers and combine them.
        optims = []
        for lookup in self._lookups:
            for _, m in lookup.named_modules():
                if isinstance(m, FusedOptimizerModule):
                    # modify param keys to match EmbeddingCollection
                    params: Mapping[str, Union[torch.Tensor, ShardedTensor]] = {}
                    for param_key, weight in m.fused_optimizer.params.items():
                        # pyre-fixme[16]: `Mapping` has no attribute `__setitem__`.
                        params["embedding_modules." + param_key] = weight
                    m.fused_optimizer.params = params
                    optims.append(("", m.fused_optimizer))
        self._optim: CombinedOptimizer = CombinedOptimizer(optims)
        self._embedding_dim: int = module.embedding_dim
        self._embedding_names_per_sharding: List[List[str]] = []
        for sharding in self._sharding_type_to_sharding.values():
            self._embedding_names_per_sharding.append(sharding.embedding_names())

    def _create_input_dist(
        self,
        input_feature_names: List[str],
    ) -> None:
        feature_names: List[str] = []
        self._feature_splits: List[int] = []
        for sharding in self._sharding_type_to_sharding.values():
            self._input_dists.append(sharding.create_input_dist())
            feature_names.extend(sharding.id_list_feature_names())
            self._feature_splits.append(len(sharding.id_list_feature_names()))
        self._features_order: List[int] = []
        for f in feature_names:
            self._features_order.append(input_feature_names.index(f))
        self._features_order = (
            []
            if self._features_order == list(range(len(self._features_order)))
            else self._features_order
        )
        self.register_buffer(
            "_features_order_tensor",
            torch.tensor(self._features_order, device=self._device, dtype=torch.int32),
        )

    def _create_lookups(self, fused_params: Optional[Dict[str, Any]]) -> None:
        for sharding in self._sharding_type_to_sharding.values():
            self._lookups.append(sharding.create_lookup(fused_params=fused_params))

    def _create_output_dist(
        self,
    ) -> None:
        for sharding in self._sharding_type_to_sharding.values():
            self._output_dists.append(sharding.create_output_dist())

    # pyre-ignore [14]
    def input_dist(
        self,
        ctx: EmbeddingCollectionContext,
        features: KeyedJaggedTensor,
    ) -> Awaitable[SparseFeaturesList]:
        if self._has_uninitialized_input_dist:
            self._create_input_dist(
                input_feature_names=features.keys() if features is not None else []
            )
            self._has_uninitialized_input_dist = False
        with torch.no_grad():
            features_by_sharding = []
            if self._features_order:
                features = features.permute(
                    self._features_order,
                    # pyre-ignore [6]
                    self._features_order_tensor,
                )
            features_by_sharding = features.split(
                self._feature_splits,
            )
            # save input splits and output splits in sharding context which
            # will be reused in sequence embedding all2all
            awaitables = []
            for module, features in zip(self._input_dists, features_by_sharding):
                tensor_awaitable = module(
                    SparseFeatures(
                        id_list_features=features,
                        id_score_list_features=None,
                    )
                )
                tensor_awaitable = tensor_awaitable.wait()  # finish lengths all2all
                input_splits = []
                output_splits = []
                if isinstance(tensor_awaitable, SparseFeaturesIndicesAwaitable):
                    input_splits = (
                        # pyre-fixme[16]: `Optional` has no attribute
                        #  `_in_lengths_per_worker`.
                        tensor_awaitable._id_list_features_awaitable._in_lengths_per_worker
                    )
                    output_splits = (
                        # pyre-fixme[16]: `Optional` has no attribute
                        #  `_out_lengths_per_worker`.
                        tensor_awaitable._id_list_features_awaitable._out_lengths_per_worker
                    )
                ctx.sharding_contexts.append(
                    SequenceShardingContext(
                        features_before_input_dist=features,
                        input_splits=input_splits,
                        output_splits=output_splits,
                        unbucketize_permute_tensor=module.unbucketize_permute_tensor
                        if isinstance(module, RwSparseFeaturesDist)
                        else None,
                    )
                )
                awaitables.append(tensor_awaitable)
            return SparseFeaturesListAwaitable(awaitables)

    def compute(
        self, ctx: ShardedModuleContext, dist_input: SparseFeaturesList
    ) -> List[torch.Tensor]:
        ret: List[torch.Tensor] = []
        for lookup, features, sharding_ctx in zip(
            self._lookups,
            dist_input,
            # pyre-ignore [16]
            ctx.sharding_contexts,
        ):
            sharding_ctx.lengths_after_input_dist = (
                features.id_list_features.lengths().view(
                    -1, features.id_list_features.stride()
                )
            )
            ret.append(lookup(features).view(-1, self._embedding_dim))
        return ret

    def output_dist(
        self, ctx: ShardedModuleContext, output: List[torch.Tensor]
    ) -> LazyAwaitable[Dict[str, torch.Tensor]]:
        awaitables_per_sharding: List[Awaitable[Dict[str, JaggedTensor]]] = []
        features_before_all2all_per_sharding: List[KeyedJaggedTensor] = []
        for odist, embeddings, sharding_ctx in zip(
            self._output_dists,
            output,
            # pyre-ignore [16]
            ctx.sharding_contexts,
        ):
            awaitables_per_sharding.append(odist(embeddings, sharding_ctx))
            features_before_all2all_per_sharding.append(
                sharding_ctx.features_before_input_dist
            )
        return EmbeddingCollectionAwaitable(
            awaitables_per_sharding=awaitables_per_sharding,
            features_per_sharding=features_before_all2all_per_sharding,
            embedding_names_per_sharding=self._embedding_names_per_sharding,
        )

    def compute_and_output_dist(
        self, ctx: ShardedModuleContext, input: SparseFeaturesList
    ) -> LazyAwaitable[Dict[str, torch.Tensor]]:
        awaitables_per_sharding: List[Awaitable[Dict[str, JaggedTensor]]] = []
        features_before_all2all_per_sharding: List[KeyedJaggedTensor] = []
        for lookup, odist, features, sharding_ctx in zip(
            self._lookups,
            self._output_dists,
            input,
            # pyre-ignore [16]
            ctx.sharding_contexts,
        ):
            sharding_ctx.lengths_after_input_dist = (
                features.id_list_features.lengths().view(
                    -1, features.id_list_features.stride()
                )
            )
            awaitables_per_sharding.append(
                odist(lookup(features).view(-1, self._embedding_dim), sharding_ctx)
            )
            features_before_all2all_per_sharding.append(
                sharding_ctx.features_before_input_dist
            )
        return EmbeddingCollectionAwaitable(
            awaitables_per_sharding=awaitables_per_sharding,
            features_per_sharding=features_before_all2all_per_sharding,
            embedding_names_per_sharding=self._embedding_names_per_sharding,
        )

    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        if destination is None:
            destination = OrderedDict()
            # pyre-ignore [16]
            destination._metadata = OrderedDict()
        for lookup in self._lookups:
            lookup.state_dict(destination, prefix + "embeddings.", keep_vars)
        return destination

    def named_modules(
        self,
        memo: Optional[Set[nn.Module]] = None,
        prefix: str = "",
        remove_duplicate: bool = True,
    ) -> Iterator[Tuple[str, nn.Module]]:
        yield from [(prefix, self)]

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        for lookup in self._lookups:
            yield from lookup.named_parameters(
                append_prefix(prefix, "embeddings"), recurse
            )

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        for lookup in self._lookups:
            yield from lookup.named_buffers(
                append_prefix(prefix, "embeddings"), recurse
            )

    def load_state_dict(
        self,
        state_dict: "OrderedDict[str, torch.Tensor]",
        strict: bool = True,
    ) -> _IncompatibleKeys:
        missing_keys = []
        unexpected_keys = []
        for lookup in self._lookups:
            missing, unexpected = lookup.load_state_dict(
                filter_state_dict(state_dict, "embeddings"),
                strict,
            )
            missing_keys.extend(missing)
            unexpected_keys.extend(unexpected)
        return _IncompatibleKeys(
            missing_keys=missing_keys, unexpected_keys=unexpected_keys
        )

    def sparse_grad_parameter_names(
        self,
        destination: Optional[List[str]] = None,
        prefix: str = "",
    ) -> List[str]:
        destination = [] if destination is None else destination
        for lookup in self._lookups:
            lookup.sparse_grad_parameter_names(
                destination, append_prefix(prefix, "embeddings")
            )
        return destination

    def sharded_parameter_names(self, prefix: str = "") -> Iterator[str]:
        for lookup, sharding_type in zip(
            self._lookups, self._sharding_type_to_sharding.keys()
        ):
            if sharding_type == ShardingType.DATA_PARALLEL.value:
                continue
            for name, _ in lookup.named_parameters(append_prefix(prefix, "embeddings")):
                yield name

    @property
    def fused_optimizer(self) -> KeyedOptimizer:
        return self._optim

    def create_context(self) -> ShardedModuleContext:
        return EmbeddingCollectionContext(sharding_contexts=[])


M = TypeVar("M", bound=nn.Module)


class EmbeddingCollectionSharder(BaseEmbeddingSharder[M]):
    """
    This implementation uses non-fused EmbeddingCollection
    """

    def shard(
        self,
        module: EmbeddingCollection,
        params: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
    ) -> ShardedEmbeddingCollection:
        return ShardedEmbeddingCollection(
            module, params, env, self.fused_params, device
        )

    def shardable_parameters(
        self, module: EmbeddingCollection
    ) -> Dict[str, nn.Parameter]:
        return {
            name.split(".")[0]: param
            for name, param in module.embeddings.named_parameters()
        }

    @property
    def module_type(self) -> Type[EmbeddingCollection]:
        return EmbeddingCollection
