#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

import torch
from torch import nn
from torch.nn.modules.module import _IncompatibleKeys
from torchrec.distributed.embedding import (
    create_sharding_infos_by_sharding,
    EmbeddingShardingInfo,
)
from torchrec.distributed.embedding_sharding import (
    EmbeddingSharding,
    ListOfSparseFeaturesListAwaitable,
)
from torchrec.distributed.embedding_types import (
    BaseQuantEmbeddingSharder,
    ListOfSparseFeaturesList,
    ShardingType,
    SparseFeatures,
    SparseFeaturesList,
)
from torchrec.distributed.sharding.sequence_sharding import InferSequenceShardingContext
from torchrec.distributed.sharding.tw_sequence_sharding import (
    InferTwSequenceEmbeddingSharding,
)
from torchrec.distributed.types import (
    Awaitable,
    FeatureShardingMixIn,
    LazyAwaitable,
    ParameterSharding,
    ShardedModule,
    ShardingEnv,
)
from torchrec.distributed.utils import filter_state_dict
from torchrec.modules.embedding_configs import (
    data_type_to_sparse_type,
    dtype_to_data_type,
)
from torchrec.quant.embedding_modules import (
    EmbeddingCollection as QuantEmbeddingCollection,
)
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor
from torchrec.streamable import Multistreamable

try:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
except OSError:
    pass


@dataclass
class EmbeddingCollectionContext(Multistreamable):
    sharding_contexts: List[InferSequenceShardingContext]

    def record_stream(self, stream: torch.cuda.streams.Stream) -> None:
        for ctx in self.sharding_contexts:
            ctx.record_stream(stream)


def create_infer_embedding_sharding(
    sharding_type: str,
    sharding_infos: List[EmbeddingShardingInfo],
    env: ShardingEnv,
    device: Optional[torch.device] = None,
) -> EmbeddingSharding[SparseFeaturesList, List[torch.Tensor]]:
    if sharding_type == ShardingType.TABLE_WISE.value:
        return InferTwSequenceEmbeddingSharding(sharding_infos, env, device)
    else:
        raise ValueError(f"Sharding type not supported {sharding_type}")


def _construct_jagged_tensors(
    embeddings: torch.Tensor,
    features: KeyedJaggedTensor,
    need_indices: bool = False,
) -> Dict[str, JaggedTensor]:
    # ignore cw consideration for inference now.
    ret: Dict[str, JaggedTensor] = {}
    lengths = features.lengths().view(-1, features.stride())
    values = features.values()
    length_per_key = features.length_per_key()
    values_list = torch.split(values, length_per_key) if need_indices else None
    embeddings_list = torch.split(embeddings, length_per_key, dim=0)
    stride = features.stride()
    lengths_tuple = torch.unbind(lengths.view(-1, stride), dim=0)
    for i, key in enumerate(features.keys()):
        ret[key] = JaggedTensor(
            lengths=lengths_tuple[i],
            values=embeddings_list[i],
            # pyre-fixme[16]: `Optional` has no attribute `__getitem__`.
            weights=values_list[i] if need_indices else None,
        )
    return ret


class EmbeddingCollectionAwaitable(LazyAwaitable[Dict[str, JaggedTensor]]):
    def __init__(
        self,
        awaitables_per_sharding: List[Awaitable[List[torch.Tensor]]],
        features_per_sharding: List[List[KeyedJaggedTensor]],
        need_indices: bool = False,
    ) -> None:
        super().__init__()
        self._awaitables_per_sharding: List[
            Awaitable[List[torch.Tensor]]
        ] = awaitables_per_sharding
        self._features_per_sharding: List[
            List[KeyedJaggedTensor]
        ] = features_per_sharding
        self._need_indices = need_indices

    def _wait_impl(self) -> Dict[str, JaggedTensor]:
        jt_dict: Dict[str, JaggedTensor] = {}
        for w_sharding, f_sharding in zip(
            self._awaitables_per_sharding,
            self._features_per_sharding,
        ):
            emb_sharding = w_sharding.wait()
            for emb, f in zip(emb_sharding, f_sharding):
                jt_dict.update(
                    _construct_jagged_tensors(
                        embeddings=emb,
                        features=f,
                        need_indices=self._need_indices,
                    )
                )
        return jt_dict


class ShardedQuantEmbeddingCollection(
    ShardedModule[
        ListOfSparseFeaturesList,
        List[List[torch.Tensor]],
        Dict[str, JaggedTensor],
        EmbeddingCollectionContext,
    ],
):
    """
    Sharded implementation of `QuantEmbeddingCollection`.
    """

    def __init__(
        self,
        module: QuantEmbeddingCollection,
        table_name_to_parameter_sharding: Dict[str, ParameterSharding],
        env: ShardingEnv,
        fused_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        sharding_type_to_sharding_infos = create_sharding_infos_by_sharding(
            module, table_name_to_parameter_sharding, fused_params
        )
        self._sharding_type_to_sharding: Dict[
            str, EmbeddingSharding[SparseFeaturesList, List[torch.Tensor]]
        ] = {
            sharding_type: create_infer_embedding_sharding(
                sharding_type, embedding_confings, env
            )
            for sharding_type, embedding_confings in sharding_type_to_sharding_infos.items()
        }

        self._input_dists: nn.ModuleList = nn.ModuleList()
        self._lookups: nn.ModuleList = nn.ModuleList()
        self._create_lookups(fused_params)
        self._output_dists: nn.ModuleList = nn.ModuleList()

        self._feature_splits: List[int] = []
        self._features_order: List[int] = []

        self._has_uninitialized_input_dist: bool = True
        self._has_uninitialized_output_dist: bool = True

        self._embedding_dim: int = module.embedding_dim()
        self._need_indices: bool = module.need_indices()

    def _create_input_dist(
        self,
        input_feature_names: List[str],
        device: torch.device,
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
            torch.tensor(self._features_order, device=device, dtype=torch.int32),
        )

    def _create_lookups(self, fused_params: Optional[Dict[str, Any]]) -> None:
        for sharding in self._sharding_type_to_sharding.values():
            self._lookups.append(sharding.create_lookup(fused_params=fused_params))

    def _create_output_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> None:
        for sharding in self._sharding_type_to_sharding.values():
            self._output_dists.append(sharding.create_output_dist(device))

    # pyre-ignore [3, 14]
    def input_dist(
        self,
        ctx: EmbeddingCollectionContext,
        features: KeyedJaggedTensor,
    ) -> Awaitable[Any]:
        if self._has_uninitialized_input_dist:
            self._create_input_dist(
                input_feature_names=features.keys() if features is not None else [],
                device=features.device(),
            )
            self._has_uninitialized_input_dist = False
        if self._has_uninitialized_output_dist:
            self._create_output_dist(features.device())
            self._has_uninitialized_output_dist = False
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
                ).wait()  # a dummy wait since now length indices comm is splited
                awaitables.append(tensor_awaitable)
            return ListOfSparseFeaturesListAwaitable(awaitables)

    def compute(
        self, ctx: EmbeddingCollectionContext, dist_input: ListOfSparseFeaturesList
    ) -> List[List[torch.Tensor]]:
        ret: List[List[torch.Tensor]] = []
        for lookup, features in zip(
            self._lookups,
            dist_input,
        ):
            ctx.sharding_contexts.append(
                InferSequenceShardingContext(
                    features=[feature.id_list_features for feature in features],
                )
            )
            ret.append([o.view(-1, self._embedding_dim) for o in lookup(features)])
        return ret

    def output_dist(
        self, ctx: EmbeddingCollectionContext, output: List[List[torch.Tensor]]
    ) -> LazyAwaitable[Dict[str, JaggedTensor]]:
        awaitables_per_sharding: List[Awaitable[List[torch.Tensor]]] = []
        features_per_sharding: List[List[KeyedJaggedTensor]] = []
        for odist, embeddings, sharding_ctx in zip(
            self._output_dists,
            output,
            ctx.sharding_contexts,
        ):
            awaitables_per_sharding.append(odist(embeddings, sharding_ctx))
            features_per_sharding.append(sharding_ctx.features)
        return EmbeddingCollectionAwaitable(
            awaitables_per_sharding=awaitables_per_sharding,
            features_per_sharding=features_per_sharding,
            need_indices=self._need_indices,
        )

    def compute_and_output_dist(
        self, ctx: EmbeddingCollectionContext, input: ListOfSparseFeaturesList
    ) -> LazyAwaitable[Dict[str, JaggedTensor]]:
        return self.output_dist(ctx, self.compute(ctx, input))

    # pyre-fixme[14]: `state_dict` overrides method defined in `Module` inconsistently.
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

    # pyre-fixme[14]: `load_state_dict` overrides method defined in `Module`
    #  inconsistently.
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

    def copy(self, device: torch.device) -> nn.Module:
        if self._has_uninitialized_output_dist:
            self._create_output_dist(device)
            self._has_uninitialized_output_dist = False
        return self

    def create_context(self) -> EmbeddingCollectionContext:
        return EmbeddingCollectionContext(sharding_contexts=[])

    @property
    def shardings(self) -> Dict[str, FeatureShardingMixIn]:
        # pyre-ignore [7]
        return self._sharding_type_to_sharding


class QuantEmbeddingCollectionSharder(
    BaseQuantEmbeddingSharder[QuantEmbeddingCollection]
):
    """
    This implementation uses non-fused EmbeddingCollection
    """

    def shard(
        self,
        module: QuantEmbeddingCollection,
        params: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
    ) -> ShardedQuantEmbeddingCollection:
        fused_params = self.fused_params if self.fused_params else {}
        fused_params["output_dtype"] = data_type_to_sparse_type(
            dtype_to_data_type(module.output_dtype())
        )
        return ShardedQuantEmbeddingCollection(module, params, env, fused_params)

    def shardable_parameters(
        self, module: QuantEmbeddingCollection
    ) -> Dict[str, nn.Parameter]:
        return {
            name.split(".")[-2]: param
            for name, param in module.state_dict().items()
            if name.endswith(".weight")
        }

    @property
    def module_type(self) -> Type[QuantEmbeddingCollection]:
        return QuantEmbeddingCollection
