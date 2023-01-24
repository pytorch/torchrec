#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import copy
from collections import defaultdict, deque, OrderedDict
from dataclasses import dataclass, field
from typing import Any, cast, Dict, List, MutableMapping, Optional, Type, Union

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torchrec.distributed.embedding_sharding import (
    EmbeddingSharding,
    EmbeddingShardingInfo,
    KJTListSplitsAwaitable,
)
from torchrec.distributed.embedding_types import (
    BaseEmbeddingSharder,
    EmbeddingComputeKernel,
    KJTList,
    ShardedEmbeddingModule,
    ShardingType,
)
from torchrec.distributed.sharding.cw_sequence_sharding import (
    CwSequenceEmbeddingSharding,
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
    ModuleShardingPlan,
    Multistreamable,
    ParameterSharding,
    QuantizedCommCodecs,
    ShardedTensor,
    ShardingEnv,
    ShardMetadata,
)
from torchrec.distributed.utils import (
    append_prefix,
    filter_state_dict,
    merge_fused_params,
    optimizer_type_to_emb_opt_type,
)
from torchrec.modules.embedding_configs import (
    EmbeddingConfig,
    EmbeddingTableConfig,
    PoolingType,
)
from torchrec.modules.embedding_modules import (
    EmbeddingCollection,
    EmbeddingCollectionInterface,
)
from torchrec.optim.fused import EmptyFusedOptimizer, FusedOptimizerModule
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizer
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor

try:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
except OSError:
    pass


def create_embedding_sharding(
    sharding_type: str,
    sharding_infos: List[EmbeddingShardingInfo],
    env: ShardingEnv,
    device: Optional[torch.device] = None,
    qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
) -> EmbeddingSharding[
    SequenceShardingContext, KeyedJaggedTensor, torch.Tensor, torch.Tensor
]:
    if sharding_type == ShardingType.TABLE_WISE.value:
        return TwSequenceEmbeddingSharding(
            sharding_infos=sharding_infos,
            env=env,
            device=device,
            qcomm_codecs_registry=qcomm_codecs_registry,
        )
    elif sharding_type == ShardingType.ROW_WISE.value:
        return RwSequenceEmbeddingSharding(
            sharding_infos=sharding_infos,
            env=env,
            device=device,
            qcomm_codecs_registry=qcomm_codecs_registry,
        )
    elif sharding_type == ShardingType.DATA_PARALLEL.value:
        return DpSequenceEmbeddingSharding(
            sharding_infos=sharding_infos,
            env=env,
            device=device,
        )
    elif sharding_type == ShardingType.COLUMN_WISE.value:
        return CwSequenceEmbeddingSharding(
            sharding_infos=sharding_infos,
            env=env,
            device=device,
            qcomm_codecs_registry=qcomm_codecs_registry,
        )
    else:
        raise ValueError(f"Sharding not supported {sharding_type}")


def create_sharding_infos_by_sharding(
    module: EmbeddingCollectionInterface,
    table_name_to_parameter_sharding: Dict[str, ParameterSharding],
    fused_params: Optional[Dict[str, Any]],
) -> Dict[str, List[EmbeddingShardingInfo]]:

    if fused_params is None:
        fused_params = {}

    sharding_type_to_sharding_infos: Dict[str, List[EmbeddingShardingInfo]] = {}
    # state_dict returns parameter.Tensor, which loses parameter level attributes
    parameter_by_name = dict(module.named_parameters())
    # QuantEBC registers weights as buffers (since they are INT8), and so we need to grab it there
    state_dict = module.state_dict()

    for (
        config,
        embedding_names,
    ) in zip(module.embedding_configs(), module.embedding_names_by_table()):
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
        assert param_name in parameter_by_name or param_name in state_dict
        param = parameter_by_name.get(param_name, state_dict[param_name])

        if parameter_sharding.sharding_type not in sharding_type_to_sharding_infos:
            sharding_type_to_sharding_infos[parameter_sharding.sharding_type] = []

        optimizer_params = getattr(param, "_optimizer_kwargs", [{}])
        optimizer_classes = getattr(param, "_optimizer_classes", [None])

        assert (
            len(optimizer_classes) == 1 and len(optimizer_params) == 1
        ), f"Only support 1 optimizer, given {len(optimizer_classes)}"

        optimizer_class = optimizer_classes[0]
        optimizer_params = optimizer_params[0]
        if optimizer_class:
            optimizer_params["optimizer"] = optimizer_type_to_emb_opt_type(
                optimizer_class
            )
        fused_params = merge_fused_params(fused_params, optimizer_params)

        sharding_type_to_sharding_infos[parameter_sharding.sharding_type].append(
            (
                EmbeddingShardingInfo(
                    embedding_config=EmbeddingTableConfig(
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
                    param_sharding=parameter_sharding,
                    param=param,
                    fused_params=fused_params,
                )
            )
        )
    return sharding_type_to_sharding_infos


def _construct_jagged_tensors(
    embeddings: torch.Tensor,
    features: KeyedJaggedTensor,
    embedding_names: List[str],
    need_indices: bool = False,
    features_to_permute_indices: Optional[Dict[str, List[int]]] = None,
) -> Dict[str, JaggedTensor]:
    ret: Dict[str, JaggedTensor] = {}
    stride = features.stride()
    length_per_key = features.length_per_key()
    values = features.values()

    lengths = features.lengths().view(-1, stride)
    lengths_tuple = torch.unbind(lengths.view(-1, stride), dim=0)
    embeddings_list = torch.split(embeddings, length_per_key, dim=0)
    values_list = torch.split(values, length_per_key) if need_indices else None

    key_indices = defaultdict(list)
    for i, key in enumerate(embedding_names):
        key_indices[key].append(i)
    for key, indices in key_indices.items():
        # combines outputs in correct order for CW sharding
        indices = (
            _permute_indices(indices, features_to_permute_indices[key])
            if features_to_permute_indices and key in features_to_permute_indices
            else indices
        )
        ret[key] = JaggedTensor(
            lengths=lengths_tuple[indices[0]],
            values=embeddings_list[indices[0]]
            if len(indices) == 1
            else torch.cat([embeddings_list[i] for i in indices], dim=1),
            weights=values_list[indices[0]] if values_list else None,
        )
    return ret


def _permute_indices(indices: List[int], permute: List[int]) -> List[int]:
    permuted_indices = [0] * len(indices)
    for i, permuted_index in enumerate(permute):
        permuted_indices[i] = indices[permuted_index]
    return permuted_indices


@dataclass
class EmbeddingCollectionContext(Multistreamable):
    sharding_contexts: List[SequenceShardingContext] = field(default_factory=list)

    def record_stream(self, stream: torch.cuda.streams.Stream) -> None:
        for ctx in self.sharding_contexts:
            ctx.record_stream(stream)


class EmbeddingCollectionAwaitable(LazyAwaitable[Dict[str, JaggedTensor]]):
    def __init__(
        self,
        awaitables_per_sharding: List[Awaitable[torch.Tensor]],
        features_per_sharding: List[KeyedJaggedTensor],
        embedding_names_per_sharding: List[List[str]],
        need_indices: bool = False,
        features_to_permute_indices: Optional[Dict[str, List[int]]] = None,
    ) -> None:
        super().__init__()
        self._awaitables_per_sharding = awaitables_per_sharding
        self._features_per_sharding = features_per_sharding
        self._need_indices = need_indices
        self._features_to_permute_indices = features_to_permute_indices
        self._embedding_names_per_sharding = embedding_names_per_sharding

    def _wait_impl(self) -> Dict[str, JaggedTensor]:
        jt_dict: Dict[str, JaggedTensor] = {}
        for w, f, e in zip(
            self._awaitables_per_sharding,
            self._features_per_sharding,
            self._embedding_names_per_sharding,
        ):
            jt_dict.update(
                _construct_jagged_tensors(
                    embeddings=w.wait(),
                    features=f,
                    embedding_names=e,
                    need_indices=self._need_indices,
                    features_to_permute_indices=self._features_to_permute_indices,
                )
            )
        return jt_dict


class ShardedEmbeddingCollection(
    ShardedEmbeddingModule[
        KJTList,
        List[torch.Tensor],
        Dict[str, JaggedTensor],
        EmbeddingCollectionContext,
    ],
    # TODO remove after compute_kernel X sharding decoupling
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
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__(qcomm_codecs_registry=qcomm_codecs_registry)
        self._embedding_configs: List[EmbeddingConfig] = module.embedding_configs()
        self._table_names: List[str] = [
            config.name for config in self._embedding_configs
        ]
        self.module_sharding_plan: ModuleShardingPlan = {
            table_name: parameter_sharding
            for table_name, parameter_sharding in table_name_to_parameter_sharding.items()
            if table_name in self._table_names
        }
        self._env = env
        sharding_type_to_sharding_infos = create_sharding_infos_by_sharding(
            module,
            table_name_to_parameter_sharding,
            fused_params,
        )
        self._sharding_type_to_sharding: Dict[
            str,
            EmbeddingSharding[
                SequenceShardingContext, KeyedJaggedTensor, torch.Tensor, torch.Tensor
            ],
        ] = {
            sharding_type: create_embedding_sharding(
                sharding_type=sharding_type,
                sharding_infos=embedding_confings,
                env=env,
                device=device,
                qcomm_codecs_registry=self.qcomm_codecs_registry,
            )
            for sharding_type, embedding_confings in sharding_type_to_sharding_infos.items()
        }

        self._device = device
        self._input_dists: List[nn.Module] = []
        self._lookups: List[nn.Module] = []
        self._create_lookups()
        self._output_dists: List[nn.Module] = []
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
                    params: MutableMapping[str, Union[torch.Tensor, ShardedTensor]] = {}
                    for param_key, weight in m.fused_optimizer.params.items():
                        params["embeddings." + param_key] = weight
                    m.fused_optimizer.params = params
                    optims.append(("", m.fused_optimizer))
        self._optim: CombinedOptimizer = CombinedOptimizer(optims)
        self._embedding_dim: int = module.embedding_dim()
        self._embedding_names_per_sharding: List[List[str]] = []
        for sharding in self._sharding_type_to_sharding.values():
            self._embedding_names_per_sharding.append(sharding.embedding_names())
        self._local_embedding_dim: int = self._embedding_dim
        self._features_to_permute_indices: Dict[str, List[int]] = {}
        if ShardingType.COLUMN_WISE.value in self._sharding_type_to_sharding:
            sharding = self._sharding_type_to_sharding[ShardingType.COLUMN_WISE.value]
            # CW partition must be same for all CW sharded parameters
            self._local_embedding_dim = cast(
                ShardMetadata, sharding.embedding_shard_metadata()[0]
            ).shard_sizes[1]
            self._generate_permute_indices_per_feature(
                module.embedding_configs(), table_name_to_parameter_sharding
            )
        self._need_indices: bool = module.need_indices()

        for index, (sharding, lookup) in enumerate(
            zip(
                self._sharding_type_to_sharding.values(),
                self._lookups,
            )
        ):
            # TODO: can move this into DpPooledEmbeddingSharding once all modules are composable
            if isinstance(sharding, DpSequenceEmbeddingSharding):
                self._lookups[index] = DistributedDataParallel(
                    module=lookup,
                    device_ids=[device]
                    if self._device and self._device.type == "gpu"
                    else None,
                    process_group=env.process_group,
                    gradient_as_bucket_view=True,
                    broadcast_buffers=True,
                    static_graph=True,
                )
        self._initialize_torch_state()

    @staticmethod
    def _pre_load_state_dict_hook(
        self: "ShardedEmbeddingCollection",
        state_dict: Dict[str, Any],
        prefix: str,
        *args: Any,
    ) -> None:
        """
        Modify the destination state_dict for model parallel
        to transform from ShardedTensors into tensors
        """
        for table_name in self._model_parallel_name_to_local_shards.keys():
            key = f"{prefix}embeddings.{table_name}.weight"
            local_shards = state_dict[key].local_shards()
            if len(local_shards) == 0:
                state_dict[key] = torch.empty(0)
            else:
                dim = state_dict[key].metadata().shards_metadata[0].shard_sizes[1]
                # CW multiple shards are merged
                state_dict[key] = torch.cat(
                    [s.tensor.view(-1) for s in local_shards], dim=0
                ).view(-1, dim)

    def _initialize_torch_state(self) -> None:  # noqa
        """
        This provides consistency between this class and the EmbeddingCollection's
        nn.Module API calls (state_dict, named_modules, etc)
        """

        self.embeddings: nn.ModuleDict = nn.ModuleDict()
        for table_name in self._table_names:
            self.embeddings[table_name] = nn.Module()
        self._model_parallel_name_to_local_shards = OrderedDict()
        self._model_parallel_name_to_sharded_tensor = OrderedDict()
        model_parallel_name_to_compute_kernel: Dict[str, str] = {}
        for (
            table_name,
            parameter_sharding,
        ) in self.module_sharding_plan.items():
            if parameter_sharding.sharding_type == ShardingType.DATA_PARALLEL.value:
                continue
            self._model_parallel_name_to_local_shards[table_name] = []
            model_parallel_name_to_compute_kernel[
                table_name
            ] = parameter_sharding.compute_kernel

        self._name_to_table_size = {}
        for table in self._embedding_configs:
            self._name_to_table_size[table.name] = (
                table.num_embeddings,
                table.embedding_dim,
            )

        for sharding_type, lookup in zip(
            self._sharding_type_to_sharding.keys(), self._lookups
        ):
            if sharding_type == ShardingType.DATA_PARALLEL.value:
                # unwrap DDP
                lookup = lookup.module
            else:
                # save local_shards for transforming MP params to shardedTensor
                for key, v in lookup.state_dict().items():
                    table_name = key[: -len(".weight")]
                    self._model_parallel_name_to_local_shards[table_name].extend(
                        v.local_shards()
                    )
            for (
                table_name,
                tbe_slice,
            ) in lookup.named_parameters_by_table():
                self.embeddings[table_name].register_parameter("weight", tbe_slice)
        for (
            table_name,
            local_shards,
        ) in self._model_parallel_name_to_local_shards.items():
            # for shards that don't exist on this rank, register with empty tensor
            if not hasattr(self.embeddings[table_name], "weight"):
                self.embeddings[table_name].register_parameter(
                    "weight", nn.Parameter(torch.empty(0))
                )
                if (
                    model_parallel_name_to_compute_kernel[table_name]
                    != EmbeddingComputeKernel.DENSE.value
                ):
                    self.embeddings[table_name].weight._in_backward_optimizers = [
                        EmptyFusedOptimizer()
                    ]
            # created ShardedTensors once in init, use in post_state_dict_hook
            self._model_parallel_name_to_sharded_tensor[
                table_name
            ] = ShardedTensor._init_from_local_shards(
                local_shards,
                self._name_to_table_size[table_name],
                process_group=self._env.process_group,
            )

        def post_state_dict_hook(
            module: ShardedEmbeddingCollection,
            destination: Dict[str, torch.Tensor],
            prefix: str,
            _local_metadata: Dict[str, Any],
        ) -> None:
            # Adjust dense MP
            for (
                table_name,
                sharded_t,
            ) in module._model_parallel_name_to_sharded_tensor.items():
                destination_key = f"{prefix}embeddings.{table_name}.weight"
                destination[destination_key] = sharded_t

        self._register_state_dict_hook(post_state_dict_hook)
        self._register_load_state_dict_pre_hook(
            self._pre_load_state_dict_hook, with_module=True
        )

    def _generate_permute_indices_per_feature(
        self,
        embedding_configs: List[EmbeddingConfig],
        table_name_to_parameter_sharding: Dict[str, ParameterSharding],
    ) -> None:
        """
        Generates permute indices per feature for column-wise sharding.

        Since outputs are stored in order of rank, column-wise shards of a table on the
        same rank will be seen as adjacent, which may not be correct.

        The permute indices store the correct ordering of outputs relative to the
        provided ordering.

        Example::
            rank_0 = [f_0(shard_0), f_0(shard_2)]
            rank_1 = [f_0(shard_1)]
            output = [f_0(shard_0), f_0(shard_2), f_0(shard_1)]

            shard_ranks = [0, 1, 0]
            output_ranks = [0, 0, 1]

            # To get the correct order from output_ranks -> shard_ranks
            permute_indices = [0, 2, 1]
        """
        shared_feature: Dict[str, bool] = {}
        for table in embedding_configs:
            for feature_name in table.feature_names:
                if feature_name not in shared_feature:
                    shared_feature[feature_name] = False
                else:
                    shared_feature[feature_name] = True

        for table in embedding_configs:
            sharding = table_name_to_parameter_sharding[table.name]
            if sharding.sharding_type != ShardingType.COLUMN_WISE.value:
                continue
            ranks = cast(List[int], sharding.ranks)
            rank_to_indices = defaultdict(deque)
            for i, rank in enumerate(sorted(ranks)):
                rank_to_indices[rank].append(i)
            permute_indices = [rank_to_indices[rank].popleft() for rank in ranks]
            for feature_name in table.feature_names:
                if shared_feature[feature_name]:
                    self._features_to_permute_indices[
                        feature_name + "@" + table.name
                    ] = permute_indices
                else:
                    self._features_to_permute_indices[feature_name] = permute_indices

    def _create_input_dist(
        self,
        input_feature_names: List[str],
    ) -> None:
        feature_names: List[str] = []
        self._feature_splits: List[int] = []
        for sharding in self._sharding_type_to_sharding.values():
            self._input_dists.append(sharding.create_input_dist())
            feature_names.extend(sharding.feature_names())
            self._feature_splits.append(len(sharding.feature_names()))
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
            persistent=False,
        )

    def _create_lookups(self) -> None:
        for sharding in self._sharding_type_to_sharding.values():
            self._lookups.append(sharding.create_lookup())

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
    ) -> Awaitable[Awaitable[KJTList]]:
        if self._has_uninitialized_input_dist:
            self._create_input_dist(input_feature_names=features.keys())
            self._has_uninitialized_input_dist = False
        with torch.no_grad():
            if self._features_order:
                features = features.permute(
                    self._features_order,
                    # pyre-ignore [6]
                    self._features_order_tensor,
                )
            features_by_shards = features.split(
                self._feature_splits,
            )
            awaitables = []
            for input_dist, features in zip(self._input_dists, features_by_shards):
                awaitables.append(input_dist(features))
                ctx.sharding_contexts.append(
                    SequenceShardingContext(
                        features_before_input_dist=features,
                        unbucketize_permute_tensor=input_dist.unbucketize_permute_tensor
                        if isinstance(input_dist, RwSparseFeaturesDist)
                        else None,
                    )
                )
        return KJTListSplitsAwaitable(awaitables, ctx)

    def compute(
        self, ctx: EmbeddingCollectionContext, dist_input: KJTList
    ) -> List[torch.Tensor]:
        ret: List[torch.Tensor] = []
        for lookup, features, sharding_ctx, sharding_type in zip(
            self._lookups,
            dist_input,
            ctx.sharding_contexts,
            self._sharding_type_to_sharding,
        ):
            sharding_ctx.lengths_after_input_dist = features.lengths().view(
                -1, features.stride()
            )
            embedding_dim = self._embedding_dim_for_sharding_type(sharding_type)
            ret.append(lookup(features).view(-1, embedding_dim))
        return ret

    def output_dist(
        self, ctx: EmbeddingCollectionContext, output: List[torch.Tensor]
    ) -> LazyAwaitable[Dict[str, JaggedTensor]]:
        awaitables_per_sharding: List[Awaitable[torch.Tensor]] = []
        features_before_all2all_per_sharding: List[KeyedJaggedTensor] = []
        for odist, embeddings, sharding_ctx in zip(
            self._output_dists,
            output,
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
            need_indices=self._need_indices,
            features_to_permute_indices=self._features_to_permute_indices,
        )

    def compute_and_output_dist(
        self, ctx: EmbeddingCollectionContext, input: KJTList
    ) -> LazyAwaitable[Dict[str, JaggedTensor]]:
        awaitables_per_sharding: List[Awaitable[torch.Tensor]] = []
        features_before_all2all_per_sharding: List[KeyedJaggedTensor] = []
        for lookup, odist, features, sharding_ctx, sharding_type in zip(
            self._lookups,
            self._output_dists,
            input,
            ctx.sharding_contexts,
            self._sharding_type_to_sharding,
        ):
            sharding_ctx.lengths_after_input_dist = features.lengths().view(
                -1, features.stride()
            )
            embedding_dim = self._embedding_dim_for_sharding_type(sharding_type)
            awaitables_per_sharding.append(
                odist(lookup(features).view(-1, embedding_dim), sharding_ctx)
            )
            features_before_all2all_per_sharding.append(
                sharding_ctx.features_before_input_dist
            )
        return EmbeddingCollectionAwaitable(
            awaitables_per_sharding=awaitables_per_sharding,
            features_per_sharding=features_before_all2all_per_sharding,
            embedding_names_per_sharding=self._embedding_names_per_sharding,
            need_indices=self._need_indices,
            features_to_permute_indices=self._features_to_permute_indices,
        )

    def _embedding_dim_for_sharding_type(self, sharding_type: str) -> int:
        return (
            self._local_embedding_dim
            if sharding_type == ShardingType.COLUMN_WISE.value
            else self._embedding_dim
        )

    @property
    def fused_optimizer(self) -> KeyedOptimizer:
        return self._optim

    def create_context(self) -> EmbeddingCollectionContext:
        return EmbeddingCollectionContext(sharding_contexts=[])


class EmbeddingCollectionSharder(BaseEmbeddingSharder[EmbeddingCollection]):
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
            module,
            params,
            env,
            self.fused_params,
            device,
            qcomm_codecs_registry=self.qcomm_codecs_registry,
        )

    def shardable_parameters(
        self, module: EmbeddingCollection
    ) -> Dict[str, nn.Parameter]:
        return {
            name.split(".")[0]: param
            for name, param in module.embeddings.named_parameters()
        }

    def sharding_types(self, compute_device_type: str) -> List[str]:
        types = [
            ShardingType.DATA_PARALLEL.value,
            ShardingType.TABLE_WISE.value,
            ShardingType.COLUMN_WISE.value,
            ShardingType.ROW_WISE.value,
        ]
        return types

    @property
    def module_type(self) -> Type[EmbeddingCollection]:
        return EmbeddingCollection
