#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import math
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from itertools import accumulate
from typing import Any, cast, DefaultDict, Dict, Iterator, List, Optional, Type

import torch

from torch import nn
from torch.distributed._shard.sharded_tensor import Shard
from torchrec.distributed.comm import get_local_rank
from torchrec.distributed.embedding import (
    EmbeddingCollectionAwaitable,
    EmbeddingCollectionContext,
    EmbeddingCollectionSharder,
)
from torchrec.distributed.embedding_sharding import (
    EmbeddingSharding,
    EmbeddingShardingContext,
    EmbeddingShardingInfo,
    KJTListSplitsAwaitable,
)
from torchrec.distributed.embedding_types import (
    BaseEmbeddingSharder,
    GroupedEmbeddingConfig,
    KJTList,
)
from torchrec.distributed.sharding.rw_sequence_sharding import (
    RwSequenceEmbeddingDist,
    RwSequenceEmbeddingSharding,
)
from torchrec.distributed.sharding.rw_sharding import (
    BaseRwEmbeddingSharding,
    RwSparseFeaturesDist,
)
from torchrec.distributed.sharding.sequence_sharding import SequenceShardingContext
from torchrec.distributed.types import (
    Awaitable,
    LazyAwaitable,
    ParameterSharding,
    QuantizedCommCodecs,
    ShardedModule,
    ShardedTensor,
    ShardingEnv,
    ShardingType,
    ShardMetadata,
)
from torchrec.distributed.utils import append_prefix
from torchrec.modules.embedding_configs import (
    DataType,
    EmbeddingTableConfig,
    PoolingType,
)
from torchrec.modules.managed_collision_modules import MCHManagedCollisionModule
from torchrec.modules.mc_embedding_modules import (
    apply_managed_collision_modules_to_kjt,
    ManagedCollisionCollection,
)
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor


def _calculate_rw_shard_sizes(
    hash_size: int,
    num_devices: int,
) -> List[int]:
    """
    see https://fburl.com/code/y94h768g
    """
    block_size: int = math.ceil(hash_size / num_devices)
    last_rank: int = hash_size // block_size
    last_block_size: int = hash_size - block_size * last_rank
    shard_sizes: List[int] = []
    for rank in range(num_devices):
        if rank < last_rank:
            local_row: int = block_size
        elif rank == last_rank:
            local_row: int = last_block_size
        else:
            local_row: int = 0
        shard_sizes.append(local_row)
    return shard_sizes


def _lengths_to_offsets(lengths: List[int], keep_last: bool = False) -> List[int]:
    assert len(lengths) > 0
    offsets = [0] + list(accumulate(lengths))
    if not keep_last:
        offsets.pop()
    return offsets


class ManagedCollisionCollectionAwaitable(EmbeddingCollectionAwaitable):
    pass


@dataclass
class ManagedCollisionCollectionContext(EmbeddingCollectionContext):
    pass


def create_sharding_infos_by_sharding(
    table_name_to_features: Dict[str, List[str]],
    table_name_to_parameter_sharding: Dict[str, ParameterSharding],
) -> Dict[str, List[EmbeddingShardingInfo]]:
    sharding_type_to_sharding_infos: Dict[str, List[EmbeddingShardingInfo]] = {}
    table_names = list(table_name_to_features.keys())
    for table_name in table_names:
        assert table_name in table_name_to_parameter_sharding
        parameter_sharding = table_name_to_parameter_sharding[table_name]
        if parameter_sharding.sharding_type not in sharding_type_to_sharding_infos:
            sharding_type_to_sharding_infos[parameter_sharding.sharding_type] = []

        sharding_type_to_sharding_infos[parameter_sharding.sharding_type].append(
            (
                EmbeddingShardingInfo(
                    embedding_config=EmbeddingTableConfig(
                        # NOTE placeholder as not actually used for materialization
                        num_embeddings=2**16,
                        # NOTE placeholder as not actually used for materialization
                        embedding_dim=2**3,
                        name=table_name,
                        data_type=DataType.FP32,
                        feature_names=copy.deepcopy(table_name_to_features[table_name]),
                        pooling=PoolingType.NONE,
                        embedding_names=copy.deepcopy(
                            table_name_to_features[table_name]
                        ),
                    ),
                    param_sharding=parameter_sharding,
                    param=nn.Parameter(
                        torch.empty((1, 1, 1), device=torch.device("meta")),
                        requires_grad=False,
                    ),
                )
            )
        )

    return sharding_type_to_sharding_infos


def create_embedding_sharding(
    sharding_type: str,
    sharding_infos: List[EmbeddingShardingInfo],
    env: ShardingEnv,
    device: Optional[torch.device] = None,
    qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
) -> EmbeddingSharding[
    SequenceShardingContext, KeyedJaggedTensor, torch.Tensor, torch.Tensor
]:
    if sharding_type == ShardingType.ROW_WISE.value:
        return RwSequenceEmbeddingSharding(
            sharding_infos=sharding_infos,
            env=env,
            device=device,
            qcomm_codecs_registry=qcomm_codecs_registry,
        )
    else:
        raise ValueError(f"Sharding not supported {sharding_type}")


class ShardedManagedCollisionCollection(
    ShardedModule[
        KJTList,
        KJTList,
        Dict[str, JaggedTensor],
        ManagedCollisionCollectionContext,
    ]
):
    def __init__(
        self,
        module: ManagedCollisionCollection,
        table_name_to_parameter_sharding: Dict[str, ParameterSharding],
        ec_sharder: EmbeddingCollectionSharder,
        # TODO - maybe we need this to manage unsharded/sharded consistency/state consistency
        env: ShardingEnv,
        device: torch.device,
        sharding_type_to_sharding: Optional[
            Dict[
                str,
                EmbeddingSharding[
                    EmbeddingShardingContext,
                    KeyedJaggedTensor,
                    torch.Tensor,
                    torch.Tensor,
                ],
            ]
        ] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__()

        self._device = device
        self._env = env
        self._table_name_to_parameter_sharding: Dict[
            str, ParameterSharding
        ] = copy.deepcopy(table_name_to_parameter_sharding)

        self._features_to_mc: Dict[str, str] = module._features_to_mc
        self._mc_to_features: Dict[str, List[str]] = module._mc_to_features

        if sharding_type_to_sharding is None:
            sharding_type_to_sharding_infos = create_sharding_infos_by_sharding(
                self._mc_to_features,
                self._table_name_to_parameter_sharding,
            )
            self._sharding_type_to_sharding: Dict[
                str,
                EmbeddingSharding[
                    SequenceShardingContext,
                    KeyedJaggedTensor,
                    torch.Tensor,
                    torch.Tensor,
                ],
            ] = {
                sharding_type: create_embedding_sharding(
                    sharding_type=sharding_type,
                    sharding_infos=embedding_confings,
                    env=env,
                    device=device,
                    qcomm_codecs_registry=qcomm_codecs_registry,
                )
                for sharding_type, embedding_confings in sharding_type_to_sharding_infos.items()
            }
        else:
            self._sharding_type_to_sharding = cast(
                Dict[
                    str,
                    EmbeddingSharding[
                        SequenceShardingContext,
                        KeyedJaggedTensor,
                        torch.Tensor,
                        torch.Tensor,
                    ],
                ],
                sharding_type_to_sharding,
            )

        self._embedding_names_per_sharding: List[List[str]] = []
        for sharding in self._sharding_type_to_sharding.values():
            self._embedding_names_per_sharding.append(sharding.embedding_names())

        self._has_uninitialized_input_dists: bool = True
        self._input_dists: List[nn.Module] = []
        self._managed_collision_modules = nn.ModuleDict()
        self._create_managed_collision_modules(module)
        self._output_dists: List[nn.Module] = []
        self._create_output_dists()

        self._initialize_torch_state()

    def _initialize_torch_state(self) -> None:
        self._model_parallel_mc_buffer_name_to_sharded_tensor = OrderedDict()
        for table_name, mc_module in self._managed_collision_modules.items():
            if (
                table_sharding_type := self._table_name_to_parameter_sharding[
                    table_name
                ].sharding_type
                == ShardingType.ROW_WISE.value
            ):
                mc_module_state_dict = mc_module.state_dict()
                prefix = table_name + "."
                for (
                    buffer_name,
                    buffer_tensor,
                ) in mc_module.named_buffers():
                    if buffer_name in mc_module_state_dict:
                        buffer_tensor = mc_module_state_dict[buffer_name]
                        if isinstance(mc_module, MCHManagedCollisionModule):
                            zch_block_sizes = self._mc_module_name_global_metadata[
                                table_name
                            ]["zch_block_sizes"]
                            assert zch_block_sizes[
                                self._env.rank
                            ] == buffer_tensor.size(0)
                            zch_block_offsets = self._mc_module_name_global_metadata[
                                table_name
                            ]["zch_block_offsets"]
                            self._model_parallel_mc_buffer_name_to_sharded_tensor[
                                prefix + buffer_name
                            ] = ShardedTensor._init_from_local_shards(
                                [
                                    Shard(
                                        tensor=buffer_tensor,
                                        metadata=ShardMetadata(
                                            shard_offsets=[
                                                zch_block_offsets[self._env.rank]
                                            ],
                                            shard_sizes=[buffer_tensor.size(0)],
                                            placement=(
                                                f"rank:{self._env.rank}/cuda:"
                                                f"{get_local_rank(self._env.world_size, self._env.rank)}"
                                            ),
                                        ),
                                    )
                                ],
                                torch.Size([zch_block_offsets[-1]]),
                                process_group=self._env.process_group,
                            )
                        else:
                            raise RuntimeError(
                                "RW-sharding is currently only supported "
                                f"for MCHManagedCollisionModules: {mc_module}"
                            )

            else:
                raise AssertionError(
                    f"Invalid MC table sharding_type: {table_sharding_type}."
                )

        def _post_state_dict_hook(
            module: ShardedManagedCollisionCollection,
            destination: Dict[str, torch.Tensor],
            prefix: str,
            _local_metadata: Dict[str, Any],
        ) -> None:
            for (
                mc_buffer_name,
                sharded_tensor,
            ) in module._model_parallel_mc_buffer_name_to_sharded_tensor.items():
                destination_key = f"{prefix}_managed_collision_modules.{mc_buffer_name}"
                destination[destination_key] = sharded_tensor

        def _load_state_dict_pre_hook(
            module: "ShardedManagedCollisionCollection",
            state_dict: Dict[str, Any],
            prefix: str,
            *args: Any,
        ) -> None:
            for (
                mc_buffer_name,
                _sharded_tensor,
            ) in module._model_parallel_mc_buffer_name_to_sharded_tensor.items():
                key = f"{prefix}_managed_collision_modules.{mc_buffer_name}"
                if key in state_dict:
                    if isinstance(state_dict[key], ShardedTensor):
                        local_shards = state_dict[key].local_shards()
                        state_dict[key] = local_shards[0].tensor
                    else:
                        raise RuntimeError(
                            f"Unexpected state_dict key type {type(state_dict[key])} found for {key}"
                        )

        self._register_state_dict_hook(_post_state_dict_hook)
        self._register_load_state_dict_pre_hook(
            _load_state_dict_pre_hook, with_module=True
        )

    def _create_managed_collision_modules(
        self, module: ManagedCollisionCollection
    ) -> None:
        self._mc_module_name_global_metadata: DefaultDict[
            str, DefaultDict[str, List[int]]
        ] = defaultdict(lambda: defaultdict(list))
        for sharding_type, sharding in self._sharding_type_to_sharding.items():
            if sharding_type == ShardingType.ROW_WISE.value:
                assert isinstance(sharding, BaseRwEmbeddingSharding)
                grouped_embedding_configs: List[
                    GroupedEmbeddingConfig
                ] = sharding._grouped_embedding_configs
                for group_config in grouped_embedding_configs:
                    for table in group_config.embedding_tables:
                        table_name = table.name
                        mc_module = module._managed_collision_modules[table_name]
                        block_sizes = _calculate_rw_shard_sizes(
                            mc_module._max_output_id, self._env.world_size
                        )
                        block_starting_offset = _lengths_to_offsets(
                            block_sizes, keep_last=False
                        )
                        # TODO make cleaner. add abstract/default method to mc_module
                        #   to update all appropiate internal values?
                        if isinstance(mc_module, MCHManagedCollisionModule):
                            zch_block_sizes = _calculate_rw_shard_sizes(
                                mc_module._zch_size, self._env.world_size
                            )
                            mc_module._zch_size = zch_block_sizes[self._env.rank]
                            zch_block_offsets = _lengths_to_offsets(
                                zch_block_sizes, keep_last=True
                            )
                            self._mc_module_name_global_metadata[table_name][
                                "zch_block_sizes"
                            ] = zch_block_sizes
                            self._mc_module_name_global_metadata[table_name][
                                "zch_block_offsets"
                            ] = zch_block_offsets
                        self._managed_collision_modules[
                            table_name
                        ] = mc_module.rebuild_with_max_output_id(
                            max_output_id=block_sizes[self._env.rank],
                            remapping_range_start_index=block_starting_offset[
                                self._env.rank
                            ],
                            device=self._device,
                        )

    def _create_input_dists(
        self,
        input_feature_names: List[str],
    ) -> None:
        feature_names: List[str] = []
        self._feature_splits: List[int] = []
        for sharding_type, sharding in self._sharding_type_to_sharding.items():
            if sharding_type == ShardingType.ROW_WISE.value:
                assert isinstance(sharding, BaseRwEmbeddingSharding)
                num_features = sharding._get_num_features()
                orig_feature_hash_sizes = sharding._get_feature_hash_sizes()
                assert len(orig_feature_hash_sizes) == num_features
                updated_feature_hash_sizes: List[int] = []
                grouped_embedding_configs: List[
                    GroupedEmbeddingConfig
                ] = sharding._grouped_embedding_configs
                for group_config in grouped_embedding_configs:
                    for table in group_config.embedding_tables:
                        table_name = table.name
                        max_input_id = self._managed_collision_modules[
                            table_name
                        ]._max_input_id
                        updated_feature_hash_sizes.extend(
                            table.num_features() * [max_input_id]
                        )
                assert len(updated_feature_hash_sizes) == num_features
                input_dist = RwSparseFeaturesDist(
                    # pyre-fixme[6]: For 1st param expected `ProcessGroup` but got
                    #  `Optional[ProcessGroup]`.
                    pg=sharding._pg,
                    num_features=sharding._get_num_features(),
                    feature_hash_sizes=updated_feature_hash_sizes,
                    device=sharding._device,
                    is_sequence=True,
                    has_feature_processor=False,
                    need_pos=False,
                )
                self._input_dists.append(input_dist)
                feature_names.extend(sharding.feature_names())
                self._feature_splits.append(len(sharding.feature_names()))
            else:
                raise RuntimeError(f"Invalid MC sharding_type: {sharding_type}.")

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

    def _create_output_dists(
        self,
    ) -> None:
        for sharding_type, sharding in self._sharding_type_to_sharding.items():
            if sharding_type == ShardingType.ROW_WISE.value:
                assert isinstance(sharding, BaseRwEmbeddingSharding)
                self._output_dists.append(
                    RwSequenceEmbeddingDist(
                        # pyre-fixme[6]: For 1st param expected `ProcessGroup` but got
                        #  `Optional[ProcessGroup]`.
                        sharding._pg,
                        sharding._get_num_features(),
                        sharding._device,
                        qcomm_codecs_registry=sharding.qcomm_codecs_registry,
                    )
                )
            else:
                raise RuntimeError(f"Invalid MC sharding_type: {sharding_type}.")

    # pyre-ignore
    def input_dist(
        self,
        ctx: ManagedCollisionCollectionContext,
        features: KeyedJaggedTensor,
        mc_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Awaitable[Awaitable[KJTList]]:
        self._mc_kwargs = mc_kwargs

        if self._has_uninitialized_input_dists:
            self._create_input_dists(input_feature_names=features.keys())
            self._has_uninitialized_input_dists = False

        with torch.no_grad():
            # NOTE shared features not currently supported
            features = apply_managed_collision_modules_to_kjt(
                features,
                self._managed_collision_modules,
                self._features_to_mc,
                mode="preprocess",
            )
            if self._features_order:
                features = features.permute(
                    self._features_order,
                    self._features_order_tensor,
                )
            features_by_sharding = features.split(
                self._feature_splits,
            )
            awaitables = []
            for input_dist, features in zip(self._input_dists, features_by_sharding):
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

    def _apply_mc_modules_to_kjt_list(
        self,
        dist_input: KJTList,
        mode: Optional[str] = None,
        mc_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> KJTList:
        return KJTList(
            [
                apply_managed_collision_modules_to_kjt(
                    features,
                    self._managed_collision_modules,
                    self._features_to_mc,
                    mode=mode,
                    mc_kwargs=mc_kwargs,
                )
                for features in dist_input
            ]
        )

    def _kjt_list_to_tensor_list(
        self,
        kjt_list: KJTList,
    ) -> List[torch.Tensor]:
        remapped_ids_ret: List[torch.Tensor] = []
        for kjt in kjt_list:
            remapped_ids_ret.append(kjt.values().view(-1, 1))
        return remapped_ids_ret

    def compute(
        self,
        ctx: ManagedCollisionCollectionContext,
        dist_input: KJTList,
    ) -> KJTList:
        mc_local_remapped_dist_input = self._apply_mc_modules_to_kjt_list(
            dist_input,
            mc_kwargs=self._mc_kwargs,
        )
        for features, sharding_ctx in zip(
            mc_local_remapped_dist_input,
            ctx.sharding_contexts,
        ):
            sharding_ctx.lengths_after_input_dist = features.lengths().view(
                -1, features.stride()
            )
        return mc_local_remapped_dist_input

    def output_dist(
        self,
        ctx: ManagedCollisionCollectionContext,
        output: KJTList,
    ) -> LazyAwaitable[Dict[str, JaggedTensor]]:
        local_remapped_kjt_list = output
        global_remapped_kjt_list = self._apply_mc_modules_to_kjt_list(
            local_remapped_kjt_list,
            mode="local_to_global",
        )
        global_remapped = self._kjt_list_to_tensor_list(global_remapped_kjt_list)
        awaitables_per_sharding: List[Awaitable[torch.Tensor]] = []
        features_before_all2all_per_sharding: List[KeyedJaggedTensor] = []
        for odist, remapped_ids, sharding_ctx in zip(
            self._output_dists,
            global_remapped,
            ctx.sharding_contexts,
        ):
            awaitables_per_sharding.append(odist(remapped_ids, sharding_ctx))
            features_before_all2all_per_sharding.append(
                # pyre-fixme[6]: For 1st argument expected `KeyedJaggedTensor` but
                #  got `Optional[KeyedJaggedTensor]`.
                sharding_ctx.features_before_input_dist
            )
        return ManagedCollisionCollectionAwaitable(
            awaitables_per_sharding=awaitables_per_sharding,
            features_per_sharding=features_before_all2all_per_sharding,
            embedding_names_per_sharding=self._embedding_names_per_sharding,
            need_indices=False,
            features_to_permute_indices=None,
        )

    def create_context(self) -> ManagedCollisionCollectionContext:
        return ManagedCollisionCollectionContext(sharding_contexts=[])

    def sharded_parameter_names(self, prefix: str = "") -> Iterator[str]:
        for fqn, _ in self.named_parameters():
            yield append_prefix(prefix, fqn)
        for fqn, _ in self.named_buffers():
            yield append_prefix(prefix, fqn)


class ManagedCollisionCollectionSharder(
    BaseEmbeddingSharder[ManagedCollisionCollection]
):
    def __init__(
        self,
        ec_sharder: Optional[EmbeddingCollectionSharder] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__(qcomm_codecs_registry=qcomm_codecs_registry)
        self._ec_sharder: EmbeddingCollectionSharder = (
            ec_sharder or EmbeddingCollectionSharder(self.qcomm_codecs_registry)
        )

    def shard(
        self,
        module: ManagedCollisionCollection,
        params: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        sharding_type_to_sharding: Optional[
            Dict[
                str,
                EmbeddingSharding[
                    EmbeddingShardingContext,
                    KeyedJaggedTensor,
                    torch.Tensor,
                    torch.Tensor,
                ],
            ]
        ] = None,
    ) -> ShardedManagedCollisionCollection:

        if device is None:
            device = torch.device("cuda")

        return ShardedManagedCollisionCollection(
            module,
            params,
            ec_sharder=self._ec_sharder,
            env=env,
            device=device,
            sharding_type_to_sharding=sharding_type_to_sharding,
        )

    def shardable_parameters(
        self, module: ManagedCollisionCollection
    ) -> Dict[str, torch.nn.Parameter]:
        return self._ec_sharder.shardable_parameters(module._embedding_collection)

    @property
    def module_type(self) -> Type[ManagedCollisionCollection]:
        return ManagedCollisionCollection

    def sharding_types(self, compute_device_type: str) -> List[str]:
        types = [
            ShardingType.ROW_WISE.value,
        ]
        return types
