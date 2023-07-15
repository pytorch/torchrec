#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Iterator, List, Optional, Type

import torch
from torch import nn
from torchrec.distributed.embedding_lookup import GroupedPooledEmbeddingsLookup

from torchrec.distributed.embedding_types import (
    BaseEmbeddingSharder,
    KJTList,
    ShardedEmbeddingModule,
)
from torchrec.distributed.embeddingbag import (
    EmbeddingBagCollectionContext,
    EmbeddingBagCollectionSharder,
    ShardedEmbeddingBagCollection,
)
from torchrec.distributed.sharding.rw_sharding import RwSparseFeaturesDist
from torchrec.distributed.types import (
    Awaitable,
    LazyAwaitable,
    ParameterSharding,
    QuantizedCommCodecs,
    ShardingEnv,
    ShardingType,
)
from torchrec.distributed.utils import append_prefix
from torchrec.modules.mc_embedding_modules import (
    apply_managed_collision_modules_to_kjt,
    ManagedCollisionEmbeddingBagCollection,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor


class ShardedManagedCollisionEmbeddingBagCollection(
    ShardedEmbeddingModule[
        KJTList, List[torch.Tensor], KeyedTensor, EmbeddingBagCollectionContext
    ]
):
    def __init__(
        self,
        module: ManagedCollisionEmbeddingBagCollection,
        table_name_to_parameter_sharding: Dict[str, ParameterSharding],
        ebc_sharder: EmbeddingBagCollectionSharder,
        # pyre-ignore
        managed_collision_module_sharders: Optional[List],
        # TODO - maybe we need this to manage unsharded/sharded consistency/state consistency
        env: ShardingEnv,
        device: torch.device,
    ) -> None:
        super().__init__()

        self._device = device
        self._env = env

        self._embedding_bag_collection: ShardedEmbeddingBagCollection = (
            ebc_sharder.shard(
                module._embedding_bag_collection,
                table_name_to_parameter_sharding,
                env=env,
                device=device,
            )
        )

        self._features_to_tables: Dict[str, str] = module._features_to_tables

        self._managed_collision_modules = nn.ModuleDict()
        for table_name, mc_module in module._managed_collision_modules.items():
            if (
                table_name_to_parameter_sharding[table_name].sharding_type
                == ShardingType.ROW_WISE.value
            ):
                max_output_id = (
                    mc_module._max_output_id + self._env.world_size - 1
                ) // self._env.world_size
                self._managed_collision_modules[
                    table_name
                ] = mc_module.rebuild_with_max_output_id(
                    max_output_id, device=self._device
                )
            elif (
                table_name_to_parameter_sharding[table_name].sharding_type
                == ShardingType.TABLE_WISE.value
            ):
                max_output_id = mc_module._max_output_id
                self._managed_collision_modules[
                    table_name
                ] = mc_module.rebuild_with_max_output_id(
                    max_output_id, device=self._device
                )

        # TODO BELOW IS HACKY - JUST FOR MVP
        self._rw_feature_hash_sizes: List[int] = []
        for table, parameter_sharding in table_name_to_parameter_sharding.items():
            if parameter_sharding.sharding_type == ShardingType.ROW_WISE.value:
                max_input_id = self._managed_collision_modules[table]._max_input_id
                rw_feature_block_size = (
                    max_input_id + self._env.world_size - 1
                ) // self._env.world_size
                self._rw_feature_hash_sizes.append(rw_feature_block_size)

        # pyre-ignore
        self._table_to_tbe_and_index = {}
        for lookup in self._embedding_bag_collection._lookups:
            if isinstance(lookup, GroupedPooledEmbeddingsLookup):
                for emb_module in lookup._emb_modules:
                    for table_idx, table in enumerate(
                        emb_module._config.embedding_tables
                    ):
                        self._table_to_tbe_and_index[table.name] = (
                            emb_module._emb_module,
                            torch.tensor(
                                [table_idx], dtype=torch.int, device=self._device
                            ),
                        )
        self._buffer_ids: torch.Tensor = torch.tensor(
            [0], device=self._device, dtype=torch.int
        )

    # pyre-ignore
    def input_dist(
        self, ctx: EmbeddingBagCollectionContext, features: KeyedJaggedTensor
    ) -> Awaitable[Awaitable[KJTList]]:
        if self._embedding_bag_collection._has_uninitialized_input_dist:
            # PART OF THE HACK FOR MVAP
            self._embedding_bag_collection._create_input_dist(features.keys())
            self._embedding_bag_collection._has_uninitialized_input_dist = False

            for input_dist in self._embedding_bag_collection._input_dists:
                if isinstance(input_dist, RwSparseFeaturesDist):
                    print(
                        "overriding with feature block size",
                        self._rw_feature_hash_sizes,
                    )
                    input_dist.register_buffer(
                        "_feature_block_sizes_tensor",
                        torch.tensor(
                            self._rw_feature_hash_sizes,
                            device=self._device,
                            dtype=torch.int32,
                        ),
                    )

        return self._embedding_bag_collection.input_dist(ctx, features)

    def _apply_mc_modules_to_kjt_list(self, dist_input: KJTList) -> KJTList:
        return KJTList(
            [
                apply_managed_collision_modules_to_kjt(
                    features,
                    self._managed_collision_modules,
                    self._features_to_tables,
                )
                for features in dist_input
            ]
        )

    def _evict(self, evictions_per_table: Dict[str, Optional[torch.Tensor]]) -> None:
        for table, evictions_indices_for_table in evictions_per_table.items():
            if evictions_indices_for_table is not None:
                (tbe, logical_table_ids) = self._table_to_tbe_and_index[table]
                pruned_indices_offsets = torch.tensor(
                    [0, evictions_indices_for_table.shape[0]],
                    dtype=torch.long,
                    device=self._device,
                )

                with torch.no_grad():
                    # embeddings, and optimizer state will be reset
                    tbe.reset_embedding_weight_momentum(
                        pruned_indices=evictions_indices_for_table.long(),
                        pruned_indices_offsets=pruned_indices_offsets,
                        logical_table_ids=logical_table_ids,
                        buffer_ids=self._buffer_ids,
                    )
                    table_weight_param = (
                        self._embedding_bag_collection.embedding_bags.get_parameter(
                            f"{table}.weight"
                        )
                    )

                    init_fn = self._embedding_bag_collection._table_name_to_config[
                        table
                    ].init_fn

                    # pyre-ignore
                    # Set evicted indices to original init_fn instead of all zeros
                    table_weight_param[evictions_indices_for_table] = init_fn(
                        table_weight_param[evictions_indices_for_table]
                    )

    def compute(
        self,
        ctx: EmbeddingBagCollectionContext,
        dist_input: KJTList,
    ) -> List[torch.Tensor]:

        evictions_per_table: Dict[str, Optional[torch.Tensor]] = {}
        # TODO refactor this to be cleaner...
        for table, managed_collision_module in self._managed_collision_modules.items():
            if table not in self._table_to_tbe_and_index:
                continue
            evictions_per_table[table] = managed_collision_module.evict()
        self._evict(evictions_per_table)

        # TODO batch the evictions instead of doing it per table. Need to group by TBE
        mc_features = self._apply_mc_modules_to_kjt_list(dist_input)
        return self._embedding_bag_collection.compute(ctx, mc_features)

    def output_dist(
        self,
        ctx: EmbeddingBagCollectionContext,
        output: List[torch.Tensor],
    ) -> LazyAwaitable[KeyedTensor]:
        # TODO investigate if we can overlap eviction with this
        return self._embedding_bag_collection.output_dist(ctx, output)

    def create_context(self) -> EmbeddingBagCollectionContext:
        return self._embedding_bag_collection.create_context()

    def sharded_parameter_names(self, prefix: str = "") -> Iterator[str]:
        for fqn, _ in self.named_parameters():
            yield append_prefix(prefix, fqn)
        for fqn, _ in self.named_buffers():
            yield append_prefix(prefix, fqn)


class ManagedCollisionEmbeddingBagCollectionSharder(
    BaseEmbeddingSharder[ManagedCollisionEmbeddingBagCollection]
):
    def __init__(
        self,
        ebc_sharder: Optional[EmbeddingBagCollectionSharder] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__(qcomm_codecs_registry=qcomm_codecs_registry)
        self._ebc_sharder: EmbeddingBagCollectionSharder = (
            ebc_sharder or EmbeddingBagCollectionSharder(self.qcomm_codecs_registry)
        )

    def shard(
        self,
        module: ManagedCollisionEmbeddingBagCollection,
        params: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
    ) -> ShardedManagedCollisionEmbeddingBagCollection:

        if device is None:
            device = torch.device("cuda")

        return ShardedManagedCollisionEmbeddingBagCollection(
            module,
            params,
            ebc_sharder=self._ebc_sharder,
            managed_collision_module_sharders=[],
            env=env,
            device=device,
        )

    def shardable_parameters(
        self, module: ManagedCollisionEmbeddingBagCollection
    ) -> Dict[str, torch.nn.Parameter]:
        return self._ebc_sharder.shardable_parameters(module._embedding_bag_collection)

    @property
    def module_type(self) -> Type[ManagedCollisionEmbeddingBagCollection]:
        return ManagedCollisionEmbeddingBagCollection

    def sharding_types(self, compute_device_type: str) -> List[str]:
        types = [
            ShardingType.TABLE_WISE.value,
            ShardingType.ROW_WISE.value,
        ]
        return types
