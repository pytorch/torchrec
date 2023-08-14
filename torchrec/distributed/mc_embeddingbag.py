#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, cast, Dict, Iterator, List, Optional, Tuple, Type, Union

import torch

from torchrec.distributed.embedding_lookup import GroupedPooledEmbeddingsLookup
from torchrec.distributed.embedding_types import (
    BaseEmbeddingSharder,
    KJTList,
    ShardedEmbeddingModule,
)
from torchrec.distributed.embeddingbag import (
    EmbeddingBagCollectionAwaitable,
    EmbeddingBagCollectionContext,
    EmbeddingBagCollectionSharder,
    ShardedEmbeddingBagCollection,
)
from torchrec.distributed.mc_module import (
    ManagedCollisionCollectionAwaitable,
    ManagedCollisionCollectionContext,
    ManagedCollisionCollectionSharder,
    ShardedManagedCollisionCollection,
)
from torchrec.distributed.types import (
    Awaitable,
    LazyAwaitable,
    ParameterSharding,
    QuantizedCommCodecs,
    ShardingEnv,
)
from torchrec.distributed.utils import append_prefix
from torchrec.modules.mc_embedding_modules import ManagedCollisionEmbeddingBagCollection
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor, KeyedTensor


class ManagedCollisionEmbeddingBagCollectionAwaitable(
    LazyAwaitable[
        Union[KeyedTensor, Tuple[KeyedTensor, Optional[Dict[str, JaggedTensor]]]]
    ],
):
    def __init__(
        self,
        ebc_awaitable: EmbeddingBagCollectionAwaitable,
        remapped_ids_awaitable: Optional[ManagedCollisionCollectionAwaitable],
    ) -> None:
        super().__init__()
        self._ebc_awaitable = ebc_awaitable
        self._remapped_ids_awaitable = remapped_ids_awaitable

    def _wait_impl(
        self,
    ) -> Union[KeyedTensor, Tuple[KeyedTensor, Optional[Dict[str, JaggedTensor]]]]:
        ebc_output = self._ebc_awaitable.wait()
        if self._remapped_ids_awaitable is not None:
            remapped_ids_output = self._remapped_ids_awaitable.wait()
            return ebc_output, remapped_ids_output
        else:
            return ebc_output


@dataclass
class ManagedCollisionEmbeddingBagCollectionContext(ManagedCollisionCollectionContext):
    pass


class ShardedManagedCollisionEmbeddingBagCollection(
    ShardedEmbeddingModule[
        KJTList,
        Tuple[List[torch.Tensor], KJTList],
        Union[KeyedTensor, Tuple[KeyedTensor, Optional[Dict[str, JaggedTensor]]]],
        ManagedCollisionEmbeddingBagCollectionContext,
    ]
):
    def __init__(
        self,
        module: ManagedCollisionEmbeddingBagCollection,
        table_name_to_parameter_sharding: Dict[str, ParameterSharding],
        ebc_sharder: EmbeddingBagCollectionSharder,
        mc_sharder: ManagedCollisionCollectionSharder,
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
        self._managed_collision_collection: ShardedManagedCollisionCollection = mc_sharder.shard(
            module._managed_collision_collection,
            table_name_to_parameter_sharding,
            env=env,
            device=device,
            sharding_type_to_sharding=self._embedding_bag_collection._sharding_type_to_sharding,
        )
        self._return_remapped_features: bool = module._return_remapped_features

        self._features_to_tables: Dict[str, str] = module._features_to_tables

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
        self,
        ctx: ManagedCollisionEmbeddingBagCollectionContext,
        features: KeyedJaggedTensor,
        mc_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Awaitable[Awaitable[KJTList]]:
        return self._managed_collision_collection.input_dist(ctx, features, mc_kwargs)

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
        ctx: ManagedCollisionEmbeddingBagCollectionContext,
        dist_input: KJTList,
    ) -> Tuple[List[torch.Tensor], KJTList]:

        mc_dist_input = self._managed_collision_collection.compute(
            ctx,
            dist_input,
        )
        evictions_per_table: Dict[str, Optional[torch.Tensor]] = {}
        # TODO refactor this to be cleaner...
        for (
            table,
            managed_collision_module,
        ) in self._managed_collision_collection._managed_collision_modules.items():
            if table not in self._table_to_tbe_and_index:
                continue
            evictions_per_table[table] = managed_collision_module.evict()
        self._evict(evictions_per_table)

        ebc_ret = self._embedding_bag_collection.compute(
            cast(EmbeddingBagCollectionContext, ctx), mc_dist_input
        )

        return ebc_ret, mc_dist_input

    def output_dist(
        self,
        ctx: ManagedCollisionEmbeddingBagCollectionContext,
        output: Tuple[List[torch.Tensor], KJTList],
    ) -> LazyAwaitable[
        Union[KeyedTensor, Tuple[KeyedTensor, Optional[Dict[str, JaggedTensor]]]]
    ]:
        # TODO investigate if we can overlap eviction with this
        ebc_output, mc_output = output
        ebc_awaitable = self._embedding_bag_collection.output_dist(
            cast(EmbeddingBagCollectionContext, ctx), ebc_output
        )
        if self._return_remapped_features:
            remapped_ids_awaitable = self._managed_collision_collection.output_dist(
                ctx, mc_output
            )
        else:
            remapped_ids_awaitable = None

        return ManagedCollisionEmbeddingBagCollectionAwaitable(
            ebc_awaitable=ebc_awaitable, remapped_ids_awaitable=remapped_ids_awaitable
        )

    def create_context(self) -> ManagedCollisionEmbeddingBagCollectionContext:
        return ManagedCollisionEmbeddingBagCollectionContext(sharding_contexts=[])

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
        mc_sharder: Optional[ManagedCollisionCollectionSharder] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__(qcomm_codecs_registry=qcomm_codecs_registry)
        self._ebc_sharder: EmbeddingBagCollectionSharder = (
            ebc_sharder or EmbeddingBagCollectionSharder(self.qcomm_codecs_registry)
        )
        self._mc_sharder: ManagedCollisionCollectionSharder = (
            mc_sharder
            or ManagedCollisionCollectionSharder(
                qcomm_codecs_registry=self.qcomm_codecs_registry
            )
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
            mc_sharder=self._mc_sharder,
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
        return list(
            set.intersection(
                set(self._ebc_sharder.sharding_types(compute_device_type)),
                set(self._mc_sharder.sharding_types(compute_device_type)),
            )
        )
