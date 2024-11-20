#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from dataclasses import dataclass
from typing import Dict, List, Optional, Type, Union

import torch

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
from torchrec.distributed.types import (
    Awaitable,
    LazyAwaitable,
    ParameterSharding,
    QuantizedCommCodecs,
    ShardingEnv,
    ShardingType,
)
from torchrec.modules.itep_embedding_modules import ITEPEmbeddingBagCollection
from torchrec.modules.itep_modules import GenericITEPModule
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor


@dataclass
class ITEPEmbeddingBagCollectionContext(EmbeddingBagCollectionContext):
    is_reindexed: bool = False


class ShardedITEPEmbeddingBagCollection(
    ShardedEmbeddingModule[
        KJTList,
        List[torch.Tensor],
        KeyedTensor,
        ITEPEmbeddingBagCollectionContext,
    ]
):
    def __init__(
        self,
        module: ITEPEmbeddingBagCollection,
        table_name_to_parameter_sharding: Dict[str, ParameterSharding],
        ebc_sharder: EmbeddingBagCollectionSharder,
        env: ShardingEnv,
        device: torch.device,
    ) -> None:
        super().__init__()

        self._device = device
        self._env = env

        # Iteration counter for ITEP Module. Pinning on CPU because used for condition checking and checkpointing.
        self.register_buffer(
            "_iter", torch.tensor(0, dtype=torch.int64, device=torch.device("cpu"))
        )

        self._embedding_bag_collection: ShardedEmbeddingBagCollection = (
            ebc_sharder.shard(
                module._embedding_bag_collection,
                table_name_to_parameter_sharding,
                env=env,
                device=device,
            )
        )

        # Instantiate ITEP Module in sharded case, re-using metadata from non-sharded case
        self._itep_module: GenericITEPModule = GenericITEPModule(
            table_name_to_unpruned_hash_sizes=module._itep_module.table_name_to_unpruned_hash_sizes,
            lookups=self._embedding_bag_collection._lookups,
            pruning_interval=module._itep_module.pruning_interval,
            enable_pruning=module._itep_module.enable_pruning,
        )

    def prefetch(
        self,
        dist_input: KJTList,
        forward_stream: Optional[Union[torch.cuda.Stream, torch.mtia.Stream]] = None,
        ctx: Optional[ITEPEmbeddingBagCollectionContext] = None,
    ) -> None:
        assert (
            ctx is not None
        ), "ITEP Prefetch call requires ITEPEmbeddingBagCollectionContext"
        dist_input = self._reindex(dist_input)
        ctx.is_reindexed = True
        self._embedding_bag_collection.prefetch(dist_input, forward_stream, ctx)

    # pyre-ignore
    def input_dist(
        self,
        ctx: ITEPEmbeddingBagCollectionContext,
        features: KeyedJaggedTensor,
        force_insert: bool = False,
    ) -> Awaitable[Awaitable[KJTList]]:
        return self._embedding_bag_collection.input_dist(ctx, features)

    def _reindex(self, dist_input: KJTList) -> KJTList:
        for i in range(len(dist_input)):
            remapped_kjt = self._itep_module(dist_input[i], self._iter.item())
            dist_input[i] = remapped_kjt
        return dist_input

    def compute(
        self,
        ctx: ITEPEmbeddingBagCollectionContext,
        dist_input: KJTList,
    ) -> List[torch.Tensor]:
        if not ctx.is_reindexed:
            dist_input = self._reindex(dist_input)
            ctx.is_reindexed = True

        self._iter += 1
        return self._embedding_bag_collection.compute(ctx, dist_input)

    def output_dist(
        self,
        ctx: ITEPEmbeddingBagCollectionContext,
        output: List[torch.Tensor],
    ) -> LazyAwaitable[KeyedTensor]:

        ebc_awaitable = self._embedding_bag_collection.output_dist(ctx, output)
        return ebc_awaitable

    def compute_and_output_dist(
        self, ctx: ITEPEmbeddingBagCollectionContext, input: KJTList
    ) -> LazyAwaitable[KeyedTensor]:
        # Insert forward() function of GenericITEPModule into compute_and_output_dist()
        for i in range(len(input)):
            remapped_kjt = self._itep_module(input[i], self._iter.item())
            input[i] = remapped_kjt
        self._iter += 1
        ebc_awaitable = self._embedding_bag_collection.compute_and_output_dist(
            ctx, input
        )
        return ebc_awaitable

    def create_context(self) -> ITEPEmbeddingBagCollectionContext:
        return ITEPEmbeddingBagCollectionContext()


class ITEPEmbeddingBagCollectionSharder(
    BaseEmbeddingSharder[ITEPEmbeddingBagCollection]
):
    def __init__(
        self,
        ebc_sharder: Optional[EmbeddingBagCollectionSharder] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__(qcomm_codecs_registry=qcomm_codecs_registry)
        self._ebc_sharder: EmbeddingBagCollectionSharder = (
            ebc_sharder
            or EmbeddingBagCollectionSharder(
                qcomm_codecs_registry=self.qcomm_codecs_registry
            )
        )

    def shard(
        self,
        module: ITEPEmbeddingBagCollection,
        params: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        module_fqn: Optional[str] = None,
    ) -> ShardedITEPEmbeddingBagCollection:

        # Enforce GPU for ITEPEmbeddingBagCollection
        if device is None:
            device = torch.device("cuda")

        return ShardedITEPEmbeddingBagCollection(
            module,
            params,
            ebc_sharder=self._ebc_sharder,
            env=env,
            device=device,
        )

    def shardable_parameters(
        self, module: ITEPEmbeddingBagCollection
    ) -> Dict[str, torch.nn.Parameter]:
        return self._ebc_sharder.shardable_parameters(module._embedding_bag_collection)

    @property
    def module_type(self) -> Type[ITEPEmbeddingBagCollection]:
        return ITEPEmbeddingBagCollection

    def sharding_types(self, compute_device_type: str) -> List[str]:
        types = [
            ShardingType.COLUMN_WISE.value,
            ShardingType.TABLE_WISE.value,
        ]
        return types
