#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import cast
from unittest.mock import MagicMock

import torch
from torchrec.distributed.embedding_types import EmbeddingComputeKernel

from torchrec.distributed.planner.types import Shard, ShardingOption
from torchrec.distributed.types import (
    BoundsCheckMode,
    CacheAlgorithm,
    CacheParams,
    DataType,
    ShardingType,
)
from torchrec.modules.embedding_configs import EmbeddingBagConfig, EmbeddingConfig
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)
from torchrec.modules.mc_embedding_modules import (
    ManagedCollisionCollection,
    ManagedCollisionEmbeddingBagCollection,
    ManagedCollisionEmbeddingCollection,
)
from torchrec.modules.mc_modules import (
    DistanceLFU_EvictionPolicy,
    ManagedCollisionModule,
    MCHManagedCollisionModule,
)


class TestShardingOption(unittest.TestCase):
    def test_hash_sharding_option(self) -> None:
        shard_size = [10000, 80]
        shard_offsets = [[0, 0], [0, 80]]
        sharding_option: ShardingOption = ShardingOption(
            name="table_0",
            tensor=torch.empty(
                (10000, 160), dtype=torch.float16, device=torch.device("meta")
            ),
            module=("ebc", MagicMock()),
            input_lengths=MagicMock(),
            batch_size=MagicMock(),
            sharding_type=ShardingType.COLUMN_WISE.value,
            partition_by=MagicMock(),
            compute_kernel=EmbeddingComputeKernel.FUSED.value,
            shards=[Shard(size=shard_size, offset=offset) for offset in shard_offsets],
            cache_params=CacheParams(
                algorithm=CacheAlgorithm.LRU,
                load_factor=0.5,
                reserved_memory=0.0,
                precision=DataType.FP16,
                prefetch_pipeline=True,
            ),
            enforce_hbm=True,
            stochastic_rounding=False,
            bounds_check_mode=BoundsCheckMode.WARNING,
        )
        self.assertTrue(map(hash, [sharding_option]))

    def test_module_pooled_ebc(self) -> None:
        eb_config = EmbeddingBagConfig(
            name="table_0",
            embedding_dim=160,
            num_embeddings=10000,
            feature_names=["f1"],
            data_type=DataType.FP16,
        )
        ebc = EmbeddingBagCollection(tables=[eb_config])

        sharding_option: ShardingOption = ShardingOption(
            name="table_0",
            tensor=torch.empty(
                (10000, 160), dtype=torch.float16, device=torch.device("meta")
            ),
            module=("ebc", ebc),
            input_lengths=MagicMock(),
            batch_size=MagicMock(),
            sharding_type=ShardingType.COLUMN_WISE.value,
            partition_by=MagicMock(),
            compute_kernel=EmbeddingComputeKernel.FUSED.value,
            shards=[
                Shard(size=[10000, 80], offset=offset) for offset in [[0, 0], [0, 80]]
            ],
        )
        self.assertEqual(sharding_option.is_pooled, True)

    def test_module_pooled_mch_ebc(self) -> None:
        eb_config = EmbeddingBagConfig(
            name="table_0",
            embedding_dim=160,
            num_embeddings=10000,
            feature_names=["f1"],
            data_type=DataType.FP16,
        )
        ebc = EmbeddingBagCollection(tables=[eb_config])
        mc_modules = {
            "table_0": cast(
                ManagedCollisionModule,
                MCHManagedCollisionModule(
                    zch_size=10000,
                    device=torch.device("meta"),
                    eviction_interval=1,
                    eviction_policy=DistanceLFU_EvictionPolicy(),
                ),
            ),
        }
        mcc = ManagedCollisionCollection(
            managed_collision_modules=mc_modules,
            embedding_configs=[eb_config],
        )
        mch_ebc = ManagedCollisionEmbeddingBagCollection(ebc, mcc)

        sharding_option: ShardingOption = ShardingOption(
            name="table_0",
            tensor=torch.empty(
                (10000, 80), dtype=torch.float16, device=torch.device("meta")
            ),
            module=("mch_ebc", mch_ebc),
            input_lengths=MagicMock(),
            batch_size=MagicMock(),
            sharding_type=ShardingType.COLUMN_WISE.value,
            partition_by=MagicMock(),
            compute_kernel=EmbeddingComputeKernel.FUSED.value,
            shards=[
                Shard(size=[10000, 80], offset=offset) for offset in [[0, 0], [0, 80]]
            ],
        )
        self.assertEqual(sharding_option.is_pooled, True)

    def test_module_pooled_ec(self) -> None:
        e_config = EmbeddingConfig(
            name="table_0",
            embedding_dim=80,
            num_embeddings=10000,
            feature_names=["f1"],
            data_type=DataType.FP16,
        )
        ec = EmbeddingCollection(tables=[e_config])

        shard_size = [10000, 80]
        shard_offsets = [[0, 0], [0, 80]]
        sharding_option: ShardingOption = ShardingOption(
            name="table_0",
            tensor=torch.empty(
                (10000, 160), dtype=torch.float16, device=torch.device("meta")
            ),
            module=("ec", ec),
            input_lengths=MagicMock(),
            batch_size=MagicMock(),
            sharding_type=ShardingType.COLUMN_WISE.value,
            partition_by=MagicMock(),
            compute_kernel=EmbeddingComputeKernel.FUSED.value,
            shards=[Shard(size=shard_size, offset=offset) for offset in shard_offsets],
        )
        self.assertEqual(sharding_option.is_pooled, False)

    def test_module_pooled_mch_ec(self) -> None:
        e_config = EmbeddingConfig(
            name="table_0",
            embedding_dim=80,
            num_embeddings=10000,
            feature_names=["f1"],
            data_type=DataType.FP16,
        )
        ec = EmbeddingCollection(tables=[e_config])
        mc_modules = {
            "table_0": cast(
                ManagedCollisionModule,
                MCHManagedCollisionModule(
                    zch_size=10000,
                    device=torch.device("meta"),
                    eviction_interval=1,
                    eviction_policy=DistanceLFU_EvictionPolicy(),
                ),
            ),
        }
        mcc = ManagedCollisionCollection(
            managed_collision_modules=mc_modules,
            embedding_configs=[e_config],
        )
        mch_ec = ManagedCollisionEmbeddingCollection(ec, mcc)

        shard_size = [10000, 80]
        shard_offsets = [[0, 0], [0, 80]]
        sharding_option: ShardingOption = ShardingOption(
            name="table_0",
            tensor=torch.empty(
                (10000, 160), dtype=torch.float16, device=torch.device("meta")
            ),
            module=("mch_ec", mch_ec),
            input_lengths=MagicMock(),
            batch_size=MagicMock(),
            sharding_type=ShardingType.COLUMN_WISE.value,
            partition_by=MagicMock(),
            compute_kernel=EmbeddingComputeKernel.FUSED.value,
            shards=[Shard(size=shard_size, offset=offset) for offset in shard_offsets],
        )
        self.assertEqual(sharding_option.is_pooled, False)
