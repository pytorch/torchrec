#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
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
