#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import List
from unittest.mock import MagicMock

from torchrec.distributed.planner.types import Perf, Shard, ShardingOption, Storage
from torchrec.distributed.planner.utils import _find_imbalance_tables, reset_shard_rank
from torchrec.distributed.types import ShardingType


class TestFindImbalanceTables(unittest.TestCase):
    def setUp(self) -> None:
        self.best_plan: List[ShardingOption] = []
        for i in range(10):
            shard_size = [100 * i, 8]
            shard_offsets = [[0, 0], [0, 8]]
            self.best_plan.append(
                ShardingOption(
                    name=f"table_{i}",
                    tensor=MagicMock(),
                    module=MagicMock(),
                    input_lengths=MagicMock(),
                    batch_size=MagicMock(),
                    sharding_type=ShardingType.COLUMN_WISE.value,
                    partition_by=MagicMock(),
                    compute_kernel=MagicMock(),
                    shards=[
                        Shard(size=shard_size, offset=offset)
                        for offset in shard_offsets
                    ],
                )
            )

    def test_find_perf_imbalance_tables(self) -> None:
        reset_shard_rank(self.best_plan)
        for i, sharding_option in enumerate(self.best_plan):
            for j, shard in enumerate(sharding_option.shards):
                shard.rank = 2 * i + j
                shard.perf = Perf(
                    fwd_compute=2 * i,
                    fwd_comms=2 * i,
                    bwd_compute=2 * i,
                    bwd_comms=2 * i,
                )

        expected_max_perf_table_names = ["table_9"]
        max_perf_table_names = [
            sharding_option.name
            for sharding_option in _find_imbalance_tables(self.best_plan)
        ]
        self.assertTrue(expected_max_perf_table_names, max_perf_table_names)

    def test_find_hbm_imbalance_tables(self) -> None:
        reset_shard_rank(self.best_plan)
        for i, sharding_option in enumerate(self.best_plan):
            for j, shard in enumerate(sharding_option.shards):
                shard.rank = 2 * i + j
                shard.storage = Storage(
                    hbm=2 * (10 - i),
                    ddr=0,
                )

        expected_max_hbm_table_names = ["table_0"]
        max_hbm_table_names = [
            sharding_option.name
            for sharding_option in _find_imbalance_tables(
                self.best_plan, target_imbalance="hbm"
            )
        ]
        self.assertTrue(expected_max_hbm_table_names, max_hbm_table_names)
