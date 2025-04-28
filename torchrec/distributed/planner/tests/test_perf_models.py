#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from unittest.mock import MagicMock

from torchrec.distributed.planner.perf_models import (
    NoopCriticalPathPerfModel,
    NoopPerfModel,
    NoopStorageModel,
)
from torchrec.distributed.planner.types import (
    Perf,
    Shard,
    ShardingOption,
    Storage,
    Topology,
)


class TestPerfModels(unittest.TestCase):
    def setUp(self) -> None:
        sharding_types = ["CW", "TW"]
        self.topology = Topology(world_size=2, compute_device="cuda")
        self.tables = [
            ShardingOption(
                name=MagicMock(),
                tensor=MagicMock(),
                module=MagicMock(),
                input_lengths=MagicMock(),
                batch_size=MagicMock(),
                sharding_type=sharding_types[rank],
                partition_by=MagicMock(),
                compute_kernel=MagicMock(),
                shards=[
                    Shard(
                        size=MagicMock(),
                        offset=MagicMock(),
                        rank=rank,
                        perf=Perf(
                            fwd_compute=2 - rank,
                            fwd_comms=0,
                            bwd_compute=0,
                            bwd_comms=0,
                        ),
                        storage=Storage(hbm=100 * (rank + 1), ddr=0),
                    ),
                ],
            )
            for rank in range(2)
        ]

    def test_noop_perf_model(self) -> None:
        perf_model = NoopPerfModel(self.topology)
        perf_rating = perf_model.rate(self.tables)
        self.assertEqual(perf_rating, 2)

    def test_noop_storage_model(self) -> None:
        perf_model = NoopStorageModel(self.topology)
        perf_rating = perf_model.rate(self.tables)
        self.assertEqual(perf_rating, 200)

    def test_noop_critical_path_perf_model(self) -> None:
        perf_model_default = NoopCriticalPathPerfModel(self.topology)
        perf_rating_default = perf_model_default.rate(self.tables)
        self.assertEqual(perf_rating_default, 2)

        perf_model_sharding_type = NoopCriticalPathPerfModel(
            self.topology,
            comms_group_keys=["sharding_type"],
            comp_group_keys=["sharding_type"],
        )
        perf_rating_sharding_type = perf_model_sharding_type.rate(self.tables)
        self.assertEqual(perf_rating_sharding_type, 3)
