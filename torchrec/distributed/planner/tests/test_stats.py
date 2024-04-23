#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import List

import torch
from torch import nn
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.planner.planners import EmbeddingShardingPlanner
from torchrec.distributed.planner.stats import EmbeddingStats, NoopEmbeddingStats
from torchrec.distributed.planner.types import Topology
from torchrec.distributed.test_utils.test_model import TestSparseNN
from torchrec.distributed.types import ModuleSharder, ShardingType
from torchrec.modules.embedding_configs import EmbeddingBagConfig


class TWvsRWSharder(EmbeddingBagCollectionSharder, ModuleSharder[nn.Module]):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [ShardingType.ROW_WISE.value, ShardingType.TABLE_WISE.value]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [EmbeddingComputeKernel.FUSED.value]


class TestEmbeddingStats(unittest.TestCase):
    def setUp(self) -> None:
        compute_device = "cuda"
        self.topology = Topology(
            world_size=2, hbm_cap=1024 * 1024 * 2, compute_device=compute_device
        )
        tables = [
            EmbeddingBagConfig(
                num_embeddings=100,
                embedding_dim=64,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(4)
        ]
        self.model = TestSparseNN(tables=tables, sparse_device=torch.device("meta"))

    def test_embedding_stats_runs(self) -> None:
        planner = EmbeddingShardingPlanner(topology=self.topology)
        _ = planner.plan(module=self.model, sharders=[TWvsRWSharder()])
        self.assertEqual(len(planner._stats), 1)
        stats_data = planner._stats[0]
        assert isinstance(stats_data, EmbeddingStats)
        stats: List[str] = stats_data._stats_table
        self.assertTrue(isinstance(stats, list))
        self.assertTrue(stats[0].startswith("####"))

    def test_empty_embedding_stats_runs(self) -> None:
        planner = EmbeddingShardingPlanner(topology=self.topology, stats=[])
        _ = planner.plan(module=self.model, sharders=[TWvsRWSharder()])
        self.assertEqual(len(planner._stats), 0)

    def test_noop_embedding_stats_runs(self) -> None:
        planner = EmbeddingShardingPlanner(
            topology=self.topology, stats=NoopEmbeddingStats()
        )
        _ = planner.plan(module=self.model, sharders=[TWvsRWSharder()])
        self.assertEqual(len(planner._stats), 1)

    def test_embedding_stats_output_with_top_hbm_usage(self) -> None:
        planner = EmbeddingShardingPlanner(topology=self.topology)
        _ = planner.plan(module=self.model, sharders=[TWvsRWSharder()])
        self.assertEqual(len(planner._stats), 1)
        stats_data = planner._stats[0]
        assert isinstance(stats_data, EmbeddingStats)
        stats: List[str] = stats_data._stats_table
        self.assertTrue(isinstance(stats, list))
        top_hbm_usage_keyword = "Top HBM Memory Usage Estimation:"
        self.assertTrue(any(top_hbm_usage_keyword in row for row in stats))
        top_hbm_mem_usage = None
        for row in stats:
            if top_hbm_usage_keyword in row:
                top_hbm_mem_usage = float(row.split(" ")[6])
        self.assertIsNotNone(top_hbm_mem_usage)
