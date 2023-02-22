#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import cast, List

import torch
from torch import nn
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.planner.planners import EmbeddingShardingPlanner
from torchrec.distributed.planner.types import PlannerError, PlannerErrorType, Topology
from torchrec.distributed.test_utils.test_model import TestSparseNN
from torchrec.distributed.types import (
    EmbeddingModuleShardingPlan,
    ModuleSharder,
    ShardingPlan,
    ShardingType,
)
from torchrec.modules.embedding_configs import EmbeddingBagConfig


class TWvsRWSharder(EmbeddingBagCollectionSharder, ModuleSharder[nn.Module]):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [ShardingType.ROW_WISE.value, ShardingType.TABLE_WISE.value]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [EmbeddingComputeKernel.FUSED.value]


class TWSharder(EmbeddingBagCollectionSharder, ModuleSharder[nn.Module]):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [ShardingType.TABLE_WISE.value]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [EmbeddingComputeKernel.FUSED.value]


class TestEmbeddingShardingPlanner(unittest.TestCase):
    def setUp(self) -> None:
        compute_device = "cuda"
        self.topology = Topology(
            world_size=2, hbm_cap=1024 * 1024 * 2, compute_device=compute_device
        )
        self.planner = EmbeddingShardingPlanner(topology=self.topology)

    def test_tw_solution(self) -> None:
        tables = [
            EmbeddingBagConfig(
                num_embeddings=100,
                embedding_dim=64,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(4)
        ]
        model = TestSparseNN(tables=tables, sparse_device=torch.device("meta"))
        sharding_plan = self.planner.plan(module=model, sharders=[TWvsRWSharder()])
        expected_ranks = [[0], [0], [1], [1]]
        ranks = [
            cast(List[int], param_shard.ranks)
            for param_shard in cast(
                EmbeddingModuleShardingPlan, sharding_plan.plan["sparse.ebc"]
            ).values()
        ]

        self.assertEqual(sorted(expected_ranks), sorted(ranks))

    def test_hidden_rw_solution(self) -> None:
        tables = [
            EmbeddingBagConfig(
                num_embeddings=100,
                embedding_dim=64,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(3)
        ]
        model = TestSparseNN(tables=tables, sparse_device=torch.device("meta"))
        sharding_plan = self.planner.plan(module=model, sharders=[TWvsRWSharder()])
        expected_ranks = [[0], [0, 1], [1]]
        ranks = [
            cast(List[int], param_shard.ranks)
            for param_shard in cast(
                EmbeddingModuleShardingPlan, sharding_plan.plan["sparse.ebc"]
            ).values()
        ]

        self.assertEqual(sorted(expected_ranks), sorted(ranks))

    def test_never_fit(self) -> None:
        tables = [
            EmbeddingBagConfig(
                num_embeddings=10000000,
                embedding_dim=10000000,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(2)
        ]
        model = TestSparseNN(tables=tables, sparse_device=torch.device("meta"))

        with self.assertRaises(PlannerError) as context:
            self.planner.plan(module=model, sharders=[TWvsRWSharder()])
        self.assertEqual(
            context.exception.error_type, PlannerErrorType.INSUFFICIENT_STORAGE
        )

        self.assertEqual(self.planner._num_proposals, 4)

    def test_fail_then_rerun(self) -> None:
        tables = [
            EmbeddingBagConfig(
                num_embeddings=4096,
                embedding_dim=128,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(1)
        ]
        model = TestSparseNN(tables=tables, sparse_device=torch.device("meta"))

        with self.assertRaises(PlannerError) as context:
            self.planner.plan(module=model, sharders=[TWSharder()])
        self.assertEqual(
            context.exception.error_type, PlannerErrorType.STRICT_CONSTRAINTS
        )

        sharding_plan = self.planner.plan(module=model, sharders=[TWvsRWSharder()])
        expected_ranks = [[0, 1]]
        ranks = [
            cast(List[int], param_shard.ranks)
            for param_shard in cast(
                EmbeddingModuleShardingPlan, sharding_plan.plan["sparse.ebc"]
            ).values()
        ]

        self.assertEqual(sorted(expected_ranks), sorted(ranks))

    def test_no_sharders(self) -> None:
        tables = [
            EmbeddingBagConfig(
                num_embeddings=100,
                embedding_dim=64,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(4)
        ]
        model = TestSparseNN(tables=tables, sparse_device=torch.device("meta"))
        sharding_plan = self.planner.plan(module=model, sharders=[])

        self.assertEqual(sharding_plan, ShardingPlan({}))
