#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import cast, List, Optional

import torch
from torch import nn
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.planner import ParameterConstraints
from torchrec.distributed.planner.planners import EmbeddingShardingPlanner
from torchrec.distributed.planner.types import (
    PlannerError,
    PlannerErrorType,
    ShardingOption,
    Topology,
)
from torchrec.distributed.sharding_plan import get_default_sharders
from torchrec.distributed.test_utils.test_model import TestSparseNN
from torchrec.distributed.types import (
    BoundsCheckMode,
    CacheAlgorithm,
    CacheParams,
    DataType,
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

        # since it has negative storage_constraint
        self.assertEqual(self.planner._num_proposals, 0)

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


class TestEmbeddingShardingPlannerWithConstraints(unittest.TestCase):
    def setUp(self) -> None:
        compute_device = "cuda"
        self.topology = Topology(
            world_size=2, hbm_cap=1024 * 1024 * 2, compute_device=compute_device
        )
        self.tables = [
            EmbeddingBagConfig(
                num_embeddings=100,
                embedding_dim=64,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(4)
        ]
        self.constraints = {
            "table_0": ParameterConstraints(
                enforce_hbm=True,
                cache_params=CacheParams(
                    algorithm=CacheAlgorithm.LFU,
                ),
                feature_names=self.tables[0].feature_names,
            ),
            "table_1": ParameterConstraints(
                enforce_hbm=False,
                stochastic_rounding=True,
                feature_names=self.tables[1].feature_names,
            ),
            "table_2": ParameterConstraints(
                bounds_check_mode=BoundsCheckMode.FATAL,
                feature_names=self.tables[2].feature_names,
            ),
            "table_3": ParameterConstraints(
                cache_params=CacheParams(
                    algorithm=CacheAlgorithm.LFU,
                    load_factor=0.1,
                    reserved_memory=1.0,
                    precision=DataType.FP16,
                ),
                feature_names=self.tables[3].feature_names,
            ),
        }
        self.planner = EmbeddingShardingPlanner(
            topology=self.topology, constraints=self.constraints
        )

    def test_fused_paramters_from_constraints(self) -> None:
        model = TestSparseNN(tables=self.tables, sparse_device=torch.device("meta"))
        sharding_plan = self.planner.plan(module=model, sharders=get_default_sharders())

        expected_fused_params = {
            "table_0": (
                CacheParams(
                    algorithm=CacheAlgorithm.LFU,
                    load_factor=None,
                    reserved_memory=None,
                    precision=None,
                ),
                True,
                None,
                None,
            ),
            "table_1": (None, False, True, None),
            "table_2": (None, None, None, BoundsCheckMode.FATAL),
            "table_3": (
                CacheParams(
                    algorithm=CacheAlgorithm.LFU,
                    load_factor=0.1,
                    reserved_memory=1.0,
                    precision=DataType.FP16,
                ),
                None,
                None,
                None,
            ),
        }

        table_names = ["table_" + str(i) for i in range(4)]
        for table in table_names:
            parameter_sharding = cast(
                EmbeddingModuleShardingPlan, sharding_plan.plan["sparse.ebc"]
            )[table]
            self.assertEqual(
                (
                    parameter_sharding.cache_params,
                    parameter_sharding.enforce_hbm,
                    parameter_sharding.stochastic_rounding,
                    parameter_sharding.bounds_check_mode,
                ),
                expected_fused_params[table],
            )

    def test_passing_info_through_constraints(self) -> None:
        model = TestSparseNN(tables=self.tables, sparse_device=torch.device("meta"))
        _ = self.planner.plan(module=model, sharders=get_default_sharders())

        best_plan: Optional[List[ShardingOption]] = self.planner._best_plan
        self.assertIsNotNone(best_plan)

        for table, constraint, sharding_option in zip(
            self.tables, self.constraints.values(), best_plan
        ):
            self.assertEqual(table.name, sharding_option.name)

            self.assertEqual(table.feature_names, sharding_option.feature_names)
            self.assertEqual(table.feature_names, constraint.feature_names)

            self.assertEqual(constraint.cache_params, sharding_option.cache_params)
            self.assertEqual(constraint.enforce_hbm, sharding_option.enforce_hbm)
            self.assertEqual(
                constraint.stochastic_rounding, sharding_option.stochastic_rounding
            )
            self.assertEqual(
                constraint.bounds_check_mode, sharding_option.bounds_check_mode
            )
            self.assertEqual(constraint.is_weighted, sharding_option.is_weighted)
