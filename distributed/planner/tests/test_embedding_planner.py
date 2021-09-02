#!/usr/bin/env python3

import unittest
from typing import List
from unittest.mock import MagicMock

import torch
import torch.distributed as dist
from torchrec.distributed.embedding import EmbeddingBagCollectionSharder
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.planner.embedding_planner import EmbeddingShardingPlanner
from torchrec.distributed.tests.test_model import TestSparseNN
from torchrec.distributed.types import ParameterSharding, ShardingType, ShardingPlan
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
)


class TWSharder(EmbeddingBagCollectionSharder[EmbeddingBagCollection]):
    @property
    def sharding_types(self) -> List[str]:
        return [ShardingType.TABLE_WISE.value]

    """
    Restricts to single impl.
    """

    def compute_kernels(self, sharding_type: str, device: torch.device) -> List[str]:
        return [EmbeddingComputeKernel.DENSE.value]


class DPTWSharder(EmbeddingBagCollectionSharder[EmbeddingBagCollection]):
    @property
    def sharding_types(self) -> List[str]:
        return [
            ShardingType.TABLE_WISE.value,
            ShardingType.DATA_PARALLEL.value,
        ]

    """
    Restricts to single impl.
    """

    def compute_kernels(self, sharding_type: str, device: torch.device) -> List[str]:
        return [EmbeddingComputeKernel.DENSE.value]


class DPRWTWSharder(EmbeddingBagCollectionSharder[EmbeddingBagCollection]):
    @property
    def sharding_types(self) -> List[str]:
        return [
            ShardingType.TABLE_WISE.value,
            ShardingType.DATA_PARALLEL.value,
            ShardingType.ROW_WISE.value,
        ]

    """
    Restricts to single impl.
    """

    def compute_kernels(self, sharding_type: str, device: torch.device) -> List[str]:
        return [EmbeddingComputeKernel.DENSE.value]


class TestEmbeddingPlanner(unittest.TestCase):
    def setUp(self) -> None:
        # Mocks
        dist.get_world_size = MagicMock(return_value=2)
        self.pg = MagicMock()
        self.device = torch.device("cuda:0")

    def test_allocation_planner_balanced(self) -> None:
        tables = [
            EmbeddingBagConfig(
                num_embeddings=100 + i,
                embedding_dim=10 + i,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(4)
        ]
        storage = {"hbm": 1}
        expected_plan: ShardingPlan = ShardingPlan(
            {
                "sparse.ebc": {
                    "table_3": ParameterSharding(
                        sharding_type=ShardingType.TABLE_WISE.value,
                        compute_kernel="dense",
                        rank=0,
                    ),
                    "table_2": ParameterSharding(
                        sharding_type=ShardingType.TABLE_WISE.value,
                        compute_kernel="dense",
                        rank=1,
                    ),
                    "table_1": ParameterSharding(
                        sharding_type=ShardingType.TABLE_WISE.value,
                        compute_kernel="dense",
                        rank=1,
                    ),
                    "table_0": ParameterSharding(
                        sharding_type=ShardingType.TABLE_WISE.value,
                        compute_kernel="dense",
                        rank=0,
                    ),
                }
            }
        )

        model = TestSparseNN(tables=tables, weighted_tables=[])
        planner = EmbeddingShardingPlanner(
            pg=self.pg, device=self.device, storage=storage
        )

        sharders = [TWSharder()]
        # pyre-ignore [6]
        plan = planner.plan(model, sharders)
        self.assertEqual(plan, expected_plan)

    def test_allocation_planner_one_big_rest_small(self) -> None:
        big_hash = int(1024 * 1024 * 1024 / 16 / 4)
        small_hash = 1000
        tables = [
            EmbeddingBagConfig(
                num_embeddings=big_hash if i == 0 else small_hash,
                embedding_dim=16,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(4)
        ]

        storage = {"hbm": 1}

        expected_plan: ShardingPlan = ShardingPlan(
            {
                "sparse.ebc": {
                    "table_3": ParameterSharding(
                        sharding_type=ShardingType.TABLE_WISE.value,
                        compute_kernel="dense",
                        rank=1,
                    ),
                    "table_2": ParameterSharding(
                        sharding_type=ShardingType.TABLE_WISE.value,
                        compute_kernel="dense",
                        rank=1,
                    ),
                    "table_1": ParameterSharding(
                        sharding_type=ShardingType.TABLE_WISE.value,
                        compute_kernel="dense",
                        rank=1,
                    ),
                    "table_0": ParameterSharding(
                        sharding_type=ShardingType.TABLE_WISE.value,
                        compute_kernel="dense",
                        rank=0,
                    ),
                }
            }
        )
        model = TestSparseNN(tables=tables, weighted_tables=[])

        planner = EmbeddingShardingPlanner(
            pg=self.pg, device=self.device, storage=storage
        )
        sharders = [DPTWSharder()]
        # pyre-ignore [6]
        plan = planner.plan(model, sharders)
        self.assertEqual(plan, expected_plan)

    def test_allocation_planner_two_big_rest_small(self) -> None:
        big_hash = int(1024 * 1024 * 1024 / 16 / 4)
        small_hash = 1000
        tables = [
            EmbeddingBagConfig(
                num_embeddings=big_hash if i <= 1 else small_hash,
                embedding_dim=16,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(4)
        ]

        storage = {"hbm": 1.1}

        expected_plan: ShardingPlan = ShardingPlan(
            {
                "sparse.ebc": {
                    "table_3": ParameterSharding(
                        sharding_type=ShardingType.DATA_PARALLEL.value,
                        compute_kernel="dense",
                        rank=None,
                    ),
                    "table_2": ParameterSharding(
                        sharding_type=ShardingType.DATA_PARALLEL.value,
                        compute_kernel="dense",
                        rank=None,
                    ),
                    "table_1": ParameterSharding(
                        sharding_type=ShardingType.TABLE_WISE.value,
                        compute_kernel="dense",
                        rank=1,
                    ),
                    "table_0": ParameterSharding(
                        sharding_type=ShardingType.TABLE_WISE.value,
                        compute_kernel="dense",
                        rank=0,
                    ),
                }
            }
        )
        model = TestSparseNN(tables=tables, weighted_tables=[])

        planner = EmbeddingShardingPlanner(
            pg=self.pg,
            device=self.device,
            # pyre-fixme[6]: Expected `Optional[typing.Dict[str, int]]` for 3rd
            #  param but got `Dict[str, float]`.
            storage=storage,
        )
        sharders = [DPRWTWSharder()]
        # pyre-ignore [6]
        plan = planner.plan(model, sharders)
        self.assertEqual(plan, expected_plan)

    def test_allocation_planner_rw_two_big_rest_small(self) -> None:
        big_hash = int(1024 * 1024 * 1024 / 16 / 4)
        small_hash = 1000
        tables = [
            EmbeddingBagConfig(
                num_embeddings=big_hash if i <= 1 else small_hash,
                embedding_dim=16,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(4)
        ]

        dist.get_world_size = MagicMock(return_value=4)

        storage = {"hbm": 0.6}

        expected_plan: ShardingPlan = ShardingPlan(
            {
                "sparse.ebc": {
                    "table_3": ParameterSharding(
                        sharding_type=ShardingType.DATA_PARALLEL.value,
                        compute_kernel="dense",
                        rank=None,
                    ),
                    "table_2": ParameterSharding(
                        sharding_type=ShardingType.DATA_PARALLEL.value,
                        compute_kernel="dense",
                        rank=None,
                    ),
                    "table_1": ParameterSharding(
                        sharding_type=ShardingType.ROW_WISE.value,
                        compute_kernel="dense",
                        rank=None,
                    ),
                    "table_0": ParameterSharding(
                        sharding_type=ShardingType.ROW_WISE.value,
                        compute_kernel="dense",
                        rank=None,
                    ),
                }
            }
        )
        model = TestSparseNN(tables=tables, weighted_tables=[])

        planner = EmbeddingShardingPlanner(
            pg=self.pg,
            device=self.device,
            # pyre-fixme[6]: Expected `Optional[typing.Dict[str, int]]` for 3rd
            #  param but got `Dict[str, float]`.
            storage=storage,
        )
        sharders = [DPRWTWSharder()]
        # pyre-ignore [6]
        plan = planner.plan(model, sharders)
        self.assertEqual(plan, expected_plan)
