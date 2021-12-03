#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import List
from unittest.mock import MagicMock, patch, call

from torch.distributed._sharding_spec import ShardMetadata, EnumerableShardingSpec
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.planner.embedding_planner import EmbeddingShardingPlanner
from torchrec.distributed.planner.parameter_sharding import _rw_shard_table_rows
from torchrec.distributed.planner.types import ParameterHints
from torchrec.distributed.planner.utils import MIN_DIM
from torchrec.distributed.tests.test_model import TestSparseNN
from torchrec.distributed.types import ParameterSharding, ShardingType
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
)


class CWSharder(EmbeddingBagCollectionSharder[EmbeddingBagCollection]):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [ShardingType.COLUMN_WISE.value]

    """
    Restricts to single impl.
    """

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [EmbeddingComputeKernel.DENSE.value]


class DPCWSharder(EmbeddingBagCollectionSharder[EmbeddingBagCollection]):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [
            ShardingType.COLUMN_WISE.value,
            ShardingType.DATA_PARALLEL.value,
        ]

    """
    Restricts to single impl.
    """

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [EmbeddingComputeKernel.DENSE.value]


class TWSharder(EmbeddingBagCollectionSharder[EmbeddingBagCollection]):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [ShardingType.TABLE_WISE.value]

    """
    Restricts to single impl.
    """

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [EmbeddingComputeKernel.DENSE.value]


class DPTWSharder(EmbeddingBagCollectionSharder[EmbeddingBagCollection]):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [
            ShardingType.TABLE_WISE.value,
            ShardingType.DATA_PARALLEL.value,
        ]

    """
    Restricts to single impl.
    """

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [EmbeddingComputeKernel.DENSE.value]


class DPRWTWSharder(EmbeddingBagCollectionSharder[EmbeddingBagCollection]):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [
            ShardingType.TABLE_WISE.value,
            ShardingType.DATA_PARALLEL.value,
            ShardingType.ROW_WISE.value,
        ]

    """
    Restricts to single impl.
    """

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [EmbeddingComputeKernel.DENSE.value]


class TWRWSharder(EmbeddingBagCollectionSharder[EmbeddingBagCollection]):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [
            ShardingType.TABLE_ROW_WISE.value,
        ]

    """
    Restricts to single impl.
    """

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [EmbeddingComputeKernel.DENSE.value]


class TestEmbeddingPlanner(unittest.TestCase):
    def setUp(self) -> None:
        # Mocks
        self.compute_device_type = "cuda"

    @patch("torchrec.distributed.planner.embedding_planner.logger", create=True)
    def test_allocation_planner_balanced(self, mock_logger: MagicMock) -> None:
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
        expected_plan = {
            "sparse.ebc": {
                "table_0": ParameterSharding(
                    sharding_type=ShardingType.TABLE_WISE.value,
                    compute_kernel="dense",
                    ranks=[0],
                    sharding_spec=EnumerableShardingSpec(
                        shards=[
                            ShardMetadata(
                                shard_sizes=[
                                    tables[0].num_embeddings,
                                    tables[0].embedding_dim,
                                ],
                                shard_offsets=[0, 0],
                                placement="rank:0/cuda:0",
                            )
                        ]
                    ),
                ),
                "table_1": ParameterSharding(
                    sharding_type=ShardingType.TABLE_WISE.value,
                    compute_kernel="dense",
                    ranks=[1],
                    sharding_spec=EnumerableShardingSpec(
                        shards=[
                            ShardMetadata(
                                shard_sizes=[
                                    tables[1].num_embeddings,
                                    tables[1].embedding_dim,
                                ],
                                shard_offsets=[0, 0],
                                placement="rank:1/cuda:1",
                            )
                        ]
                    ),
                ),
                "table_2": ParameterSharding(
                    sharding_type=ShardingType.TABLE_WISE.value,
                    compute_kernel="dense",
                    ranks=[1],
                    sharding_spec=EnumerableShardingSpec(
                        shards=[
                            ShardMetadata(
                                shard_sizes=[
                                    tables[2].num_embeddings,
                                    tables[2].embedding_dim,
                                ],
                                shard_offsets=[0, 0],
                                placement="rank:1/cuda:1",
                            )
                        ]
                    ),
                ),
                "table_3": ParameterSharding(
                    sharding_type=ShardingType.TABLE_WISE.value,
                    compute_kernel="dense",
                    ranks=[0],
                    sharding_spec=EnumerableShardingSpec(
                        shards=[
                            ShardMetadata(
                                shard_sizes=[
                                    tables[3].num_embeddings,
                                    tables[3].embedding_dim,
                                ],
                                shard_offsets=[0, 0],
                                placement="rank:0/cuda:0",
                            )
                        ]
                    ),
                ),
            }
        }

        model = TestSparseNN(tables=tables, weighted_tables=[])
        world_size = 2
        planner = EmbeddingShardingPlanner(
            world_size=world_size,
            compute_device_type=self.compute_device_type,
            storage=storage,
        )

        sharders = [TWSharder()]
        # pyre-ignore [6]
        output = planner.plan(model, sharders)
        self.assertEqual(output.plan, expected_plan)

        # check logger
        self.assertEqual(
            mock_logger.mock_calls[1 : world_size + 1],
            [
                call.info(
                    "  Rank 0 -- HBM/DDR: 0.0/0.0, Cost: 2308, Mean Pooling: 0, Emb Dims: 23, Shards: {'table_wise': 2}"
                ),
                call.info(
                    "  Rank 1 -- HBM/DDR: 0.0/0.0, Cost: 2307, Mean Pooling: 0, Emb Dims: 23, Shards: {'table_wise': 2}"
                ),
            ],
        )

    @patch("torchrec.distributed.planner.embedding_planner.logger", create=True)
    def test_twrw_no_constraints(self, mock_logger: MagicMock) -> None:
        tables = [
            EmbeddingBagConfig(
                num_embeddings=100 + i,
                embedding_dim=10 + i,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(1)
        ]
        expected_plan = {
            "sparse.ebc": {
                "table_0": ParameterSharding(
                    sharding_type=ShardingType.TABLE_ROW_WISE.value,
                    compute_kernel="dense",
                    ranks=[1],
                    sharding_spec=EnumerableShardingSpec(
                        shards=[
                            ShardMetadata(
                                shard_sizes=[
                                    tables[0].num_embeddings // 2,
                                    tables[0].embedding_dim,
                                ],
                                shard_offsets=[0, 0],
                                placement="rank:0/cuda:0",
                            ),
                            ShardMetadata(
                                shard_sizes=[
                                    tables[0].num_embeddings // 2,
                                    tables[0].embedding_dim,
                                ],
                                shard_offsets=[50, 0],
                                placement="rank:1/cuda:1",
                            ),
                        ]
                    ),
                ),
            }
        }

        model = TestSparseNN(tables=tables, weighted_tables=[])
        world_size = 2
        planner = EmbeddingShardingPlanner(
            world_size=world_size,
            compute_device_type=self.compute_device_type,
        )

        sharders = [TWRWSharder()]
        # pyre-ignore [6]
        output = planner.plan(model, sharders)
        self.assertEqual(output.plan, expected_plan)

        # check logger
        self.assertEqual(
            mock_logger.mock_calls[1 : world_size + 1],
            [
                call.info(
                    "  Rank 0 -- HBM/DDR: 0.0/0.0, Cost: 1500, Mean Pooling: 0, Emb Dims: 10, Shards: {'table_row_wise': 1}"
                ),
                call.info(
                    "  Rank 1 -- HBM/DDR: 0.0/0.0, Cost: 1500, Mean Pooling: 0, Emb Dims: 10, Shards: {'table_row_wise': 1}"
                ),
            ],
        )

    @patch("torchrec.distributed.planner.embedding_planner.logger", create=True)
    def test_twrw_no_constraints_edge_case(self, mock_logger: MagicMock) -> None:
        tables = [
            EmbeddingBagConfig(
                num_embeddings=5,
                embedding_dim=10,
                name="table_0",
                feature_names=["feature_0"],
            )
        ]
        expected_plan = {
            "sparse.ebc": {
                "table_0": ParameterSharding(
                    sharding_type=ShardingType.TABLE_ROW_WISE.value,
                    compute_kernel="dense",
                    ranks=[3],
                    sharding_spec=EnumerableShardingSpec(
                        shards=[
                            ShardMetadata(
                                shard_sizes=[
                                    2,
                                    10,
                                ],
                                shard_offsets=[0, 0],
                                placement="rank:0/cuda:0",
                            ),
                            ShardMetadata(
                                shard_sizes=[
                                    2,
                                    10,
                                ],
                                shard_offsets=[2, 0],
                                placement="rank:1/cuda:1",
                            ),
                            ShardMetadata(
                                shard_sizes=[
                                    1,
                                    10,
                                ],
                                shard_offsets=[4, 0],
                                placement="rank:2/cuda:2",
                            ),
                            ShardMetadata(
                                shard_sizes=[
                                    0,
                                    10,
                                ],
                                shard_offsets=[5, 0],
                                placement="rank:3/cuda:3",
                            ),
                        ]
                    ),
                ),
            }
        }

        model = TestSparseNN(tables=tables, weighted_tables=[])
        world_size = 4
        planner = EmbeddingShardingPlanner(
            world_size=world_size,
            compute_device_type=self.compute_device_type,
        )

        sharders = [TWRWSharder()]
        # pyre-ignore [6]
        output = planner.plan(model, sharders)
        self.assertEqual(output.plan, expected_plan)

    @patch("torchrec.distributed.planner.embedding_planner.logger", create=True)
    def test_allocation_planner_one_big_rest_small(
        self, mock_logger: MagicMock
    ) -> None:
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

        expected_plan = {
            "sparse.ebc": {
                "table_0": ParameterSharding(
                    sharding_type=ShardingType.TABLE_WISE.value,
                    compute_kernel="dense",
                    ranks=[0],
                    sharding_spec=EnumerableShardingSpec(
                        shards=[
                            ShardMetadata(
                                shard_sizes=[
                                    tables[0].num_embeddings,
                                    tables[0].embedding_dim,
                                ],
                                shard_offsets=[0, 0],
                                placement="rank:0/cuda:0",
                            )
                        ]
                    ),
                ),
                "table_1": ParameterSharding(
                    sharding_type=ShardingType.TABLE_WISE.value,
                    compute_kernel="dense",
                    ranks=[1],
                    sharding_spec=EnumerableShardingSpec(
                        shards=[
                            ShardMetadata(
                                shard_sizes=[
                                    tables[1].num_embeddings,
                                    tables[1].embedding_dim,
                                ],
                                shard_offsets=[0, 0],
                                placement="rank:1/cuda:1",
                            )
                        ]
                    ),
                ),
                "table_2": ParameterSharding(
                    sharding_type=ShardingType.TABLE_WISE.value,
                    compute_kernel="dense",
                    ranks=[1],
                    sharding_spec=EnumerableShardingSpec(
                        shards=[
                            ShardMetadata(
                                shard_sizes=[
                                    tables[2].num_embeddings,
                                    tables[2].embedding_dim,
                                ],
                                shard_offsets=[0, 0],
                                placement="rank:1/cuda:1",
                            )
                        ]
                    ),
                ),
                "table_3": ParameterSharding(
                    sharding_type=ShardingType.TABLE_WISE.value,
                    compute_kernel="dense",
                    ranks=[1],
                    sharding_spec=EnumerableShardingSpec(
                        shards=[
                            ShardMetadata(
                                shard_sizes=[
                                    tables[3].num_embeddings,
                                    tables[3].embedding_dim,
                                ],
                                shard_offsets=[0, 0],
                                placement="rank:1/cuda:1",
                            )
                        ]
                    ),
                ),
            }
        }
        model = TestSparseNN(tables=tables, weighted_tables=[])

        world_size = 2
        planner = EmbeddingShardingPlanner(
            world_size=world_size,
            compute_device_type=self.compute_device_type,
            storage=storage,
        )
        sharders = [DPTWSharder()]
        # pyre-ignore [6]
        output = planner.plan(model, sharders)
        self.assertEqual(output.plan, expected_plan)

        # check logger
        self.assertEqual(
            mock_logger.mock_calls[1 : world_size + 1],
            [
                call.info(
                    "  Rank 0 -- HBM/DDR: 1.0/0.0, Cost: 5780, Mean Pooling: 0, Emb Dims: 16, Shards: {'table_wise': 1}"
                ),
                call.info(
                    "  Rank 1 -- HBM/DDR: 0.0/0.0, Cost: 7200, Mean Pooling: 0, Emb Dims: 48, Shards: {'table_wise': 3}"
                ),
            ],
        )

    @patch("torchrec.distributed.planner.embedding_planner.logger", create=True)
    def test_allocation_planner_two_big_rest_small(
        self, mock_logger: MagicMock
    ) -> None:
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

        expected_plan = {
            "sparse.ebc": {
                "table_0": ParameterSharding(
                    sharding_type=ShardingType.TABLE_WISE.value,
                    compute_kernel="dense",
                    ranks=[0],
                    sharding_spec=EnumerableShardingSpec(
                        shards=[
                            ShardMetadata(
                                shard_sizes=[
                                    tables[0].num_embeddings,
                                    tables[0].embedding_dim,
                                ],
                                shard_offsets=[0, 0],
                                placement="rank:0/cuda:0",
                            )
                        ]
                    ),
                ),
                "table_1": ParameterSharding(
                    sharding_type=ShardingType.TABLE_WISE.value,
                    compute_kernel="dense",
                    ranks=[1],
                    sharding_spec=EnumerableShardingSpec(
                        shards=[
                            ShardMetadata(
                                shard_sizes=[
                                    tables[1].num_embeddings,
                                    tables[1].embedding_dim,
                                ],
                                shard_offsets=[0, 0],
                                placement="rank:1/cuda:1",
                            )
                        ]
                    ),
                ),
                "table_2": ParameterSharding(
                    sharding_type=ShardingType.DATA_PARALLEL.value,
                    compute_kernel="dense",
                    ranks=None,
                    sharding_spec=None,
                ),
                "table_3": ParameterSharding(
                    sharding_type=ShardingType.DATA_PARALLEL.value,
                    compute_kernel="dense",
                    ranks=None,
                    sharding_spec=None,
                ),
            }
        }
        model = TestSparseNN(tables=tables, weighted_tables=[])

        world_size = 2
        planner = EmbeddingShardingPlanner(
            world_size=world_size,
            compute_device_type=self.compute_device_type,
            # pyre-fixme[6]: Expected `Optional[typing.Dict[str, int]]` for 3rd
            #  param but got `Dict[str, float]`.
            storage=storage,
        )
        sharders = [DPRWTWSharder()]
        # pyre-ignore [6]
        output = planner.plan(model, sharders)
        self.assertEqual(output.plan, expected_plan)

        # check logger
        self.assertEqual(
            mock_logger.mock_calls[1 : world_size + 1],
            [
                call.info(
                    "  Rank 0 -- HBM/DDR: 1.0/0.0, Cost: 8180, Mean Pooling: 0, Emb Dims: 48, Shards: {'table_wise': 1, 'data_parallel': 2}"
                ),
                call.info(
                    "  Rank 1 -- HBM/DDR: 1.0/0.0, Cost: 8180, Mean Pooling: 0, Emb Dims: 48, Shards: {'table_wise': 1, 'data_parallel': 2}"
                ),
            ],
        )

    @patch("torchrec.distributed.planner.embedding_planner.logger", create=True)
    def test_allocation_planner_rw_two_big_rest_small(
        self, mock_logger: MagicMock
    ) -> None:
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
        local_rows, block_size, last_rank = _rw_shard_table_rows(big_hash, 4)
        storage = {"hbm": 0.6}

        expected_plan = {
            "sparse.ebc": {
                "table_0": ParameterSharding(
                    sharding_type=ShardingType.ROW_WISE.value,
                    compute_kernel="dense",
                    ranks=None,
                    sharding_spec=EnumerableShardingSpec(
                        shards=[
                            ShardMetadata(
                                shard_sizes=[
                                    local_rows[0],
                                    tables[0].embedding_dim,
                                ],
                                shard_offsets=[0, 0],
                                placement="rank:0/cuda:0",
                            ),
                            ShardMetadata(
                                shard_sizes=[
                                    local_rows[1],
                                    tables[0].embedding_dim,
                                ],
                                shard_offsets=[block_size, 0],
                                placement="rank:1/cuda:1",
                            ),
                            ShardMetadata(
                                shard_sizes=[
                                    local_rows[2],
                                    tables[0].embedding_dim,
                                ],
                                shard_offsets=[2 * block_size, 0],
                                placement="rank:2/cuda:2",
                            ),
                            ShardMetadata(
                                shard_sizes=[
                                    local_rows[3],
                                    tables[0].embedding_dim,
                                ],
                                shard_offsets=[3 * block_size, 0],
                                placement="rank:3/cuda:3",
                            ),
                        ],
                    ),
                ),
                "table_1": ParameterSharding(
                    sharding_type=ShardingType.ROW_WISE.value,
                    compute_kernel="dense",
                    ranks=None,
                    sharding_spec=EnumerableShardingSpec(
                        shards=[
                            ShardMetadata(
                                shard_sizes=[
                                    local_rows[0],
                                    tables[1].embedding_dim,
                                ],
                                shard_offsets=[0, 0],
                                placement="rank:0/cuda:0",
                            ),
                            ShardMetadata(
                                shard_sizes=[
                                    local_rows[1],
                                    tables[1].embedding_dim,
                                ],
                                shard_offsets=[block_size, 0],
                                placement="rank:1/cuda:1",
                            ),
                            ShardMetadata(
                                shard_sizes=[
                                    local_rows[2],
                                    tables[1].embedding_dim,
                                ],
                                shard_offsets=[2 * block_size, 0],
                                placement="rank:2/cuda:2",
                            ),
                            ShardMetadata(
                                shard_sizes=[
                                    local_rows[3],
                                    tables[1].embedding_dim,
                                ],
                                shard_offsets=[3 * block_size, 0],
                                placement="rank:3/cuda:3",
                            ),
                        ],
                    ),
                ),
                "table_2": ParameterSharding(
                    sharding_type=ShardingType.DATA_PARALLEL.value,
                    compute_kernel="dense",
                    ranks=None,
                ),
                "table_3": ParameterSharding(
                    sharding_type=ShardingType.DATA_PARALLEL.value,
                    compute_kernel="dense",
                    ranks=None,
                ),
            }
        }
        model = TestSparseNN(tables=tables, weighted_tables=[])

        world_size = 4
        planner = EmbeddingShardingPlanner(
            world_size=world_size,
            compute_device_type=self.compute_device_type,
            # pyre-fixme[6]: Expected `Optional[typing.Dict[str, int]]` for 3rd
            #  param but got `Dict[str, float]`.
            storage=storage,
        )
        sharders = [DPRWTWSharder()]
        # pyre-ignore [6]
        output = planner.plan(model, sharders)
        self.assertEqual(output.plan, expected_plan)

        # check logger
        self.assertEqual(
            mock_logger.mock_calls[1 : world_size + 1],
            [
                call.info(
                    f"  Rank {rank} -- HBM/DDR: 0.5/0.0, Cost: 31298, Mean Pooling: 0, Emb Dims: 64, Shards: {{'row_wise': 2, 'data_parallel': 2}}"
                )
                for rank in range(world_size)
            ],
        )

    @patch("torchrec.distributed.planner.embedding_planner.logger", create=True)
    def test_allocation_planner_cw_balanced(self, mock_logger: MagicMock) -> None:
        tables = [
            EmbeddingBagConfig(
                num_embeddings=100,
                embedding_dim=128,
                name="table_0",
                feature_names=["feature_0"],
            )
        ]
        storage = {"hbm": 1}
        block_size, residual = divmod(128, 2)
        expected_plan = {
            "sparse.ebc": {
                "table_0": ParameterSharding(
                    sharding_type=ShardingType.COLUMN_WISE.value,
                    compute_kernel="dense",
                    ranks=[
                        0,
                        1,
                    ],
                    sharding_spec=EnumerableShardingSpec(
                        shards=[
                            ShardMetadata(
                                shard_sizes=[
                                    tables[0].num_embeddings,
                                    block_size,
                                ],
                                shard_offsets=[0, 0],
                                placement="rank:0/cuda:0",
                            ),
                            ShardMetadata(
                                shard_sizes=[
                                    tables[0].num_embeddings,
                                    block_size,
                                ],
                                shard_offsets=[0, block_size],
                                placement="rank:1/cuda:1",
                            ),
                        ]
                    ),
                ),
            }
        }

        model = TestSparseNN(tables=tables, weighted_tables=[])
        world_size = 2
        planner = EmbeddingShardingPlanner(
            world_size=world_size,
            compute_device_type=self.compute_device_type,
            storage=storage,
            hints={
                "table_0": ParameterHints(
                    sharding_types=[ShardingType.COLUMN_WISE.value],
                    col_wise_shard_dim=64,
                ),
            },
        )

        sharders = [CWSharder()]
        # pyre-ignore [6]
        output = planner.plan(model, sharders)
        self.assertEqual(output.plan, expected_plan)
        # check logger
        self.assertEqual(
            mock_logger.mock_calls[1 : world_size + 1],
            [
                call.info(
                    "  Rank 0 -- HBM/DDR: 0.0/0.0, Cost: 6400, Mean Pooling: 0, Emb Dims: 64, Shards: {'column_wise': 1}"
                ),
                call.info(
                    "  Rank 1 -- HBM/DDR: 0.0/0.0, Cost: 6400, Mean Pooling: 0, Emb Dims: 64, Shards: {'column_wise': 1}"
                ),
            ],
        )

    @patch("torchrec.distributed.planner.embedding_planner.logger", create=True)
    def test_allocation_planner_cw_two_big_rest_small_with_residual(
        self, mock_logger: MagicMock
    ) -> None:
        big_hash = int(1024 * 1024 * 1024 / 16 / 4)
        small_hash = 1000
        tables = [
            EmbeddingBagConfig(
                num_embeddings=(big_hash if i <= 1 else small_hash) // 4,
                embedding_dim=62,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(4)
        ]

        block_size, residual = divmod(62, MIN_DIM)

        storage = {"hbm": 0.6}

        expected_plan = {
            "sparse.ebc": {
                "table_0": ParameterSharding(
                    sharding_type=ShardingType.COLUMN_WISE.value,
                    compute_kernel="dense",
                    ranks=[0, 1],
                    sharding_spec=EnumerableShardingSpec(
                        shards=[
                            ShardMetadata(
                                shard_sizes=[
                                    tables[0].num_embeddings,
                                    MIN_DIM,
                                ],
                                shard_offsets=[0, 0],
                                placement="rank:0/cuda:0",
                            ),
                            ShardMetadata(
                                shard_sizes=[
                                    tables[0].num_embeddings,
                                    residual,
                                ],
                                shard_offsets=[0, MIN_DIM],
                                placement="rank:1/cuda:1",
                            ),
                        ]
                    ),
                ),
                "table_1": ParameterSharding(
                    sharding_type=ShardingType.COLUMN_WISE.value,
                    compute_kernel="dense",
                    ranks=[2, 3],
                    sharding_spec=EnumerableShardingSpec(
                        shards=[
                            ShardMetadata(
                                shard_sizes=[
                                    tables[1].num_embeddings,
                                    MIN_DIM,
                                ],
                                shard_offsets=[0, 0],
                                placement="rank:2/cuda:2",
                            ),
                            ShardMetadata(
                                shard_sizes=[
                                    tables[1].num_embeddings,
                                    residual,
                                ],
                                shard_offsets=[0, MIN_DIM],
                                placement="rank:3/cuda:3",
                            ),
                        ]
                    ),
                ),
                "table_2": ParameterSharding(
                    sharding_type=ShardingType.DATA_PARALLEL.value,
                    compute_kernel="dense",
                    ranks=None,
                ),
                "table_3": ParameterSharding(
                    sharding_type=ShardingType.DATA_PARALLEL.value,
                    compute_kernel="dense",
                    ranks=None,
                ),
            }
        }
        model = TestSparseNN(tables=tables, weighted_tables=[])

        world_size = 4
        planner = EmbeddingShardingPlanner(
            world_size=world_size,
            compute_device_type=self.compute_device_type,
            # pyre-fixme[6]: Expected `Optional[typing.Dict[str, int]]` for 3rd
            #  param but got `Dict[str, float]`.
            storage=storage,
            hints={
                "table_0": ParameterHints(
                    sharding_types=[ShardingType.COLUMN_WISE.value],
                    col_wise_shard_dim=32,
                ),
                "table_1": ParameterHints(
                    sharding_types=[ShardingType.COLUMN_WISE.value],
                    col_wise_shard_dim=32,
                ),
            },
        )
        sharders = [DPCWSharder()]
        # pyre-ignore [6]
        output = planner.plan(model, sharders)
        self.assertEqual(output.plan, expected_plan)

        # check logger
        # default cost function: math.log(hash_size, 10) * emb_dim * COMMS_MULTIPLER[sharding_type] * KERNEL_MULTIPLER[compute_kernel]
        # rank 0:
        #   table 0 shard 0: log10(4194304) * 31 * 2 * 25
        #   table 2: log10(250) * 62 * 1 * 25
        #   table 3: log10(250) * 62 * 1 * 25
        self.assertEqual(
            mock_logger.mock_calls[1 : world_size + 1],
            [
                call.info(
                    "  Rank 0 -- HBM/DDR: 0.5/0.0, Cost: 17699, Mean Pooling: 0, Emb Dims: 156, Shards: {'column_wise': 1, 'data_parallel': 2}"
                ),
                call.info(
                    "  Rank 1 -- HBM/DDR: 0.5/0.0, Cost: 17699, Mean Pooling: 0, Emb Dims: 154, Shards: {'column_wise': 1, 'data_parallel': 2}"
                ),
                call.info(
                    "  Rank 2 -- HBM/DDR: 0.5/0.0, Cost: 17699, Mean Pooling: 0, Emb Dims: 156, Shards: {'column_wise': 1, 'data_parallel': 2}"
                ),
                call.info(
                    "  Rank 3 -- HBM/DDR: 0.5/0.0, Cost: 17699, Mean Pooling: 0, Emb Dims: 154, Shards: {'column_wise': 1, 'data_parallel': 2}"
                ),
            ],
        )
