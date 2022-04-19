#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import List, cast
from unittest.mock import MagicMock

import torch
from torchrec.distributed.embeddingbag import (
    EmbeddingBagCollectionSharder,
)
from torchrec.distributed.planner.enumerators import EmbeddingEnumerator
from torchrec.distributed.planner.proposers import GreedyProposer, UniformProposer
from torchrec.distributed.planner.types import Topology, ShardingOption
from torchrec.distributed.test_utils.test_model import TestSparseNN
from torchrec.distributed.types import ShardingType, ModuleSharder
from torchrec.modules.embedding_configs import EmbeddingBagConfig


class TestProposers(unittest.TestCase):
    def setUp(self) -> None:
        topology = Topology(world_size=2, compute_device="cuda")
        self.enumerator = EmbeddingEnumerator(topology=topology)
        self.greedy_proposer = GreedyProposer()
        self.uniform_proposer = UniformProposer()

    def test_greedy_two_table_perf(self) -> None:
        tables = [
            EmbeddingBagConfig(
                num_embeddings=100,
                embedding_dim=10,
                name="table_0",
                feature_names=["feature_0"],
            ),
            EmbeddingBagConfig(
                num_embeddings=100,
                embedding_dim=10,
                name="table_1",
                feature_names=["feature_1"],
            ),
        ]

        model = TestSparseNN(tables=tables, sparse_device=torch.device("meta"))
        search_space = self.enumerator.enumerate(
            module=model,
            sharders=[
                cast(ModuleSharder[torch.nn.Module], EmbeddingBagCollectionSharder())
            ],
        )
        self.greedy_proposer.load(search_space)

        # simulate first five iterations:
        output = []
        for _ in range(5):
            proposal = cast(List[ShardingOption], self.greedy_proposer.propose())
            proposal.sort(
                key=lambda sharding_option: (
                    max([shard.perf for shard in sharding_option.shards]),
                    sharding_option.name,
                )
            )
            output.append(
                [
                    (
                        candidate.name,
                        candidate.sharding_type,
                        candidate.compute_kernel,
                    )
                    for candidate in proposal
                ]
            )
            self.greedy_proposer.feedback(partitionable=True)

        expected_output = [
            [
                ("table_0", "row_wise", "batched_fused"),
                ("table_1", "row_wise", "batched_fused"),
            ],
            [
                ("table_0", "table_row_wise", "batched_fused"),
                ("table_1", "row_wise", "batched_fused"),
            ],
            [
                ("table_1", "row_wise", "batched_fused"),
                ("table_0", "data_parallel", "batched_dense"),
            ],
            [
                ("table_1", "table_row_wise", "batched_fused"),
                ("table_0", "data_parallel", "batched_dense"),
            ],
            [
                ("table_0", "data_parallel", "batched_dense"),
                ("table_1", "data_parallel", "batched_dense"),
            ],
        ]

        self.assertEqual(expected_output, output)

    def test_uniform_three_table_perf(self) -> None:
        tables = [
            EmbeddingBagConfig(
                num_embeddings=100 * i,
                embedding_dim=10 * i,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(1, 4)
        ]
        model = TestSparseNN(tables=tables, sparse_device=torch.device("meta"))

        mock_ebc_sharder = cast(
            ModuleSharder[torch.nn.Module], EmbeddingBagCollectionSharder()
        )
        # TODO update this test for CW and TWCW sharding
        mock_ebc_sharder.sharding_types = MagicMock(
            return_value=[
                ShardingType.DATA_PARALLEL.value,
                ShardingType.TABLE_WISE.value,
                ShardingType.ROW_WISE.value,
                ShardingType.TABLE_ROW_WISE.value,
            ]
        )

        self.maxDiff = None

        search_space = self.enumerator.enumerate(
            module=model, sharders=[mock_ebc_sharder]
        )
        self.uniform_proposer.load(search_space)

        output = []
        proposal = self.uniform_proposer.propose()
        while proposal:
            proposal.sort(
                key=lambda sharding_option: (
                    max([shard.perf for shard in sharding_option.shards]),
                    sharding_option.name,
                )
            )
            output.append(
                [
                    (
                        candidate.name,
                        candidate.sharding_type,
                        candidate.compute_kernel,
                    )
                    for candidate in proposal
                ]
            )
            self.uniform_proposer.feedback(partitionable=True)
            proposal = self.uniform_proposer.propose()

        expected_output = [
            [
                (
                    "table_1",
                    "data_parallel",
                    "batched_dense",
                ),
                (
                    "table_2",
                    "data_parallel",
                    "batched_dense",
                ),
                (
                    "table_3",
                    "data_parallel",
                    "batched_dense",
                ),
            ],
            [
                (
                    "table_1",
                    "table_wise",
                    "batched_fused",
                ),
                (
                    "table_2",
                    "table_wise",
                    "batched_fused",
                ),
                (
                    "table_3",
                    "table_wise",
                    "batched_fused",
                ),
            ],
            [
                (
                    "table_1",
                    "row_wise",
                    "batched_fused",
                ),
                (
                    "table_2",
                    "row_wise",
                    "batched_fused",
                ),
                (
                    "table_3",
                    "row_wise",
                    "batched_fused",
                ),
            ],
            [
                (
                    "table_1",
                    "table_row_wise",
                    "batched_fused",
                ),
                (
                    "table_2",
                    "table_row_wise",
                    "batched_fused",
                ),
                (
                    "table_3",
                    "table_row_wise",
                    "batched_fused",
                ),
            ],
        ]

        self.assertEqual(expected_output, output)
