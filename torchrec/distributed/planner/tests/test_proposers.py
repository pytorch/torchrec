#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import cast, List
from unittest.mock import MagicMock

import torch
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.planner.enumerators import EmbeddingEnumerator
from torchrec.distributed.planner.proposers import (
    GreedyProposer,
    GridSearchProposer,
    UniformProposer,
)
from torchrec.distributed.planner.types import ShardingOption, Topology
from torchrec.distributed.test_utils.test_model import TestSparseNN
from torchrec.distributed.types import ModuleSharder, ShardingType
from torchrec.modules.embedding_configs import EmbeddingBagConfig


class TestProposers(unittest.TestCase):
    def setUp(self) -> None:
        topology = Topology(world_size=2, compute_device="cuda")
        self.enumerator = EmbeddingEnumerator(topology=topology)
        self.greedy_proposer = GreedyProposer()
        self.uniform_proposer = UniformProposer()
        self.grid_search_proposer = GridSearchProposer()

    def test_greedy_two_table(self) -> None:
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

    def test_uniform_three_table(self) -> None:
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

    def test_grid_search_three_table(self) -> None:
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
        search_space = self.enumerator.enumerate(
            module=model,
            sharders=[
                cast(ModuleSharder[torch.nn.Module], EmbeddingBagCollectionSharder())
            ],
        )

        """
        All sharding types but DP will have 3 possible compute kernels after pruning:
            - batched_fused
            - batched_fused_uvm_caching
            - batched_fused_uvm
        DP will have 1 possible compute kernel: batched_dense
        So the total number of pruned options will be:
            (num_sharding_types - 1) * 3 + 1 = 16
        """
        num_pruned_options = (len(ShardingType) - 1) * 3 + 1
        self.grid_search_proposer.load(search_space)
        for (
            sharding_options
        ) in self.grid_search_proposer._sharding_options_by_fqn.values():
            # number of sharding types after pruning is number of sharding types * 3
            # 3 compute kernels batched_fused/batched_dense, batched_fused_uvm_caching, batched_fused_uvm
            self.assertEqual(len(sharding_options), num_pruned_options)

        num_proposals = 0
        proposal = self.grid_search_proposer.propose()
        while proposal:
            self.grid_search_proposer.feedback(partitionable=True)
            proposal = self.grid_search_proposer.propose()
            num_proposals += 1

        self.assertEqual(num_pruned_options ** len(tables), num_proposals)
