#!/usr/bin/env python3

import unittest
from typing import List, cast

from torchrec.distributed.embeddingbag import (
    EmbeddingBagCollectionSharder,
)
from torchrec.distributed.planner.new.enumerators import EmbeddingEnumerator
from torchrec.distributed.planner.new.proposers import GreedyProposer
from torchrec.distributed.planner.new.types import Topology, ShardingOption
from torchrec.distributed.tests.test_model import TestSparseNN
from torchrec.modules.embedding_configs import EmbeddingBagConfig


class TestGreedyProposer(unittest.TestCase):
    def setUp(self) -> None:
        topology = Topology(world_size=2, compute_device="cuda")
        self.enumerator = EmbeddingEnumerator(topology=topology)
        self.proposer = GreedyProposer()

    def test_two_table_cost(self) -> None:
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

        model = TestSparseNN(tables=tables, weighted_tables=[])
        search_space = self.enumerator.enumerate(
            module=model, sharders=[EmbeddingBagCollectionSharder()]
        )
        self.proposer.load(search_space)

        # simulate first five iterations:
        output = []
        for _ in range(5):
            proposal = cast(List[ShardingOption], self.proposer.propose())
            proposal.sort(
                key=lambda sharding_option: (
                    max([shard.cost for shard in sharding_option.shards]),
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
            self.proposer.feedback(partitionable=True)

        expected_output = [
            [
                (
                    "table_0",
                    "data_parallel",
                    "batched_dense",
                ),
                (
                    "table_1",
                    "data_parallel",
                    "batched_dense",
                ),
            ],
            [
                (
                    "table_1",
                    "data_parallel",
                    "batched_dense",
                ),
                (
                    "table_0",
                    "data_parallel",
                    "dense",
                ),
            ],
            [
                (
                    "table_1",
                    "data_parallel",
                    "batched_dense",
                ),
                (
                    "table_0",
                    "row_wise",
                    "batched_fused",
                ),
            ],
            [
                (
                    "table_1",
                    "data_parallel",
                    "batched_dense",
                ),
                (
                    "table_0",
                    "table_wise",
                    "batched_fused",
                ),
            ],
            [
                (
                    "table_1",
                    "data_parallel",
                    "batched_dense",
                ),
                (
                    "table_0",
                    "row_wise",
                    "batched_dense",
                ),
            ],
        ]

        self.assertEqual(expected_output, output)
