#!/usr/bin/env python3

import unittest

from torchrec.distributed.embeddingbag import (
    EmbeddingBagCollectionSharder,
)
from torchrec.distributed.planner.new.calculators import EmbeddingWTCostCalculator
from torchrec.distributed.planner.new.enumerators import EmbeddingEnumerator
from torchrec.distributed.planner.new.rankers import DepthRanker
from torchrec.distributed.planner.new.types import Topology
from torchrec.distributed.tests.test_model import TestSparseNN
from torchrec.modules.embedding_configs import EmbeddingBagConfig


class TestDepthRanker(unittest.TestCase):
    def setUp(self) -> None:
        topology = Topology(world_size=2, compute_device="cuda")
        self.calculator = EmbeddingWTCostCalculator(topology=topology)
        self.enumerator = EmbeddingEnumerator(topology=topology)
        self.ranker = DepthRanker()

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
        sharding_options = self.enumerator.run(
            module=model, sharders=[EmbeddingBagCollectionSharder()]
        )
        self.calculator.run(sharding_options)
        rank_stack = self.ranker.run(sharding_options=sharding_options)

        # simulate first five iterations:
        output = []
        for _ in range(5):
            candidates = rank_stack.bulk_pop()
            candidates.sort(key=lambda x: (x.cost, x.name))
            output.append(
                [
                    (
                        candidate.name,
                        candidate.sharding_type,
                        candidate.compute_kernel,
                    )
                    for candidate in candidates
                ]
            )
            drop = candidates[0]
            keep = candidates[1:]
            rank_stack.remove(drop)
            rank_stack.bulk_push(keep)

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
                    "table_0",
                    "data_parallel",
                    "dense",
                ),
                (
                    "table_1",
                    "data_parallel",
                    "dense",
                ),
            ],
            [
                (
                    "table_1",
                    "data_parallel",
                    "dense",
                ),
                (
                    "table_0",
                    "row_wise",
                    "batched_fused",
                ),
            ],
            [
                (
                    "table_0",
                    "row_wise",
                    "batched_fused",
                ),
                (
                    "table_1",
                    "row_wise",
                    "batched_fused",
                ),
            ],
        ]

        self.assertEqual(expected_output, output)
