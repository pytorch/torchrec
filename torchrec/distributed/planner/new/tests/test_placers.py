#!/usr/bin/env python3

import unittest
from typing import List, cast

import torch
from torch import nn
from torchrec.distributed.embedding import EmbeddingBagCollectionSharder
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.planner.new.calculators import EmbeddingWTCostCalculator
from torchrec.distributed.planner.new.enumerators import ShardingEnumerator
from torchrec.distributed.planner.new.partitioners import GreedyCostPartitioner
from torchrec.distributed.planner.new.placers import EmbeddingPlacer
from torchrec.distributed.planner.new.rankers import FlatRanker
from torchrec.distributed.planner.new.types import Topology, PartitionError
from torchrec.distributed.tests.test_model import TestSparseNN
from torchrec.distributed.types import ModuleSharder, ShardingType
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
)


class TWvsRWSharder(
    EmbeddingBagCollectionSharder[EmbeddingBagCollection], ModuleSharder[nn.Module]
):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [ShardingType.ROW_WISE.value, ShardingType.TABLE_WISE.value]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [EmbeddingComputeKernel.DENSE.value]


class TestEmbeddingPlacer(unittest.TestCase):
    def setUp(self) -> None:
        compute_device = "cuda"
        self.topology = Topology(world_size=2, compute_device=compute_device)
        self.enumerator = ShardingEnumerator(topology=self.topology)
        self.ranker = FlatRanker(calculator=EmbeddingWTCostCalculator(self.topology))
        self.placer = EmbeddingPlacer(
            topology=self.topology, partitioner=GreedyCostPartitioner()
        )

    def test_tw_solution(self) -> None:
        tables = [
            EmbeddingBagConfig(
                num_embeddings=100,
                embedding_dim=10,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(4)
        ]
        model = TestSparseNN(tables=tables, sparse_device=torch.device("meta"))
        sharding_options = self.enumerator.run(module=model, sharders=[TWvsRWSharder()])
        sharding_plan = self.placer.run(rank_stack=self.ranker.run(sharding_options))
        expected_ranks = [[0], [0], [1], [1]]
        ranks = [
            cast(List[int], param_shard.ranks)
            for param_shard in sharding_plan.plan["sparse.ebc"].values()
        ]
        self.assertEqual(sorted(expected_ranks), sorted(ranks))

    def test_hidden_rw_solution(self) -> None:
        tables = [
            EmbeddingBagConfig(
                num_embeddings=100,
                embedding_dim=10,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(3)
        ]
        model = TestSparseNN(tables=tables, sparse_device=torch.device("meta"))
        sharding_options = self.enumerator.run(module=model, sharders=[TWvsRWSharder()])
        sharding_plan = self.placer.run(rank_stack=self.ranker.run(sharding_options))
        expected_ranks = [[0], [0, 1], [1]]
        ranks = [
            cast(List[int], param_shard.ranks)
            for param_shard in sharding_plan.plan["sparse.ebc"].values()
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
            for i in range(10)
        ]
        model = TestSparseNN(tables=tables, sparse_device=torch.device("meta"))
        sharding_options = self.enumerator.run(module=model, sharders=[TWvsRWSharder()])
        with self.assertRaises(PartitionError):
            self.placer.run(rank_stack=self.ranker.run(sharding_options))
        self.assertEqual(self.placer._counter, 11)
