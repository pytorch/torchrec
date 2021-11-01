#!/usr/bin/env python3

import copy
import unittest
from typing import List

from torch import nn
from torchrec.distributed.embedding import EmbeddingBagCollectionSharder
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.planner.new.enumerators import ShardingEnumerator
from torchrec.distributed.planner.new.partitioners import GreedyCostPartitioner
from torchrec.distributed.planner.new.types import Storage, Topology, PartitionByType
from torchrec.distributed.tests.test_model import TestSparseNN
from torchrec.distributed.types import ModuleSharder, ShardingType
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
)


class RWSharder(
    EmbeddingBagCollectionSharder[EmbeddingBagCollection], ModuleSharder[nn.Module]
):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [ShardingType.ROW_WISE.value]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [EmbeddingComputeKernel.DENSE.value]


class TWSharder(
    EmbeddingBagCollectionSharder[EmbeddingBagCollection], ModuleSharder[nn.Module]
):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [ShardingType.TABLE_WISE.value]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [EmbeddingComputeKernel.DENSE.value]


class TWRWSharder(
    EmbeddingBagCollectionSharder[EmbeddingBagCollection], ModuleSharder[nn.Module]
):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [ShardingType.TABLE_ROW_WISE.value]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [EmbeddingComputeKernel.DENSE.value]


class TestGreedyCostPartitioner(unittest.TestCase):
    def setUp(self) -> None:
        compute_device = "cuda"
        self.topology = Topology(world_size=2, compute_device=compute_device)
        tables = [
            EmbeddingBagConfig(
                num_embeddings=100 + i,
                embedding_dim=10 + i,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(4)
        ]
        self.topology = Topology(world_size=2, compute_device=compute_device)
        self.model = TestSparseNN(tables=tables, weighted_tables=[])
        self.enumerator = ShardingEnumerator(topology=self.topology)
        self.partitioner = GreedyCostPartitioner()

    def test_tw_balanced_cost_device(self) -> None:
        sharding_options = self.enumerator.run(
            module=self.model, sharders=[TWSharder()]
        )

        for sharding_option in sharding_options:
            sharding_option.cost = 100
            sharding_option.shards[0].cost = 100
            sharding_option.shards[0].storage = Storage(hbm=1000, ddr=1000)

        candidate_topology = copy.deepcopy(self.topology)
        self.partitioner.run(
            sharding_options=sharding_options,
            topology=candidate_topology,
        )
        expected_ranks = {
            "table_0": [1],
            "table_1": [0],
            "table_2": [1],
            "table_3": [0],
        }

        ranks = {
            sharding_option.name: [shard.rank for shard in sharding_option.shards]
            for sharding_option in sharding_options
        }
        self.assertEqual(expected_ranks, ranks)

        self.assertEqual(candidate_topology.devices[0].cost, 200)
        self.assertEqual(candidate_topology.devices[1].cost, 200)

        self.assertEqual(
            candidate_topology.devices[0].storage,
            self.topology.devices[0].storage - Storage(2000, 2000),
        )
        self.assertEqual(
            candidate_topology.devices[1].storage,
            self.topology.devices[1].storage - Storage(2000, 2000),
        )

    def test_tw_unbalanced_cost_device(self) -> None:
        sharding_options = self.enumerator.run(
            module=self.model, sharders=[TWSharder()]
        )

        for i, sharding_option in enumerate(sharding_options):
            cost = 100 if i > 0 else 300
            sharding_option.cost = cost
            sharding_option.shards[0].cost = cost
            sharding_option.shards[0].storage = Storage(hbm=1000, ddr=1000)

        candidate_topology = copy.deepcopy(self.topology)
        self.partitioner.run(
            sharding_options=sharding_options,
            topology=candidate_topology,
        )
        expected_ranks = {
            "table_0": [0],
            "table_1": [1],
            "table_2": [1],
            "table_3": [1],
        }

        ranks = {
            sharding_option.name: [shard.rank for shard in sharding_option.shards]
            for sharding_option in sharding_options
        }
        self.assertEqual(expected_ranks, ranks)

        self.assertEqual(candidate_topology.devices[0].cost, 300)
        self.assertEqual(candidate_topology.devices[1].cost, 300)

        self.assertEqual(
            candidate_topology.devices[0].storage,
            self.topology.devices[0].storage - Storage(1000, 1000),
        )
        self.assertEqual(
            candidate_topology.devices[1].storage,
            self.topology.devices[1].storage - Storage(3000, 3000),
        )

    def test_tw_balanced_cost_host(self) -> None:
        self.topology = Topology(
            world_size=16, local_world_size=8, compute_device="cuda"
        )
        tables = [
            EmbeddingBagConfig(
                num_embeddings=64,
                embedding_dim=10 + i,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(4)
        ]

        self.model = TestSparseNN(tables=tables, weighted_tables=[])
        self.enumerator = ShardingEnumerator(topology=self.topology)
        self.partitioner = GreedyCostPartitioner()
        sharding_options = self.enumerator.run(
            module=self.model, sharders=[TWRWSharder()]
        )
        for sharding_option in sharding_options:
            cost = 100.0
            for shard in sharding_option.shards:
                shard.cost = cost
                shard.storage = Storage(hbm=1000, ddr=1000)
            sharding_option.cost = cost * sharding_option.num_shards
            sharding_option.partition_by = PartitionByType.HOST.value

        candidate_topology = copy.deepcopy(self.topology)
        self.partitioner.run(
            sharding_options=sharding_options,
            topology=candidate_topology,
        )

        expected_ranks = {
            "table_0": [8, 9, 10, 11, 12, 13, 14, 15],
            "table_1": [0, 1, 2, 3, 4, 5, 6, 7],
            "table_2": [8, 9, 10, 11, 12, 13, 14, 15],
            "table_3": [0, 1, 2, 3, 4, 5, 6, 7],
        }

        ranks = {
            sharding_option.name: [shard.rank for shard in sharding_option.shards]
            for sharding_option in sharding_options
        }
        self.assertEqual(expected_ranks, ranks)

        for i in range(self.topology.world_size):
            self.assertEqual(
                candidate_topology.devices[i].storage,
                # there are two shards allocated to each device
                self.topology.devices[i].storage - Storage(2000, 2000),
            )

    def test_rw_unbalanced_cost_uniform(self) -> None:
        self.topology = Topology(world_size=4, compute_device="cuda")
        tables = [
            EmbeddingBagConfig(
                num_embeddings=64,
                embedding_dim=10 + i,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(4)
        ]

        self.model = TestSparseNN(tables=tables, weighted_tables=[])
        self.enumerator = ShardingEnumerator(topology=self.topology)
        self.partitioner = GreedyCostPartitioner()
        sharding_options = self.enumerator.run(
            module=self.model, sharders=[RWSharder()]
        )
        for sharding_option in sharding_options:
            cost = 100.0
            for shard in sharding_option.shards:
                shard.cost = cost
                shard.storage = Storage(hbm=1000, ddr=1000)
            sharding_option.cost = cost * sharding_option.num_shards
            sharding_option.partition_by = PartitionByType.UNIFORM.value

        candidate_topology = copy.deepcopy(self.topology)
        self.partitioner.run(
            sharding_options=sharding_options,
            topology=candidate_topology,
        )

        expected_ranks = {
            "table_0": [0, 1, 2, 3],
            "table_1": [0, 1, 2, 3],
            "table_2": [0, 1, 2, 3],
            "table_3": [0, 1, 2, 3],
        }

        ranks = {
            sharding_option.name: [shard.rank for shard in sharding_option.shards]
            for sharding_option in sharding_options
        }
        self.assertEqual(expected_ranks, ranks)

        for i in range(self.topology.world_size):
            self.assertEqual(
                candidate_topology.devices[i].storage,
                self.topology.devices[i].storage - Storage(4000, 4000),
            )
