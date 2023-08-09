#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest
from typing import cast, List

from torch import nn
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.planner.constants import BATCH_SIZE
from torchrec.distributed.planner.enumerators import EmbeddingEnumerator
from torchrec.distributed.planner.partitioners import (
    GreedyPerfPartitioner,
    MemoryBalancedPartitioner,
)
from torchrec.distributed.planner.types import (
    ParameterConstraints,
    PartitionByType,
    Perf,
    PlannerError,
    Storage,
    Topology,
)
from torchrec.distributed.planner.utils import reset_shard_rank
from torchrec.distributed.test_utils.test_model import TestSparseNN
from torchrec.distributed.types import ModuleSharder, ShardingType
from torchrec.modules.embedding_configs import EmbeddingBagConfig


class RWSharder(EmbeddingBagCollectionSharder, ModuleSharder[nn.Module]):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [ShardingType.ROW_WISE.value]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [EmbeddingComputeKernel.DENSE.value]


class TWSharder(EmbeddingBagCollectionSharder, ModuleSharder[nn.Module]):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [ShardingType.TABLE_WISE.value]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [EmbeddingComputeKernel.DENSE.value]


class TWRWSharder(EmbeddingBagCollectionSharder, ModuleSharder[nn.Module]):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [ShardingType.TABLE_ROW_WISE.value]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [EmbeddingComputeKernel.DENSE.value]


class TWCWSharder(EmbeddingBagCollectionSharder, ModuleSharder[nn.Module]):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [ShardingType.TABLE_COLUMN_WISE.value]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [EmbeddingComputeKernel.DENSE.value]


class HostLevelSharder(EmbeddingBagCollectionSharder, ModuleSharder[nn.Module]):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [ShardingType.TABLE_ROW_WISE.value, ShardingType.TABLE_COLUMN_WISE.value]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [EmbeddingComputeKernel.DENSE.value]


class TestGreedyPerfPartitioner(unittest.TestCase):
    def setUp(self) -> None:
        compute_device = "cuda"
        self.topology = Topology(world_size=2, compute_device=compute_device)
        tables = [
            EmbeddingBagConfig(
                num_embeddings=100 + i,
                embedding_dim=4 * (10 + i),
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(4)
        ]
        self.topology = Topology(world_size=2, compute_device=compute_device)
        self.model = TestSparseNN(tables=tables, weighted_tables=[])
        self.enumerator = EmbeddingEnumerator(
            topology=self.topology, batch_size=BATCH_SIZE
        )
        self.partitioner = GreedyPerfPartitioner()

    def test_tw_balanced_perf_device(self) -> None:
        sharding_options = self.enumerator.enumerate(
            module=self.model, sharders=[TWSharder()]
        )

        for sharding_option in sharding_options:
            sharding_option.shards[0].perf = Perf(
                fwd_compute=40, fwd_comms=30, bwd_compute=20, bwd_comms=10
            )
            sharding_option.shards[0].storage = Storage(hbm=1000, ddr=1000)

        candidate_topology = copy.deepcopy(self.topology)
        sharding_plan = self.partitioner.partition(
            proposal=sharding_options,
            storage_constraint=candidate_topology,
        )
        # pyre-ignore [16]
        solution_topology = self.partitioner._topology

        expected_ranks = {
            "table_0": [0],
            "table_1": [1],
            "table_2": [0],
            "table_3": [1],
        }

        ranks = {
            sharding_option.name: [shard.rank for shard in sharding_option.shards]
            for sharding_option in sharding_plan
        }
        self.assertEqual(expected_ranks, ranks)

        expected_perf = Perf(
            fwd_compute=80,
            fwd_comms=60,
            bwd_compute=40,
            bwd_comms=20,
        )

        self.assertEqual(solution_topology.devices[0].perf, expected_perf)
        self.assertEqual(solution_topology.devices[1].perf, expected_perf)

        self.assertEqual(
            solution_topology.devices[0].storage,
            self.topology.devices[0].storage - Storage(2000, 2000),
        )
        self.assertEqual(
            solution_topology.devices[1].storage,
            self.topology.devices[1].storage - Storage(2000, 2000),
        )

    def test_tw_unbalanced_perf_device(self) -> None:
        sharding_options = self.enumerator.enumerate(
            module=self.model, sharders=[TWSharder()]
        )

        for i, sharding_option in enumerate(sharding_options):
            perf = (
                Perf(fwd_compute=40, fwd_comms=30, bwd_compute=20, bwd_comms=10)
                if i > 0
                else Perf(fwd_compute=75, fwd_comms=75, bwd_compute=75, bwd_comms=75)
            )
            sharding_option.shards[0].perf = perf
            sharding_option.shards[0].storage = Storage(hbm=1000, ddr=1000)

        candidate_topology = copy.deepcopy(self.topology)
        sharding_plan = self.partitioner.partition(
            proposal=sharding_options,
            storage_constraint=candidate_topology,
        )
        # pyre-ignore[16]
        solution_topology = self.partitioner._topology

        expected_ranks = {
            "table_0": [0],
            "table_1": [1],
            "table_2": [1],
            "table_3": [1],
        }

        ranks = {
            sharding_option.name: [shard.rank for shard in sharding_option.shards]
            for sharding_option in sharding_plan
        }
        self.assertEqual(expected_ranks, ranks)

        expected_perfs = [
            Perf(fwd_compute=75, fwd_comms=75, bwd_compute=75, bwd_comms=75),
            Perf(fwd_compute=120, fwd_comms=90, bwd_compute=60, bwd_comms=30),
        ]

        self.assertEqual(solution_topology.devices[0].perf, expected_perfs[0])
        self.assertEqual(solution_topology.devices[1].perf, expected_perfs[1])

        self.assertEqual(
            solution_topology.devices[0].storage,
            self.topology.devices[0].storage - Storage(1000, 1000),
        )
        self.assertEqual(
            solution_topology.devices[1].storage,
            self.topology.devices[1].storage - Storage(3000, 3000),
        )

    def test_tw_balanced_perf_host(self) -> None:
        self.topology = Topology(
            world_size=16, local_world_size=8, compute_device="cuda"
        )
        tables = [
            EmbeddingBagConfig(
                num_embeddings=64,
                embedding_dim=4 * (10 + i),
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(4)
        ]

        self.model = TestSparseNN(tables=tables, weighted_tables=[])
        self.enumerator = EmbeddingEnumerator(
            topology=self.topology, batch_size=BATCH_SIZE
        )
        self.partitioner = GreedyPerfPartitioner()
        sharding_options = self.enumerator.enumerate(
            module=self.model, sharders=[TWRWSharder()]
        )
        for sharding_option in sharding_options:
            for shard in sharding_option.shards:
                shard.perf = Perf(
                    fwd_compute=40, fwd_comms=30, bwd_compute=20, bwd_comms=10
                )
                shard.storage = Storage(hbm=1000, ddr=1000)
            sharding_option.partition_by = PartitionByType.HOST.value

        candidate_topology = copy.deepcopy(self.topology)
        sharding_plan = self.partitioner.partition(
            proposal=sharding_options,
            storage_constraint=candidate_topology,
        )
        # pyre-ignore[16]
        solution_topology = self.partitioner._topology

        expected_ranks = {
            "table_0": [0, 1, 2, 3, 4, 5, 6, 7],
            "table_1": [8, 9, 10, 11, 12, 13, 14, 15],
            "table_2": [0, 1, 2, 3, 4, 5, 6, 7],
            "table_3": [8, 9, 10, 11, 12, 13, 14, 15],
        }

        ranks = {
            sharding_option.name: [shard.rank for shard in sharding_option.shards]
            for sharding_option in sharding_plan
        }
        self.assertEqual(expected_ranks, ranks)

        for i in range(self.topology.world_size):
            self.assertEqual(
                solution_topology.devices[i].storage,
                # there are two shards allocated to each device
                self.topology.devices[i].storage - Storage(2000, 2000),
            )
            expected_perf = Perf(
                fwd_compute=80,
                fwd_comms=60,
                bwd_compute=40,
                bwd_comms=20,
            )
            self.assertEqual(solution_topology.devices[i].perf, expected_perf)

    def test_rw_unbalanced_perf_uniform(self) -> None:
        self.topology = Topology(world_size=4, compute_device="cuda")
        tables = [
            EmbeddingBagConfig(
                num_embeddings=64,
                embedding_dim=4 * (10 + i),
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(4)
        ]

        self.model = TestSparseNN(tables=tables, weighted_tables=[])
        self.enumerator = EmbeddingEnumerator(
            topology=self.topology, batch_size=BATCH_SIZE
        )
        self.partitioner = GreedyPerfPartitioner()
        sharding_options = self.enumerator.enumerate(
            module=self.model, sharders=[RWSharder()]
        )
        for sharding_option in sharding_options:
            for shard in sharding_option.shards:
                shard.perf = Perf(
                    fwd_compute=25, fwd_comms=25, bwd_compute=25, bwd_comms=25
                )
                shard.storage = Storage(hbm=1000, ddr=1000)
            sharding_option.partition_by = PartitionByType.UNIFORM.value

        candidate_topology = copy.deepcopy(self.topology)
        sharding_plan = self.partitioner.partition(
            proposal=sharding_options,
            storage_constraint=candidate_topology,
        )
        # pyre-ignore[16]
        solution_topology = self.partitioner._topology

        expected_ranks = {
            "table_0": [0, 1, 2, 3],
            "table_1": [0, 1, 2, 3],
            "table_2": [0, 1, 2, 3],
            "table_3": [0, 1, 2, 3],
        }

        ranks = {
            sharding_option.name: [shard.rank for shard in sharding_option.shards]
            for sharding_option in sharding_plan
        }
        self.assertEqual(expected_ranks, ranks)

        for i in range(self.topology.world_size):
            self.assertEqual(
                solution_topology.devices[i].storage,
                self.topology.devices[i].storage - Storage(4000, 4000),
            )

    def test_twcw_unbalanced_perf_host(self) -> None:
        self.topology = Topology(
            world_size=16, local_world_size=8, compute_device="cuda"
        )
        constraints = {
            "table_0": ParameterConstraints(min_partition=4 * 2),
            "table_1": ParameterConstraints(min_partition=4 * 10),
            "table_2": ParameterConstraints(min_partition=4 * 5),
            "table_3": ParameterConstraints(min_partition=4 * 8),
        }
        tables = [
            EmbeddingBagConfig(
                num_embeddings=64,
                embedding_dim=80 * (i + 1),
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(4)
        ]

        self.model = TestSparseNN(tables=tables)
        self.enumerator = EmbeddingEnumerator(
            topology=self.topology, batch_size=BATCH_SIZE, constraints=constraints
        )
        self.partitioner = GreedyPerfPartitioner()
        sharding_options = self.enumerator.enumerate(
            module=self.model,
            sharders=[TWCWSharder()],
        )
        for sharding_option in sharding_options:
            for shard in sharding_option.shards:
                shard.perf = Perf(
                    fwd_compute=25, fwd_comms=25, bwd_compute=25, bwd_comms=25
                )
                shard.storage = Storage(hbm=1000, ddr=1000)
            sharding_option.partition_by = PartitionByType.HOST.value

        sharding_plan = self.partitioner.partition(
            proposal=sharding_options,
            storage_constraint=self.topology,
        )

        expected_ranks = {
            "table_0": [8, 9, 10, 11, 12, 13, 14, 15, 8, 9],
            "table_1": [4, 5, 6, 7],
            "table_2": [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3],
            "table_3": [10, 11, 12, 13, 14, 15, 8, 9, 10, 11],
        }

        ranks = {
            sharding_option.name: [shard.rank for shard in sharding_option.shards]
            for sharding_option in sharding_plan
        }
        self.assertEqual(expected_ranks, ranks)

    def test_twrw_and_twcw_perf_host(self) -> None:
        self.topology = Topology(
            world_size=16, local_world_size=8, compute_device="cuda"
        )
        constraints = {
            "table_0": ParameterConstraints(
                sharding_types=[ShardingType.TABLE_ROW_WISE.value]
            ),
            "table_1": ParameterConstraints(
                sharding_types=[ShardingType.TABLE_COLUMN_WISE.value],
                min_partition=4 * 8,
            ),
            "table_2": ParameterConstraints(
                sharding_types=[ShardingType.TABLE_COLUMN_WISE.value],
                min_partition=4 * 10,
            ),
            "table_3": ParameterConstraints(
                sharding_types=[ShardingType.TABLE_ROW_WISE.value]
            ),
        }
        tables = [
            EmbeddingBagConfig(
                num_embeddings=128,
                embedding_dim=80 * (i + 1),
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(4)
        ]

        self.model = TestSparseNN(tables=tables)
        self.enumerator = EmbeddingEnumerator(
            topology=self.topology, batch_size=BATCH_SIZE, constraints=constraints
        )
        self.partitioner = GreedyPerfPartitioner()
        sharding_options = self.enumerator.enumerate(
            module=self.model,
            sharders=[HostLevelSharder()],
        )

        for sharding_option in sharding_options:
            for shard in sharding_option.shards:
                shard.perf = Perf(
                    fwd_compute=25, fwd_comms=25, bwd_compute=25, bwd_comms=25
                )
                shard.storage = Storage(hbm=1000, ddr=1000)
            sharding_option.partition_by = PartitionByType.HOST.value

        sharding_plan = self.partitioner.partition(
            proposal=sharding_options,
            storage_constraint=self.topology,
        )
        expected_ranks = {
            "table_0": [0, 1, 2, 3, 4, 5, 6, 7],
            "table_1": [8, 9, 10, 11, 12],
            "table_2": [0, 1, 2, 3, 4, 5],
            "table_3": [8, 9, 10, 11, 12, 13, 14, 15],
        }

        ranks = {
            sharding_option.name: [shard.rank for shard in sharding_option.shards]
            for sharding_option in sharding_plan
        }

        self.assertEqual(expected_ranks, ranks)

    def test_twrw_and_twcw_cohost(self) -> None:
        self.topology = Topology(
            world_size=16, local_world_size=8, compute_device="cuda"
        )
        constraints = {
            "table_0": ParameterConstraints(
                sharding_types=[ShardingType.TABLE_ROW_WISE.value]
            ),
            "table_1": ParameterConstraints(
                sharding_types=[ShardingType.TABLE_COLUMN_WISE.value],
                min_partition=4 * 8,
            ),
            "table_2": ParameterConstraints(
                sharding_types=[ShardingType.TABLE_COLUMN_WISE.value],
                min_partition=4 * 10,
            ),
            "table_3": ParameterConstraints(
                sharding_types=[ShardingType.TABLE_ROW_WISE.value]
            ),
        }
        tables = [
            EmbeddingBagConfig(
                num_embeddings=128,
                embedding_dim=80 * (i + 1),
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(4)
        ]

        self.model = TestSparseNN(tables=tables)
        self.enumerator = EmbeddingEnumerator(
            topology=self.topology, batch_size=BATCH_SIZE, constraints=constraints
        )
        self.partitioner = GreedyPerfPartitioner()
        sharding_options = self.enumerator.enumerate(
            module=self.model,
            sharders=[HostLevelSharder()],
        )

        for i, sharding_option in enumerate(sharding_options):
            for shard in sharding_option.shards:
                shard.perf = Perf(
                    fwd_compute=25, fwd_comms=25, bwd_compute=25, bwd_comms=25
                )
                shard.storage = Storage(hbm=1000, ddr=1000)
            sharding_option.partition_by = PartitionByType.HOST.value
            if i <= 2:
                sharding_option.dependency = "host_0"

        sharding_plan = self.partitioner.partition(
            proposal=sharding_options,
            storage_constraint=self.topology,
        )
        expected_ranks = {
            "table_0": [0, 1, 2, 3, 4, 5, 6, 7],
            "table_1": [0, 1, 2, 3, 4],
            "table_2": [5, 6, 7, 0, 1, 2],
            "table_3": [8, 9, 10, 11, 12, 13, 14, 15],
        }

        ranks = {
            sharding_option.name: [shard.rank for shard in sharding_option.shards]
            for sharding_option in sharding_plan
        }

        self.assertEqual(expected_ranks, ranks)

        # pyre-ignore [16]
        solution_topology = self.partitioner._topology
        for i in range(self.topology.world_size):
            total_storage = Storage(0, 0)
            total_perf = Perf(
                fwd_compute=0,
                fwd_comms=0,
                bwd_compute=0,
                bwd_comms=0,
            )
            for sharding_option in sharding_plan:
                for shard in sharding_option.shards:
                    if shard.rank == i:
                        total_storage += cast(Storage, shard.storage)
                        total_perf += cast(Perf, shard.perf)
            self.assertEqual(
                solution_topology.devices[i].storage + total_storage,
                self.topology.devices[i].storage,
            )
            self.assertEqual(solution_topology.devices[i].perf, total_perf)

    def test_oom(self) -> None:
        self.topology = Topology(
            world_size=2, local_world_size=1, compute_device="cuda"
        )
        constraints = {
            "table_0": ParameterConstraints(
                sharding_types=[ShardingType.TABLE_ROW_WISE.value]
            ),
            "table_1": ParameterConstraints(
                sharding_types=[ShardingType.TABLE_COLUMN_WISE.value],
                min_partition=4 * 4,
            ),
            "table_2": ParameterConstraints(
                sharding_types=[ShardingType.TABLE_COLUMN_WISE.value],
                min_partition=4 * 7,
            ),
            "table_3": ParameterConstraints(
                sharding_types=[ShardingType.TABLE_ROW_WISE.value]
            ),
        }
        tables = [
            EmbeddingBagConfig(
                num_embeddings=128,
                embedding_dim=20 * (i + 1),
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(4)
        ]

        self.model = TestSparseNN(tables=tables)
        self.enumerator = EmbeddingEnumerator(
            topology=self.topology, batch_size=BATCH_SIZE, constraints=constraints
        )
        self.partitioner = GreedyPerfPartitioner()
        sharding_options = self.enumerator.enumerate(
            module=self.model,
            sharders=[HostLevelSharder()],
        )

        for i, sharding_option in enumerate(sharding_options):
            for shard in sharding_option.shards:
                shard.perf = Perf(
                    fwd_compute=25, fwd_comms=25, bwd_compute=25, bwd_comms=25
                )
                shard.storage = Storage(
                    # pyre-ignore [6]
                    hbm=self.topology.devices[0].storage.hbm / 2,
                    # pyre-ignore [6]
                    ddr=self.topology.devices[0].storage.ddr / 2,
                )
            sharding_option.partition_by = PartitionByType.HOST.value
            if i <= 2:
                sharding_option.dependency = "host_0"

        with self.assertRaises(PlannerError):
            self.partitioner.partition(
                proposal=sharding_options,
                storage_constraint=self.topology,
            )


class TestMemoryBalancedPartitioner(unittest.TestCase):
    def setUp(self) -> None:
        compute_device = "cuda"
        self.topology = Topology(world_size=2, compute_device=compute_device)
        tables = [
            EmbeddingBagConfig(
                num_embeddings=100 + i,
                embedding_dim=4 * (10 + i),
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(3)
        ]
        self.topology = Topology(
            world_size=2,
            compute_device=compute_device,
            hbm_cap=2000 * 1024**2,
        )
        self.model = TestSparseNN(tables=tables, weighted_tables=[])
        self.enumerator = EmbeddingEnumerator(
            topology=self.topology, batch_size=BATCH_SIZE
        )
        self.greedy_perf_partitioner = GreedyPerfPartitioner()
        self.memory_balanced_partitioner = MemoryBalancedPartitioner(tolerance=100)

    def test_same_sharding_plan(self) -> None:
        sharding_options = self.enumerator.enumerate(
            module=self.model, sharders=[TWSharder()]
        )

        for sharding_option in sharding_options:
            sharding_option.shards[0].perf = Perf(
                fwd_compute=40, fwd_comms=30, bwd_compute=20, bwd_comms=10
            )
            sharding_option.shards[0].storage = Storage(
                hbm=1000 * 1024**2, ddr=1000 * 1024**2
            )

        greedy_perf_sharding_plan = self.greedy_perf_partitioner.partition(
            proposal=sharding_options,
            storage_constraint=self.topology,
        )
        greedy_perf_ranks = {
            sharding_option.name: [shard.rank for shard in sharding_option.shards]
            for sharding_option in greedy_perf_sharding_plan
        }

        reset_shard_rank(sharding_options)
        memory_balanced_sharding_plan = self.memory_balanced_partitioner.partition(
            proposal=sharding_options,
            storage_constraint=self.topology,
        )
        memory_balanced_ranks = {
            sharding_option.name: [shard.rank for shard in sharding_option.shards]
            for sharding_option in memory_balanced_sharding_plan
        }
        self.assertEqual(greedy_perf_ranks, memory_balanced_ranks)

    def test_different_sharding_plan(self) -> None:
        sharding_options = self.enumerator.enumerate(
            module=self.model, sharders=[TWSharder()]
        )

        for i, sharding_option in enumerate(sharding_options):
            sharding_option.shards[0].perf = Perf(
                fwd_compute=40 * (i + 1), fwd_comms=0, bwd_compute=0, bwd_comms=0
            )
            sharding_option.shards[0].storage = Storage(
                hbm=(1500 - i * 500) * 1024**2, ddr=1000 * 1024**2
            )

        greedy_perf_sharding_plan = self.greedy_perf_partitioner.partition(
            proposal=sharding_options,
            storage_constraint=self.topology,
        )
        greedy_perf_ranks = {
            sharding_option.name: [shard.rank for shard in sharding_option.shards]
            for sharding_option in greedy_perf_sharding_plan
        }
        greedy_perf_expected_ranks = {
            "table_0": [0],
            "table_1": [1],
            "table_2": [0],
        }
        self.assertEqual(greedy_perf_ranks, greedy_perf_expected_ranks)

        greedy_perf_hbm_uses = [0] * self.topology.world_size
        for sharding_option in sharding_options:
            for shard in sharding_option.shards:
                if shard.storage and shard.rank is not None:
                    greedy_perf_hbm_uses[
                        shard.rank
                    ] += shard.storage.hbm  # pyre-ignore[16]

        reset_shard_rank(sharding_options)
        memory_balanced_sharding_plan = self.memory_balanced_partitioner.partition(
            proposal=sharding_options,
            storage_constraint=self.topology,
        )
        memory_balanced_ranks = {
            sharding_option.name: [shard.rank for shard in sharding_option.shards]
            for sharding_option in memory_balanced_sharding_plan
        }
        memory_balanced_expected_ranks = {
            "table_0": [0],
            "table_1": [1],
            "table_2": [1],
        }
        self.assertEqual(memory_balanced_ranks, memory_balanced_expected_ranks)

        memory_balanced_hbm_uses = [0.0] * self.topology.world_size
        for sharding_option in sharding_options:
            for shard in sharding_option.shards:
                if shard.storage and shard.rank:
                    memory_balanced_hbm_uses[shard.rank] += shard.storage.hbm

        self.assertTrue(max(memory_balanced_hbm_uses) < max(greedy_perf_hbm_uses))
