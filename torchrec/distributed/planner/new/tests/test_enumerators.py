#!/usr/bin/env python3

import math
import unittest
from typing import List

from torchrec.distributed.embedding import EmbeddingBagCollectionSharder
from torchrec.distributed.embedding_types import (
    EmbeddingComputeKernel,
)
from torchrec.distributed.planner.new.constants import (
    BIGINT_DTYPE,
)
from torchrec.distributed.planner.new.enumerators import (
    ShardingEnumerator,
    _get_tw_shard_io_sizes,
    _get_dp_shard_io_sizes,
)
from torchrec.distributed.planner.new.types import (
    InputStats,
    PlannerConstraints,
    Storage,
    Topology,
)
from torchrec.distributed.tests.test_model import TestSparseNN
from torchrec.distributed.types import ShardingType
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection


EXPECTED_RW_SHARD_LENGTHS = [
    [[13, 10], [13, 10], [13, 10], [13, 10], [12, 10], [12, 10], [12, 10], [12, 10]],
    [[14, 20], [14, 20], [14, 20], [14, 20], [14, 20], [14, 20], [13, 20], [13, 20]],
    [[15, 30], [15, 30], [15, 30], [15, 30], [15, 30], [15, 30], [15, 30], [15, 30]],
    [[17, 40], [17, 40], [16, 40], [16, 40], [16, 40], [16, 40], [16, 40], [16, 40]],
]

EXPECTED_RW_SHARD_OFFSETS = [
    [[0, 0], [13, 0], [26, 0], [39, 0], [52, 0], [64, 0], [76, 0], [88, 0]],
    [[0, 0], [14, 0], [28, 0], [42, 0], [56, 0], [70, 0], [84, 0], [97, 0]],
    [[0, 0], [15, 0], [30, 0], [45, 0], [60, 0], [75, 0], [90, 0], [105, 0]],
    [[0, 0], [17, 0], [34, 0], [50, 0], [66, 0], [82, 0], [98, 0], [114, 0]],
]

EXPECTED_RW_SHARD_STORAGE = [
    [
        Storage(hbm=84488, ddr=0),
        Storage(hbm=84488, ddr=0),
        Storage(hbm=84488, ddr=0),
        Storage(hbm=84488, ddr=0),
        Storage(hbm=84448, ddr=0),
        Storage(hbm=84448, ddr=0),
        Storage(hbm=84448, ddr=0),
        Storage(hbm=84448, ddr=0),
    ],
    [
        Storage(hbm=511072, ddr=0),
        Storage(hbm=511072, ddr=0),
        Storage(hbm=511072, ddr=0),
        Storage(hbm=511072, ddr=0),
        Storage(hbm=511072, ddr=0),
        Storage(hbm=511072, ddr=0),
        Storage(hbm=510992, ddr=0),
        Storage(hbm=510992, ddr=0),
    ],
    [
        Storage(hbm=513800, ddr=0),
        Storage(hbm=513800, ddr=0),
        Storage(hbm=513800, ddr=0),
        Storage(hbm=513800, ddr=0),
        Storage(hbm=513800, ddr=0),
        Storage(hbm=513800, ddr=0),
        Storage(hbm=513800, ddr=0),
        Storage(hbm=513800, ddr=0),
    ],
    [
        Storage(hbm=1340064, ddr=0),
        Storage(hbm=1340064, ddr=0),
        Storage(hbm=1339904, ddr=0),
        Storage(hbm=1339904, ddr=0),
        Storage(hbm=1339904, ddr=0),
        Storage(hbm=1339904, ddr=0),
        Storage(hbm=1339904, ddr=0),
        Storage(hbm=1339904, ddr=0),
    ],
]

EXPECTED_TWRW_SHARD_LENGTHS = [
    [[25, 10], [25, 10], [25, 10], [25, 10]],
    [[28, 20], [28, 20], [27, 20], [27, 20]],
    [[30, 30], [30, 30], [30, 30], [30, 30]],
    [[33, 40], [33, 40], [32, 40], [32, 40]],
]

EXPECTED_TWRW_SHARD_OFFSETS = [
    [[0, 0], [25, 0], [50, 0], [75, 0]],
    [[0, 0], [28, 0], [56, 0], [83, 0]],
    [[0, 0], [30, 0], [60, 0], [90, 0]],
    [[0, 0], [33, 0], [66, 0], [98, 0]],
]

EXPECTED_TWRW_SHARD_STORAGE = [
    [
        Storage(hbm=87016, ddr=0),
        Storage(hbm=87016, ddr=0),
        Storage(hbm=87016, ddr=0),
        Storage(hbm=87016, ddr=0),
    ],
    [
        Storage(hbm=530624, ddr=0),
        Storage(hbm=530624, ddr=0),
        Storage(hbm=530544, ddr=0),
        Storage(hbm=530544, ddr=0),
    ],
    [
        Storage(hbm=536080, ddr=0),
        Storage(hbm=536080, ddr=0),
        Storage(hbm=536080, ddr=0),
        Storage(hbm=536080, ddr=0),
    ],
    [
        Storage(hbm=1369248, ddr=0),
        Storage(hbm=1369248, ddr=0),
        Storage(hbm=1369088, ddr=0),
        Storage(hbm=1369088, ddr=0),
    ],
]

EXPECTED_CW_SHARD_LENGTHS = [
    [[100, 10]],
    [[110, 8], [110, 12]],
    [[120, 9], [120, 9], [120, 12]],
    [[130, 12], [130, 12], [130, 16]],
]

EXPECTED_CW_SHARD_OFFSETS = [
    [[0, 0]],
    [[0, 0], [0, 8]],
    [[0, 0], [0, 9], [0, 18]],
    [[0, 0], [0, 12], [0, 24]],
]

EXPECTED_CW_SHARD_STORAGE = [
    [Storage(hbm=102304, ddr=0)],
    [Storage(hbm=347584, ddr=0), Storage(hbm=447648, ddr=0)],
    [
        Storage(hbm=315616, ddr=0),
        Storage(hbm=315616, ddr=0),
        Storage(hbm=366208, ddr=0),
    ],
    [
        Storage(hbm=612448, ddr=0),
        Storage(hbm=612448, ddr=0),
        Storage(hbm=745600, ddr=0),
    ],
]


class TWSharder(EmbeddingBagCollectionSharder[EmbeddingBagCollection]):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [ShardingType.TABLE_WISE.value]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [EmbeddingComputeKernel.DENSE.value]


class RWSharder(EmbeddingBagCollectionSharder[EmbeddingBagCollection]):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [ShardingType.ROW_WISE.value]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [EmbeddingComputeKernel.DENSE.value]


class TWRWSharder(EmbeddingBagCollectionSharder[EmbeddingBagCollection]):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [ShardingType.TABLE_ROW_WISE.value]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [EmbeddingComputeKernel.DENSE.value]


class CWSharder(EmbeddingBagCollectionSharder[EmbeddingBagCollection]):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [ShardingType.COLUMN_WISE.value]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [EmbeddingComputeKernel.DENSE.value]


class DPSharder(EmbeddingBagCollectionSharder[EmbeddingBagCollection]):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [ShardingType.DATA_PARALLEL.value]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [EmbeddingComputeKernel.DENSE.value]


class AllTypesSharder(EmbeddingBagCollectionSharder[EmbeddingBagCollection]):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [
            ShardingType.DATA_PARALLEL.value,
            ShardingType.TABLE_WISE.value,
            ShardingType.ROW_WISE.value,
            ShardingType.TABLE_ROW_WISE.value,
            ShardingType.COLUMN_WISE.value,
        ]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [
            EmbeddingComputeKernel.DENSE.value,
            EmbeddingComputeKernel.SPARSE.value,
            EmbeddingComputeKernel.BATCHED_DENSE.value,
            EmbeddingComputeKernel.BATCHED_FUSED.value,
            EmbeddingComputeKernel.BATCHED_FUSED_UVM.value,
            EmbeddingComputeKernel.BATCHED_FUSED_UVM_CACHING.value,
            EmbeddingComputeKernel.BATCHED_QUANT.value,
        ]


class TestEnumerators(unittest.TestCase):
    def setUp(self) -> None:
        self.compute_device = "cuda"
        self.batch_size = 256
        self.world_size = 8
        self.local_world_size = 4
        self.constraints = {
            "sparse.ebc.table_0": PlannerConstraints(min_partition=20),
            "sparse.ebc.table_1": PlannerConstraints(min_partition=8),
            "sparse.ebc.table_2": PlannerConstraints(min_partition=9),
            "sparse.ebc.table_3": PlannerConstraints(min_partition=12),
        }
        self.input_stats = {
            "sparse.ebc.table_0": InputStats(),
            "sparse.ebc.table_1": InputStats(pooling_factors=[1, 3, 5]),
            "sparse.ebc.table_2": InputStats(pooling_factors=[8, 2]),
            "sparse.ebc.table_3": InputStats(pooling_factors=[2, 1, 3, 7]),
        }
        self.num_tables = 4
        tables = [
            EmbeddingBagConfig(
                num_embeddings=100 + i * 10,
                embedding_dim=10 + i * 10,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(self.num_tables)
        ]
        self.model = TestSparseNN(tables=tables, weighted_tables=[])
        self.enumerator = ShardingEnumerator(
            topology=Topology(
                world_size=self.world_size,
                compute_device=self.compute_device,
                local_world_size=self.local_world_size,
            ),
            constraints=self.constraints,
            input_stats=self.input_stats,
            batch_size=self.batch_size,
        )

    def test_dp_sharding(self) -> None:
        # pyre-ignore[6]
        sharding_options = self.enumerator.run(self.model, [DPSharder()])

        for sharding_option in sharding_options:
            self.assertEqual(
                sharding_option.sharding_type, ShardingType.DATA_PARALLEL.value
            )
            self.assertEqual(
                [shard.length for shard in sharding_option.shards],
                [list(sharding_option.tensor.shape)] * self.world_size,
            )
            self.assertEqual(
                [shard.offset for shard in sharding_option.shards],
                [[0, 0]] * self.world_size,
            )

            input_sizes, output_sizes = _get_dp_shard_io_sizes(
                batch_size=self.batch_size,
                input_lengths=self.input_stats[sharding_option.fqn].pooling_factors,
                emb_dim=sharding_option.tensor.shape[1],
                num_shards=self.world_size,
                input_data_type_size=int(BIGINT_DTYPE),
                output_data_type_size=sharding_option.tensor.element_size(),
            )

            tensor_sizes = [
                math.prod(sharding_option.tensor.shape)
                * sharding_option.tensor.element_size()
            ] * self.world_size

            storage_sizes = [
                input_size + tensor_size + output_size
                for input_size, tensor_size, output_size in zip(
                    input_sizes, tensor_sizes, output_sizes
                )
            ]

            expected_storage = [
                Storage(hbm=storage_size, ddr=0) for storage_size in storage_sizes
            ]

            self.assertEqual(
                [shard.storage for shard in sharding_option.shards], expected_storage
            )

    def test_tw_sharding(self) -> None:
        # pyre-ignore[6]
        sharding_options = self.enumerator.run(self.model, [TWSharder()])

        for sharding_option in sharding_options:
            self.assertEqual(
                sharding_option.sharding_type, ShardingType.TABLE_WISE.value
            )
            self.assertEqual(
                sharding_option.shards[0].length, list(sharding_option.tensor.shape)
            )
            self.assertEqual(sharding_option.shards[0].offset, [0, 0])

            input_sizes, output_sizes = _get_tw_shard_io_sizes(
                batch_size=self.batch_size,
                world_size=self.world_size,
                input_lengths=self.input_stats[sharding_option.fqn].pooling_factors,
                emb_dim=sharding_option.tensor.shape[1],
                input_data_type_size=int(BIGINT_DTYPE),
                output_data_type_size=sharding_option.tensor.element_size(),
            )

            tensor_size = (
                math.prod(sharding_option.tensor.shape)
                * sharding_option.tensor.element_size()
            )

            storage_size = input_sizes[0] + output_sizes[0] + tensor_size

            self.assertEqual(
                sharding_option.shards[0].storage, Storage(hbm=storage_size, ddr=0)
            )

    def test_rw_sharding(self) -> None:
        # pyre-ignore[6]
        sharding_options = self.enumerator.run(self.model, [RWSharder()])

        for i, sharding_option in enumerate(sharding_options):
            assert sharding_option.sharding_type == ShardingType.ROW_WISE.value
            assert [
                shard.length for shard in sharding_option.shards
            ] == EXPECTED_RW_SHARD_LENGTHS[i]
            assert [
                shard.offset for shard in sharding_option.shards
            ] == EXPECTED_RW_SHARD_OFFSETS[i]
            assert [
                shard.storage for shard in sharding_option.shards
            ] == EXPECTED_RW_SHARD_STORAGE[i]

    def test_twrw_sharding(self) -> None:
        # pyre-ignore[6]
        sharding_options = self.enumerator.run(self.model, [TWRWSharder()])

        for i, sharding_option in enumerate(sharding_options):
            assert sharding_option.sharding_type == ShardingType.TABLE_ROW_WISE.value
            assert [
                shard.length for shard in sharding_option.shards
            ] == EXPECTED_TWRW_SHARD_LENGTHS[i]
            assert [
                shard.offset for shard in sharding_option.shards
            ] == EXPECTED_TWRW_SHARD_OFFSETS[i]
            assert [
                shard.storage for shard in sharding_option.shards
            ] == EXPECTED_TWRW_SHARD_STORAGE[i]

    def test_cw_sharding(self) -> None:
        # pyre-ignore[6]
        sharding_options = self.enumerator.run(self.model, [CWSharder()])

        for i, sharding_option in enumerate(sharding_options):
            assert sharding_option.sharding_type == ShardingType.COLUMN_WISE.value
            assert [
                shard.length for shard in sharding_option.shards
            ] == EXPECTED_CW_SHARD_LENGTHS[i]
            assert [
                shard.offset for shard in sharding_option.shards
            ] == EXPECTED_CW_SHARD_OFFSETS[i]
            assert [
                shard.storage for shard in sharding_option.shards
            ] == EXPECTED_CW_SHARD_STORAGE[i]

    def test_filtering(self) -> None:
        constraint = PlannerConstraints(
            sharding_types=[
                ShardingType.TABLE_ROW_WISE.value,
                ShardingType.COLUMN_WISE.value,
            ],
            compute_kernels=[
                EmbeddingComputeKernel.SPARSE.value,
                EmbeddingComputeKernel.BATCHED_FUSED_UVM.value,
                EmbeddingComputeKernel.BATCHED_QUANT.value,
            ],
        )
        constraints = {
            "sparse.ebc.table_0": constraint,
            "sparse.ebc.table_1": constraint,
            "sparse.ebc.table_2": constraint,
            "sparse.ebc.table_3": constraint,
        }

        enumerator = ShardingEnumerator(
            topology=Topology(
                world_size=self.world_size,
                compute_device=self.compute_device,
                local_world_size=self.local_world_size,
            ),
            constraints=constraints,
            batch_size=self.batch_size,
        )
        sharder = AllTypesSharder()
        # pyre-ignore[6]
        sharding_options = enumerator.run(self.model, [sharder])

        expected_sharding_types = {
            ShardingType.TABLE_ROW_WISE.value,
            ShardingType.COLUMN_WISE.value,
        }
        expected_compute_kernels = {
            EmbeddingComputeKernel.SPARSE.value,
            EmbeddingComputeKernel.BATCHED_FUSED_UVM.value,
            EmbeddingComputeKernel.BATCHED_QUANT.value,
        }
        unexpected_sharding_types = (
            set(sharder.sharding_types(self.compute_device)) - expected_sharding_types
        )
        unexpected_compute_kernels = (
            set(sharder.compute_kernels("", "")) - expected_compute_kernels
        )

        assert len(sharding_options) == self.num_tables * len(
            expected_sharding_types
        ) * len(expected_compute_kernels)

        for sharding_option in sharding_options:
            assert sharding_option.sharding_type in expected_sharding_types
            assert sharding_option.sharding_type not in unexpected_sharding_types
            assert sharding_option.compute_kernel in expected_compute_kernels
            assert sharding_option.compute_kernel not in unexpected_compute_kernels
