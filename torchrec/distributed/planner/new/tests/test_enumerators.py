#!/usr/bin/env python3

import math
import unittest
from typing import List

from torchrec.distributed.embedding_types import (
    EmbeddingComputeKernel,
)
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.planner.new.constants import (
    BIGINT_DTYPE,
)
from torchrec.distributed.planner.new.enumerators import (
    EmbeddingEnumerator,
    _calculate_tw_shard_io_sizes,
    _calculate_dp_shard_io_sizes,
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
    [[13, 10], [13, 10], [13, 10], [13, 10], [13, 10], [13, 10], [13, 10], [9, 10]],
    [[14, 20], [14, 20], [14, 20], [14, 20], [14, 20], [14, 20], [14, 20], [12, 20]],
    [[15, 30], [15, 30], [15, 30], [15, 30], [15, 30], [15, 30], [15, 30], [15, 30]],
    [[17, 40], [17, 40], [17, 40], [17, 40], [17, 40], [17, 40], [17, 40], [11, 40]],
]

EXPECTED_RW_SHARD_OFFSETS = [
    [[0, 0], [13, 0], [26, 0], [39, 0], [52, 0], [65, 0], [78, 0], [91, 0]],
    [[0, 0], [14, 0], [28, 0], [42, 0], [56, 0], [70, 0], [84, 0], [98, 0]],
    [[0, 0], [15, 0], [30, 0], [45, 0], [60, 0], [75, 0], [90, 0], [105, 0]],
    [[0, 0], [17, 0], [34, 0], [51, 0], [68, 0], [85, 0], [102, 0], [119, 0]],
]

EXPECTED_RW_SHARD_STORAGE = [
    [
        Storage(hbm=85008, ddr=0),
        Storage(hbm=85008, ddr=0),
        Storage(hbm=85008, ddr=0),
        Storage(hbm=85008, ddr=0),
        Storage(hbm=85008, ddr=0),
        Storage(hbm=85008, ddr=0),
        Storage(hbm=85008, ddr=0),
        Storage(hbm=84688, ddr=0),
    ],
    [
        Storage(hbm=512192, ddr=0),
        Storage(hbm=512192, ddr=0),
        Storage(hbm=512192, ddr=0),
        Storage(hbm=512192, ddr=0),
        Storage(hbm=512192, ddr=0),
        Storage(hbm=512192, ddr=0),
        Storage(hbm=512192, ddr=0),
        Storage(hbm=511872, ddr=0),
    ],
    [
        Storage(hbm=515600, ddr=0),
        Storage(hbm=515600, ddr=0),
        Storage(hbm=515600, ddr=0),
        Storage(hbm=515600, ddr=0),
        Storage(hbm=515600, ddr=0),
        Storage(hbm=515600, ddr=0),
        Storage(hbm=515600, ddr=0),
        Storage(hbm=515600, ddr=0),
    ],
    [
        Storage(hbm=1342784, ddr=0),
        Storage(hbm=1342784, ddr=0),
        Storage(hbm=1342784, ddr=0),
        Storage(hbm=1342784, ddr=0),
        Storage(hbm=1342784, ddr=0),
        Storage(hbm=1342784, ddr=0),
        Storage(hbm=1342784, ddr=0),
        Storage(hbm=1340864, ddr=0),
    ],
]


EXPECTED_UVM_CACHING_RW_SHARD_STORAGE = [
    [
        Storage(hbm=84176, ddr=832),
        Storage(hbm=84176, ddr=832),
        Storage(hbm=84176, ddr=832),
        Storage(hbm=84176, ddr=832),
        Storage(hbm=84176, ddr=832),
        Storage(hbm=84176, ddr=832),
        Storage(hbm=84176, ddr=832),
        Storage(hbm=84112, ddr=576),
    ],
    [
        Storage(hbm=510400, ddr=1792),
        Storage(hbm=510400, ddr=1792),
        Storage(hbm=510400, ddr=1792),
        Storage(hbm=510400, ddr=1792),
        Storage(hbm=510400, ddr=1792),
        Storage(hbm=510400, ddr=1792),
        Storage(hbm=510400, ddr=1792),
        Storage(hbm=510336, ddr=1536),
    ],
    [
        Storage(hbm=513296, ddr=2304),
        Storage(hbm=513296, ddr=2304),
        Storage(hbm=513296, ddr=2304),
        Storage(hbm=513296, ddr=2304),
        Storage(hbm=513296, ddr=2304),
        Storage(hbm=513296, ddr=2304),
        Storage(hbm=513296, ddr=2304),
        Storage(hbm=513296, ddr=2304),
    ],
    [
        Storage(hbm=1341968, ddr=816),
        Storage(hbm=1341968, ddr=816),
        Storage(hbm=1341968, ddr=816),
        Storage(hbm=1341968, ddr=816),
        Storage(hbm=1341968, ddr=816),
        Storage(hbm=1341968, ddr=816),
        Storage(hbm=1341968, ddr=816),
        Storage(hbm=1340336, ddr=528),
    ],
]


EXPECTED_TWRW_SHARD_LENGTHS = [
    [[25, 10], [25, 10], [25, 10], [25, 10]],
    [[28, 20], [28, 20], [28, 20], [26, 20]],
    [[30, 30], [30, 30], [30, 30], [30, 30]],
    [[33, 40], [33, 40], [33, 40], [31, 40]],
]

EXPECTED_TWRW_SHARD_OFFSETS = [
    [[0, 0], [25, 0], [50, 0], [75, 0]],
    [[0, 0], [28, 0], [56, 0], [84, 0]],
    [[0, 0], [30, 0], [60, 0], [90, 0]],
    [[0, 0], [33, 0], [66, 0], [99, 0]],
]

EXPECTED_TWRW_SHARD_STORAGE = [
    [
        Storage(hbm=88016, ddr=0),
        Storage(hbm=88016, ddr=0),
        Storage(hbm=88016, ddr=0),
        Storage(hbm=88016, ddr=0),
    ],
    [
        Storage(hbm=532864, ddr=0),
        Storage(hbm=532864, ddr=0),
        Storage(hbm=532864, ddr=0),
        Storage(hbm=532544, ddr=0),
    ],
    [
        Storage(hbm=539680, ddr=0),
        Storage(hbm=539680, ddr=0),
        Storage(hbm=539680, ddr=0),
        Storage(hbm=539680, ddr=0),
    ],
    [
        Storage(hbm=1374528, ddr=0),
        Storage(hbm=1374528, ddr=0),
        Storage(hbm=1374528, ddr=0),
        Storage(hbm=1373888, ddr=0),
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
    [Storage(hbm=106304, ddr=0)],
    [Storage(hbm=351104, ddr=0), Storage(hbm=452928, ddr=0)],
    [
        Storage(hbm=319936, ddr=0),
        Storage(hbm=319936, ddr=0),
        Storage(hbm=371968, ddr=0),
    ],
    [
        Storage(hbm=618688, ddr=0),
        Storage(hbm=618688, ddr=0),
        Storage(hbm=753920, ddr=0),
    ],
]


class TWSharder(EmbeddingBagCollectionSharder[EmbeddingBagCollection]):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [ShardingType.TABLE_WISE.value]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [EmbeddingComputeKernel.DENSE.value, EmbeddingComputeKernel.SPARSE.value]


class RWSharder(EmbeddingBagCollectionSharder[EmbeddingBagCollection]):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [ShardingType.ROW_WISE.value]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [EmbeddingComputeKernel.DENSE.value]


class UVMCachingRWSharder(EmbeddingBagCollectionSharder[EmbeddingBagCollection]):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [ShardingType.ROW_WISE.value]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [EmbeddingComputeKernel.BATCHED_FUSED_UVM_CACHING.value]


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
        return [EmbeddingComputeKernel.DENSE.value, EmbeddingComputeKernel.SPARSE.value]


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
            "table_0": PlannerConstraints(min_partition=20),
            "table_1": PlannerConstraints(min_partition=8),
            "table_2": PlannerConstraints(min_partition=9, caching_ratio=0.36),
            "table_3": PlannerConstraints(min_partition=12, caching_ratio=0.85),
        }
        self.input_stats = {
            "table_0": InputStats(),
            "table_1": InputStats(pooling_factors=[1, 3, 5]),
            "table_2": InputStats(pooling_factors=[8, 2]),
            "table_3": InputStats(pooling_factors=[2, 1, 3, 7]),
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
        self.enumerator = EmbeddingEnumerator(
            topology=Topology(
                world_size=self.world_size,
                compute_device=self.compute_device,
                local_world_size=self.local_world_size,
                batch_size=self.batch_size,
            ),
            constraints=self.constraints,
            input_stats=self.input_stats,
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

            input_data_type_size = BIGINT_DTYPE
            output_data_type_size = sharding_option.tensor.element_size()

            input_sizes, output_sizes = _calculate_dp_shard_io_sizes(
                batch_size=self.batch_size,
                input_lengths=self.input_stats[sharding_option.name].pooling_factors,
                emb_dim=sharding_option.tensor.shape[1],
                num_shards=self.world_size,
                input_data_type_size=input_data_type_size,
                output_data_type_size=output_data_type_size,
            )

            tensor_sizes = [
                math.prod(sharding_option.tensor.shape)
                * sharding_option.tensor.element_size()
            ] * self.world_size

            gradient_sizes = (
                [
                    input_sizes[0]
                    * sharding_option.tensor.shape[1]
                    * output_data_type_size
                    / input_data_type_size
                ]
                * self.world_size
                if sharding_option.compute_kernel == EmbeddingComputeKernel.SPARSE.value
                else tensor_sizes
            )

            optimizer_sizes = [tensor_size * 2 for tensor_size in tensor_sizes]

            storage_sizes = [
                input_size + tensor_size + output_size + gradient_size + optimizer_size
                for input_size, tensor_size, output_size, gradient_size, optimizer_size in zip(
                    input_sizes,
                    tensor_sizes,
                    output_sizes,
                    gradient_sizes,
                    optimizer_sizes,
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

            input_data_type_size = BIGINT_DTYPE
            output_data_type_size = sharding_option.tensor.element_size()

            input_sizes, output_sizes = _calculate_tw_shard_io_sizes(
                batch_size=self.batch_size,
                world_size=self.world_size,
                input_lengths=self.input_stats[sharding_option.name].pooling_factors,
                emb_dim=sharding_option.tensor.shape[1],
                input_data_type_size=input_data_type_size,
                output_data_type_size=output_data_type_size,
            )

            tensor_size = (
                math.prod(sharding_option.tensor.shape)
                * sharding_option.tensor.element_size()
            )
            gradient_size = (
                (
                    input_sizes[0]
                    * sharding_option.tensor.shape[1]
                    * output_data_type_size
                    / input_data_type_size
                )
                if sharding_option.compute_kernel == EmbeddingComputeKernel.SPARSE.value
                else tensor_size
            )
            optimizer_size = 0

            storage_size = (
                input_sizes[0]
                + output_sizes[0]
                + tensor_size
                + gradient_size
                + optimizer_size
            )

            self.assertEqual(
                sharding_option.shards[0].storage, Storage(hbm=storage_size, ddr=0)
            )

    def test_rw_sharding(self) -> None:
        # pyre-ignore[6]
        sharding_options = self.enumerator.run(self.model, [RWSharder()])

        for i, sharding_option in enumerate(sharding_options):
            self.assertEqual(sharding_option.sharding_type, ShardingType.ROW_WISE.value)
            self.assertEqual(
                [shard.length for shard in sharding_option.shards],
                EXPECTED_RW_SHARD_LENGTHS[i],
            )
            self.assertEqual(
                [shard.offset for shard in sharding_option.shards],
                EXPECTED_RW_SHARD_OFFSETS[i],
            )
            self.assertEqual(
                [shard.storage for shard in sharding_option.shards],
                EXPECTED_RW_SHARD_STORAGE[i],
            )

    def test_uvm_caching_rw_sharding(self) -> None:
        # pyre-ignore[6]
        sharding_options = self.enumerator.run(self.model, [UVMCachingRWSharder()])
        for i, sharding_option in enumerate(sharding_options):
            self.assertEqual(sharding_option.sharding_type, ShardingType.ROW_WISE.value)
            self.assertEqual(
                [shard.length for shard in sharding_option.shards],
                EXPECTED_RW_SHARD_LENGTHS[i],
            )
            self.assertEqual(
                [shard.offset for shard in sharding_option.shards],
                EXPECTED_RW_SHARD_OFFSETS[i],
            )
            self.assertEqual(
                [shard.storage for shard in sharding_option.shards],
                EXPECTED_UVM_CACHING_RW_SHARD_STORAGE[i],
            )

    def test_twrw_sharding(self) -> None:
        # pyre-ignore[6]
        sharding_options = self.enumerator.run(self.model, [TWRWSharder()])

        for i, sharding_option in enumerate(sharding_options):
            self.assertEqual(
                sharding_option.sharding_type, ShardingType.TABLE_ROW_WISE.value
            )
            self.assertEqual(
                [shard.length for shard in sharding_option.shards],
                EXPECTED_TWRW_SHARD_LENGTHS[i],
            )
            self.assertEqual(
                [shard.offset for shard in sharding_option.shards],
                EXPECTED_TWRW_SHARD_OFFSETS[i],
            )
            self.assertEqual(
                [shard.storage for shard in sharding_option.shards],
                EXPECTED_TWRW_SHARD_STORAGE[i],
            )

    def test_cw_sharding(self) -> None:
        # pyre-ignore[6]
        sharding_options = self.enumerator.run(self.model, [CWSharder()])

        for i, sharding_option in enumerate(sharding_options):
            self.assertEqual(
                sharding_option.sharding_type, ShardingType.COLUMN_WISE.value
            )
            self.assertEqual(
                [shard.length for shard in sharding_option.shards],
                EXPECTED_CW_SHARD_LENGTHS[i],
            )
            self.assertEqual(
                [shard.offset for shard in sharding_option.shards],
                EXPECTED_CW_SHARD_OFFSETS[i],
            )
            self.assertEqual(
                [shard.storage for shard in sharding_option.shards],
                EXPECTED_CW_SHARD_STORAGE[i],
            )

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
            "table_0": constraint,
            "table_1": constraint,
            "table_2": constraint,
            "table_3": constraint,
        }

        enumerator = EmbeddingEnumerator(
            topology=Topology(
                world_size=self.world_size,
                compute_device=self.compute_device,
                local_world_size=self.local_world_size,
                batch_size=self.batch_size,
            ),
            constraints=constraints,
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

        self.assertEqual(
            len(sharding_options),
            self.num_tables
            * len(expected_sharding_types)
            * len(expected_compute_kernels),
        )

        for sharding_option in sharding_options:
            self.assertIn(sharding_option.sharding_type, expected_sharding_types)
            self.assertNotIn(sharding_option.sharding_type, unexpected_sharding_types)
            self.assertIn(sharding_option.compute_kernel, expected_compute_kernels)
            self.assertNotIn(sharding_option.compute_kernel, unexpected_compute_kernels)
