#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import List

from torchrec.distributed.embedding_types import (
    EmbeddingComputeKernel,
)
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.planner.constants import (
    BIGINT_DTYPE,
)
from torchrec.distributed.planner.enumerators import EmbeddingEnumerator
from torchrec.distributed.planner.shard_estimators import (
    _calculate_tw_shard_io_sizes,
    _calculate_dp_shard_io_sizes,
)
from torchrec.distributed.planner.types import (
    ParameterConstraints,
    Storage,
    Topology,
)
from torchrec.distributed.planner.utils import prod
from torchrec.distributed.tests.test_model import TestSparseNN
from torchrec.distributed.types import ShardingType
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection


EXPECTED_RW_SHARD_SIZES = [
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
        Storage(hbm=84488, ddr=0),
        Storage(hbm=84488, ddr=0),
        Storage(hbm=84488, ddr=0),
        Storage(hbm=84488, ddr=0),
        Storage(hbm=84488, ddr=0),
        Storage(hbm=84488, ddr=0),
        Storage(hbm=84488, ddr=0),
        Storage(hbm=84328, ddr=0),
    ],
    [
        Storage(hbm=511072, ddr=0),
        Storage(hbm=511072, ddr=0),
        Storage(hbm=511072, ddr=0),
        Storage(hbm=511072, ddr=0),
        Storage(hbm=511072, ddr=0),
        Storage(hbm=511072, ddr=0),
        Storage(hbm=511072, ddr=0),
        Storage(hbm=510912, ddr=0),
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
        Storage(hbm=1340064, ddr=0),
        Storage(hbm=1340064, ddr=0),
        Storage(hbm=1340064, ddr=0),
        Storage(hbm=1340064, ddr=0),
        Storage(hbm=1340064, ddr=0),
        Storage(hbm=1339104, ddr=0),
    ],
]


EXPECTED_UVM_CACHING_RW_SHARD_STORAGE = [
    [
        Storage(hbm=84072, ddr=416),
        Storage(hbm=84072, ddr=416),
        Storage(hbm=84072, ddr=416),
        Storage(hbm=84072, ddr=416),
        Storage(hbm=84072, ddr=416),
        Storage(hbm=84072, ddr=416),
        Storage(hbm=84072, ddr=416),
        Storage(hbm=84040, ddr=288),
    ],
    [
        Storage(hbm=510176, ddr=896),
        Storage(hbm=510176, ddr=896),
        Storage(hbm=510176, ddr=896),
        Storage(hbm=510176, ddr=896),
        Storage(hbm=510176, ddr=896),
        Storage(hbm=510176, ddr=896),
        Storage(hbm=510176, ddr=896),
        Storage(hbm=510144, ddr=768),
    ],
    [
        Storage(hbm=512648, ddr=1152),
        Storage(hbm=512648, ddr=1152),
        Storage(hbm=512648, ddr=1152),
        Storage(hbm=512648, ddr=1152),
        Storage(hbm=512648, ddr=1152),
        Storage(hbm=512648, ddr=1152),
        Storage(hbm=512648, ddr=1152),
        Storage(hbm=512648, ddr=1152),
    ],
    [
        Storage(hbm=1339656, ddr=408),
        Storage(hbm=1339656, ddr=408),
        Storage(hbm=1339656, ddr=408),
        Storage(hbm=1339656, ddr=408),
        Storage(hbm=1339656, ddr=408),
        Storage(hbm=1339656, ddr=408),
        Storage(hbm=1339656, ddr=408),
        Storage(hbm=1338840, ddr=264),
    ],
]


EXPECTED_TWRW_SHARD_SIZES = [
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
        Storage(hbm=87016, ddr=0),
        Storage(hbm=87016, ddr=0),
        Storage(hbm=87016, ddr=0),
        Storage(hbm=87016, ddr=0),
    ],
    [
        Storage(hbm=530624, ddr=0),
        Storage(hbm=530624, ddr=0),
        Storage(hbm=530624, ddr=0),
        Storage(hbm=530464, ddr=0),
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
        Storage(hbm=1369248, ddr=0),
        Storage(hbm=1368928, ddr=0),
    ],
]

EXPECTED_CW_SHARD_SIZES = [
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

EXPECTED_TWCW_SHARD_SIZES: List[List[List[int]]] = EXPECTED_CW_SHARD_SIZES

EXPECTED_TWCW_SHARD_OFFSETS: List[List[List[int]]] = EXPECTED_CW_SHARD_OFFSETS

EXPECTED_TWCW_SHARD_STORAGE = [
    [Storage(hbm=53152, ddr=0)],
    [Storage(hbm=175552, ddr=0), Storage(hbm=226464, ddr=0)],
    [
        Storage(hbm=159968, ddr=0),
        Storage(hbm=159968, ddr=0),
        Storage(hbm=185984, ddr=0),
    ],
    [
        Storage(hbm=309344, ddr=0),
        Storage(hbm=309344, ddr=0),
        Storage(hbm=376960, ddr=0),
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


class TWCWSharder(EmbeddingBagCollectionSharder[EmbeddingBagCollection]):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [ShardingType.TABLE_COLUMN_WISE.value]

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
            ShardingType.TABLE_COLUMN_WISE.value,
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
            "table_0": ParameterConstraints(min_partition=20),
            "table_1": ParameterConstraints(min_partition=8, pooling_factors=[1, 3, 5]),
            "table_2": ParameterConstraints(
                min_partition=9, caching_ratio=0.36, pooling_factors=[8, 2]
            ),
            "table_3": ParameterConstraints(
                min_partition=12, caching_ratio=0.85, pooling_factors=[2, 1, 3, 7]
            ),
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
        )

    def test_dp_sharding(self) -> None:
        # pyre-ignore[6]
        sharding_options = self.enumerator.enumerate(self.model, [DPSharder()])

        for sharding_option in sharding_options:
            self.assertEqual(
                sharding_option.sharding_type, ShardingType.DATA_PARALLEL.value
            )
            self.assertEqual(
                [shard.size for shard in sharding_option.shards],
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
                input_lengths=self.constraints[sharding_option.name].pooling_factors,
                emb_dim=sharding_option.tensor.shape[1],
                num_shards=self.world_size,
                input_data_type_size=input_data_type_size,
                output_data_type_size=output_data_type_size,
            )

            tensor_sizes = [
                prod(sharding_option.tensor.shape)
                * sharding_option.tensor.element_size()
            ] * self.world_size

            optimizer_sizes = [tensor_size * 2 for tensor_size in tensor_sizes]

            storage_sizes = [
                input_size + tensor_size + output_size + optimizer_size
                for input_size, tensor_size, output_size, optimizer_size in zip(
                    input_sizes,
                    tensor_sizes,
                    output_sizes,
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
        sharding_options = self.enumerator.enumerate(self.model, [TWSharder()])

        for sharding_option in sharding_options:
            self.assertEqual(
                sharding_option.sharding_type, ShardingType.TABLE_WISE.value
            )
            self.assertEqual(
                sharding_option.shards[0].size, list(sharding_option.tensor.shape)
            )
            self.assertEqual(sharding_option.shards[0].offset, [0, 0])

            input_data_type_size = BIGINT_DTYPE
            output_data_type_size = sharding_option.tensor.element_size()

            input_sizes, output_sizes = _calculate_tw_shard_io_sizes(
                batch_size=self.batch_size,
                world_size=self.world_size,
                input_lengths=self.constraints[sharding_option.name].pooling_factors,
                emb_dim=sharding_option.tensor.shape[1],
                input_data_type_size=input_data_type_size,
                output_data_type_size=output_data_type_size,
            )

            tensor_size = (
                prod(sharding_option.tensor.shape)
                * sharding_option.tensor.element_size()
            )
            optimizer_size = 0

            storage_size = (
                input_sizes[0] + output_sizes[0] + tensor_size + optimizer_size
            )

            self.assertEqual(
                sharding_option.shards[0].storage, Storage(hbm=storage_size, ddr=0)
            )

    def test_rw_sharding(self) -> None:
        # pyre-ignore[6]
        sharding_options = self.enumerator.enumerate(self.model, [RWSharder()])

        for i, sharding_option in enumerate(sharding_options):
            self.assertEqual(sharding_option.sharding_type, ShardingType.ROW_WISE.value)
            self.assertEqual(
                [shard.size for shard in sharding_option.shards],
                EXPECTED_RW_SHARD_SIZES[i],
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
        sharding_options = self.enumerator.enumerate(
            self.model,
            # pyre-ignore[6]
            [UVMCachingRWSharder()],
        )

        for i, sharding_option in enumerate(sharding_options):
            self.assertEqual(sharding_option.sharding_type, ShardingType.ROW_WISE.value)
            self.assertEqual(
                [shard.size for shard in sharding_option.shards],
                EXPECTED_RW_SHARD_SIZES[i],
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
        sharding_options = self.enumerator.enumerate(self.model, [TWRWSharder()])

        for i, sharding_option in enumerate(sharding_options):
            self.assertEqual(
                sharding_option.sharding_type, ShardingType.TABLE_ROW_WISE.value
            )
            self.assertEqual(
                [shard.size for shard in sharding_option.shards],
                EXPECTED_TWRW_SHARD_SIZES[i],
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
        sharding_options = self.enumerator.enumerate(self.model, [CWSharder()])

        for i, sharding_option in enumerate(sharding_options):
            self.assertEqual(
                sharding_option.sharding_type, ShardingType.COLUMN_WISE.value
            )
            self.assertEqual(
                [shard.size for shard in sharding_option.shards],
                EXPECTED_CW_SHARD_SIZES[i],
            )
            self.assertEqual(
                [shard.offset for shard in sharding_option.shards],
                EXPECTED_CW_SHARD_OFFSETS[i],
            )
            self.assertEqual(
                [shard.storage for shard in sharding_option.shards],
                EXPECTED_CW_SHARD_STORAGE[i],
            )

    def test_twcw_sharding(self) -> None:
        # pyre-ignore[6]
        sharding_options = self.enumerator.enumerate(self.model, [TWCWSharder()])

        for i, sharding_option in enumerate(sharding_options):
            self.assertEqual(
                sharding_option.sharding_type, ShardingType.TABLE_COLUMN_WISE.value
            )
            self.assertEqual(
                [shard.size for shard in sharding_option.shards],
                EXPECTED_TWCW_SHARD_SIZES[i],
            )
            self.assertEqual(
                [shard.offset for shard in sharding_option.shards],
                EXPECTED_TWCW_SHARD_OFFSETS[i],
            )
            self.assertEqual(
                [shard.storage for shard in sharding_option.shards],
                EXPECTED_TWCW_SHARD_STORAGE[i],
            )

    def test_filtering(self) -> None:
        constraint = ParameterConstraints(
            sharding_types=[
                ShardingType.TABLE_ROW_WISE.value,
                ShardingType.COLUMN_WISE.value,
            ],
            compute_kernels=[
                EmbeddingComputeKernel.SPARSE.value,
                EmbeddingComputeKernel.BATCHED_FUSED_UVM.value,
                EmbeddingComputeKernel.BATCHED_DENSE.value,
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
        sharding_options = enumerator.enumerate(self.model, [sharder])

        expected_sharding_types = {
            ShardingType.TABLE_ROW_WISE.value,
            ShardingType.COLUMN_WISE.value,
        }
        expected_compute_kernels = {
            EmbeddingComputeKernel.SPARSE.value,
            EmbeddingComputeKernel.BATCHED_FUSED_UVM.value,
            EmbeddingComputeKernel.BATCHED_DENSE.value,
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
