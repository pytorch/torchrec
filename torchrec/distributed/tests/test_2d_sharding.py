#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Any, cast, Dict, Optional, Tuple, Type

import torch
import torch.nn as nn
from hypothesis import assume, given, settings, strategies as st, Verbosity
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.fbgemm_qcomm_codec import CommType, QCommsConfig
from torchrec.distributed.planner import ParameterConstraints
from torchrec.distributed.test_utils.test_model_parallel import ModelParallelTestShared
from torchrec.distributed.test_utils.test_sharding import (
    create_test_sharder,
    SharderType,
)
from torchrec.distributed.types import ModuleSharder, ShardingType
from torchrec.modules.embedding_configs import PoolingType
from torchrec.test_utils import skip_if_asan_class


@skip_if_asan_class
class Test2DSharding(ModelParallelTestShared):
    """
    Tests for 2D parallelism of embedding tables
    """

    WORLD_SIZE = 8
    WORLD_SIZE_2D = 4

    def setUp(self, backend: str = "nccl") -> None:
        super().setUp(backend=backend)

    @unittest.skipIf(
        torch.cuda.device_count() <= 7,
        "Not enough GPUs, this test requires at least four GPUs",
    )
    # pyre-fixme[56]
    @given(
        sharder_type=st.sampled_from(
            [
                SharderType.EMBEDDING_BAG_COLLECTION.value,
            ]
        ),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.FUSED.value,
                EmbeddingComputeKernel.FUSED_UVM_CACHING.value,
                EmbeddingComputeKernel.FUSED_UVM.value,
            ],
        ),
        qcomms_config=st.sampled_from(
            [
                None,
                # QCommsConfig(
                #     forward_precision=CommType.FP16, backward_precision=CommType.BF16
                # ),
            ]
        ),
        apply_optimizer_in_backward_config=st.sampled_from(
            [
                # None,
                {
                    "embedding_bags": (
                        torch.optim.SGD,
                        {
                            "lr": 0.01,
                        },
                    ),
                },
            ]
        ),
        pooling=st.sampled_from([PoolingType.SUM]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=1, deadline=None)
    def test_sharding_cw_2D(
        self,
        sharder_type: str,
        kernel_type: str,
        qcomms_config: Optional[QCommsConfig],
        apply_optimizer_in_backward_config: Optional[
            Dict[str, Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]]
        ],
        pooling: PoolingType,
    ) -> None:
        if (
            self.device == torch.device("cpu")
            and kernel_type != EmbeddingComputeKernel.FUSED.value
        ):
            self.skipTest("CPU does not support uvm.")

        sharding_type = ShardingType.COLUMN_WISE.value
        assume(sharder_type == SharderType.EMBEDDING_BAG_COLLECTION.value)

        self._test_sharding(
            world_size=self.WORLD_SIZE,
            world_size_2D=self.WORLD_SIZE_2D,
            sharders=[
                cast(
                    ModuleSharder[nn.Module],
                    create_test_sharder(
                        sharder_type,
                        sharding_type,
                        kernel_type,
                        qcomms_config=qcomms_config,
                        device=self.device,
                    ),
                ),
            ],
            qcomms_config=qcomms_config,
            constraints={
                table.name: ParameterConstraints(min_partition=4)
                for table in self.tables
            },
            backend=self.backend,
            apply_optimizer_in_backward_config=apply_optimizer_in_backward_config,
            pooling=pooling,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 7,
        "Not enough GPUs, this test requires at least four GPUs",
    )
    # pyre-fixme[56]
    @given(
        sharder_type=st.sampled_from(
            [
                SharderType.EMBEDDING_BAG_COLLECTION.value,
            ]
        ),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.FUSED.value,
                EmbeddingComputeKernel.FUSED_UVM_CACHING.value,
                EmbeddingComputeKernel.FUSED_UVM.value,
            ],
        ),
        qcomms_config=st.sampled_from(
            [
                # None,
                QCommsConfig(
                    forward_precision=CommType.FP16, backward_precision=CommType.BF16
                ),
            ]
        ),
        apply_optimizer_in_backward_config=st.sampled_from(
            [
                None,
                {
                    "embedding_bags": (
                        torch.optim.SGD,
                        {
                            "lr": 0.01,
                        },
                    ),
                },
            ]
        ),
        pooling=st.sampled_from([PoolingType.SUM]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=1, deadline=None)
    def test_sharding_tw_2D(
        self,
        sharder_type: str,
        kernel_type: str,
        qcomms_config: Optional[QCommsConfig],
        apply_optimizer_in_backward_config: Optional[
            Dict[str, Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]]
        ],
        pooling: PoolingType,
    ) -> None:
        if (
            self.device == torch.device("cpu")
            and kernel_type != EmbeddingComputeKernel.FUSED.value
        ):
            self.skipTest("CPU does not support uvm.")

        sharding_type = ShardingType.TABLE_WISE.value
        assume(sharder_type == SharderType.EMBEDDING_BAG_COLLECTION.value)

        self._test_sharding(
            world_size=self.WORLD_SIZE,
            world_size_2D=self.WORLD_SIZE_2D,
            node_group_size=self.WORLD_SIZE_2D // 2,
            sharders=[
                cast(
                    ModuleSharder[nn.Module],
                    create_test_sharder(
                        sharder_type,
                        sharding_type,
                        kernel_type,
                        qcomms_config=qcomms_config,
                        device=self.device,
                    ),
                ),
            ],
            qcomms_config=qcomms_config,
            constraints={
                table.name: ParameterConstraints(min_partition=2)
                for table in self.tables
            },
            backend=self.backend,
            apply_optimizer_in_backward_config=apply_optimizer_in_backward_config,
            pooling=pooling,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 7,
        "Not enough GPUs, this test requires at least four GPUs",
    )
    # pyre-fixme[56]
    @given(
        sharder_type=st.sampled_from(
            [
                SharderType.EMBEDDING_BAG_COLLECTION.value,
            ]
        ),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.FUSED.value,
                EmbeddingComputeKernel.FUSED_UVM_CACHING.value,
                EmbeddingComputeKernel.FUSED_UVM.value,
            ],
        ),
        qcomms_config=st.sampled_from(
            [
                None,
                # QCommsConfig(
                #     forward_precision=CommType.FP16, backward_precision=CommType.BF16
                # ),
            ]
        ),
        apply_optimizer_in_backward_config=st.sampled_from(
            [
                # None,
                {
                    "embedding_bags": (
                        torch.optim.SGD,
                        {
                            "lr": 0.01,
                        },
                    ),
                },
            ]
        ),
        pooling=st.sampled_from([PoolingType.SUM]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=1, deadline=None)
    def test_sharding_grid_2D(
        self,
        sharder_type: str,
        kernel_type: str,
        qcomms_config: Optional[QCommsConfig],
        apply_optimizer_in_backward_config: Optional[
            Dict[str, Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]]
        ],
        pooling: PoolingType,
    ) -> None:
        if (
            self.device == torch.device("cpu")
            and kernel_type != EmbeddingComputeKernel.FUSED.value
        ):
            self.skipTest("CPU does not support uvm.")

        sharding_type = ShardingType.GRID_SHARD.value
        assume(sharder_type == SharderType.EMBEDDING_BAG_COLLECTION.value)

        self._test_sharding(
            world_size=self.WORLD_SIZE,
            world_size_2D=self.WORLD_SIZE_2D,
            node_group_size=self.WORLD_SIZE // 4,
            sharders=[
                cast(
                    ModuleSharder[nn.Module],
                    create_test_sharder(
                        sharder_type,
                        sharding_type,
                        kernel_type,
                        qcomms_config=qcomms_config,
                        device=self.device,
                    ),
                ),
            ],
            qcomms_config=qcomms_config,
            constraints={
                "table_0": ParameterConstraints(
                    min_partition=8, sharding_types=[ShardingType.GRID_SHARD.value]
                ),
                "table_1": ParameterConstraints(
                    min_partition=12, sharding_types=[ShardingType.GRID_SHARD.value]
                ),
                "table_2": ParameterConstraints(
                    min_partition=8, sharding_types=[ShardingType.GRID_SHARD.value]
                ),
                "table_3": ParameterConstraints(
                    min_partition=10, sharding_types=[ShardingType.GRID_SHARD.value]
                ),
                "table_4": ParameterConstraints(
                    min_partition=4, sharding_types=[ShardingType.GRID_SHARD.value]
                ),
                "table_5": ParameterConstraints(
                    min_partition=6, sharding_types=[ShardingType.GRID_SHARD.value]
                ),
                "weighted_table_0": ParameterConstraints(
                    min_partition=2, sharding_types=[ShardingType.GRID_SHARD.value]
                ),
                "weighted_table_1": ParameterConstraints(
                    min_partition=3, sharding_types=[ShardingType.GRID_SHARD.value]
                ),
            },
            backend=self.backend,
            apply_optimizer_in_backward_config=apply_optimizer_in_backward_config,
            pooling=pooling,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 7,
        "Not enough GPUs, this test requires at least eight GPUs",
    )
    # pyre-fixme[56]
    @given(
        sharder_type=st.sampled_from(
            [
                SharderType.EMBEDDING_BAG_COLLECTION.value,
            ]
        ),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.FUSED.value,
                EmbeddingComputeKernel.FUSED_UVM_CACHING.value,
                EmbeddingComputeKernel.FUSED_UVM.value,
            ],
        ),
        qcomms_config=st.sampled_from(
            [
                # None,
                QCommsConfig(
                    forward_precision=CommType.FP16, backward_precision=CommType.BF16
                ),
            ]
        ),
        apply_optimizer_in_backward_config=st.sampled_from(
            [
                None,
                {
                    "embedding_bags": (torch.optim.SGD, {"lr": 0.01}),
                    "embeddings": (torch.optim.SGD, {"lr": 0.2}),
                },
            ]
        ),
        variable_batch_size=st.booleans(),
        pooling=st.sampled_from([PoolingType.SUM]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=1, deadline=None)
    def test_sharding_rw_2D(
        self,
        sharder_type: str,
        kernel_type: str,
        qcomms_config: Optional[QCommsConfig],
        apply_optimizer_in_backward_config: Optional[
            Dict[str, Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]]
        ],
        variable_batch_size: bool,
        pooling: PoolingType,
    ) -> None:
        if self.backend == "gloo":
            self.skipTest(
                "Gloo reduce_scatter_base fallback not supported with async_op=True"
            )

        sharding_type = ShardingType.ROW_WISE.value
        assume(
            sharder_type == SharderType.EMBEDDING_BAG_COLLECTION.value
            or not variable_batch_size
        )

        self._test_sharding(
            world_size=self.WORLD_SIZE,
            world_size_2D=self.WORLD_SIZE_2D,
            sharders=[
                cast(
                    ModuleSharder[nn.Module],
                    create_test_sharder(
                        sharder_type,
                        sharding_type,
                        kernel_type,
                        qcomms_config=qcomms_config,
                        device=self.device,
                    ),
                ),
            ],
            qcomms_config=qcomms_config,
            backend=self.backend,
            apply_optimizer_in_backward_config=apply_optimizer_in_backward_config,
            variable_batch_size=variable_batch_size,
            pooling=pooling,
        )
