#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Optional

import torch
from hypothesis import given, settings, strategies as st, Verbosity
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.fbgemm_qcomm_codec import CommType, QCommsConfig
from torchrec.distributed.planner import ParameterConstraints
from torchrec.distributed.test_utils.test_model import (
    TestTowerCollectionSparseNN,
    TestTowerSparseNN,
)
from torchrec.distributed.test_utils.test_model_parallel import ModelParallelTestShared
from torchrec.distributed.test_utils.test_sharding import (
    create_test_sharder,
    SharderType,
)
from torchrec.distributed.types import ShardingType
from torchrec.test_utils import skip_if_asan_class


@skip_if_asan_class
class ModelParallelHierarchicalTest(ModelParallelTestShared):
    """
    Testing hierarchical sharding types.

    NOTE:
        Requires at least 4 GPUs to test.
    """

    @unittest.skipIf(
        torch.cuda.device_count() <= 3,
        "Not enough GPUs, this test requires at least four GPUs",
    )
    # pyre-fixme[56]
    @given(
        sharder_type=st.sampled_from(
            [
                SharderType.EMBEDDING_BAG.value,
                SharderType.EMBEDDING_BAG_COLLECTION.value,
            ]
        ),
        sharding_type=st.just(ShardingType.TABLE_ROW_WISE.value),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.DENSE.value,
                EmbeddingComputeKernel.FUSED.value,
            ]
        ),
        local_size=st.sampled_from([2]),
        qcomms_config=st.sampled_from(
            [
                None,
                QCommsConfig(
                    forward_precision=CommType.FP16, backward_precision=CommType.BF16
                ),
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=4, deadline=None)
    def test_sharding_nccl_twrw(
        self,
        sharder_type: str,
        sharding_type: str,
        kernel_type: str,
        local_size: int,
        qcomms_config: Optional[QCommsConfig],
    ) -> None:
        self._test_sharding(
            # pyre-ignore[6]
            sharders=[
                create_test_sharder(
                    sharder_type,
                    sharding_type,
                    kernel_type,
                    qcomms_config=qcomms_config,
                    device=torch.device("cuda"),
                ),
            ],
            backend="nccl",
            world_size=4,
            local_size=local_size,
            qcomms_config=qcomms_config,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 3,
        "Not enough GPUs, this test requires at least four GPUs",
    )
    # pyre-fixme[56]
    @given(
        sharder_type=st.sampled_from(
            [
                SharderType.EMBEDDING_BAG.value,
                SharderType.EMBEDDING_BAG_COLLECTION.value,
            ]
        ),
        sharding_type=st.sampled_from(
            [
                ShardingType.TABLE_COLUMN_WISE.value,
            ]
        ),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.DENSE.value,
                EmbeddingComputeKernel.FUSED.value,
            ]
        ),
        local_size=st.sampled_from([2]),
        qcomms_config=st.sampled_from(
            [
                None,
                QCommsConfig(
                    forward_precision=CommType.FP16, backward_precision=CommType.BF16
                ),
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=4, deadline=None)
    def test_sharding_nccl_twcw(
        self,
        sharder_type: str,
        sharding_type: str,
        kernel_type: str,
        local_size: int,
        qcomms_config: Optional[QCommsConfig],
    ) -> None:
        world_size = 4
        self._test_sharding(
            # pyre-ignore[6]
            sharders=[
                create_test_sharder(
                    sharder_type,
                    sharding_type,
                    kernel_type,
                    qcomms_config=qcomms_config,
                    device=torch.device("cuda"),
                )
            ],
            backend="nccl",
            world_size=world_size,
            local_size=local_size,
            constraints={
                table.name: ParameterConstraints(min_partition=4)
                for table in self.tables
            },
            qcomms_config=qcomms_config,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 3,
        "Not enough GPUs, this test requires at least four GPUs",
    )
    # pyre-fixme[56]
    @given(
        sharding_type=st.sampled_from(
            [
                ShardingType.TABLE_ROW_WISE.value,
                ShardingType.TABLE_COLUMN_WISE.value,
            ]
        ),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.DENSE.value,
                EmbeddingComputeKernel.FUSED.value,
            ]
        ),
        qcomms_config=st.sampled_from(
            [
                None,
                QCommsConfig(
                    forward_precision=CommType.FP16, backward_precision=CommType.BF16
                ),
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=4, deadline=None)
    def test_embedding_tower_nccl(
        self,
        sharding_type: str,
        kernel_type: str,
        qcomms_config: Optional[QCommsConfig],
    ) -> None:
        self._test_sharding(
            # pyre-ignore[6]
            sharders=[
                create_test_sharder(
                    SharderType.EMBEDDING_TOWER.value,
                    sharding_type,
                    kernel_type,
                    qcomms_config=qcomms_config,
                    device=torch.device("cuda"),
                )
            ],
            backend="nccl",
            world_size=4,
            local_size=2,
            model_class=TestTowerSparseNN,
            qcomms_config=qcomms_config,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 3,
        "Not enough GPUs, this test requires at least four GPUs",
    )
    # pyre-fixme[56]
    @given(
        sharding_type=st.sampled_from(
            [
                ShardingType.TABLE_ROW_WISE.value,
                ShardingType.TABLE_COLUMN_WISE.value,
            ]
        ),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.DENSE.value,
                EmbeddingComputeKernel.FUSED.value,
            ]
        ),
        qcomms_config=st.sampled_from(
            [
                None,
                QCommsConfig(
                    forward_precision=CommType.FP16, backward_precision=CommType.BF16
                ),
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=4, deadline=None)
    def test_embedding_tower_collection_nccl(
        self,
        sharding_type: str,
        kernel_type: str,
        qcomms_config: Optional[QCommsConfig],
    ) -> None:
        self._test_sharding(
            # pyre-ignore[6]
            sharders=[
                create_test_sharder(
                    SharderType.EMBEDDING_TOWER_COLLECTION.value,
                    sharding_type,
                    kernel_type,
                    qcomms_config=qcomms_config,
                    device=torch.device("cuda"),
                )
            ],
            backend="nccl",
            world_size=4,
            local_size=2,
            model_class=TestTowerCollectionSparseNN,
            qcomms_config=qcomms_config,
        )
