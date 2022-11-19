#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Any, cast, Dict, Optional, Tuple, Type

import hypothesis.strategies as st
import torch
import torch.nn as nn
from hypothesis import assume, given, settings, Verbosity
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.fbgemm_qcomm_codec import CommType, QCommsConfig
from torchrec.distributed.planner import ParameterConstraints
from torchrec.distributed.test_utils.test_model_parallel import ModelParallelTestShared
from torchrec.distributed.test_utils.test_model_parallel_base import (
    ModelParallelSparseOnlyTest,
    ModelParallelStateDictTest,
)
from torchrec.distributed.test_utils.test_sharding import (
    create_test_sharder,
    SharderType,
)
from torchrec.distributed.types import ModuleSharder, ShardingType
from torchrec.test_utils import skip_if_asan_class


@skip_if_asan_class
class ModelParallelTest(ModelParallelTestShared):
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
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
                ShardingType.ROW_WISE.value,
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
        apply_optimizer_in_backward_config=st.sampled_from(
            [
                None,
                {
                    "embeddingbags": (torch.optim.SGD, {"lr": 0.01}),
                    "embeddings": (torch.optim.SGD, {"lr": 0.2}),
                },
            ]
        ),
        variable_batch_size=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=4, deadline=None)
    def test_sharding_nccl_rw(
        self,
        sharder_type: str,
        sharding_type: str,
        kernel_type: str,
        qcomms_config: Optional[QCommsConfig],
        apply_optimizer_in_backward_config: Optional[
            Dict[str, Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]]
        ],
        variable_batch_size: bool,
    ) -> None:
        assume(
            sharder_type == SharderType.EMBEDDING_BAG_COLLECTION.value
            or not variable_batch_size
        )
        self._test_sharding(
            sharders=[
                cast(
                    ModuleSharder[nn.Module],
                    create_test_sharder(
                        sharder_type,
                        sharding_type,
                        kernel_type,
                        qcomms_config=qcomms_config,
                        device=torch.device("cuda"),
                        variable_batch_size=variable_batch_size,
                    ),
                ),
            ],
            qcomms_config=qcomms_config,
            backend="nccl",
            apply_optimizer_in_backward_config=apply_optimizer_in_backward_config,
            variable_batch_size=variable_batch_size,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
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
                ShardingType.DATA_PARALLEL.value,
            ]
        ),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.DENSE.value,
            ]
        ),
        apply_optimizer_in_backward_config=st.sampled_from([None]),
        # TODO - need to enable optimizer overlapped behavior for data_parallel tables
    )
    @settings(verbosity=Verbosity.verbose, max_examples=2, deadline=None)
    def test_sharding_nccl_dp(
        self,
        sharder_type: str,
        sharding_type: str,
        kernel_type: str,
        apply_optimizer_in_backward_config: Optional[
            Dict[str, Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]]
        ],
    ) -> None:

        self._test_sharding(
            # pyre-ignore[6]
            sharders=[
                create_test_sharder(sharder_type, sharding_type, kernel_type),
            ],
            backend="nccl",
            apply_optimizer_in_backward_config=apply_optimizer_in_backward_config,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    # pyre-fixme[56]
    @given(
        sharder_type=st.sampled_from(
            [
                # SharderType.EMBEDDING_BAG.value,
                SharderType.EMBEDDING_BAG_COLLECTION.value,
            ]
        ),
        sharding_type=st.sampled_from(
            [
                ShardingType.COLUMN_WISE.value,
            ]
        ),
        kernel_type=st.sampled_from(
            [
                # EmbeddingComputeKernel.DENSE.value,
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
        apply_optimizer_in_backward_config=st.sampled_from(
            [
                None,
                {
                    "embeddingbags": (torch.optim.SGD, {"lr": 0.01}),
                    "embeddings": (torch.optim.SGD, {"lr": 0.2}),
                },
            ]
        ),
        variable_batch_size=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=8, deadline=None)
    def test_sharding_nccl_cw(
        self,
        sharder_type: str,
        sharding_type: str,
        kernel_type: str,
        qcomms_config: Optional[QCommsConfig],
        apply_optimizer_in_backward_config: Optional[
            Dict[str, Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]]
        ],
        variable_batch_size: bool,
    ) -> None:
        assume(
            sharder_type == SharderType.EMBEDDING_BAG_COLLECTION.value
            or not variable_batch_size
        )
        self._test_sharding(
            # pyre-ignore[6]
            sharders=[
                create_test_sharder(
                    sharder_type,
                    sharding_type,
                    kernel_type,
                    qcomms_config=qcomms_config,
                    device=torch.device("cuda"),
                    variable_batch_size=variable_batch_size,
                ),
            ],
            backend="nccl",
            qcomms_config=qcomms_config,
            constraints={
                table.name: ParameterConstraints(min_partition=4)
                for table in self.tables
            },
            apply_optimizer_in_backward_config=apply_optimizer_in_backward_config,
            variable_batch_size=variable_batch_size,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    # pyre-fixme[56]
    @given(
        sharder_type=st.sampled_from(
            [
                # SharderType.EMBEDDING_BAG.value,
                SharderType.EMBEDDING_BAG_COLLECTION.value,
            ]
        ),
        sharding_type=st.sampled_from(
            [
                ShardingType.TABLE_WISE.value,
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
                # None,
                QCommsConfig(
                    forward_precision=CommType.FP16,
                    backward_precision=CommType.BF16,
                ),
            ]
        ),
        apply_optimizer_in_backward_config=st.sampled_from(
            [
                None,
                {
                    "embeddingbags": (torch.optim.SGD, {"lr": 0.01}),
                    "embeddings": (torch.optim.SGD, {"lr": 0.2}),
                },
            ]
        ),
        variable_batch_size=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=8, deadline=None)
    def test_sharding_nccl_tw(
        self,
        sharder_type: str,
        sharding_type: str,
        kernel_type: str,
        qcomms_config: Optional[QCommsConfig],
        apply_optimizer_in_backward_config: Optional[
            Dict[str, Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]]
        ],
        variable_batch_size: bool,
    ) -> None:
        assume(
            sharder_type == SharderType.EMBEDDING_BAG_COLLECTION.value
            or not variable_batch_size
        )
        self._test_sharding(
            # pyre-ignore[6]
            sharders=[
                create_test_sharder(
                    sharder_type,
                    sharding_type,
                    kernel_type,
                    qcomms_config=qcomms_config,
                    device=torch.device("cuda"),
                    variable_batch_size=variable_batch_size,
                ),
            ],
            backend="nccl",
            qcomms_config=qcomms_config,
            apply_optimizer_in_backward_config=apply_optimizer_in_backward_config,
            variable_batch_size=variable_batch_size,
        )


class ModelParallelStateDictTestNccl(ModelParallelStateDictTest):
    pass


class ModelParallelSparseOnlyTestNccl(ModelParallelSparseOnlyTest):
    pass
