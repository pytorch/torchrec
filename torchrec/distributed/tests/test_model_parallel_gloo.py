#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional, Tuple, Type

import hypothesis.strategies as st
import torch
from hypothesis import given, settings, Verbosity
from torchrec.distributed.embedding_types import EmbeddingComputeKernel, ShardingType
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
from torchrec.test_utils import skip_if_asan_class


@skip_if_asan_class
class ModelParallelTest(ModelParallelTestShared):
    # pyre-fixme[56]
    @given(
        sharder_type=st.sampled_from(
            [
                # TODO: enable it with correct semantics, see T104397332
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
                None,
                # On gloo, BF16 is not supported as dtype.
                QCommsConfig(
                    forward_precision=CommType.FP16, backward_precision=CommType.FP16
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
    )
    @settings(verbosity=Verbosity.verbose, max_examples=8, deadline=None)
    def test_sharding_gloo_tw(
        self,
        sharder_type: str,
        sharding_type: str,
        kernel_type: str,
        qcomms_config: Optional[QCommsConfig],
        apply_optimizer_in_backward_config: Optional[
            Dict[str, Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]]
        ],
    ) -> None:
        self._test_sharding(
            # pyre-ignore[6]
            sharders=[
                create_test_sharder(
                    sharder_type,
                    sharding_type,
                    kernel_type,
                    qcomms_config=qcomms_config,
                    device=torch.device("cpu"),
                ),
            ],
            qcomms_config=qcomms_config,
            backend="gloo",
            apply_optimizer_in_backward_config=apply_optimizer_in_backward_config,
        )

    # pyre-fixme[56]
    @given(
        sharder_type=st.sampled_from(
            [
                # TODO: enable it with correct semantics, see T104397332
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
                EmbeddingComputeKernel.DENSE.value,
                EmbeddingComputeKernel.FUSED.value,
            ]
        ),
        qcomms_config=st.sampled_from(
            [
                None,
                # On gloo, BF16 is not supported as dtype.
                QCommsConfig(
                    forward_precision=CommType.FP16,
                    backward_precision=CommType.FP16,
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
    )
    @settings(verbosity=Verbosity.verbose, max_examples=8, deadline=None)
    def test_sharding_gloo_cw(
        self,
        sharder_type: str,
        sharding_type: str,
        kernel_type: str,
        qcomms_config: Optional[QCommsConfig],
        apply_optimizer_in_backward_config: Optional[
            Dict[str, Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]]
        ],
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
                    device=torch.device("cpu"),
                ),
            ],
            qcomms_config=qcomms_config,
            backend="gloo",
            world_size=world_size,
            constraints={
                table.name: ParameterConstraints(min_partition=4)
                for table in self.tables
            },
            apply_optimizer_in_backward_config=apply_optimizer_in_backward_config,
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
                # TODO dp+batch_fused is numerically buggy in cpu
                # EmbeddingComputeKernel.FUSED.value,
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=8, deadline=None)
    def test_sharding_gloo_dp(
        self, sharder_type: str, sharding_type: str, kernel_type: str
    ) -> None:
        self._test_sharding(
            # pyre-ignore[6]
            sharders=[
                create_test_sharder(sharder_type, sharding_type, kernel_type),
            ],
            backend="gloo",
        )


class ModelParallelStateDictTestGloo(ModelParallelStateDictTest):
    pass


class ModelParallelSparseOnlyTestGloo(ModelParallelSparseOnlyTest):
    pass
