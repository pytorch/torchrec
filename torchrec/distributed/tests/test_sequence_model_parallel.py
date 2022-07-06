#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest
from typing import Dict, List, Optional, Type

import hypothesis.strategies as st
import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from hypothesis import given, settings, Verbosity
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.planner import ParameterConstraints
from torchrec.distributed.quantized_comms.types import CommType, QuantizedCommsConfig
from torchrec.distributed.test_utils.multi_process import MultiProcessTestBase
from torchrec.distributed.test_utils.test_model import TestSparseNNBase
from torchrec.distributed.test_utils.test_sharding import sharding_single_rank_test
from torchrec.distributed.tests.test_sequence_model import (
    TestEmbeddingCollectionSharder,
    TestSequenceSparseNN,
)
from torchrec.distributed.types import ShardingType
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.test_utils import seed_and_log, skip_if_asan_class


@skip_if_asan_class
class SequenceModelParallelTest(MultiProcessTestBase):
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    # pyre-fixme[56]
    @given(
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
        quantized_comms_config=st.sampled_from(
            [
                None,
                QuantizedCommsConfig(
                    forward_precision=CommType.FP16, backward_precision=CommType.FP16
                ),
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=2, deadline=None)
    def test_sharding_nccl_rw(
        self,
        sharding_type: str,
        kernel_type: str,
        quantized_comms_config: Optional[QuantizedCommsConfig],
    ) -> None:
        self._test_sharding(
            sharders=[
                TestEmbeddingCollectionSharder(
                    sharding_type=sharding_type,
                    kernel_type=kernel_type,
                    quantized_comms_config=quantized_comms_config,
                )
            ],
            backend="nccl",
            using_quantized_comms=quantized_comms_config is not None,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    # pyre-fixme[56]
    @given(
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
    )
    @settings(verbosity=Verbosity.verbose, max_examples=1, deadline=None)
    def test_sharding_nccl_dp(self, sharding_type: str, kernel_type: str) -> None:
        self._test_sharding(
            sharders=[
                TestEmbeddingCollectionSharder(
                    sharding_type=sharding_type,
                    kernel_type=kernel_type,
                )
            ],
            backend="nccl",
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    # pyre-fixme[56]
    @given(
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
        quantized_comms_config=st.sampled_from(
            [
                None,
                QuantizedCommsConfig(
                    forward_precision=CommType.FP16, backward_precision=CommType.FP16
                ),
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=2, deadline=None)
    def test_sharding_nccl_tw(
        self,
        sharding_type: str,
        kernel_type: str,
        quantized_comms_config: Optional[QuantizedCommsConfig],
    ) -> None:
        self._test_sharding(
            sharders=[
                TestEmbeddingCollectionSharder(
                    sharding_type=sharding_type,
                    kernel_type=kernel_type,
                    quantized_comms_config=quantized_comms_config,
                )
            ],
            backend="nccl",
            using_quantized_comms=quantized_comms_config is not None,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    # pyre-fixme[56]
    @given(
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
    )
    @settings(verbosity=Verbosity.verbose, max_examples=2, deadline=None)
    def test_sharding_nccl_cw(self, sharding_type: str, kernel_type: str) -> None:
        self._test_sharding(
            sharders=[
                TestEmbeddingCollectionSharder(
                    sharding_type=sharding_type,
                    kernel_type=kernel_type,
                )
            ],
            backend="nccl",
            constraints={
                table.name: ParameterConstraints(min_partition=4)
                for table in self.tables
            },
        )

    @seed_and_log
    def setUp(self) -> None:
        super().setUp()

        num_features = 4

        self.tables = [
            EmbeddingConfig(
                num_embeddings=(i + 1) * 11,
                embedding_dim=16,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(num_features)
        ]
        self.embedding_groups = {
            "group_0": ["feature_" + str(i) for i in range(num_features)]
        }

    def _test_sharding(
        self,
        sharders: List[TestEmbeddingCollectionSharder],
        backend: str = "gloo",
        world_size: int = 2,
        local_size: Optional[int] = None,
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
        model_class: Type[TestSparseNNBase] = TestSequenceSparseNN,
        using_quantized_comms: bool = False,
    ) -> None:
        self._run_multi_process_test(
            callable=sharding_single_rank_test,
            world_size=world_size,
            local_size=local_size,
            model_class=model_class,
            tables=self.tables,
            embedding_groups=self.embedding_groups,
            sharders=sharders,
            optim=EmbOptimType.EXACT_SGD,
            backend=backend,
            constraints=constraints,
            using_quantized_comms=using_quantized_comms,
        )
