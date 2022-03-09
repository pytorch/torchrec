#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest
from typing import List, Optional, Type

import hypothesis.strategies as st
import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from hypothesis import Verbosity, given, settings
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.test_utils.test_model import TestSparseNNBase
from torchrec.distributed.test_utils.test_model_parallel_base import (
    ModelParallelTestBase,
)
from torchrec.distributed.tests.test_sequence_model import (
    TestEmbeddingCollectionSharder,
    TestSequenceSparseNN,
)
from torchrec.distributed.types import ShardingType
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.test_utils import (
    skip_if_asan_class,
    seed_and_log,
)


@skip_if_asan_class
class SequenceModelParallelTest(ModelParallelTestBase):
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
                EmbeddingComputeKernel.SPARSE.value,
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=2, deadline=None)
    def test_sharding_nccl_rw(self, sharding_type: str, kernel_type: str) -> None:
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
                EmbeddingComputeKernel.SPARSE.value,
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=2, deadline=None)
    def test_sharding_nccl_tw(self, sharding_type: str, kernel_type: str) -> None:
        self._test_sharding(
            sharders=[
                TestEmbeddingCollectionSharder(
                    sharding_type=sharding_type,
                    kernel_type=kernel_type,
                )
            ],
            backend="nccl",
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
        model_class: Type[TestSparseNNBase] = TestSequenceSparseNN,
    ) -> None:
        self._run_multi_process_test(
            # pyre-ignore [6]
            callable=self._test_sharding_single_rank,
            world_size=world_size,
            local_size=local_size,
            # pyre-ignore [6]
            model_class=model_class,
            tables=self.tables,
            embedding_groups=self.embedding_groups,
            # pyre-fixme[6]
            sharders=sharders,
            optim=EmbOptimType.EXACT_SGD,
            backend=backend,
        )
