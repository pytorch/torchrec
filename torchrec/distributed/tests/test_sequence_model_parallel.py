#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import unittest
from typing import Any, Dict, List, Optional, Tuple, Type

import hypothesis.strategies as st
import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from hypothesis import assume, given, settings, Verbosity
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.fbgemm_qcomm_codec import CommType, QCommsConfig
from torchrec.distributed.planner import ParameterConstraints
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
        sharding_type=st.just(ShardingType.ROW_WISE.value),
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
    )
    @settings(verbosity=Verbosity.verbose, max_examples=2, deadline=None)
    def test_sharding_nccl_rw(
        self,
        sharding_type: str,
        kernel_type: str,
        qcomms_config: Optional[QCommsConfig],
        apply_optimizer_in_backward_config: Optional[
            Dict[str, Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]]
        ],
    ) -> None:
        assume(
            apply_optimizer_in_backward_config is None
            or kernel_type != EmbeddingComputeKernel.DENSE.value
        )
        self._test_sharding(
            sharders=[
                TestEmbeddingCollectionSharder(
                    sharding_type=sharding_type,
                    kernel_type=kernel_type,
                    qcomms_config=qcomms_config,
                )
            ],
            backend="nccl",
            qcomms_config=qcomms_config,
            apply_optimizer_in_backward_config=apply_optimizer_in_backward_config,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    # pyre-fixme[56]
    @given(
        sharding_type=st.just(ShardingType.DATA_PARALLEL.value),
        kernel_type=st.just(EmbeddingComputeKernel.DENSE.value),
        apply_optimizer_in_backward_config=st.just(None),
        # TODO - need to enable optimizer overlapped behavior for data_parallel tables
        # apply_optimizer_in_backward_config=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=1, deadline=None)
    def test_sharding_nccl_dp(
        self,
        sharding_type: str,
        kernel_type: str,
        apply_optimizer_in_backward_config: Optional[
            Dict[str, Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]]
        ],
    ) -> None:
        self._test_sharding(
            sharders=[
                TestEmbeddingCollectionSharder(
                    sharding_type=sharding_type,
                    kernel_type=kernel_type,
                )
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
        sharding_type=st.just(ShardingType.TABLE_WISE.value),
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
    )
    @settings(verbosity=Verbosity.verbose, max_examples=2, deadline=None)
    def test_sharding_nccl_tw(
        self,
        sharding_type: str,
        kernel_type: str,
        qcomms_config: Optional[QCommsConfig],
        apply_optimizer_in_backward_config: Optional[
            Dict[str, Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]]
        ],
    ) -> None:
        assume(
            apply_optimizer_in_backward_config is None
            or kernel_type != EmbeddingComputeKernel.DENSE.value
        )
        self._test_sharding(
            sharders=[
                TestEmbeddingCollectionSharder(
                    sharding_type=sharding_type,
                    kernel_type=kernel_type,
                    qcomms_config=qcomms_config,
                )
            ],
            backend="nccl",
            qcomms_config=qcomms_config,
            apply_optimizer_in_backward_config=apply_optimizer_in_backward_config,
            variable_batch_size=False,
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    # pyre-fixme[56]
    @given(
        sharding_type=st.just(ShardingType.COLUMN_WISE.value),
        kernel_type=st.sampled_from(
            [
                EmbeddingComputeKernel.DENSE.value,
                EmbeddingComputeKernel.FUSED.value,
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
    @settings(verbosity=Verbosity.verbose, max_examples=2, deadline=None)
    def test_sharding_nccl_cw(
        self,
        sharding_type: str,
        kernel_type: str,
        apply_optimizer_in_backward_config: Optional[
            Dict[str, Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]]
        ],
    ) -> None:
        assume(
            apply_optimizer_in_backward_config is None
            or kernel_type != EmbeddingComputeKernel.DENSE.value
        )
        self._test_sharding(
            sharders=[
                TestEmbeddingCollectionSharder(
                    sharding_type=sharding_type,
                    kernel_type=kernel_type,
                )
            ],
            backend="nccl",
            constraints={
                table.name: ParameterConstraints(min_partition=8)
                for table in self.tables
            },
            apply_optimizer_in_backward_config=apply_optimizer_in_backward_config,
            variable_batch_size=False,
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
                ShardingType.COLUMN_WISE.value,
                ShardingType.ROW_WISE.value,
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=3, deadline=None)
    def test_sharding_variable_batch(
        self,
        sharding_type: str,
    ) -> None:
        self._test_sharding(
            sharders=[
                TestEmbeddingCollectionSharder(
                    sharding_type=sharding_type,
                    kernel_type=EmbeddingComputeKernel.FUSED.value,
                )
            ],
            backend="nccl",
            constraints={
                table.name: ParameterConstraints(min_partition=4)
                for table in self.tables
            },
            variable_batch_size=True,
        )

    # pyre-fixme[56]
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    def test_sharding_empty_rank(self) -> None:
        table = self.tables[0]
        embedding_groups = {"group_0": table.feature_names}
        self._run_multi_process_test(
            callable=sharding_single_rank_test,
            world_size=2,
            model_class=TestSequenceSparseNN,
            tables=[table],
            embedding_groups=embedding_groups,
            sharders=[
                TestEmbeddingCollectionSharder(
                    sharding_type=ShardingType.TABLE_WISE.value,
                    kernel_type=EmbeddingComputeKernel.FUSED.value,
                )
            ],
            optim=EmbOptimType.EXACT_SGD,
            backend="nccl",
            variable_batch_size=True,
        )

    @seed_and_log
    def setUp(self) -> None:
        super().setUp()

        num_features = 4
        shared_features = 2

        initial_tables = [
            EmbeddingConfig(
                num_embeddings=(i + 1) * 11,
                embedding_dim=16,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(num_features)
        ]

        shared_features_tables = [
            EmbeddingConfig(
                num_embeddings=(i + 1) * 11,
                embedding_dim=16,
                name="table_" + str(i + num_features),
                feature_names=["feature_" + str(i)],
            )
            for i in range(shared_features)
        ]

        self.tables = initial_tables + shared_features_tables
        self.shared_features = [f"feature_{i}" for i in range(shared_features)]

        self.embedding_groups = {
            "group_0": [
                (
                    f"{feature}@{table.name}"
                    if feature in self.shared_features
                    else feature
                )
                for table in self.tables
                for feature in table.feature_names
            ]
        }

    def _test_sharding(
        self,
        sharders: List[TestEmbeddingCollectionSharder],
        backend: str = "gloo",
        world_size: int = 2,
        local_size: Optional[int] = None,
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
        model_class: Type[TestSparseNNBase] = TestSequenceSparseNN,
        qcomms_config: Optional[QCommsConfig] = None,
        apply_optimizer_in_backward_config: Optional[
            Dict[str, Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]]
        ] = None,
        variable_batch_size: bool = False,
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
            qcomms_config=qcomms_config,
            apply_optimizer_in_backward_config=apply_optimizer_in_backward_config,
            variable_batch_size=variable_batch_size,
        )
