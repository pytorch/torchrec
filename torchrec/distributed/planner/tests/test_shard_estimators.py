#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import cast

import torch
from torchrec.distributed.embedding import EmbeddingCollectionSharder
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.planner.enumerators import EmbeddingEnumerator
from torchrec.distributed.planner.shard_estimators import EmbeddingPerfEstimator
from torchrec.distributed.planner.types import Topology
from torchrec.distributed.test_utils.test_model import TestSparseNN
from torchrec.distributed.tests.test_sequence_model import TestSequenceSparseNN
from torchrec.distributed.types import ModuleSharder
from torchrec.modules.embedding_configs import EmbeddingBagConfig, EmbeddingConfig


class TestEmbeddingPerfEstimator(unittest.TestCase):
    def setUp(self) -> None:
        topology = Topology(world_size=2, compute_device="cuda")
        self.estimator = EmbeddingPerfEstimator(topology=topology)
        self.enumerator = EmbeddingEnumerator(
            topology=topology, estimator=self.estimator
        )

    def test_1_table_perf(self) -> None:
        tables = [
            EmbeddingBagConfig(
                num_embeddings=100,
                embedding_dim=10,
                name="table_0",
                feature_names=["feature_0"],
            )
        ]
        model = TestSparseNN(tables=tables, weighted_tables=[])
        sharding_options = self.enumerator.enumerate(
            module=model,
            sharders=[
                cast(ModuleSharder[torch.nn.Module], EmbeddingBagCollectionSharder())
            ],
        )

        expected_perfs = {
            ("batched_dense", "data_parallel"): [
                0.0004935158269195386,
                0.0004935158269195386,
            ],
            ("batched_fused", "table_wise"): [0.0011095368078055323],
            ("batched_fused_uvm", "table_wise"): [0.1729105033126532],
            ("batched_fused_uvm_caching", "table_wise"): [0.040145097917908434],
            ("batched_fused", "column_wise"): [0.0011095368078055323],
            ("batched_fused_uvm", "column_wise"): [0.1729105033126532],
            ("batched_fused_uvm_caching", "column_wise"): [0.040145097917908434],
            ("batched_fused", "table_column_wise"): [0.0011095368078055323],
            ("batched_fused_uvm", "table_column_wise"): [0.1729105033126532],
            ("batched_fused_uvm_caching", "table_column_wise"): [0.040145097917908434],
            ("batched_fused", "row_wise"): [
                0.00043569201211068144,
                0.00043569201211068144,
            ],
            ("batched_fused_uvm", "row_wise"): [
                0.054393095128676475,
                0.054393095128676475,
            ],
            ("batched_fused_uvm_caching", "row_wise"): [
                0.012695561962491483,
                0.012695561962491483,
            ],
            ("batched_fused", "table_row_wise"): [
                0.00043569201211068144,
                0.00043569201211068144,
            ],
            ("batched_fused_uvm", "table_row_wise"): [
                0.054393095128676475,
                0.054393095128676475,
            ],
            ("batched_fused_uvm_caching", "table_row_wise"): [
                0.012695561962491483,
                0.012695561962491483,
            ],
        }

        perfs = {
            (
                sharding_option.compute_kernel,
                sharding_option.sharding_type,
            ): [shard.perf for shard in sharding_option.shards]
            for sharding_option in sharding_options
        }

        self.assertEqual(expected_perfs, perfs)

    def test_sequence_2_table_perf(self) -> None:
        tables = [
            EmbeddingConfig(
                num_embeddings=128,
                embedding_dim=32,
                name="table_0",
                feature_names=["feature_0"],
            ),
            EmbeddingConfig(
                num_embeddings=256,
                embedding_dim=32,
                name="table_1",
                feature_names=["feature_1"],
            ),
        ]
        model = TestSequenceSparseNN(tables=tables)
        sharding_options = self.enumerator.enumerate(
            module=model,
            sharders=[
                cast(ModuleSharder[torch.nn.Module], EmbeddingCollectionSharder())
            ],
        )

        expected_perfs = {
            ("batched_dense", "data_parallel"): [
                0.002677347614879459,
                0.002677347614879459,
            ],
            ("batched_fused", "table_wise"): [0.001880471390093715],
            ("batched_fused_uvm", "table_wise"): [0.25958192114736517],
            ("batched_fused_uvm_caching", "table_wise"): [0.060433813055248066],
            ("batched_fused", "row_wise"): [
                0.0007915177871551004,
                0.0007915177871551004,
            ],
            ("batched_fused_uvm", "row_wise"): [0.1036341050091912, 0.1036341050091912],
            ("batched_fused_uvm_caching", "row_wise"): [
                0.024158779217047007,
                0.024158779217047007,
            ],
        }

        perfs = {
            (
                sharding_option.compute_kernel,
                sharding_option.sharding_type,
            ): [shard.perf for shard in sharding_option.shards]
            for sharding_option in sharding_options
        }

        self.assertEqual(expected_perfs, perfs)
