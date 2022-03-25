#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import cast

import torch
from torchrec.distributed.embeddingbag import (
    EmbeddingBagCollectionSharder,
)
from torchrec.distributed.planner.enumerators import EmbeddingEnumerator
from torchrec.distributed.planner.shard_estimators import EmbeddingPerfEstimator
from torchrec.distributed.planner.types import Topology
from torchrec.distributed.test_utils.test_model import TestSparseNN
from torchrec.distributed.types import ModuleSharder
from torchrec.modules.embedding_configs import EmbeddingBagConfig


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
            ("dense", "data_parallel"): [0.0003985666507405638, 0.0003985666507405638],
            ("batched_dense", "data_parallel"): [
                0.00037899665551839465,
                0.00037899665551839465,
            ],
            ("dense", "table_wise"): [0.0035437999681477944],
            ("batched_dense", "table_wise"): [0.0035046599777034558],
            ("batched_fused", "table_wise"): [0.0034589966555183945],
            ("sparse", "table_wise"): [0.0035437999681477944],
            ("batched_fused_uvm", "table_wise"): [0.08372705882352942],
            ("batched_fused_uvm_caching", "table_wise"): [0.022014604904632154],
            ("dense", "row_wise"): [0.0034785666507405636, 0.0034785666507405636],
            ("batched_dense", "row_wise"): [
                0.0034589966555183945,
                0.0034589966555183945,
            ],
            ("batched_fused", "row_wise"): [0.003436164994425864, 0.003436164994425864],
            ("sparse", "row_wise"): [0.0034785666507405636, 0.0034785666507405636],
            ("batched_fused_uvm", "row_wise"): [
                0.04357019607843137,
                0.04357019607843137,
            ],
            ("batched_fused_uvm_caching", "row_wise"): [
                0.012713969118982742,
                0.012713969118982742,
            ],
            ("dense", "column_wise"): [0.0035437999681477944],
            ("batched_dense", "column_wise"): [0.0035046599777034558],
            ("batched_fused", "column_wise"): [0.0034589966555183945],
            ("sparse", "column_wise"): [0.0035437999681477944],
            ("batched_fused_uvm", "column_wise"): [0.08372705882352942],
            ("batched_fused_uvm_caching", "column_wise"): [0.022014604904632154],
            ("dense", "table_column_wise"): [0.0035437999681477944],
            ("batched_dense", "table_column_wise"): [0.0035046599777034558],
            ("batched_fused", "table_column_wise"): [0.0034589966555183945],
            ("sparse", "table_column_wise"): [0.0035437999681477944],
            ("batched_fused_uvm", "table_column_wise"): [0.08372705882352942],
            ("batched_fused_uvm_caching", "table_column_wise"): [0.022014604904632154],
            ("dense", "table_row_wise"): [0.0035468333174072304, 0.0035468333174072304],
            ("batched_dense", "table_row_wise"): [
                0.0035272633221850613,
                0.0035272633221850613,
            ],
            ("batched_fused", "table_row_wise"): [
                0.0035044316610925307,
                0.0035044316610925307,
            ],
            ("sparse", "table_row_wise"): [
                0.0035468333174072304,
                0.0035468333174072304,
            ],
            ("batched_fused_uvm", "table_row_wise"): [
                0.04363846274509804,
                0.04363846274509804,
            ],
            ("batched_fused_uvm_caching", "table_row_wise"): [
                0.01278223578564941,
                0.01278223578564941,
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
