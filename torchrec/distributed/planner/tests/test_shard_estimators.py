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
            ("dense", "data_parallel"): [
                0.00037119411932357005,
                0.00037119411932357005,
            ],
            ("batched_dense", "data_parallel"): [
                0.00035296814098804687,
                0.00035296814098804687,
            ],
            ("dense", "table_wise"): [0.0033004209102576545],
            ("batched_dense", "table_wise"): [0.0032639689535866085],
            ("batched_fused", "table_wise"): [0.003221441670803721],
            ("sparse", "table_wise"): [0.0033004209102576545],
            ("batched_fused_uvm", "table_wise"): [0.07797689998851104],
            ("batched_fused_uvm_caching", "table_wise"): [0.020502698518924556],
            ("dense", "row_wise"): [0.003239667649139244, 0.003239667649139244],
            ("batched_dense", "row_wise"): [0.003221441670803721, 0.003221441670803721],
            ("batched_fused", "row_wise"): [0.003200178029412277, 0.003200178029412277],
            ("sparse", "row_wise"): [0.003239667649139244, 0.003239667649139244],
            ("batched_fused_uvm", "row_wise"): [
                0.04057790718826594,
                0.04057790718826594,
            ],
            ("batched_fused_uvm_caching", "row_wise"): [
                0.011840806453472696,
                0.011840806453472696,
            ],
            ("dense", "column_wise"): [0.0033004209102576545],
            ("batched_dense", "column_wise"): [0.0032639689535866085],
            ("batched_fused", "column_wise"): [0.003221441670803721],
            ("sparse", "column_wise"): [0.0033004209102576545],
            ("batched_fused_uvm", "column_wise"): [0.07797689998851104],
            ("batched_fused_uvm_caching", "column_wise"): [0.020502698518924556],
            ("dense", "table_column_wise"): [0.0033004209102576545],
            ("batched_dense", "table_column_wise"): [0.0032639689535866085],
            ("batched_fused", "table_column_wise"): [0.003221441670803721],
            ("sparse", "table_column_wise"): [0.0033004209102576545],
            ("batched_fused_uvm", "table_column_wise"): [0.07797689998851104],
            ("batched_fused_uvm_caching", "table_column_wise"): [0.020502698518924556],
            ("dense", "table_row_wise"): [0.0033032459368996605, 0.0033032459368996605],
            ("batched_dense", "table_row_wise"): [
                0.0032850199585641375,
                0.0032850199585641375,
            ],
            ("batched_fused", "table_row_wise"): [
                0.0032637563171726935,
                0.0032637563171726935,
            ],
            ("sparse", "table_row_wise"): [
                0.0033032459368996605,
                0.0033032459368996605,
            ],
            ("batched_fused_uvm", "table_row_wise"): [
                0.040641485476026355,
                0.040641485476026355,
            ],
            ("batched_fused_uvm_caching", "table_row_wise"): [
                0.011904384741233112,
                0.011904384741233112,
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
