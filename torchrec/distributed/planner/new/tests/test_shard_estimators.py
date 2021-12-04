#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torchrec.distributed.embeddingbag import (
    EmbeddingBagCollectionSharder,
)
from torchrec.distributed.planner.new.enumerators import EmbeddingEnumerator
from torchrec.distributed.planner.new.shard_estimators import EmbeddingPerfEstimator
from torchrec.distributed.planner.new.types import Topology
from torchrec.distributed.tests.test_model import TestSparseNN
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
            sharders=[EmbeddingBagCollectionSharder()],
        )

        expected_perfs = {
            ("dense", "data_parallel"): [398.5666507405638, 398.5666507405638],
            ("batched_dense", "data_parallel"): [378.9966555183946, 378.9966555183946],
            ("dense", "table_wise"): [3543.7999681477945],
            ("batched_dense", "table_wise"): [3504.659977703456],
            ("batched_fused", "table_wise"): [3458.9966555183946],
            ("sparse", "table_wise"): [3543.7999681477945],
            ("batched_fused_uvm", "table_wise"): [83727.05882352941],
            ("batched_fused_uvm_caching", "table_wise"): [22014.604904632153],
            ("dense", "row_wise"): [3478.566650740564, 3478.566650740564],
            ("batched_dense", "row_wise"): [3458.9966555183946, 3458.9966555183946],
            ("batched_fused", "row_wise"): [3436.1649944258643, 3436.1649944258643],
            ("sparse", "row_wise"): [3478.566650740564, 3478.566650740564],
            ("batched_fused_uvm", "row_wise"): [43570.19607843138, 43570.19607843138],
            ("batched_fused_uvm_caching", "row_wise"): [
                12713.969118982744,
                12713.969118982744,
            ],
            ("dense", "table_row_wise"): [3546.833317407231, 3546.833317407231],
            ("batched_dense", "table_row_wise"): [
                3527.2633221850615,
                3527.2633221850615,
            ],
            ("batched_fused", "table_row_wise"): [3504.431661092531, 3504.431661092531],
            ("sparse", "table_row_wise"): [3546.833317407231, 3546.833317407231],
            ("batched_fused_uvm", "table_row_wise"): [
                43638.46274509804,
                43638.46274509804,
            ],
            ("batched_fused_uvm_caching", "table_row_wise"): [
                12782.23578564941,
                12782.23578564941,
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
