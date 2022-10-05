#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import cast

import torch
import torchrec

from torchrec.distributed.embedding import EmbeddingCollectionSharder
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.planner.constants import BATCH_SIZE
from torchrec.distributed.planner.enumerators import EmbeddingEnumerator
from torchrec.distributed.planner.shard_estimators import (
    _calculate_storage_specific_sizes,
    EmbeddingPerfEstimator,
)
from torchrec.distributed.planner.types import Topology
from torchrec.distributed.quant_embeddingbag import QuantEmbeddingBagCollectionSharder
from torchrec.distributed.test_utils.test_model import TestSparseNN
from torchrec.distributed.tests.test_quant_model_parallel import _quantize
from torchrec.distributed.tests.test_sequence_model import TestSequenceSparseNN
from torchrec.distributed.types import ModuleSharder, ShardingType
from torchrec.modules.embedding_configs import EmbeddingBagConfig, EmbeddingConfig


class TestEmbeddingPerfEstimator(unittest.TestCase):
    def setUp(self) -> None:
        self.topology = Topology(world_size=2, compute_device="cuda")
        self.estimator = EmbeddingPerfEstimator(topology=self.topology)
        self.enumerator = EmbeddingEnumerator(
            topology=self.topology, batch_size=BATCH_SIZE, estimator=self.estimator
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
                0.0004935158269195386,
                0.0004935158269195386,
            ],
            ("fused", "table_wise"): [0.0011095368078055323],
            ("fused_uvm", "table_wise"): [0.1729105033126532],
            ("fused_uvm_caching", "table_wise"): [0.040145097917908434],
            ("fused", "column_wise"): [0.0011095368078055323],
            ("fused_uvm", "column_wise"): [0.1729105033126532],
            ("fused_uvm_caching", "column_wise"): [0.040145097917908434],
            ("fused", "table_column_wise"): [0.0011095368078055323],
            ("fused_uvm", "table_column_wise"): [0.1729105033126532],
            ("fused_uvm_caching", "table_column_wise"): [0.040145097917908434],
            ("fused", "row_wise"): [
                0.00043569201211068144,
                0.00043569201211068144,
            ],
            ("fused_uvm", "row_wise"): [
                0.054393095128676475,
                0.054393095128676475,
            ],
            ("fused_uvm_caching", "row_wise"): [
                0.012695561962491483,
                0.012695561962491483,
            ],
            ("fused", "table_row_wise"): [
                0.00043569201211068144,
                0.00043569201211068144,
            ],
            ("fused_uvm", "table_row_wise"): [
                0.054393095128676475,
                0.054393095128676475,
            ],
            ("fused_uvm_caching", "table_row_wise"): [
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
            ("dense", "data_parallel"): [
                0.002677347614879459,
                0.002677347614879459,
            ],
            ("fused", "table_wise"): [0.001880471390093715],
            ("fused_uvm", "table_wise"): [0.25958192114736517],
            ("fused_uvm_caching", "table_wise"): [0.060433813055248066],
            ("fused", "column_wise"): [0.001880471390093715],
            ("fused_uvm", "column_wise"): [0.25958192114736517],
            ("fused_uvm_caching", "column_wise"): [0.060433813055248066],
            ("fused", "row_wise"): [
                0.0007915177871551004,
                0.0007915177871551004,
            ],
            ("fused_uvm", "row_wise"): [0.1036341050091912, 0.1036341050091912],
            ("fused_uvm_caching", "row_wise"): [
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

    def test_inference_1_table_perf(self) -> None:
        tables = [
            EmbeddingBagConfig(
                num_embeddings=100,
                embedding_dim=10,
                name="table_0",
                feature_names=["feature_0"],
            )
        ]
        model = TestSparseNN(tables=tables, weighted_tables=[])
        quant_model = _quantize(model, inplace=True)

        inference_estimator = EmbeddingPerfEstimator(
            topology=self.topology, is_inference=True
        )
        inference_enumerator = EmbeddingEnumerator(
            topology=self.topology, batch_size=BATCH_SIZE, estimator=inference_estimator
        )
        sharding_options = inference_enumerator.enumerate(
            module=quant_model,
            sharders=[
                cast(
                    ModuleSharder[torch.nn.Module], QuantEmbeddingBagCollectionSharder()
                )
            ],
        )

        expected_perfs = {
            ("quant", "table_wise"): [0.0001296231579222408],
            ("quant_uvm", "table_wise"): [0.018350937787224266],
            ("quant_uvm_caching", "table_wise"): [0.004269758427175579],
        }

        perfs = {
            (
                sharding_option.compute_kernel,
                sharding_option.sharding_type,
            ): [shard.perf for shard in sharding_option.shards]
            for sharding_option in sharding_options
        }

        self.assertEqual(perfs, expected_perfs)


# pyre-ignore[3]
def calculate_storage_specific_size_data_provider():
    return (
        {
            "sharding_type": ShardingType.TABLE_ROW_WISE,
            "optimizer_class": torch.optim.SGD,
            "expected_storage": [50, 50],
        },
        {
            "sharding_type": ShardingType.COLUMN_WISE,
            "optimizer_class": torch.optim.Adam,
            "expected_storage": [150, 150],
        },
        {
            "sharding_type": ShardingType.TABLE_ROW_WISE,
            "optimizer_class": None,
            "expected_storage": [50, 50],
        },
        {
            "sharding_type": ShardingType.DATA_PARALLEL,
            "optimizer_class": torchrec.optim.RowWiseAdagrad,
            "expected_storage": [134, 134],
        },
    )


class TestEmbeddingStorageEstimator(unittest.TestCase):
    def test_calculate_storage_specific_sizes(self) -> None:
        for inputs in calculate_storage_specific_size_data_provider():
            sharding_type, optimizer_class, expected_storage = inputs.values()
            estimates = _calculate_storage_specific_sizes(
                storage=100,
                shape=torch.Size((10, 5, 3)),
                shard_sizes=[[5, 5, 3], [5, 5, 3]],
                sharding_type=sharding_type.value,
                optimizer_class=optimizer_class,
            )

            self.assertEqual(estimates, expected_storage)
