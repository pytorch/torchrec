#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import cast

import torch
import torchrec.optim as trec_optim

from torchrec.distributed.embedding import EmbeddingCollectionSharder
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.fbgemm_qcomm_codec import (
    CommType,
    get_qcomm_codecs_registry,
    QCommsConfig,
)
from torchrec.distributed.planner.constants import BATCH_SIZE
from torchrec.distributed.planner.enumerators import EmbeddingEnumerator
from torchrec.distributed.planner.shard_estimators import (
    _calculate_storage_specific_sizes,
    EmbeddingPerfEstimator,
)
from torchrec.distributed.planner.types import Perf, Topology
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
                Perf(
                    fwd_compute=9.356002212235228e-05,
                    fwd_comms=0,
                    bwd_compute=0.00018712004424470456,
                    bwd_comms=0.000225593945387348,
                ),
                Perf(
                    fwd_compute=9.356002212235228e-05,
                    fwd_comms=0,
                    bwd_compute=0.00018712004424470456,
                    bwd_comms=0.000225593945387348,
                ),
            ],
            ("fused", "table_wise"): [
                Perf(
                    fwd_compute=0.000327460077428233,
                    fwd_comms=6.357828776041667e-05,
                    bwd_compute=0.000654920154856466,
                    bwd_comms=6.357828776041667e-05,
                )
            ],
            ("fused_uvm", "table_wise"): [
                Perf(
                    fwd_compute=0.05759444891237746,
                    fwd_comms=6.357828776041667e-05,
                    bwd_compute=0.11518889782475492,
                    bwd_comms=6.357828776041667e-05,
                )
            ],
            ("fused_uvm_caching", "table_wise"): [
                Perf(
                    fwd_compute=0.013339313780795867,
                    fwd_comms=6.357828776041667e-05,
                    bwd_compute=0.026678627561591735,
                    bwd_comms=6.357828776041667e-05,
                )
            ],
            ("fused", "column_wise"): [
                Perf(
                    fwd_compute=0.000327460077428233,
                    fwd_comms=6.357828776041667e-05,
                    bwd_compute=0.000654920154856466,
                    bwd_comms=6.357828776041667e-05,
                )
            ],
            ("fused_uvm", "column_wise"): [
                Perf(
                    fwd_compute=0.05759444891237746,
                    fwd_comms=6.357828776041667e-05,
                    bwd_compute=0.11518889782475492,
                    bwd_comms=6.357828776041667e-05,
                )
            ],
            ("fused_uvm_caching", "column_wise"): [
                Perf(
                    fwd_compute=0.013339313780795867,
                    fwd_comms=6.357828776041667e-05,
                    bwd_compute=0.026678627561591735,
                    bwd_comms=6.357828776041667e-05,
                )
            ],
            ("fused", "table_column_wise"): [
                Perf(
                    fwd_compute=0.000327460077428233,
                    fwd_comms=6.357828776041667e-05,
                    bwd_compute=0.000654920154856466,
                    bwd_comms=6.357828776041667e-05,
                )
            ],
            ("fused_uvm", "table_column_wise"): [
                Perf(
                    fwd_compute=0.05759444891237746,
                    fwd_comms=6.357828776041667e-05,
                    bwd_compute=0.11518889782475492,
                    bwd_comms=6.357828776041667e-05,
                )
            ],
            ("fused_uvm_caching", "table_column_wise"): [
                Perf(
                    fwd_compute=0.013339313780795867,
                    fwd_comms=6.357828776041667e-05,
                    bwd_compute=0.026678627561591735,
                    bwd_comms=6.357828776041667e-05,
                )
            ],
            ("fused", "row_wise"): [
                Perf(
                    fwd_compute=6.804365245261984e-05,
                    fwd_comms=6.357828776041667e-05,
                    bwd_compute=0.0001360873049052397,
                    bwd_comms=0.00016798276699240525,
                ),
                Perf(
                    fwd_compute=6.804365245261984e-05,
                    fwd_comms=6.357828776041667e-05,
                    bwd_compute=0.0001360873049052397,
                    bwd_comms=0.00016798276699240525,
                ),
            ],
            ("fused_uvm", "row_wise"): [
                Perf(
                    fwd_compute=0.011967677696078432,
                    fwd_comms=6.357828776041667e-05,
                    bwd_compute=0.023935355392156864,
                    bwd_comms=0.018426483752680762,
                ),
                Perf(
                    fwd_compute=0.011967677696078432,
                    fwd_comms=6.357828776041667e-05,
                    bwd_compute=0.023935355392156864,
                    bwd_comms=0.018426483752680762,
                ),
            ],
            ("fused_uvm_caching", "row_wise"): [
                Perf(
                    fwd_compute=0.0027718054609445954,
                    fwd_comms=6.357828776041667e-05,
                    bwd_compute=0.005543610921889191,
                    bwd_comms=0.004316567291897281,
                ),
                Perf(
                    fwd_compute=0.0027718054609445954,
                    fwd_comms=6.357828776041667e-05,
                    bwd_compute=0.005543610921889191,
                    bwd_comms=0.004316567291897281,
                ),
            ],
            ("fused", "table_row_wise"): [
                Perf(
                    fwd_compute=6.804365245261984e-05,
                    fwd_comms=6.357828776041667e-05,
                    bwd_compute=0.0001360873049052397,
                    bwd_comms=0.00016798276699240525,
                ),
                Perf(
                    fwd_compute=6.804365245261984e-05,
                    fwd_comms=6.357828776041667e-05,
                    bwd_compute=0.0001360873049052397,
                    bwd_comms=0.00016798276699240525,
                ),
            ],
            ("fused_uvm", "table_row_wise"): [
                Perf(
                    fwd_compute=0.011967677696078432,
                    fwd_comms=6.357828776041667e-05,
                    bwd_compute=0.023935355392156864,
                    bwd_comms=0.018426483752680762,
                ),
                Perf(
                    fwd_compute=0.011967677696078432,
                    fwd_comms=6.357828776041667e-05,
                    bwd_compute=0.023935355392156864,
                    bwd_comms=0.018426483752680762,
                ),
            ],
            ("fused_uvm_caching", "table_row_wise"): [
                Perf(
                    fwd_compute=0.0027718054609445954,
                    fwd_comms=6.357828776041667e-05,
                    bwd_compute=0.005543610921889191,
                    bwd_comms=0.004316567291897281,
                ),
                Perf(
                    fwd_compute=0.0027718054609445954,
                    fwd_comms=6.357828776041667e-05,
                    bwd_compute=0.005543610921889191,
                    bwd_comms=0.004316567291897281,
                ),
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

    def test_1_table_perf_with_fp8_comm(self) -> None:
        tables = [
            EmbeddingBagConfig(
                num_embeddings=100,
                embedding_dim=10,
                name="table_0",
                feature_names=["feature_0"],
            )
        ]
        model = TestSparseNN(tables=tables, weighted_tables=[])

        # will get warning for POOLED_EMBEDDINGS_REDUCE_SCATTER not supporting fp8
        qcomm_codecs_registry = get_qcomm_codecs_registry(
            qcomms_config=QCommsConfig(
                forward_precision=CommType.FP8, backward_precision=CommType.FP8
            )
        )

        sharding_options = self.enumerator.enumerate(
            module=model,
            sharders=[
                cast(
                    ModuleSharder[torch.nn.Module],
                    EmbeddingBagCollectionSharder(
                        qcomm_codecs_registry=qcomm_codecs_registry
                    ),
                )
            ],
        )

        expected_total_perfs = {
            ("dense", "data_parallel"): [0.0005062740117544049, 0.0005062740117544049],
            ("fused", "table_wise"): [0.000846718200207288],
            ("fused_uvm", "table_wise"): [0.14336342905081956],
            ("fused_uvm_caching", "table_wise"): [0.03322849048472446],
            ("fused", "column_wise"): [0.000846718200207288],
            ("fused_uvm", "column_wise"): [0.14336342905081956],
            ("fused_uvm_caching", "column_wise"): [0.03322849048472446],
            ("fused", "table_column_wise"): [0.000846718200207288],
            ("fused_uvm", "table_column_wise"): [0.14336342905081956],
            ("fused_uvm_caching", "table_column_wise"): [0.03322849048472446],
            ("fused", "row_wise"): [0.0002561205605599394, 0.0002561205605599394],
            ("fused_uvm", "row_wise"): [0.03392836626838235, 0.03392836626838235],
            ("fused_uvm_caching", "row_wise"): [
                0.007906921553027076,
                0.007906921553027076,
            ],
            ("fused", "table_row_wise"): [0.0002561205605599394, 0.0002561205605599394],
            ("fused_uvm", "table_row_wise"): [0.03392836626838235, 0.03392836626838235],
            ("fused_uvm_caching", "table_row_wise"): [
                0.007906921553027076,
                0.007906921553027076,
            ],
        }

        total_perfs = {
            (
                sharding_option.compute_kernel,
                sharding_option.sharding_type,
            ): [cast(Perf, shard.perf).total for shard in sharding_option.shards]
            for sharding_option in sharding_options
        }

        self.assertEqual(expected_total_perfs, total_perfs)

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

        expected_total_perfs = {
            ("dense", "data_parallel"): [0.0026901057997143255, 0.0026901057997143255],
            ("fused", "table_wise"): [0.001880471390093715],
            ("fused_uvm", "table_wise"): [0.25958192114736517],
            ("fused_uvm_caching", "table_wise"): [0.06043381305524807],
            ("fused", "column_wise"): [0.001880471390093715],
            ("fused_uvm", "column_wise"): [0.25958192114736517],
            ("fused_uvm_caching", "column_wise"): [0.06043381305524807],
            ("fused", "row_wise"): [0.0007915177871551004, 0.0007915177871551004],
            ("fused_uvm", "row_wise"): [0.10363410500919118, 0.10363410500919118],
            ("fused_uvm_caching", "row_wise"): [
                0.024158779217047004,
                0.024158779217047004,
            ],
        }

        total_perfs = {
            (
                sharding_option.compute_kernel,
                sharding_option.sharding_type,
            ): [cast(Perf, shard.perf).total for shard in sharding_option.shards]
            for sharding_option in sharding_options
        }

        self.assertEqual(expected_total_perfs, total_perfs)

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

        expected_total_perfs = {
            ("quant", "table_wise"): [0.00012272310629071731],
            ("quant_uvm", "table_wise"): [0.017693276498831956],
            ("quant_uvm_caching", "table_wise"): [0.00411499640164215],
        }

        total_perfs = {
            (
                sharding_option.compute_kernel,
                sharding_option.sharding_type,
            ): [cast(Perf, shard.perf).total for shard in sharding_option.shards]
            for sharding_option in sharding_options
        }

        self.assertEqual(total_perfs, expected_total_perfs)


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
            "optimizer_class": trec_optim.RowWiseAdagrad,
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
