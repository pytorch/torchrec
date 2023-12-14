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
    EmbeddingOffloadStats,
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
            ("quant", "table_wise"): [0.0001296231579222408],
            ("quant_uvm", "table_wise"): [0.018350937787224266],
            ("quant_uvm_caching", "table_wise"): [0.004269758427175579],
            ("quant", "row_wise"): [0.0001819317157451923, 0.0001819317157451923],
            ("quant_uvm", "row_wise"): [0.023103601792279417, 0.023103601792279417],
            ("quant_uvm_caching", "row_wise"): [
                0.005390052899352861,
                0.005390052899352861,
            ],
            ("quant", "column_wise"): [0.0001296231579222408],
            ("quant_uvm", "column_wise"): [0.018350937787224266],
            ("quant_uvm_caching", "column_wise"): [0.004269758427175579],
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


class TestEmbeddingOffloadStats(unittest.TestCase):
    def test_basic(self) -> None:
        stats = EmbeddingOffloadStats(
            cacheability=0.42,
            expected_lookups=31,
            mrc_hist_counts=torch.tensor([99, 98, 97]),
            height=92,
        )
        self.assertEqual(stats.cacheability, 0.42)
        self.assertEqual(stats.expected_lookups, 31)
        self.assertEqual(stats.expected_miss_rate(0), 1.0)
        self.assertEqual(stats.expected_miss_rate(1), 0.0)
        self.assertAlmostEqual(
            stats.expected_miss_rate(0.5), 1 - (99 + 98) / (99 + 98 + 97)
        )

    def test_estimate_cache_miss_rate(self) -> None:
        hist = torch.tensor([0, 6, 0, 8])
        bins = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
        miss_rates = EmbeddingOffloadStats.estimate_cache_miss_rate(
            torch.tensor([0, 1, 2, 3, 4]), hist, bins
        )
        m = 1 - (6 / (6 + 8))  # from hist counts above
        want = [
            1,  # size 0 - 100% miss
            1,  # size 1 - 100%, no immediate repetitions
            m,  # size 2 - m (~57%) miss, 6 occurrences
            m,  # size 3 - same as size 2, no 3 stack distances,
            #              so increasing cache by 1 doesn't help
            0,  # size 4 - 0% miss rate, everything fits
        ]
        torch.testing.assert_close(miss_rates, torch.tensor(want))

        # test with bigger bins to better validate boundary conditions
        # create simple linear miss rate curve
        trace = torch.arange(100.0)
        hist = torch.histc(trace, bins=10, min=0, max=100)
        bins = torch.linspace(0, 100, len(hist) + 1)
        cache_heights = [0, 9, 10, 11, 89, 99, 100]
        miss_rates = EmbeddingOffloadStats.estimate_cache_miss_rate(
            torch.tensor(cache_heights), hist, bins
        )
        want = [
            1,  # 0 ->  no cache, 100% miss
            0.9,  # 9 ->  bin 0, which is all cache sizes <= 10, has 90 misses of 100, so 90% miss
            0.9,  # 10 -> bin 0, same as above
            0.8,  # 11 -> bin 1, cache sizes (10, 20], 80 misses out of 100, so 80% miss
            0.1,  # 89 -> bin 8, cache sizes (80, 90], 10 misses out of 100, so 10% miss
            0,  # 99 -> bin 9, cache sizes (90, 100], final last bin gets scaled to 1, so 0% misses
            0,  # 100 -> off the end of the histogram, 0% misses
        ]
        torch.testing.assert_close(miss_rates, torch.tensor(want))
        # test using 0-d tensors works as well
        miss_rates = torch.tensor(
            [
                EmbeddingOffloadStats.estimate_cache_miss_rate(
                    torch.tensor(x), hist, bins
                )
                for x in cache_heights
            ]
        )
        torch.testing.assert_close(miss_rates, torch.tensor(want))

        # test features no with no data return non-nan
        hist = torch.tensor([0, 0])
        bins = torch.tensor([0, 1, 2])
        miss_rates = EmbeddingOffloadStats.estimate_cache_miss_rate(
            torch.tensor([0, 1, 2]), hist, bins
        )
        torch.testing.assert_close(miss_rates, torch.tensor([0.0, 0.0, 0.0]))
        # test 0-d case
        miss_rates = torch.tensor(
            [
                EmbeddingOffloadStats.estimate_cache_miss_rate(
                    torch.tensor(x), hist, bins
                )
                for x in [0, 1, 2]
            ]
        )
        torch.testing.assert_close(miss_rates, torch.tensor([0.0, 0.0, 0.0]))
