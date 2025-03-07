#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math
import unittest
from typing import cast, Dict, List, Tuple

from unittest.mock import MagicMock, Mock, patch

import torch
import torchrec.optim as trec_optim

from torchrec.distributed.embedding import EmbeddingCollectionSharder
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
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
    EmbeddingStorageEstimator,
)
from torchrec.distributed.planner.types import ParameterConstraints, Perf, Topology
from torchrec.distributed.quant_embeddingbag import QuantEmbeddingBagCollectionSharder
from torchrec.distributed.test_utils.infer_utils import quantize
from torchrec.distributed.test_utils.test_model import TestEBCSharder, TestSparseNN
from torchrec.distributed.tests.test_sequence_model import TestSequenceSparseNN
from torchrec.distributed.types import (
    CacheParams,
    CacheStatistics,
    ModuleSharder,
    MultiPassPrefetchConfig,
    PipelineType,
    ShardingType,
)
from torchrec.modules.embedding_configs import (
    DataType,
    EmbeddingBagConfig,
    EmbeddingConfig,
)


class TestEmbeddingPerfEstimator(unittest.TestCase):
    def setUp(self) -> None:
        self.topology = Topology(world_size=2, compute_device="cuda")
        self.estimator = EmbeddingPerfEstimator(topology=self.topology)
        self.enumerator = EmbeddingEnumerator(
            topology=self.topology, batch_size=BATCH_SIZE, estimator=self.estimator
        )
        self._sharding_types = [x.value for x in ShardingType]

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
        """
        GRID_SHARD only is available if specified by user in parameter constraints, however,
        adding parameter constraints does not work because of the non deterministic nature of
        _filter_sharding_types (set & set) operation when constraints are present, we mock the
        call to _filter_sharding_types to ensure the order of the sharding types list is always
        the same.
        """
        self.enumerator._filter_sharding_types = MagicMock(
            return_value=self._sharding_types
        )
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
                    bwd_comms=0.00012314846217964537,
                ),
                Perf(
                    fwd_compute=9.356002212235228e-05,
                    fwd_comms=0,
                    bwd_compute=0.00018712004424470456,
                    bwd_comms=0.00012314846217964537,
                ),
            ],
            ("fused", "table_wise"): [
                Perf(
                    fwd_compute=0.000327460077428233,
                    fwd_comms=0.00012715657552083334,
                    bwd_compute=0.000654920154856466,
                    bwd_comms=0.00012715657552083334,
                )
            ],
            ("fused_uvm", "table_wise"): [
                Perf(
                    fwd_compute=0.09179115295410156,
                    fwd_comms=0.00012715657552083334,
                    bwd_compute=0.18358230590820312,
                    bwd_comms=0.00012715657552083334,
                )
            ],
            ("fused_uvm_caching", "table_wise"): [
                Perf(
                    fwd_compute=0.01432837509527439,
                    fwd_comms=0.00012715657552083334,
                    bwd_compute=0.02865675019054878,
                    bwd_comms=0.00012715657552083334,
                )
            ],
            ("fused", "column_wise"): [
                Perf(
                    fwd_compute=0.000327460077428233,
                    fwd_comms=0.00012715657552083334,
                    bwd_compute=0.000654920154856466,
                    bwd_comms=0.00012715657552083334,
                )
            ],
            ("fused_uvm", "column_wise"): [
                Perf(
                    fwd_compute=0.09179115295410156,
                    fwd_comms=0.00012715657552083334,
                    bwd_compute=0.18358230590820312,
                    bwd_comms=0.00012715657552083334,
                )
            ],
            ("fused_uvm_caching", "column_wise"): [
                Perf(
                    fwd_compute=0.01432837509527439,
                    fwd_comms=0.00012715657552083334,
                    bwd_compute=0.02865675019054878,
                    bwd_comms=0.00012715657552083334,
                )
            ],
            ("fused", "table_column_wise"): [
                Perf(
                    fwd_compute=0.000327460077428233,
                    fwd_comms=0.00012715657552083334,
                    bwd_compute=0.000654920154856466,
                    bwd_comms=0.00012715657552083334,
                )
            ],
            ("fused_uvm", "table_column_wise"): [
                Perf(
                    fwd_compute=0.09179115295410156,
                    fwd_comms=0.00012715657552083334,
                    bwd_compute=0.18358230590820312,
                    bwd_comms=0.00012715657552083334,
                )
            ],
            ("fused_uvm_caching", "table_column_wise"): [
                Perf(
                    fwd_compute=0.01432837509527439,
                    fwd_comms=0.00012715657552083334,
                    bwd_compute=0.02865675019054878,
                    bwd_comms=0.00012715657552083334,
                )
            ],
            ("fused", "row_wise"): [
                Perf(
                    fwd_compute=6.804365245261984e-05,
                    fwd_comms=0.00012715657552083334,
                    bwd_compute=0.0001360873049052397,
                    bwd_comms=0.00023156105475282193,
                ),
                Perf(
                    fwd_compute=6.804365245261984e-05,
                    fwd_comms=0.00012715657552083334,
                    bwd_compute=0.0001360873049052397,
                    bwd_comms=0.00023156105475282193,
                ),
            ],
            ("fused_uvm", "row_wise"): [
                Perf(
                    fwd_compute=0.019073486328125,
                    fwd_comms=0.00012715657552083334,
                    bwd_compute=0.03814697265625,
                    bwd_comms=0.02939303716023763,
                ),
                Perf(
                    fwd_compute=0.019073486328125,
                    fwd_comms=0.00012715657552083334,
                    bwd_compute=0.03814697265625,
                    bwd_comms=0.02939303716023763,
                ),
            ],
            ("fused_uvm_caching", "row_wise"): [
                Perf(
                    fwd_compute=0.0029773246951219513,
                    fwd_comms=0.00012715657552083334,
                    bwd_compute=0.0059546493902439025,
                    bwd_comms=0.004695489154598577,
                ),
                Perf(
                    fwd_compute=0.0029773246951219513,
                    fwd_comms=0.00012715657552083334,
                    bwd_compute=0.0059546493902439025,
                    bwd_comms=0.004695489154598577,
                ),
            ],
            ("fused", "table_row_wise"): [
                Perf(
                    fwd_compute=6.804365245261984e-05,
                    fwd_comms=0.00012715657552083334,
                    bwd_compute=0.0001360873049052397,
                    bwd_comms=0.00023156105475282193,
                ),
                Perf(
                    fwd_compute=6.804365245261984e-05,
                    fwd_comms=0.00012715657552083334,
                    bwd_compute=0.0001360873049052397,
                    bwd_comms=0.00023156105475282193,
                ),
            ],
            ("fused_uvm", "table_row_wise"): [
                Perf(
                    fwd_compute=0.019073486328125,
                    fwd_comms=0.00012715657552083334,
                    bwd_compute=0.03814697265625,
                    bwd_comms=0.02939303716023763,
                ),
                Perf(
                    fwd_compute=0.019073486328125,
                    fwd_comms=0.00012715657552083334,
                    bwd_compute=0.03814697265625,
                    bwd_comms=0.02939303716023763,
                ),
            ],
            ("fused_uvm_caching", "table_row_wise"): [
                Perf(
                    fwd_compute=0.0029773246951219513,
                    fwd_comms=0.00012715657552083334,
                    bwd_compute=0.0059546493902439025,
                    bwd_comms=0.004695489154598577,
                ),
                Perf(
                    fwd_compute=0.0029773246951219513,
                    fwd_comms=0.00012715657552083334,
                    bwd_compute=0.0059546493902439025,
                    bwd_comms=0.004695489154598577,
                ),
            ],
            # grid_shard is the same as table_row_wise
            ("fused", "grid_shard"): [
                Perf(
                    fwd_compute=6.804365245261984e-05,
                    fwd_comms=0.00012715657552083334,
                    bwd_compute=0.0001360873049052397,
                    bwd_comms=0.00023156105475282193,
                ),
                Perf(
                    fwd_compute=6.804365245261984e-05,
                    fwd_comms=0.00012715657552083334,
                    bwd_compute=0.0001360873049052397,
                    bwd_comms=0.00023156105475282193,
                ),
            ],
            ("fused_uvm", "grid_shard"): [
                Perf(
                    fwd_compute=0.019073486328125,
                    fwd_comms=0.00012715657552083334,
                    bwd_compute=0.03814697265625,
                    bwd_comms=0.02939303716023763,
                ),
                Perf(
                    fwd_compute=0.019073486328125,
                    fwd_comms=0.00012715657552083334,
                    bwd_compute=0.03814697265625,
                    bwd_comms=0.02939303716023763,
                ),
            ],
            ("fused_uvm_caching", "grid_shard"): [
                Perf(
                    fwd_compute=0.0029773246951219513,
                    fwd_comms=0.00012715657552083334,
                    bwd_compute=0.0059546493902439025,
                    bwd_comms=0.004695489154598577,
                ),
                Perf(
                    fwd_compute=0.0029773246951219513,
                    fwd_comms=0.00012715657552083334,
                    bwd_compute=0.0059546493902439025,
                    bwd_comms=0.004695489154598577,
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

        """
        GRID_SHARD only is available if specified by user in parameter constraints, however,
        adding parameter constraints does not work because of the non deterministic nature of
        _filter_sharding_types (set & set) operation when constraints are present, we mock the
        call to _filter_sharding_types to ensure the order of the sharding types list is always
        the same.
        """
        self.enumerator._filter_sharding_types = MagicMock(
            return_value=self._sharding_types
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
            ("dense", "data_parallel"): [
                0.00040382852854670224,
                0.00040382852854670224,
            ],
            ("fused", "table_wise"): [0.0008785073440874965],
            ("fused_uvm", "table_wise"): [0.228498379389445],
            ("fused_uvm_caching", "table_wise"): [0.03572169358168191],
            ("fused", "column_wise"): [0.0008785073440874965],
            ("fused_uvm", "column_wise"): [0.228498379389445],
            ("fused_uvm_caching", "column_wise"): [0.03572169358168191],
            ("fused", "table_column_wise"): [0.0008785073440874965],
            ("fused_uvm", "table_column_wise"): [0.228498379389445],
            ("fused_uvm_caching", "table_column_wise"): [0.03572169358168191],
            ("fused", "row_wise"): [0.0003196988483203561, 0.0003196988483203561],
            ("fused_uvm", "row_wise"): [0.05409916241963704, 0.05409916241963704],
            ("fused_uvm_caching", "row_wise"): [
                0.008552055048748728,
                0.008552055048748728,
            ],
            ("fused", "table_row_wise"): [0.0003196988483203561, 0.0003196988483203561],
            ("fused_uvm", "table_row_wise"): [0.05409916241963704, 0.05409916241963704],
            ("fused_uvm_caching", "table_row_wise"): [
                0.008552055048748728,
                0.008552055048748728,
            ],
            ("fused", "grid_shard"): [0.0003196988483203561, 0.0003196988483203561],
            ("fused_uvm", "grid_shard"): [0.05409916241963704, 0.05409916241963704],
            ("fused_uvm_caching", "grid_shard"): [
                0.008552055048748728,
                0.008552055048748728,
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
            ("dense", "data_parallel"): [0.0018508724012768255, 0.0018508724012768255],
            ("fused", "table_wise"): [0.002287372431760382],
            ("fused_uvm", "table_wise"): [0.4138739903767904],
            ("fused_uvm_caching", "table_wise"): [0.0652914900120681],
            ("fused", "column_wise"): [0.002287372431760382],
            ("fused_uvm", "column_wise"): [0.4138739903767904],
            ("fused_uvm_caching", "column_wise"): [0.0652914900120681],
            ("fused", "row_wise"): [0.0009949683079884337, 0.0009949683079884337],
            ("fused_uvm", "row_wise"): [0.16524950663248697, 0.16524950663248697],
            ("fused_uvm_caching", "row_wise"): [
                0.026138429719258126,
                0.026138429719258126,
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
        quant_model = quantize(model, inplace=True)

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
            ("quant", "table_wise"): [0.00015505447302640747],
            ("quant_uvm", "table_wise"): [0.02925713857014974],
            ("quant_uvm_caching", "table_wise"): [0.00460989106961382],
            ("quant", "row_wise"): [8.063172815635452e-05, 8.063172815635452e-05],
            ("quant_uvm", "row_wise"): [0.008395512898763021, 0.008395512898763021],
            ("quant_uvm_caching", "row_wise"): [
                0.0013534421843241868,
                0.0013534421843241868,
            ],
            ("quant", "column_wise"): [0.00015505447302640747],
            ("quant_uvm", "column_wise"): [0.02925713857014974],
            ("quant_uvm_caching", "column_wise"): [0.00460989106961382],
        }

        total_perfs = {
            (
                sharding_option.compute_kernel,
                sharding_option.sharding_type,
            ): [cast(Perf, shard.perf).total for shard in sharding_option.shards]
            for sharding_option in sharding_options
        }

        self.assertEqual(total_perfs, expected_total_perfs)

    def test_prefetch_compute(self) -> None:
        class MyCacheStatistics(CacheStatistics):
            def __init__(self, expected_lookups: int, cacheability: float) -> None:
                self._expected_lookups = expected_lookups
                self._cacheability = cacheability

            @property
            def expected_lookups(self) -> int:
                return self._expected_lookups

            def expected_miss_rate(self, clf: float) -> float:
                return clf

            @property
            def cacheability(self) -> float:
                return self._cacheability

        tables = [
            EmbeddingBagConfig(
                num_embeddings=100,
                embedding_dim=10,
                name="table_0",
                feature_names=["feature_0"],
            ),
            EmbeddingBagConfig(
                num_embeddings=100,
                embedding_dim=10,
                name="table_1",
                feature_names=["feature_1"],
            ),
        ]
        constraints = {
            "table_0": ParameterConstraints(
                compute_kernels=[EmbeddingComputeKernel.FUSED_UVM_CACHING.value],
                cache_params=CacheParams(
                    load_factor=0.1,
                    stats=MyCacheStatistics(expected_lookups=200_000, cacheability=0.2),
                ),
            ),
            # simulate promoting a uvm caching table to HBM during scaleup.
            "table_1": ParameterConstraints(
                compute_kernels=[EmbeddingComputeKernel.FUSED.value],
                cache_params=CacheParams(
                    load_factor=None,
                    stats=MyCacheStatistics(expected_lookups=200_000, cacheability=0.2),
                ),
            ),
        }
        enumerator = EmbeddingEnumerator(
            topology=self.topology,
            batch_size=BATCH_SIZE,
            estimator=self.estimator,
            constraints=constraints,
        )
        """
        GRID_SHARD only is available if specified by user in parameter constraints, however,
        adding parameter constraints does not work because of the non deterministic nature of
        _filter_sharding_types (set & set) operation when constraints are present, we mock the
        call to _filter_sharding_types to ensure the order of the sharding types list is always
        the same.
        """
        enumerator._filter_sharding_types = MagicMock(return_value=self._sharding_types)
        model = TestSparseNN(tables=tables, weighted_tables=[])
        sharding_options = enumerator.enumerate(
            module=model,
            sharders=[
                cast(
                    ModuleSharder[torch.nn.Module],
                    EmbeddingBagCollectionSharder(
                        fused_params={"cache_load_factor": 0.2}
                    ),
                )
            ],
        )

        expected_prefetch_computes = {
            ("table_0", "fused_uvm_caching", "column_wise"): [0.023283064365386963],
            ("table_0", "fused_uvm_caching", "row_wise"): [
                0.011641532182693481,
                0.011641532182693481,
            ],
            ("table_0", "fused_uvm_caching", "table_column_wise"): [
                0.023283064365386963
            ],
            ("table_0", "fused_uvm_caching", "table_row_wise"): [
                0.011641532182693481,
                0.011641532182693481,
            ],
            ("table_0", "fused_uvm_caching", "grid_shard"): [
                0.011641532182693481,
                0.011641532182693481,
            ],
            ("table_0", "fused_uvm_caching", "table_wise"): [0.023283064365386963],
            ("table_1", "fused", "column_wise"): [0.0],
            ("table_1", "fused", "row_wise"): [0.0, 0.0],
            ("table_1", "fused", "table_column_wise"): [0.0],
            ("table_1", "fused", "table_row_wise"): [0.0, 0.0],
            ("table_1", "fused", "table_wise"): [0.0],
            ("table_1", "fused", "grid_shard"): [0.0, 0.0],
        }

        prefetch_computes = {
            (
                sharding_option.name,
                sharding_option.compute_kernel,
                sharding_option.sharding_type,
            ): [
                shard.perf.prefetch_compute if shard.perf else -1
                for shard in sharding_option.shards
            ]
            for sharding_option in sharding_options
        }
        self.assertEqual(expected_prefetch_computes, prefetch_computes)

    def test_weighted_feature_bwd_compute_multiplier(self) -> None:
        def _get_bwd_computes(
            model: torch.nn.Module,
            weighted_feature_bwd_compute_multiplier: float,
        ) -> Dict[Tuple[str, str, str], List[float]]:
            topology = Topology(
                world_size=2,
                compute_device="cuda",
                weighted_feature_bwd_compute_multiplier=weighted_feature_bwd_compute_multiplier,
            )
            estimator = EmbeddingPerfEstimator(topology=topology)
            enumerator = EmbeddingEnumerator(
                topology=topology, batch_size=BATCH_SIZE, estimator=estimator
            )
            sharding_options = enumerator.enumerate(
                module=model,
                sharders=[
                    cast(
                        ModuleSharder[torch.nn.Module], EmbeddingBagCollectionSharder()
                    )
                ],
            )
            bwd_computes = {
                (
                    sharding_option.name,
                    sharding_option.compute_kernel,
                    sharding_option.sharding_type,
                ): [
                    shard.perf.bwd_compute if shard.perf else -1
                    for shard in sharding_option.shards
                ]
                for sharding_option in sharding_options
            }
            return bwd_computes

        tables = [
            EmbeddingBagConfig(
                num_embeddings=100,
                embedding_dim=10,
                name="table_0",
                feature_names=["feature_0"],
            )
        ]
        weighted_tables = [
            EmbeddingBagConfig(
                num_embeddings=100,
                embedding_dim=10,
                name="weighted_table_0",
                feature_names=["weighted_feature_0"],
            )
        ]
        model = TestSparseNN(tables=tables, weighted_tables=weighted_tables)

        MULTIPLIER = 7
        bwd_computes_1 = _get_bwd_computes(
            model, weighted_feature_bwd_compute_multiplier=1
        )
        bwd_computes_2 = _get_bwd_computes(
            model,
            weighted_feature_bwd_compute_multiplier=2,
        )
        bwd_computes_n = _get_bwd_computes(
            model,
            weighted_feature_bwd_compute_multiplier=MULTIPLIER,
        )
        self.assertEqual(bwd_computes_1.keys(), bwd_computes_2.keys())
        self.assertEqual(bwd_computes_1.keys(), bwd_computes_n.keys())
        for key in bwd_computes_1.keys():
            table_name, _, sharding_type = key
            if table_name.startswith("weighted"):
                self.assertEqual(len(bwd_computes_1), len(bwd_computes_2))
                self.assertEqual(len(bwd_computes_1), len(bwd_computes_n))
                for bwd_compute_1, bwd_compute_2, bwd_compute_n in zip(
                    bwd_computes_1[key], bwd_computes_2[key], bwd_computes_n[key]
                ):
                    # bwd_compute_1 = base_bwd_compute + offset
                    # bwd_compute_2 = base_bwd_compute * 2 + offset
                    # bwd_compute_n = base_bwd_compute * MULTIPLIER + offset
                    # (where offset = bwd_grad_indice_weights_kernel in production
                    # https://fburl.com/code/u9hq6vhf)
                    base_bwd_compute = bwd_compute_2 - bwd_compute_1
                    offset = bwd_compute_1 - base_bwd_compute
                    self.assertAlmostEqual(
                        base_bwd_compute * MULTIPLIER,
                        bwd_compute_n - offset,
                    )
            else:
                self.assertEqual(bwd_computes_1[key], bwd_computes_2[key])


# pyre-ignore[3]
def calculate_storage_specific_size_data_provider():
    return (
        {
            "sharding_type": ShardingType.TABLE_ROW_WISE,
            "optimizer_class": torch.optim.SGD,
            "expected_storage": [50, 50],
            "clf": None,
        },
        {
            "sharding_type": ShardingType.COLUMN_WISE,
            "optimizer_class": torch.optim.Adam,
            "expected_storage": [
                150 + math.ceil(5 * (4 + 0.5 * 16)),
                150 + math.ceil(5 * (4 + 0.5 * 16)),
            ],
            "clf": 0.5,
        },
        {
            "sharding_type": ShardingType.TABLE_ROW_WISE,
            "optimizer_class": None,
            "expected_storage": [
                50 + math.ceil(5 * (4 + 0.0 * 16)),
                50 + math.ceil(5 * (4 + 0.0 * 16)),
            ],
            "clf": 0.0,
        },
        {
            "sharding_type": ShardingType.DATA_PARALLEL,
            "optimizer_class": trec_optim.RowWiseAdagrad,
            "expected_storage": [
                134 + math.ceil(5 * (4 + 1.0 * 16)),
                134 + math.ceil(5 * (4 + 1.0 * 16)),
            ],
            "clf": 1.0,
        },
    )


class TestEmbeddingStorageEstimator(unittest.TestCase):
    def test_calculate_storage_specific_sizes(self) -> None:
        for inputs in calculate_storage_specific_size_data_provider():
            sharding_type, optimizer_class, expected_storage, clf = inputs.values()
            estimates = _calculate_storage_specific_sizes(
                storage=100,
                shape=torch.Size((10, 5, 3)),
                shard_sizes=[[5, 5, 3], [5, 5, 3]],
                sharding_type=sharding_type.value,
                optimizer_class=optimizer_class,
                clf=clf,
            )

            self.assertEqual(estimates, expected_storage)

    @patch(
        "torchrec.distributed.planner.shard_estimators._calculate_shard_io_sizes",
        return_value=([1024], [3333]),
    )
    @patch(
        "torchrec.distributed.planner.shard_estimators._calculate_storage_specific_sizes",
        return_value=[100],
    )
    def test_pipelined_storage(self, p1: Mock, p2: Mock) -> None:
        for pipeline_type in list(PipelineType):
            for run_embedding_at_peak_memory in [False, True]:
                topology = Topology(world_size=2, compute_device="cuda")
                estimator = EmbeddingStorageEstimator(
                    topology=topology,
                    pipeline_type=pipeline_type,
                    run_embedding_at_peak_memory=run_embedding_at_peak_memory,
                )
                tables = [
                    EmbeddingBagConfig(
                        num_embeddings=100,
                        embedding_dim=10,
                        name="table_0",
                        feature_names=["feature_0"],
                    ),
                    EmbeddingBagConfig(
                        num_embeddings=100,
                        embedding_dim=10,
                        name="table_1",
                        feature_names=["feature_1"],
                    ),
                    EmbeddingBagConfig(
                        num_embeddings=100,
                        embedding_dim=10,
                        name="table_2",
                        feature_names=["feature_2"],
                    ),
                ]
                constraints = {
                    "table_0": ParameterConstraints(
                        compute_kernels=[
                            EmbeddingComputeKernel.FUSED_UVM_CACHING.value
                        ],
                        sharding_types=[ShardingType.TABLE_WISE.value],
                        cache_params=CacheParams(
                            load_factor=0.1,
                        ),
                    ),
                    # simulate promoting a uvm caching table to HBM during scaleup.
                    "table_1": ParameterConstraints(
                        compute_kernels=[EmbeddingComputeKernel.FUSED.value],
                        sharding_types=[ShardingType.TABLE_WISE.value],
                        cache_params=CacheParams(
                            load_factor=None,
                        ),
                    ),
                    "table_2": ParameterConstraints(
                        compute_kernels=[
                            EmbeddingComputeKernel.FUSED_UVM_CACHING.value
                        ],
                        sharding_types=[ShardingType.TABLE_WISE.value],
                        cache_params=CacheParams(
                            load_factor=0.1,
                            multipass_prefetch_config=MultiPassPrefetchConfig(
                                num_passes=10,
                            ),
                        ),
                    ),
                }
                enumerator = EmbeddingEnumerator(
                    topology=topology,
                    batch_size=BATCH_SIZE,
                    estimator=estimator,
                    constraints=constraints,
                )

                model = TestSparseNN(tables=tables, weighted_tables=[])
                sharding_options = enumerator.enumerate(
                    module=model,
                    sharders=[
                        cast(
                            ModuleSharder[torch.nn.Module],
                            EmbeddingBagCollectionSharder(
                                fused_params={
                                    "cache_load_factor": 0.2,
                                }
                            ),
                        )
                    ],
                )

                output_on_pipeline = 3333 if run_embedding_at_peak_memory else 0
                if pipeline_type == PipelineType.TRAIN_SPARSE_DIST:
                    expected_storage = {
                        ("table_0", "fused_uvm_caching", "table_wise"): [
                            (100 + 2048 + output_on_pipeline, 100)
                        ],
                        ("table_1", "fused", "table_wise"): [
                            (100 + 2048 + output_on_pipeline, 100)
                        ],
                        ("table_2", "fused_uvm_caching", "table_wise"): [
                            (100 + 2048 + output_on_pipeline, 100)
                        ],
                    }
                elif pipeline_type == PipelineType.TRAIN_PREFETCH_SPARSE_DIST:
                    expected_storage = {
                        ("table_0", "fused_uvm_caching", "table_wise"): [
                            (100 + 1024 * 10 + output_on_pipeline, 100)
                        ],
                        ("table_1", "fused", "table_wise"): [
                            (100 + 3072 + output_on_pipeline, 100)
                        ],
                        ("table_2", "fused_uvm_caching", "table_wise"): [
                            (100 + 1024 * 3 + int(1024 * 1.6) + output_on_pipeline, 100)
                        ],
                    }
                else:
                    # Backward compatible path, using old formula when pipeline
                    # type is None or unrecognized.
                    expected_storage = {
                        ("table_0", "fused_uvm_caching", "table_wise"): [
                            (100 + 3333 + 1024, 100)
                        ],
                        ("table_1", "fused", "table_wise"): [(100 + 3333 + 1024, 100)],
                        ("table_2", "fused_uvm_caching", "table_wise"): [
                            (100 + 3333 + 1024, 100)
                        ],
                    }
                actual_storage = {
                    (
                        sharding_option.name,
                        sharding_option.compute_kernel,
                        sharding_option.sharding_type,
                    ): [
                        (shard.storage.hbm, shard.storage.ddr)
                        for shard in sharding_option.shards
                        if shard.storage is not None
                    ]
                    for sharding_option in sharding_options
                }
                self.assertEqual(expected_storage, actual_storage)

    def test_default_output_sizes(self) -> None:
        topology = Topology(world_size=2, compute_device="cuda")
        constraint_list = [
            None,
            {"table_0": ParameterConstraints(output_dtype=DataType.FP32)},
        ]

        table_list = [
            [
                EmbeddingBagConfig(
                    num_embeddings=50,
                    embedding_dim=10,
                    name="table_0",
                    feature_names=["feature_0"],
                    data_type=DataType.FP32,
                )
            ],
            [
                EmbeddingBagConfig(
                    num_embeddings=100,
                    embedding_dim=10,
                    name="table_0",
                    feature_names=["feature_0"],
                    data_type=DataType.FP16,
                )
            ],
        ]
        hbms = []

        for tables, constraints in zip(table_list, constraint_list):
            enumerator = EmbeddingEnumerator(
                topology=topology, batch_size=BATCH_SIZE, constraints=constraints
            )
            model = TestSparseNN(tables=tables, weighted_tables=[])
            sharding_options = enumerator.enumerate(
                module=model,
                sharders=[
                    cast(
                        ModuleSharder[torch.nn.Module],
                        TestEBCSharder(
                            sharding_type=ShardingType.TABLE_WISE.value,
                            kernel_type=EmbeddingComputeKernel.FUSED.value,
                        ),
                    )
                ],
            )
            self.assertEqual(len(sharding_options), 1)
            self.assertEqual(len(sharding_options[0].shards), 1)
            self.assertIsNotNone(sharding_options[0].shards[0].storage)
            hbms.append(sharding_options[0].shards[0].storage.hbm)  # pyre-ignore

        self.assertEqual(hbms[0], hbms[1])


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
