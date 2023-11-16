#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import random
import unittest
from typing import List
from unittest.mock import MagicMock

import hypothesis.strategies as st

from hypothesis import given, settings

from torchrec.distributed.embedding_lookup import EmbeddingComputeKernel

from torchrec.distributed.embedding_sharding import (
    _get_grouping_fused_params,
    _get_weighted_avg_cache_load_factor,
    group_tables,
)
from torchrec.distributed.embedding_types import (
    GroupedEmbeddingConfig,
    ShardedEmbeddingTable,
)
from torchrec.modules.embedding_configs import DataType, PoolingType


class TestGetWeightedAverageCacheLoadFactor(unittest.TestCase):
    def test_get_avg_cache_load_factor_hbm(self) -> None:
        cache_load_factors = [random.random() for _ in range(5)]
        embedding_tables: List[ShardedEmbeddingTable] = [
            ShardedEmbeddingTable(
                num_embeddings=1000,
                embedding_dim=MagicMock(),
                fused_params={"cache_load_factor": cache_load_factor},
            )
            for cache_load_factor in cache_load_factors
        ]

        weighted_avg_cache_load_factor = _get_weighted_avg_cache_load_factor(
            embedding_tables
        )
        self.assertIsNone(weighted_avg_cache_load_factor)

    def test_get_avg_cache_load_factor(self) -> None:
        cache_load_factors = [random.random() for _ in range(5)]
        embedding_tables: List[ShardedEmbeddingTable] = [
            ShardedEmbeddingTable(
                num_embeddings=1000,
                embedding_dim=MagicMock(),
                compute_kernel=EmbeddingComputeKernel.FUSED_UVM_CACHING,
                fused_params={"cache_load_factor": cache_load_factor},
            )
            for cache_load_factor in cache_load_factors
        ]

        weighted_avg_cache_load_factor = _get_weighted_avg_cache_load_factor(
            embedding_tables
        )
        expected_avg = sum(cache_load_factors) / len(cache_load_factors)
        self.assertIsNotNone(weighted_avg_cache_load_factor)
        self.assertAlmostEqual(weighted_avg_cache_load_factor, expected_avg)

    def test_get_weighted_avg_cache_load_factor(self) -> None:
        hash_sizes = [random.randint(100, 1000) for _ in range(5)]
        cache_load_factors = [random.random() for _ in range(5)]
        embedding_tables: List[ShardedEmbeddingTable] = [
            ShardedEmbeddingTable(
                num_embeddings=hash_size,
                embedding_dim=MagicMock(),
                compute_kernel=EmbeddingComputeKernel.FUSED_UVM_CACHING,
                fused_params={"cache_load_factor": cache_load_factor},
            )
            for cache_load_factor, hash_size in zip(cache_load_factors, hash_sizes)
        ]

        weighted_avg_cache_load_factor = _get_weighted_avg_cache_load_factor(
            embedding_tables
        )
        expected_weighted_avg = sum(
            cache_load_factor * hash_size
            for cache_load_factor, hash_size in zip(cache_load_factors, hash_sizes)
        ) / sum(hash_sizes)

        self.assertIsNotNone(weighted_avg_cache_load_factor)
        self.assertAlmostEqual(weighted_avg_cache_load_factor, expected_weighted_avg)


class TestGetGroupingFusedParams(unittest.TestCase):
    def test_get_grouping_fused_params(self) -> None:
        fused_params_groups = [
            None,
            {},
            {"stochastic_rounding": False},
            {"stochastic_rounding": False, "cache_load_factor": 0.4},
        ]
        grouping_fused_params_groups = [
            _get_grouping_fused_params(fused_params)
            for fused_params in fused_params_groups
        ]
        expected_grouping_fused_params_groups = [
            None,
            {},
            {"stochastic_rounding": False},
            {"stochastic_rounding": False},
        ]

        self.assertEqual(
            grouping_fused_params_groups, expected_grouping_fused_params_groups
        )


class TestPerTBECacheLoadFactor(unittest.TestCase):
    # pyre-ignore[56]
    @given(
        data_type=st.sampled_from([DataType.FP16, DataType.FP32]),
        has_feature_processor=st.sampled_from([False, True]),
        embedding_dim=st.sampled_from(list(range(160, 320, 40))),
        pooling_type=st.sampled_from(list(PoolingType)),
    )
    @settings(max_examples=10, deadline=10000)
    def test_per_tbe_clf_weighted_average(
        self,
        data_type: DataType,
        has_feature_processor: bool,
        embedding_dim: int,
        pooling_type: PoolingType,
    ) -> None:
        compute_kernels = [
            EmbeddingComputeKernel.FUSED_UVM_CACHING,
            EmbeddingComputeKernel.FUSED_UVM_CACHING,
            EmbeddingComputeKernel.FUSED,
            EmbeddingComputeKernel.FUSED_UVM,
        ]
        fused_params_groups = [
            {"cache_load_factor": 0.5},
            {"cache_load_factor": 0.3},
            {"cache_load_factor": 0.9},  # hbm table, would have no effect
            {"cache_load_factor": 0.4},  # uvm table, would have no effect
        ]
        tables = [
            ShardedEmbeddingTable(
                name=f"table_{i}",
                data_type=data_type,
                pooling=pooling_type,
                has_feature_processor=has_feature_processor,
                fused_params=fused_params_groups[i],
                compute_kernel=compute_kernels[i],
                embedding_dim=embedding_dim,
                num_embeddings=10000 * (2 * i + 1),  # 10000 and 30000
            )
            for i in range(4)
        ]

        # since we don't have access to _group_tables_per_rank
        tables_per_rank: List[List[ShardedEmbeddingTable]] = [tables]

        # taking only the list for the first rank
        table_groups: List[GroupedEmbeddingConfig] = group_tables(tables_per_rank)[0]

        # assert that they are grouped together
        self.assertEqual(len(table_groups), 1)

        table_group = table_groups[0]
        self.assertIsNotNone(table_group.fused_params)
        self.assertEqual(table_group.fused_params.get("cache_load_factor"), 0.35)
