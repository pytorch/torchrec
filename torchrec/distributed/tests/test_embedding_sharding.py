#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import random
import unittest
from typing import Any, Dict, List
from unittest.mock import MagicMock

import hypothesis.strategies as st

from hypothesis import given, settings

from torchrec.distributed.embedding_lookup import EmbeddingComputeKernel

from torchrec.distributed.embedding_sharding import (
    _get_compute_kernel_type,
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


def _get_table_names_by_groups(
    embedding_tables: List[ShardedEmbeddingTable],
) -> List[List[str]]:
    # since we don't have access to _group_tables_per_rank
    tables_per_rank: List[List[ShardedEmbeddingTable]] = [embedding_tables]

    # taking only the list for the first rank
    table_groups: List[GroupedEmbeddingConfig] = group_tables(tables_per_rank)[0]
    return [table_group.table_names() for table_group in table_groups]


class TestGroupTablesPerRank(unittest.TestCase):
    # pyre-ignore[56]
    @given(
        data_type=st.sampled_from([DataType.FP16, DataType.FP32]),
        has_feature_processor=st.sampled_from([False, True]),
        fused_params_group=st.sampled_from(
            [
                {
                    "cache_load_factor": 0.5,
                    "prefetch_pipeline": False,
                },
                {
                    "cache_load_factor": 0.3,
                    "prefetch_pipeline": True,
                },
            ]
        ),
        embedding_dim=st.sampled_from(list(range(160, 320, 40))),
        pooling_type=st.sampled_from(list(PoolingType)),
        compute_kernel=st.sampled_from(list(EmbeddingComputeKernel)),
    )
    @settings(max_examples=10, deadline=10000)
    def test_should_group_together(
        self,
        data_type: DataType,
        has_feature_processor: bool,
        fused_params_group: Dict[str, Any],
        embedding_dim: int,
        pooling_type: PoolingType,
        compute_kernel: EmbeddingComputeKernel,
    ) -> None:
        tables = [
            ShardedEmbeddingTable(
                name=f"table_{i}",
                data_type=data_type,
                pooling=pooling_type,
                has_feature_processor=has_feature_processor,
                fused_params=fused_params_group,
                compute_kernel=compute_kernel,
                embedding_dim=embedding_dim,
                num_embeddings=10000,
            )
            for i in range(2)
        ]

        expected_table_names_by_groups = [["table_0", "table_1"]]
        self.assertEqual(
            _get_table_names_by_groups(tables),
            expected_table_names_by_groups,
        )

    # pyre-ignore[56]
    @given(
        data_type=st.sampled_from([DataType.FP16, DataType.FP32]),
        has_feature_processor=st.sampled_from([False, True]),
        embedding_dim=st.sampled_from(list(range(160, 320, 40))),
        pooling_type=st.sampled_from(list(PoolingType)),
        compute_kernel=st.sampled_from(list(EmbeddingComputeKernel)),
    )
    @settings(max_examples=10, deadline=10000)
    def test_should_group_together_with_prefetch(
        self,
        data_type: DataType,
        has_feature_processor: bool,
        embedding_dim: int,
        pooling_type: PoolingType,
        compute_kernel: EmbeddingComputeKernel,
    ) -> None:
        fused_params_groups = [
            {
                "cache_load_factor": 0.3,
                "prefetch_pipeline": True,
            },
            {
                "cache_load_factor": 0.5,
                "prefetch_pipeline": True,
            },
        ]
        tables = [
            ShardedEmbeddingTable(
                name=f"table_{i}",
                data_type=data_type,
                pooling=pooling_type,
                has_feature_processor=has_feature_processor,
                fused_params=fused_params_groups[i],
                compute_kernel=compute_kernel,
                embedding_dim=embedding_dim,
                num_embeddings=10000,
            )
            for i in range(2)
        ]

        expected_table_names_by_groups = [["table_0", "table_1"]]
        self.assertEqual(
            _get_table_names_by_groups(tables),
            expected_table_names_by_groups,
        )

    # pyre-ignore[56]
    @given(
        data_types=st.lists(
            st.sampled_from([DataType.FP16, DataType.FP32]),
            min_size=2,
            max_size=2,
            unique=True,
        ),
        has_feature_processors=st.lists(
            st.sampled_from([False, True]), min_size=2, max_size=2, unique=True
        ),
        fused_params_group=st.sampled_from(
            [
                {
                    "cache_load_factor": 0.5,
                    "prefetch_pipeline": True,
                },
                {
                    "cache_load_factor": 0.3,
                    "prefetch_pipeline": True,
                },
            ],
        ),
        embedding_dims=st.lists(
            st.sampled_from(list(range(160, 320, 40))),
            min_size=2,
            max_size=2,
            unique=True,
        ),
        pooling_types=st.lists(
            st.sampled_from(list(PoolingType)), min_size=2, max_size=2, unique=True
        ),
        compute_kernels=st.lists(
            st.sampled_from(list(EmbeddingComputeKernel)),
            min_size=2,
            max_size=2,
            unique=True,
        ),
        distinct_key=st.sampled_from(
            [
                "data_type",
                "has_feature_processor",
                "embedding_dim",
                "pooling_type",
                "compute_kernel",
            ]
        ),
    )
    @settings(max_examples=10, deadline=10000)
    def test_should_not_group_together(
        self,
        data_types: List[DataType],
        has_feature_processors: List[bool],
        fused_params_group: Dict[str, Any],
        embedding_dims: List[int],
        pooling_types: List[PoolingType],
        compute_kernels: List[EmbeddingComputeKernel],
        distinct_key: str,
    ) -> None:
        tables = [
            ShardedEmbeddingTable(
                name=f"table_{i}",
                data_type=data_types[i]
                if distinct_key == "data_type"
                else data_types[0],
                pooling=pooling_types[i]
                if distinct_key == "pooling_type"
                else pooling_types[0],
                has_feature_processor=has_feature_processors[i]
                if distinct_key == "has_feature_processor"
                else has_feature_processors[0],
                fused_params=fused_params_group,  # can't hash dicts
                compute_kernel=compute_kernels[i]
                if distinct_key == "compute_kernel"
                else compute_kernels[0],
                embedding_dim=embedding_dims[i]
                if distinct_key == "embedding_dim"
                else embedding_dims[0],
                num_embeddings=10000,
            )
            for i in range(2)
        ]

        if distinct_key == "compute_kernel" and _get_compute_kernel_type(
            compute_kernels[0]
        ) == _get_compute_kernel_type(compute_kernels[1]):
            self.assertEqual(
                _get_table_names_by_groups(tables),
                [["table_0", "table_1"]],
            )
            return

        # emb dim bucketizier only in use when computer kernel is uvm caching
        # and prefetch pipeline is True
        if (
            distinct_key == "embedding_dim"
            and compute_kernels[0] != EmbeddingComputeKernel.FUSED_UVM_CACHING
        ):
            self.assertEqual(
                _get_table_names_by_groups(tables),
                [["table_0", "table_1"]],
            )
            return

        self.assertEqual(
            sorted(_get_table_names_by_groups(tables)),
            [["table_0"], ["table_1"]],
        )
