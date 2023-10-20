#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import random
import unittest
from typing import List

from torchrec.distributed.embedding_lookup import EmbeddingComputeKernel

from torchrec.distributed.embedding_sharding import group_tables
from torchrec.distributed.embedding_types import (
    GroupedEmbeddingConfig,
    ShardedEmbeddingTable,
)
from torchrec.modules.embedding_configs import DataType, PoolingType


class TestGroupTablesPerRank(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

        random.seed(42)

        self.num_tables = 50
        self.data_type_options = [DataType.FP16, DataType.FP32]
        self.has_feature_processor_options = [False, True]
        self.fused_params_group_options = [
            {
                "cache_load_factor": 0.5,
                "stochastic_rounding": True,
            },
            {
                "cache_load_factor": 0.5,
                "stochastic_rounding": False,
            },
        ]
        self.embedding_dim_options = list(range(160, 320, 40))

        self.data_types = [
            random.choice(self.data_type_options) for _ in range(self.num_tables)
        ]
        self.pooling_types = [
            random.choice(list(PoolingType)) for _ in range(self.num_tables)
        ]
        self.has_feature_processors = [
            random.choice(self.has_feature_processor_options)
            for _ in range(self.num_tables)
        ]
        self.fused_params_groups = [
            random.choice(self.fused_params_group_options)
            for _ in range(self.num_tables)
        ]
        self.compute_kernels = [
            random.choice(list(EmbeddingComputeKernel)) for _ in range(self.num_tables)
        ]
        self.embedding_dims = [
            random.choice(self.embedding_dim_options) for _ in range(self.num_tables)
        ]

    def generate_embedding_tables(self) -> List[ShardedEmbeddingTable]:
        return [
            ShardedEmbeddingTable(
                name=f"table_{i}",
                data_type=data_type,
                pooling=pooling_type,
                has_feature_processor=has_feature_processor,
                fused_params=fused_param_group,
                compute_kernel=compute_kernel,
                embedding_dim=embedding_dim,
                num_embeddings=10000,
            )
            for i, (
                data_type,
                pooling_type,
                has_feature_processor,
                fused_param_group,
                compute_kernel,
                embedding_dim,
            ) in enumerate(
                zip(
                    self.data_types,
                    self.pooling_types,
                    self.has_feature_processors,
                    self.fused_params_groups,
                    self.compute_kernels,
                    self.embedding_dims,
                )
            )
        ]

    def test_group_tables_per_rank(self) -> None:
        embedding_tables = self.generate_embedding_tables()

        # since we don't have access to _group_tables_per_rank
        tables_per_rank: List[List[ShardedEmbeddingTable]] = [embedding_tables]

        # taking only the list for the first rank
        table_groups: List[GroupedEmbeddingConfig] = group_tables(tables_per_rank)[0]
        table_names_by_groups = [
            table_group.table_names() for table_group in table_groups
        ]

        expected_table_names_by_groups = [
            ["table_38"],
            ["table_32"],
            ["table_33"],
            ["table_8"],
            ["table_34", "table_41"],
            ["table_24"],
            ["table_36"],
            ["table_30"],
            ["table_2", "table_22", "table_23"],
            ["table_46"],
            ["table_27"],
            ["table_19"],
            ["table_16"],
            ["table_18", "table_40"],
            ["table_31", "table_42"],
            ["table_20", "table_37"],
            ["table_21", "table_48"],
            ["table_14"],
            ["table_17"],
            ["table_5", "table_28"],
            ["table_0", "table_6", "table_45"],
            ["table_7", "table_47"],
            ["table_10"],
            ["table_1", "table_35"],
            ["table_49"],
            ["table_44"],
            ["table_4"],
            ["table_11", "table_12", "table_13"],
            ["table_29"],
            ["table_9", "table_15"],
            ["table_39", "table_43"],
            ["table_25", "table_26"],
            ["table_3"],
        ]
        self.assertEqual(table_names_by_groups, expected_table_names_by_groups)

    def test_group_tables_per_rank_for_hbm(self) -> None:
        self.compute_kernels = [
            EmbeddingComputeKernel.FUSED for _ in range(self.num_tables)
        ]
        embedding_tables = self.generate_embedding_tables()

        # since we don't have access to _group_tables_per_rank
        tables_per_rank: List[List[ShardedEmbeddingTable]] = [embedding_tables]

        # taking only the list for the first rank
        table_groups: List[GroupedEmbeddingConfig] = group_tables(tables_per_rank)[0]
        table_names_by_groups = [
            table_group.table_names() for table_group in table_groups
        ]

        expected_table_names_by_groups = [
            ["table_32", "table_38"],
            ["table_33"],
            ["table_8"],
            ["table_24", "table_34", "table_41"],
            ["table_36"],
            ["table_2", "table_22", "table_23", "table_30"],
            ["table_46"],
            ["table_27"],
            ["table_19"],
            ["table_16", "table_18", "table_40"],
            ["table_20", "table_21", "table_31", "table_37", "table_42", "table_48"],
            ["table_14"],
            ["table_5", "table_17", "table_28"],
            ["table_0", "table_6", "table_7", "table_10", "table_45", "table_47"],
            ["table_1", "table_35", "table_49"],
            ["table_4", "table_44"],
            ["table_11", "table_12", "table_13", "table_29"],
            ["table_9", "table_15"],
            ["table_39", "table_43"],
            ["table_3", "table_25", "table_26"],
        ]
        self.assertEqual(table_names_by_groups, expected_table_names_by_groups)

    def test_group_tables_per_rank_for_uvm_caching(self) -> None:
        self.compute_kernels = [
            EmbeddingComputeKernel.FUSED_UVM_CACHING for _ in range(self.num_tables)
        ]
        embedding_tables = self.generate_embedding_tables()

        # since we don't have access to _group_tables_per_rank
        tables_per_rank: List[List[ShardedEmbeddingTable]] = [embedding_tables]

        # taking only the list for the first rank
        table_groups: List[GroupedEmbeddingConfig] = group_tables(tables_per_rank)[0]
        table_names_by_groups = [
            table_group.table_names() for table_group in table_groups
        ]

        expected_table_names_by_groups = [
            ["table_32", "table_38"],
            ["table_33"],
            ["table_8"],
            ["table_24", "table_34", "table_41"],
            ["table_36"],
            ["table_2", "table_22", "table_23", "table_30"],
            ["table_46"],
            ["table_27"],
            ["table_19"],
            ["table_16", "table_18", "table_40"],
            ["table_20", "table_21", "table_31", "table_37", "table_42", "table_48"],
            ["table_14"],
            ["table_5", "table_17", "table_28"],
            ["table_0", "table_6", "table_7", "table_10", "table_45", "table_47"],
            ["table_1", "table_35", "table_49"],
            ["table_4", "table_44"],
            ["table_11", "table_12", "table_13", "table_29"],
            ["table_9", "table_15"],
            ["table_39", "table_43"],
            ["table_3", "table_25", "table_26"],
        ]
        self.assertEqual(table_names_by_groups, expected_table_names_by_groups)
