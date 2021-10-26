#!/usr/bin/env python3

import unittest

from torchrec.distributed.embedding import (
    EmbeddingBagCollectionSharder,
)
from torchrec.distributed.embedding_types import EmbeddingTableConfig
from torchrec.distributed.planner.new.calculators import EmbeddingWTCostCalculator
from torchrec.distributed.planner.new.enumerators import ShardingEnumerator
from torchrec.distributed.planner.new.types import Topology
from torchrec.distributed.tests.test_model import TestSparseNN
from torchrec.fb.distributed.pooled_embedding_arch import PooledEmbeddingArchSharder
from torchrec.fb.modules.embedding_arch import PooledEmbeddingArch
from torchrec.modules.embedding_configs import EmbeddingBagConfig


class TestEmbeddingWTCostCalculator(unittest.TestCase):
    def setUp(self) -> None:
        topology = Topology(world_size=2, compute_device="cuda")
        self.enumerator = ShardingEnumerator(topology=topology)
        self.calculator = EmbeddingWTCostCalculator(topology=topology)

    def test_1_table_cost(self) -> None:
        tables = [
            EmbeddingBagConfig(
                num_embeddings=100,
                embedding_dim=10,
                name="table_0",
                feature_names=["feature_0"],
            )
        ]
        model = TestSparseNN(tables=tables, weighted_tables=[])
        sharding_options = self.enumerator.run(
            module=model, sharders=[EmbeddingBagCollectionSharder()]
        )
        for sharding_option in sharding_options:
            self.calculator.run(sharding_option=sharding_option)
        expected_costs = {
            ("dense", "data_parallel"): [365.06238859180036, 365.06238859180036],
            ("batched_dense", "data_parallel"): [308.8989441930619, 308.8989441930619],
            ("dense", "table_wise"): [4143.4581105169345],
            ("batched_dense", "table_wise"): [4031.1312217194572],
            ("batched_fused", "table_wise"): [3443.7755481233744],
            ("sparse", "table_wise"): [4082.6143790849674],
            ("batched_fused_uvm", "table_wise"): [3948.7581699346406],
            ("batched_fused_uvm_caching", "table_wise"): [3537.3418104753255],
            ("dense", "row_wise"): [3778.395721925134, 3778.395721925134],
            ("batched_dense", "row_wise"): [3722.2322775263956, 3722.2322775263956],
            ("batched_fused", "row_wise"): [3428.5544407283537, 3428.5544407283537],
            ("sparse", "row_wise"): [3747.9738562091507, 3747.9738562091507],
            ("batched_fused_uvm", "row_wise"): [3681.045751633987, 3681.045751633987],
            ("batched_fused_uvm_caching", "row_wise"): [
                3475.3375719043297,
                3475.3375719043297,
            ],
            ("dense", "table_row_wise"): [3846.6623885918007, 3846.6623885918007],
            ("batched_dense", "table_row_wise"): [3790.498944193062, 3790.498944193062],
            ("batched_fused", "table_row_wise"): [
                3496.8211073950206,
                3496.8211073950206,
            ],
            ("sparse", "table_row_wise"): [3816.2405228758175, 3816.2405228758175],
            ("batched_fused_uvm", "table_row_wise"): [
                3749.312418300654,
                3749.312418300654,
            ],
            ("batched_fused_uvm_caching", "table_row_wise"): [
                3543.6042385709966,
                3543.6042385709966,
            ],
        }
        costs = {
            (
                sharding_option.compute_kernel,
                sharding_option.sharding_type,
            ): sharding_option.shard_costs
            for sharding_option in sharding_options
        }
        self.assertEqual(expected_costs, costs)

    def test_2_table_cost_for_pooledEmbArch_model(self) -> None:
        tables = [
            EmbeddingTableConfig(
                num_embeddings=100,
                embedding_dim=10,
                name="table_1",
                feature_names=["feature_1"],
            )
        ]
        model = PooledEmbeddingArch(
            tables=tables,
            embedding_groups={"group_1": ["feature_1"]},
        )

        sharding_options = self.enumerator.run(
            module=model, sharders=[PooledEmbeddingArchSharder()]
        )
        for sharding_option in sharding_options:
            self.calculator.run(sharding_option=sharding_option)
        expected_costs = {
            ("dense", "data_parallel"): [365.06238859180036, 365.06238859180036],
            ("batched_dense", "data_parallel"): [308.8989441930619, 308.8989441930619],
            ("dense", "table_wise"): [4143.4581105169345],
            ("batched_dense", "table_wise"): [4031.1312217194572],
            ("batched_fused", "table_wise"): [3443.7755481233744],
            ("sparse", "table_wise"): [4082.6143790849674],
            ("batched_fused_uvm", "table_wise"): [3948.7581699346406],
            ("batched_fused_uvm_caching", "table_wise"): [3537.3418104753255],
            ("dense", "table_row_wise"): [3846.6623885918007, 3846.6623885918007],
            ("batched_dense", "table_row_wise"): [3790.498944193062, 3790.498944193062],
            ("batched_fused", "table_row_wise"): [
                3496.8211073950206,
                3496.8211073950206,
            ],
            ("sparse", "table_row_wise"): [3816.2405228758175, 3816.2405228758175],
            ("batched_fused_uvm", "table_row_wise"): [
                3749.312418300654,
                3749.312418300654,
            ],
            ("batched_fused_uvm_caching", "table_row_wise"): [
                3543.6042385709966,
                3543.6042385709966,
            ],
            ("dense", "row_wise"): [3778.395721925134, 3778.395721925134],
            ("batched_dense", "row_wise"): [3722.2322775263956, 3722.2322775263956],
            ("batched_fused", "row_wise"): [3428.5544407283537, 3428.5544407283537],
            ("sparse", "row_wise"): [3747.9738562091507, 3747.9738562091507],
            ("batched_fused_uvm", "row_wise"): [3681.045751633987, 3681.045751633987],
            ("batched_fused_uvm_caching", "row_wise"): [
                3475.3375719043297,
                3475.3375719043297,
            ],
            ("dense", "column_wise"): [4143.4581105169345],
            ("batched_dense", "column_wise"): [4031.1312217194572],
            ("batched_fused", "column_wise"): [3443.7755481233744],
            ("sparse", "column_wise"): [4082.6143790849674],
            ("batched_fused_uvm", "column_wise"): [3948.7581699346406],
            ("batched_fused_uvm_caching", "column_wise"): [3537.3418104753255],
        }
        costs = {
            (
                sharding_option.compute_kernel,
                sharding_option.sharding_type,
            ): sharding_option.shard_costs
            for sharding_option in sharding_options
        }
        self.assertEqual(expected_costs, costs)
