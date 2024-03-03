#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import cast, List, Optional
from unittest.mock import MagicMock

import torch
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.planner.constants import BATCH_SIZE
from torchrec.distributed.planner.enumerators import EmbeddingEnumerator
from torchrec.distributed.planner.proposers import (
    EmbeddingOffloadScaleupProposer,
    GreedyProposer,
    GridSearchProposer,
    proposers_to_proposals_list,
    UniformProposer,
)
from torchrec.distributed.planner.types import (
    Enumerator,
    ParameterConstraints,
    Proposer,
    ShardingOption,
    Topology,
)
from torchrec.distributed.test_utils.test_model import TestSparseNN
from torchrec.distributed.types import (
    CacheParams,
    CacheStatistics,
    ModuleSharder,
    ShardingType,
)
from torchrec.modules.embedding_configs import EmbeddingBagConfig


class MockProposer(Proposer):
    def load(
        self,
        search_space: List[ShardingOption],
        enumerator: Optional[Enumerator] = None,
    ) -> None:
        pass

    def feedback(
        self,
        partitionable: bool,
        plan: Optional[List[ShardingOption]] = None,
        perf_rating: Optional[float] = None,
        storage_constraint: Optional[Topology] = None,
    ) -> None:
        pass

    def propose(self) -> Optional[List[ShardingOption]]:
        pass


class MockCacheStatistics(CacheStatistics):
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


class TestProposers(unittest.TestCase):
    def setUp(self) -> None:
        topology = Topology(world_size=2, compute_device="cuda")
        self.enumerator = EmbeddingEnumerator(topology=topology, batch_size=BATCH_SIZE)
        self.greedy_proposer = GreedyProposer()
        self.uniform_proposer = UniformProposer()
        self.grid_search_proposer = GridSearchProposer()

    def test_greedy_two_table(self) -> None:
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

        model = TestSparseNN(tables=tables, sparse_device=torch.device("meta"))
        search_space = self.enumerator.enumerate(
            module=model,
            sharders=[
                cast(ModuleSharder[torch.nn.Module], EmbeddingBagCollectionSharder())
            ],
        )
        self.greedy_proposer.load(search_space)

        # simulate first five iterations:
        output = []
        for _ in range(5):
            proposal = cast(List[ShardingOption], self.greedy_proposer.propose())
            proposal.sort(
                key=lambda sharding_option: (
                    max([shard.perf.total for shard in sharding_option.shards]),
                    sharding_option.name,
                )
            )
            output.append(
                [
                    (
                        candidate.name,
                        candidate.sharding_type,
                        candidate.compute_kernel,
                    )
                    for candidate in proposal
                ]
            )
            self.greedy_proposer.feedback(partitionable=True)

        expected_output = [
            [
                ("table_0", "row_wise", "fused"),
                ("table_1", "row_wise", "fused"),
            ],
            [
                ("table_0", "table_row_wise", "fused"),
                ("table_1", "row_wise", "fused"),
            ],
            [
                ("table_1", "row_wise", "fused"),
                ("table_0", "data_parallel", "dense"),
            ],
            [
                ("table_1", "table_row_wise", "fused"),
                ("table_0", "data_parallel", "dense"),
            ],
            [
                ("table_0", "data_parallel", "dense"),
                ("table_1", "data_parallel", "dense"),
            ],
        ]

        self.assertEqual(expected_output, output)

        # Test threshold for early_stopping
        self.greedy_proposer._threshold = 10
        self.greedy_proposer.load(search_space)

        # With early stopping, after berf_perf_rating is assigned, after 10 iterations with
        # consecutive worse perf_rating, the returned proposal should be None.
        proposal = None
        for i in range(13):
            proposal = cast(List[ShardingOption], self.greedy_proposer.propose())
            self.greedy_proposer.feedback(partitionable=True, perf_rating=100 + i)
        self.assertEqual(self.greedy_proposer._best_perf_rating, 100)
        self.assertEqual(proposal, None)

    def test_uniform_three_table(self) -> None:
        tables = [
            EmbeddingBagConfig(
                num_embeddings=100 * i,
                embedding_dim=10 * i,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(1, 4)
        ]
        model = TestSparseNN(tables=tables, sparse_device=torch.device("meta"))

        mock_ebc_sharder = cast(
            ModuleSharder[torch.nn.Module], EmbeddingBagCollectionSharder()
        )
        # TODO update this test for CW and TWCW sharding
        mock_ebc_sharder.sharding_types = MagicMock(
            return_value=[
                ShardingType.DATA_PARALLEL.value,
                ShardingType.TABLE_WISE.value,
                ShardingType.ROW_WISE.value,
                ShardingType.TABLE_ROW_WISE.value,
            ]
        )

        self.maxDiff = None

        search_space = self.enumerator.enumerate(
            module=model, sharders=[mock_ebc_sharder]
        )
        self.uniform_proposer.load(search_space)

        output = []
        proposal = self.uniform_proposer.propose()
        while proposal:
            proposal.sort(
                key=lambda sharding_option: (
                    max([shard.perf.total for shard in sharding_option.shards]),
                    sharding_option.name,
                )
            )
            output.append(
                [
                    (
                        candidate.name,
                        candidate.sharding_type,
                        candidate.compute_kernel,
                    )
                    for candidate in proposal
                ]
            )
            self.uniform_proposer.feedback(partitionable=True)
            proposal = self.uniform_proposer.propose()

        expected_output = [
            [
                (
                    "table_1",
                    "data_parallel",
                    "dense",
                ),
                (
                    "table_2",
                    "data_parallel",
                    "dense",
                ),
                (
                    "table_3",
                    "data_parallel",
                    "dense",
                ),
            ],
            [
                (
                    "table_1",
                    "table_wise",
                    "fused",
                ),
                (
                    "table_2",
                    "table_wise",
                    "fused",
                ),
                (
                    "table_3",
                    "table_wise",
                    "fused",
                ),
            ],
            [
                (
                    "table_1",
                    "row_wise",
                    "fused",
                ),
                (
                    "table_2",
                    "row_wise",
                    "fused",
                ),
                (
                    "table_3",
                    "row_wise",
                    "fused",
                ),
            ],
            [
                (
                    "table_1",
                    "table_row_wise",
                    "fused",
                ),
                (
                    "table_2",
                    "table_row_wise",
                    "fused",
                ),
                (
                    "table_3",
                    "table_row_wise",
                    "fused",
                ),
            ],
        ]

        self.assertEqual(expected_output, output)

    def test_grid_search_three_table(self) -> None:
        tables = [
            EmbeddingBagConfig(
                num_embeddings=100 * i,
                embedding_dim=10 * i,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(1, 4)
        ]
        model = TestSparseNN(tables=tables, sparse_device=torch.device("meta"))
        search_space = self.enumerator.enumerate(
            module=model,
            sharders=[
                cast(ModuleSharder[torch.nn.Module], EmbeddingBagCollectionSharder())
            ],
        )

        """
        All sharding types but DP will have 3 possible compute kernels after pruning:
            - fused
            - fused_uvm_caching
            - fused_uvm
        DP will have 1 possible compute kernel: dense
        So the total number of pruned options will be:
            (num_sharding_types - 1) * 3 + 1 = 16
        """
        num_pruned_options = (len(ShardingType) - 1) * 3 + 1
        self.grid_search_proposer.load(search_space)
        for (
            sharding_options
        ) in self.grid_search_proposer._sharding_options_by_fqn.values():
            # number of sharding types after pruning is number of sharding types * 3
            # 3 compute kernels fused/dense, fused_uvm_caching, fused_uvm
            self.assertEqual(len(sharding_options), num_pruned_options)

        num_proposals = 0
        proposal = self.grid_search_proposer.propose()
        while proposal:
            self.grid_search_proposer.feedback(partitionable=True)
            proposal = self.grid_search_proposer.propose()
            num_proposals += 1

        self.assertEqual(num_pruned_options ** len(tables), num_proposals)

    def test_allocate_budget(self) -> None:
        model = torch.tensor([[1.0, 0.0], [2.0, 3.0], [4.0, 5.0]])
        got = EmbeddingOffloadScaleupProposer.clf_to_bytes(
            model, torch.tensor([0, 0.5, 1])
        )
        torch.testing.assert_close(got, torch.tensor([0, 4, 9]))

        # Scenario 1, enough budget to scale everything to 1.0
        model = torch.tensor(
            [[30_000_000, 2_000_000], [30_000_000, 2_000_000], [30_000_000, 2_000_000]]
        )
        mins = torch.tensor([0.1, 0.1, 1])
        budget = 100_000_000
        got = EmbeddingOffloadScaleupProposer.allocate_budget(
            model,
            clfs=torch.tensor(mins),
            budget=budget,
            allocation_priority=torch.tensor([2, 2, 2]),
        )
        torch.testing.assert_close(got, torch.tensor([1.0, 1.0, 1.0]))
        increase = (
            EmbeddingOffloadScaleupProposer.clf_to_bytes(model, got).sum()
            - EmbeddingOffloadScaleupProposer.clf_to_bytes(model, mins).sum()
        ).item()
        self.assertLess(increase, budget)

        # Scenario 2, limited budget, uniform scale up
        model = torch.tensor(
            [[30_000_000, 2_000_000], [30_000_000, 2_000_000], [30_000_000, 2_000_000]]
        )
        mins = torch.tensor([0.1, 0.1, 1])
        budget = 10_000_000
        got = EmbeddingOffloadScaleupProposer.allocate_budget(
            model, clfs=mins, budget=budget, allocation_priority=torch.tensor([2, 2, 2])
        )
        torch.testing.assert_close(got, torch.tensor([0.26667, 0.26667, 1.0]))
        increase = (
            EmbeddingOffloadScaleupProposer.clf_to_bytes(model, got).sum()
            - EmbeddingOffloadScaleupProposer.clf_to_bytes(model, mins).sum()
        )
        self.assertEqual(increase, budget)

        # Scenario 3, limited budget, skewed scale up
        model = torch.tensor(
            [[30_000_000, 2_000_000], [30_000_000, 2_000_000], [30_000_000, 2_000_000]]
        )
        mins = torch.tensor([0.1, 0.1, 1])
        budget = 10_000_000
        got = EmbeddingOffloadScaleupProposer.allocate_budget(
            model, clfs=mins, budget=budget, allocation_priority=torch.tensor([2, 4, 2])
        )
        # increase is twice as much for table 2 (started at 0.1)
        torch.testing.assert_close(
            got, torch.tensor([0.1 + 0.11111, 0.1 + 2 * 0.11111, 1.0])
        )
        increase = (
            EmbeddingOffloadScaleupProposer.clf_to_bytes(model, got).sum()
            - EmbeddingOffloadScaleupProposer.clf_to_bytes(model, mins).sum()
        )
        self.assertEqual(increase, budget)

        # Scenario 4, multi-pass scale up
        model = torch.tensor(
            [[30_000_000, 2_000_000], [30_000_000, 2_000_000], [30_000_000, 2_000_000]]
        )
        mins = torch.tensor([0.1, 0.3, 0.5])
        budget = 50_000_000
        got = EmbeddingOffloadScaleupProposer.allocate_budget(
            model,
            clfs=mins,
            budget=budget,
            allocation_priority=torch.tensor([1, 2, 100]),
        )
        torch.testing.assert_close(got, torch.tensor([0.56667, 1.0, 1.0]))
        increase = (
            EmbeddingOffloadScaleupProposer.clf_to_bytes(model, got).sum()
            - EmbeddingOffloadScaleupProposer.clf_to_bytes(model, mins).sum()
        )
        self.assertEqual(increase, budget)

    def test_scaleup(self) -> None:
        tables = [
            EmbeddingBagConfig(
                num_embeddings=2_000_000,
                embedding_dim=10000,
                name=f"table_{i}",
                feature_names=[f"feature_{i}"],
            )
            for i in range(4)
        ]

        # Place first three tables into cache, 4th table leave on hbm. table_1 has a
        # larger cacheability score so budget should be skewed to scaling table_1 more
        # than table_0. table_2 is a deprecated feature we have no stats for (so
        # expected_lookups 0), we want to see that left at its original load factor,
        # i.e. doesn't participate in scaleup.
        constraints = {
            "table_0": ParameterConstraints(
                compute_kernels=[EmbeddingComputeKernel.FUSED_UVM_CACHING.value],
                cache_params=CacheParams(
                    load_factor=0.1,
                    stats=MockCacheStatistics(expected_lookups=2, cacheability=0.2),
                ),
            ),
            "table_1": ParameterConstraints(
                compute_kernels=[EmbeddingComputeKernel.FUSED_UVM_CACHING.value],
                cache_params=CacheParams(
                    load_factor=0.1,
                    stats=MockCacheStatistics(expected_lookups=2, cacheability=0.5),
                ),
            ),
            "table_2": ParameterConstraints(
                compute_kernels=[EmbeddingComputeKernel.FUSED_UVM_CACHING.value],
                cache_params=CacheParams(
                    load_factor=0.002,
                    stats=MockCacheStatistics(expected_lookups=0, cacheability=0.5),
                ),
            ),
        }

        GB = 1024 * 1024 * 1024
        storage_constraint = Topology(
            world_size=2, compute_device="cuda", hbm_cap=100 * GB, ddr_cap=1000 * GB
        )
        # Ignoring table_2, the remainder require 224GB if all placed on HBM. We only
        # have 200GB, so we can't promote both uvm tables. Initial plan needs uses 90GB,
        # with 110GB of available budget.
        model = TestSparseNN(tables=tables, sparse_device=torch.device("meta"))
        enumerator = EmbeddingEnumerator(
            topology=storage_constraint, batch_size=BATCH_SIZE, constraints=constraints
        )
        search_space = enumerator.enumerate(
            module=model,
            sharders=[
                cast(ModuleSharder[torch.nn.Module], EmbeddingBagCollectionSharder())
            ],
        )
        proposer = EmbeddingOffloadScaleupProposer()
        proposer.load(search_space, enumerator=enumerator)

        proposal = proposer.propose()
        best_plan = None
        best_perf = 1e99
        proposals = -1
        while proposal is not None:
            proposals += 1
            mem = sum(so.total_storage.hbm for so in proposal)
            # simple perf model, assume partitioner gives a lowest score around 150GB of memory.
            perf = abs(mem - (150 * GB))
            plan = {
                "mem": mem,
                "proposal": [
                    (
                        candidate.name,
                        candidate.compute_kernel,
                        (
                            candidate.cache_params.load_factor
                            if candidate.cache_params
                            else None
                        ),
                    )
                    for candidate in proposal
                ],
            }
            if perf < best_perf:
                best_plan = plan
                best_perf = perf
            proposer.feedback(
                partitionable=True,
                plan=proposal,
                perf_rating=perf,
                storage_constraint=storage_constraint,
            )
            proposal = proposer.propose()

        self.assertEqual(proposals, 16)
        self.assertEqual(
            best_plan,
            {
                # 146GB, close to target of 150GB
                "mem": 157178896800,
                # table 1 has been scaled up 2.5x more than table 0 (vs original 0.1)
                # which aligns with their different cacheability scores
                # table_2 has been left alone (deprecated feature, expected zero lookups in stats)
                "proposal": [
                    ("table_0", "fused_uvm_caching", 0.3173336386680603),
                    ("table_1", "fused_uvm_caching", 0.6433340907096863),
                    ("table_2", "fused_uvm_caching", 0.002),
                    ("table_3", "fused", None),
                ],
            },
        )

    def test_proposers_to_proposals_list(self) -> None:
        def make_mock_proposal(name: str) -> List[ShardingOption]:
            return [
                ShardingOption(
                    name=name,
                    tensor=torch.zeros(1),
                    # pyre-ignore
                    module=("model", None),
                    input_lengths=[],
                    batch_size=8,
                    sharding_type="row_wise",
                    partition_by="DEVICE",
                    compute_kernel="fused",
                    shards=[],
                )
            ]

        mock_proposer_1 = MockProposer()
        mock_proposer_1_sharding_options = [
            make_mock_proposal("p1so1"),
            make_mock_proposal("p1so2"),
            make_mock_proposal("p1so1"),
            None,
        ]
        mock_proposer_1.propose = MagicMock(
            side_effect=mock_proposer_1_sharding_options
        )

        mock_proposer_2 = MockProposer()
        mock_proposer_2_sharding_options = [
            make_mock_proposal("p2so1"),
            make_mock_proposal("p2so1"),
            make_mock_proposal("p1so2"),
            make_mock_proposal("p2so2"),
            None,
        ]
        mock_proposer_2.propose = MagicMock(
            side_effect=mock_proposer_2_sharding_options
        )

        mock_proposer_3 = MockProposer()
        mock_proposer_3_sharding_options = [
            make_mock_proposal("p3so1"),
            make_mock_proposal("p2so1"),
            make_mock_proposal("p3so2"),
            None,
        ]
        mock_proposer_3.propose = MagicMock(
            side_effect=mock_proposer_3_sharding_options
        )

        proposers_list: List[Proposer] = [
            mock_proposer_1,
            mock_proposer_2,
            mock_proposer_3,
        ]

        proposals_list = proposers_to_proposals_list(proposers_list, search_space=[])
        proposals_list_names = []

        for sharding_option in proposals_list:
            proposals_list_names.append(sharding_option[0].name)

        expected_list_names = ["p1so1", "p1so2", "p2so1", "p2so2", "p3so1", "p3so2"]

        self.assertEqual(proposals_list_names, expected_list_names)
