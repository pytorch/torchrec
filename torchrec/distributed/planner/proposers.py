#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import itertools
import logging
from decimal import Decimal
from typing import cast, Dict, List, Optional, Set, Tuple, Union

import torch

from torchrec.distributed.embedding_types import EmbeddingComputeKernel

from torchrec.distributed.planner.types import (
    Enumerator,
    Perf,
    Proposer,
    ShardingOption,
    Topology,
)
from torchrec.distributed.planner.utils import bytes_to_gb, LuusJaakolaSearch, prod

logger: logging.Logger = logging.getLogger(__name__)

MAX_PROPOSALS: int = int(1e4)


class GreedyProposer(Proposer):
    """
    Proposes sharding plans in greedy fashion.

    Sorts sharding options for each shardable parameter by perf.
    On each iteration, finds parameter with largest current storage usage and tries its
    next sharding option.

    Args:
        use_depth (bool): When enabled, sharding_options of a fqn are sorted based on
            `max(shard.perf.total)`, otherwise sharding_options are sorted by
            `sum(shard.perf.total)`.
        threshold (Optional[int]): Threshold for early stopping. When specified, the
            proposer stops proposing when the proposals have consecutive worse perf_rating
            than best_perf_rating.
    """

    def __init__(self, use_depth: bool = True, threshold: Optional[int] = None) -> None:
        self._use_depth: bool = use_depth
        self._threshold: Optional[int] = threshold if threshold else None
        self._sharding_options_by_fqn: Dict[str, List[ShardingOption]] = {}
        self._current_proposal: Dict[str, int] = {}
        self._best_perf_rating: float = float("inf")
        self._num_inferior_perf: int = 0

    def load(
        self,
        search_space: List[ShardingOption],
        enumerator: Optional[Enumerator] = None,
    ) -> None:
        self._reset()
        for sharding_option in search_space:
            fqn = sharding_option.fqn
            if fqn not in self._sharding_options_by_fqn:
                self._sharding_options_by_fqn[fqn] = []
            self._sharding_options_by_fqn[fqn].append(sharding_option)

        for sharding_options in self._sharding_options_by_fqn.values():
            sharding_options.sort(
                key=lambda x: _sharding_option_score(x, self._use_depth)
            )

        self._current_proposal = {
            fqn: 0 for fqn in self._sharding_options_by_fqn.keys()
        }

    def _reset(self) -> None:
        self._sharding_options_by_fqn = {}
        self._current_proposal = {}

    def propose(self) -> Optional[List[ShardingOption]]:
        if self._current_proposal:
            return [
                self._sharding_options_by_fqn[fqn][index]
                for fqn, index in self._current_proposal.items()
            ]
        else:
            return None

    def feedback(
        self,
        partitionable: bool,
        plan: Optional[List[ShardingOption]] = None,
        perf_rating: Optional[float] = None,
        storage_constraint: Optional[Topology] = None,
    ) -> None:
        # When threshold is passed, observe the perf_rating trend. If the perf_rating
        # of the newly proposed plans have worse perf_rating, stop proposing.
        if self._threshold and perf_rating:
            self._num_inferior_perf += 1
            if perf_rating < self._best_perf_rating:
                self._best_perf_rating = perf_rating
                self._num_inferior_perf = 0
            # pyre-fixme [58]: `>` is not supported for operand types `int` and `Optional[int]`.
            if self._num_inferior_perf > self._threshold:
                self._current_proposal = {}
                return
        # static strategy, ignore feedback and just provide next proposal
        largest_fqn: Optional[str] = None
        largest_storage: Tuple[float, float, float, float] = (0, 0, 0, 0)
        for fqn, sharding_options in self._sharding_options_by_fqn.items():
            index = self._current_proposal[fqn]
            if index + 1 < len(sharding_options):
                sharding_option = sharding_options[index]
                current_storage = (
                    # pyre-fixme [16]: `Optional` has no attribute `hbm`
                    max([shard.storage.hbm for shard in sharding_option.shards]),
                    sum([shard.storage.hbm for shard in sharding_option.shards]),
                    # pyre-fixme [16]: `Optional` has no attribute `ddr`
                    max([shard.storage.ddr for shard in sharding_option.shards]),
                    sum([shard.storage.ddr for shard in sharding_option.shards]),
                )
                if current_storage > largest_storage:
                    largest_fqn = fqn
                    largest_storage = current_storage

        if largest_fqn is not None:
            self._current_proposal[largest_fqn] += 1
        else:
            self._current_proposal = {}


class UniformProposer(Proposer):
    """
    Proposes uniform sharding plans, plans that have the same sharding type for all
    sharding options.
    """

    def __init__(self, use_depth: bool = True) -> None:
        self._use_depth: bool = use_depth
        self._grouped_sharding_options: List[List[ShardingOption]] = []
        self._proposal_index: int = 0

    def load(
        self,
        search_space: List[ShardingOption],
        enumerator: Optional[Enumerator] = None,
    ) -> None:
        self._reset()
        all_fqns = set()
        sharding_options_by_type_and_fqn: Dict[str, Dict[str, List[ShardingOption]]] = (
            {}
        )

        for sharding_option in search_space:
            sharding_type = sharding_option.sharding_type
            fqn = sharding_option.fqn
            all_fqns.add(fqn)

            if sharding_type not in sharding_options_by_type_and_fqn:
                sharding_options_by_type_and_fqn[sharding_type] = {}
            if fqn not in sharding_options_by_type_and_fqn[sharding_type]:
                sharding_options_by_type_and_fqn[sharding_type][fqn] = []

            sharding_options_by_type_and_fqn[sharding_type][fqn].append(sharding_option)

        for sharding_options_by_fqn in sharding_options_by_type_and_fqn.values():
            for sharding_options in sharding_options_by_fqn.values():
                sharding_options.sort(
                    key=lambda x: _sharding_option_score(x, self._use_depth)
                )

        for sharding_options_by_fqn in sharding_options_by_type_and_fqn.values():
            if sharding_options_by_fqn.keys() == all_fqns:
                self._grouped_sharding_options.append(
                    [
                        sorted_sharding_options[0]
                        for sorted_sharding_options in sharding_options_by_fqn.values()
                    ]
                )

    def _reset(self) -> None:
        self._grouped_sharding_options = []
        self._proposal_index = 0

    def propose(self) -> Optional[List[ShardingOption]]:
        if self._proposal_index < len(self._grouped_sharding_options):
            return self._grouped_sharding_options[self._proposal_index]
        else:
            return None

    def feedback(
        self,
        partitionable: bool,
        plan: Optional[List[ShardingOption]] = None,
        perf_rating: Optional[float] = None,
        storage_constraint: Optional[Topology] = None,
    ) -> None:
        # static strategy, ignore feedback and just provide next proposal
        self._proposal_index += 1


class GridSearchProposer(Proposer):
    def __init__(self, max_proposals: int = MAX_PROPOSALS) -> None:
        self._max_proposals: int = max_proposals
        self._sharding_options_by_fqn: Dict[str, List[ShardingOption]] = {}
        self._proposal_index: int = 0
        self._proposals: List[List[int]] = []

    def load(
        self,
        search_space: List[ShardingOption],
        enumerator: Optional[Enumerator] = None,
    ) -> None:
        self._reset()
        for sharding_option in search_space:
            fqn = sharding_option.fqn
            if fqn not in self._sharding_options_by_fqn:
                self._sharding_options_by_fqn[fqn] = []
            self._sharding_options_by_fqn[fqn].append(sharding_option)

        for sharding_options in self._sharding_options_by_fqn.values():
            sharding_options.sort(key=lambda x: _sharding_option_score(x))

        total_proposals = prod(
            [
                len(sharding_options)
                for sharding_options in self._sharding_options_by_fqn.values()
            ]
        )
        if total_proposals > self._max_proposals:
            total_proposals = (
                "{:.2e}".format(Decimal(total_proposals))
                if total_proposals > 1e6
                else total_proposals
            )
            logger.info(
                "Skipping grid search proposer as there are too many proposals.\n"
                f"Total proposals to search: {total_proposals}\n"
                f"Max proposals allowed: {self._max_proposals}\n"
            )
            return
        sharding_options_by_fqn_indices = [
            range(len(sharding_options))
            for sharding_options in self._sharding_options_by_fqn.values()
        ]
        # pyre-fixme[8]: Attribute has type `List[List[int]]`; used as
        #  `List[Tuple[int]]`.
        self._proposals = list(itertools.product(*sharding_options_by_fqn_indices))

    def _reset(self) -> None:
        self._sharding_options_by_fqn = {}
        self._proposal_index = 0
        self._proposals = []

    def propose(self) -> Optional[List[ShardingOption]]:
        if self._proposals and self._proposal_index < len(self._proposals):
            proposal_indices = self._proposals[self._proposal_index]
            return [
                sharding_options[index]
                for index, sharding_options in zip(
                    proposal_indices, self._sharding_options_by_fqn.values()
                )
            ]
        else:
            return None

    def feedback(
        self,
        partitionable: bool,
        plan: Optional[List[ShardingOption]] = None,
        perf_rating: Optional[float] = None,
        storage_constraint: Optional[Topology] = None,
    ) -> None:
        # static strategy, ignore feedback and just provide next proposal
        self._proposal_index += 1


class EmbeddingOffloadScaleupProposer(Proposer):
    def __init__(self, use_depth: bool = True) -> None:
        self.use_depth: bool = use_depth
        self.enumerator: Optional[Enumerator] = None
        self.starting_proposal: List[ShardingOption] = []
        self.proposal: Optional[List[ShardingOption]] = None
        self.search: Optional[LuusJaakolaSearch] = None

    def load(
        self,
        search_space: List[ShardingOption],
        enumerator: Optional[Enumerator] = None,
    ) -> None:
        self.enumerator = enumerator
        sharding_options_by_fqn: Dict[str, List[ShardingOption]] = {}
        for sharding_option in search_space:
            sharding_options_by_fqn.setdefault(sharding_option.fqn, []).append(
                sharding_option
            )
        for sharding_options in sharding_options_by_fqn.values():
            sharding_options.sort(
                key=lambda x: _sharding_option_score(x, self.use_depth)
            )
        # currently only use 1st sharding option for proposal only.
        # TODO: could traverse through multiple options like GreedyProposer
        proposal = [
            sharding_options[0] for sharding_options in sharding_options_by_fqn.values()
        ]
        # deepcopy so it won't affect other proposers
        self.starting_proposal = copy.deepcopy(proposal)
        self.proposal = copy.deepcopy(self.starting_proposal)

    def propose(self) -> Optional[List[ShardingOption]]:
        return self.proposal

    def feedback(
        self,
        partitionable: bool,
        plan: Optional[List[ShardingOption]] = None,
        perf_rating: Optional[float] = None,
        storage_constraint: Optional[Topology] = None,
    ) -> None:
        if not self.enumerator or plan is None:
            self.proposal = None
            return

        hbm_used_previously = sum(
            sharding_option.total_storage.hbm for sharding_option in plan
        )

        if self.search is None:
            if not partitionable or storage_constraint is None:
                self.proposal = None
                return

            hbm_available = EmbeddingOffloadScaleupProposer.get_budget(
                plan, storage_constraint
            )
            logger.info(
                f"EmbeddingOffloadScaleupProposer - cache scale up budget={round(bytes_to_gb(hbm_available), 2)} GB, exploring [{round(bytes_to_gb(hbm_used_previously), 2)}, {round(bytes_to_gb(hbm_used_previously + hbm_available), 2)}] GB"
            )
            self.search = LuusJaakolaSearch(
                0, hbm_available, max_iterations=16, left_cost=perf_rating
            )

        logger.info(
            f"EmbeddingOffloadScaleupProposer - proposed size={round(bytes_to_gb(hbm_used_previously), 2)} GB, score={perf_rating}"
        )

        assert self.search is not None  # keep pyre happy
        budget = self.search.next(perf_rating or 1e99)
        if budget is not None:
            budget = int(budget)
        self.proposal = EmbeddingOffloadScaleupProposer.next_plan(
            self.starting_proposal, budget, self.enumerator
        )

    @staticmethod
    def get_budget(proposal: List[ShardingOption], storage_constraint: Topology) -> int:
        """returns additional HBM budget available for GPU caches."""
        available_hbm = sum(device.storage.hbm for device in storage_constraint.devices)
        used_hbm = sum(
            sharding_option.total_storage.hbm for sharding_option in proposal
        )
        return available_hbm - used_hbm

    # Given an available budget of additional memory, and a provisional sharding plan,
    # attempt to use the budget wisely to scale up caches that would most benefit from it.
    @staticmethod
    def next_plan(
        starting_proposal: List[ShardingOption],
        budget: Optional[int],
        enumerator: Optional[Enumerator],
    ) -> Optional[List[ShardingOption]]:
        if budget is None or enumerator is None:
            return None

        def none_to_zero(x: Optional[float]) -> float:
            return x if x is not None else 0.0

        proposal = copy.deepcopy(starting_proposal)
        # This is the subset of tables that we can scale
        cache_tables = [
            sharding_option
            for sharding_option in proposal
            if sharding_option.compute_kernel
            == EmbeddingComputeKernel.FUSED_UVM_CACHING.value
            and none_to_zero(
                EmbeddingOffloadScaleupProposer.get_cacheability(sharding_option)
            )
            * none_to_zero(
                EmbeddingOffloadScaleupProposer.get_expected_lookups(sharding_option)
            )
            * none_to_zero(sharding_option.cache_load_factor)
            > 0
        ]
        # Nothing to scale
        if len(cache_tables) == 0:
            return None

        size_model = EmbeddingOffloadScaleupProposer.build_affine_storage_model(
            cache_tables, enumerator
        )
        clfs = torch.tensor(
            [sharding_option.cache_load_factor for sharding_option in cache_tables]
        )
        # cooked_cacheability is cacheability scaled by the expected number of cache
        # lookups.

        cooked_cacheability = torch.tensor(
            [
                none_to_zero(
                    EmbeddingOffloadScaleupProposer.get_cacheability(sharding_option)
                )
                * none_to_zero(
                    EmbeddingOffloadScaleupProposer.get_expected_lookups(
                        sharding_option
                    )
                )
                for sharding_option in cache_tables
            ]
        )
        new_clfs = EmbeddingOffloadScaleupProposer.allocate_budget(
            model=size_model,
            clfs=clfs,
            budget=budget,
            allocation_priority=cooked_cacheability,
        )
        # apply new_clfs, promoting tables that made it to 1.0
        for sharding_option, clf in zip(cache_tables, new_clfs):
            clf = clf.item()  # tensor scalar -> scalar
            assert sharding_option.cache_params  # appease pyre
            sharding_option.cache_params.load_factor = clf
            if clf > 0.9999:  # tolerate float roundoff
                assert sharding_option.cache_params  # appease pyre
                sharding_option.cache_params.load_factor = None
                sharding_option.compute_kernel = EmbeddingComputeKernel.FUSED.value
        # recalculate cost estimates of modified tables
        enumerator.populate_estimates(cache_tables)
        return proposal

    @staticmethod
    def get_cacheability(sharding_option: ShardingOption) -> Optional[float]:
        # helper to appease pyre type checker, as cache_params is Optional it maybe None
        if (
            sharding_option.cache_params is None
            or sharding_option.cache_params.stats is None
        ):
            return None
        return sharding_option.cache_params.stats.cacheability

    @staticmethod
    def get_expected_lookups(sharding_option: ShardingOption) -> Optional[float]:
        # helper to appease pyre type checker, as cache_params is Optional it maybe None
        if (
            sharding_option.cache_params is None
            or sharding_option.cache_params.stats is None
        ):
            return None
        return sharding_option.cache_params.stats.expected_lookups

    # The relationship between clf and shard memory usage is non-linear due to non-clf
    # overheads like optimization stats and input/output storage. We model it as an
    # affine relationship: bytes = clf * A + B where B is fixed overhead independent of
    # CLF (e.g. input / output IO sizes and A is per cache-row overhead.
    @staticmethod
    def build_affine_storage_model(
        uvm_caching_sharding_options: List[ShardingOption], enumerator: Enumerator
    ) -> torch.Tensor:
        plan: List[ShardingOption] = copy.deepcopy(uvm_caching_sharding_options)

        def compute_hbm_sizes(clf: float) -> torch.Tensor:
            for sharding_option in plan:
                assert sharding_option.cache_params  # appease pyre
                sharding_option.cache_params.load_factor = clf
            enumerator.populate_estimates(plan)
            return torch.tensor(
                [sharding_option.total_storage.hbm for sharding_option in plan]
            )

        low_clf, high_clf = 0.1, 0.9
        low_hbms = compute_hbm_sizes(low_clf)
        high_hbms = compute_hbm_sizes(high_clf)

        A = (high_hbms - low_hbms) / (high_clf - low_clf)
        B = low_hbms - A * low_clf
        return torch.stack((A, B), dim=1)  # Nx2 (a,b)

    @staticmethod
    def clf_to_bytes(
        model: torch.Tensor, clfs: Union[float, torch.Tensor]
    ) -> torch.Tensor:
        # evaluate affine model AX + B
        return (model[:, 0] * clfs + model[:, 1]).to(torch.int64)

    # Given a model of an affine system, an existing configuration (clfs), available
    # budget, and an allocation policy, return new configuration that best uses the
    # available budget. We only add additional budget, we assume the existing
    # configuration is specifying a floor or minimum size.
    @staticmethod
    def allocate_budget(
        model: torch.Tensor,
        clfs: torch.Tensor,
        budget: int,
        allocation_priority: torch.Tensor,
    ) -> torch.Tensor:
        # min size is size of table at 0 CLF
        min_size_bytes = EmbeddingOffloadScaleupProposer.clf_to_bytes(model, 0)
        max_size_bytes = EmbeddingOffloadScaleupProposer.clf_to_bytes(model, 1)
        table_size_bytes = EmbeddingOffloadScaleupProposer.clf_to_bytes(model, clfs)
        cache_size_bytes = table_size_bytes - min_size_bytes
        max_cache_size_bytes = max_size_bytes - min_size_bytes

        # We have budget bytes to share across the tables in. We want to increase the
        # cache_size_bytes of each table in proportion to their allocation priority
        # fraction. If we raise the cache_size_bytes to beyond max_cache_size_bytes,
        # this is equivalent to reaching CLF=1.0, so we clip the memory to 1.0, and
        # reassign the released budget in a subsequent pass.
        num_pass = 0
        while budget > 1 and num_pass < 128:
            num_pass += 1
            # mask is False for tables at >= max_size, and True otherwise. This allows
            # us to remove tables that have already reached full size in one round from
            # being dealt more budget in subsequent rounds.
            mask = (min_size_bytes + cache_size_bytes) < max_size_bytes
            if mask.sum() == 0:
                break

            logging.debug(
                f"[allocate_budget] pass={num_pass}, budget={budget}, #cache_tables={mask.sum()}"
            )

            # switch to 64bit float to avoid rounding errors, as table cache sizes can
            # easily be > 2^24.
            masked_priority = (mask * allocation_priority).to(torch.float64)
            increase_ratio = masked_priority / torch.sum(masked_priority)
            proposed_increase_bytes = budget * increase_ratio
            new_cache_size_bytes = torch.minimum(
                cache_size_bytes + proposed_increase_bytes, max_cache_size_bytes
            )
            actual_increase_bytes = new_cache_size_bytes - cache_size_bytes

            budget -= torch.sum(actual_increase_bytes)
            cache_size_bytes = new_cache_size_bytes
            # TODO: consider trade off of using remaining budget to push >0.95 tables
            # to HBM vs spending that budget on improving hit rate on other tables in
            # next pass.

        # cache_size_bytes are the new cache sizes we want to use. We convert them back
        # to clfs by dividing by max_cache_size_bytes, which has isolated the clf
        # portion of the table size from the fixed overheads.
        # convert 64bit values back to original clf precision
        return (cache_size_bytes / max_cache_size_bytes).to(clfs.dtype)


def _sharding_option_score(
    sharding_option: ShardingOption, use_depth: bool = True
) -> float:
    return (
        max([cast(Perf, shard.perf).total for shard in sharding_option.shards])
        if use_depth
        else sum([cast(Perf, shard.perf).total for shard in sharding_option.shards])
    )


def proposers_to_proposals_list(
    proposers_list: List[Proposer], search_space: List[ShardingOption]
) -> List[List[ShardingOption]]:
    """
    only works for static_feedback proposers (the path of proposals to check is independent of the performance of the proposals)
    """

    proposals_list = []

    proposal_cache: Set[Tuple[int, ...]] = set()

    for proposer in proposers_list:
        proposer.load(search_space=search_space)

    for proposer in proposers_list:
        proposal = proposer.propose()

        while proposal:
            proposal_key = tuple(sorted(map(hash, proposal)))
            proposer.feedback(partitionable=True)
            if proposal_key in proposal_cache:
                proposal = proposer.propose()
                continue

            proposals_list.append(proposal)
            proposal_cache.add(proposal_key)
            proposal = proposer.propose()

    return proposals_list
