#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
from decimal import Decimal
from typing import cast, Dict, List, Optional, Set, Tuple

from torchrec.distributed.planner.types import Proposer, ShardingOption
from torchrec.distributed.planner.utils import prod

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
            `max(shard.perf)`, otherwise sharding_options are sorted by `sum(shard.perf)`.
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

    def load(self, search_space: List[ShardingOption]) -> None:
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

    def load(self, search_space: List[ShardingOption]) -> None:
        self._reset()
        all_fqns = set()
        sharding_options_by_type_and_fqn: Dict[
            str, Dict[str, List[ShardingOption]]
        ] = {}

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
    ) -> None:
        # static strategy, ignore feedback and just provide next proposal
        self._proposal_index += 1


class GridSearchProposer(Proposer):
    def __init__(self, max_proposals: int = MAX_PROPOSALS) -> None:
        self._max_proposals: int = max_proposals
        self._sharding_options_by_fqn: Dict[str, List[ShardingOption]] = {}
        self._proposal_index: int = 0
        self._proposals: List[List[int]] = []

    def load(self, search_space: List[ShardingOption]) -> None:
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
    ) -> None:
        # static strategy, ignore feedback and just provide next proposal
        self._proposal_index += 1


def _sharding_option_score(
    sharding_option: ShardingOption, use_depth: bool = True
) -> float:
    return (
        max([cast(float, shard.perf) for shard in sharding_option.shards])
        if use_depth
        else sum([cast(float, shard.perf) for shard in sharding_option.shards])
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
