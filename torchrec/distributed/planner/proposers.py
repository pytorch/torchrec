#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import List, Dict, Tuple, Optional, cast

from torchrec.distributed.planner.types import (
    Proposer,
    ShardingOption,
)


class GreedyProposer(Proposer):
    def __init__(self, use_depth: bool = True) -> None:
        self._use_depth: bool = use_depth
        self._sharding_options_by_fqn: Dict[
            str, List[Tuple[float, ShardingOption]]
        ] = {}
        self._current_proposal: Dict[str, int] = {}

    def load(self, search_space: List[ShardingOption]) -> None:
        for sharding_option in search_space:
            fqn = sharding_option.fqn
            score = self._sharding_option_score(sharding_option)
            if fqn not in self._sharding_options_by_fqn:
                self._sharding_options_by_fqn[fqn] = []
            self._sharding_options_by_fqn[fqn].append((score, sharding_option))

        for score_shard_tuple in self._sharding_options_by_fqn.values():
            score_shard_tuple.sort(key=lambda x: x[0])

        self._current_proposal = {
            fqn: 0 for fqn in self._sharding_options_by_fqn.keys()
        }

    def _sharding_option_score(self, sharding_option: ShardingOption) -> float:
        return (
            max([cast(float, shard.perf) for shard in sharding_option.shards])
            if self._use_depth
            else sum([cast(float, shard.perf) for shard in sharding_option.shards])
        )

    def propose(self) -> Optional[List[ShardingOption]]:
        if self._current_proposal:
            return copy.deepcopy(
                [
                    self._sharding_options_by_fqn[fqn][index][1]
                    for fqn, index in self._current_proposal.items()
                ]
            )
        else:
            return None

    def feedback(
        self,
        partitionable: bool,
        plan: Optional[List[ShardingOption]] = None,
        perf_rating: Optional[float] = None,
    ) -> None:
        # static strategy, ignore feedback and just provide next proposal
        largest_fqn: Optional[str] = None
        largest_storage: Tuple[float, float, float, float] = (0, 0, 0, 0)
        for fqn, score_shard_tuples in self._sharding_options_by_fqn.items():
            index = self._current_proposal[fqn]
            if index + 1 < len(score_shard_tuples):
                sharding_option = score_shard_tuples[index][1]
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
