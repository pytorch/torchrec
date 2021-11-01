#!/usr/bin/env python3

from typing import List, Dict, Set

from torchrec.distributed.planner.new.types import (
    Ranker,
    RankStack,
    CostCalc,
    ShardingOption,
)


class FlatRankStack(RankStack):
    """
    RankStack which orders sharding options on cost alone  --
    it does not consider module hierarchy or sharding type (ie. 'flat' ranker).

    This implementation does not attempt to apply costs on push().
    """

    def __init__(
        self,
        calculator: CostCalc,
        sharding_options: List[ShardingOption],
    ) -> None:
        self._calculator = calculator
        fqns = {sharding_option.fqn for sharding_option in sharding_options}

        self._options_by_fqn: Dict[str, List[ShardingOption]] = {}
        self._valid_fqns: Set[str] = fqns
        self._remaining_fqns: Set[str] = set()
        self._ordered_fqns: List[str] = []
        for fqn in fqns:
            self._options_by_fqn[fqn] = []

        for sharding_option in sharding_options:
            self._calculator.run(sharding_option=sharding_option)
        self.bulk_push(sharding_options)

    def bulk_push(self, sharding_options: List[ShardingOption]) -> None:
        seen_fqns = set()
        for sharding_option in sharding_options:
            seen_fqns.add(sharding_option.fqn)
            self._push(sharding_option)

        for fqn in seen_fqns:
            self._options_by_fqn[fqn].sort(key=lambda x: -x.cost)
        self._reorder()

    def push(self, sharding_option: ShardingOption) -> None:
        self._push(sharding_option)
        self._options_by_fqn[sharding_option.fqn].sort(key=lambda x: -x.cost)
        self._reorder()

    def _push(self, sharding_option: ShardingOption) -> None:
        fqn = sharding_option.fqn
        assert fqn in self._valid_fqns, f"Attempt to push unknown tensor {fqn}"
        self._remaining_fqns.add(fqn)
        self._options_by_fqn[fqn].append(sharding_option)

    def _reorder(self) -> None:
        options = [
            fqn_options[-1]
            for fqn, fqn_options in self._options_by_fqn.items()
            if fqn in self._remaining_fqns
        ]
        options.sort(key=lambda x: -x.cost)
        self._ordered_fqns = [option.fqn for option in options]

    def pop(self) -> ShardingOption:
        fqn = self._ordered_fqns.pop()
        sharding_option = self._options_by_fqn[fqn].pop()
        self._remaining_fqns.remove(fqn)
        return sharding_option

    def bulk_pop(self) -> List[ShardingOption]:
        sharding_options = []
        num_fqns = len(self._ordered_fqns)
        for _ in range(num_fqns):
            sharding_options.append(self.pop())
        return sharding_options

    def remove(self, sharding_option: ShardingOption) -> bool:
        fqn = sharding_option.fqn
        assert fqn in self._valid_fqns, f"Attempt to remove unknown tensor {fqn}"
        # check if another alternative option exists, if not return false
        if not self._options_by_fqn[fqn]:
            return False
        self._remaining_fqns.add(fqn)
        self._reorder()
        return True

    def __len__(self) -> int:
        return len(self._remaining_fqns)


class FlatRanker(Ranker):
    def __init__(self, calculator: CostCalc) -> None:
        self._calculator = calculator

    def run(self, sharding_options: List[ShardingOption]) -> RankStack:
        return FlatRankStack(
            calculator=self._calculator,
            sharding_options=sharding_options,
        )
