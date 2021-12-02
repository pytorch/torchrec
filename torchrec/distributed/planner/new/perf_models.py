#!/usr/bin/env python3

from typing import List, cast

from torchrec.distributed.planner.new.types import PerfModel, Topology, ShardingOption


class NoopPerfModel(PerfModel):
    def __init__(self, topology: Topology) -> None:
        self._topology = topology

    def rate(self, plan: List[ShardingOption]) -> float:
        prefs = [0] * self._topology.world_size
        for sharding_option in plan:
            for shard in sharding_option.shards:
                # pyre-ignore [6]: Expected `typing_extensions.SupportsIndex`
                prefs[shard.rank] += cast(float, shard.cost)

        return max(prefs)
