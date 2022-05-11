#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, List

from torchrec.distributed.planner.types import PerfModel, ShardingOption, Topology


class NoopPerfModel(PerfModel):
    def __init__(self, topology: Topology) -> None:
        self._topology = topology

    def rate(self, plan: List[ShardingOption]) -> float:
        perfs = [0] * self._topology.world_size
        for sharding_option in plan:
            for shard in sharding_option.shards:
                # pyre-ignore [6]: Expected `typing_extensions.SupportsIndex`
                perfs[shard.rank] += cast(float, shard.perf)

        return max(perfs)
