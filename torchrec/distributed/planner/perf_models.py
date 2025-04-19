#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections import defaultdict
from typing import cast, DefaultDict, List, Optional

from torchrec.distributed.planner.types import (
    Perf,
    PerfModel,
    ShardingOption,
    Storage,
    Topology,
)


class NoopPerfModel(PerfModel):
    """
    A no-op model that returns the maximum perf among all shards. Here, no-op
    means we estimate the performance of a model without actually running it.
    """

    def __init__(self, topology: Topology) -> None:
        self._topology = topology

    def rate(self, plan: List[ShardingOption]) -> float:
        perfs = [0] * self._topology.world_size
        for sharding_option in plan:
            for shard in sharding_option.shards:
                # pyre-ignore [6]: Expected `typing_extensions.SupportsIndex`
                perfs[shard.rank] += cast(Perf, shard.perf).total

        return max(perfs)


class NoopStorageModel(PerfModel):
    """
    A no-op model that returns the maximum hbm usage among all shards. Here, no-op
    means we estimate the performance of a model without actually running it.
    """

    def __init__(self, topology: Topology) -> None:
        self._topology = topology

    def rate(self, plan: List[ShardingOption]) -> float:
        hbms = [0] * self._topology.world_size
        for sharding_option in plan:
            for shard in sharding_option.shards:
                # pyre-ignore [6]: Expected `typing_extensions.SupportsIndex`
                hbms[shard.rank] += cast(Storage, shard.storage).hbm

        return max(hbms)


class NoopCriticalPathPerfModel(PerfModel):
    """
    Models the critical path of the sparse arch. Makes the following assumptions:

        1. There is a synchronization point across the ranks after each of the 4 events: Fwd/Bwd x Comms/Comp.
        2. There could be additional synchronization points across ranks during communication (both fwd & bwd)
        3. There could be additional synchronization points across ranks during computation (both fwd & bwd)

    Args:
        topology (Topology): System topology.
        comms_group_keys (Optional[List[str]]): Additional synchronization points for communication. For example, if we assume that ranks
            synchronize after each module and sharding type operation, then this would be ["module", "sharding_type"].
        comp_group_keys (Optional[List[str]]): Additional synchronization points for computation. For example, if we assume that ranks
            synchronize after each module and sharding type operation, then this would be ["module", "sharding_type"].
    """

    def __init__(
        self,
        topology: Topology,
        comms_group_keys: Optional[List[str]] = None,
        comp_group_keys: Optional[List[str]] = None,
    ) -> None:
        self._topology = topology
        self.comms_group_keys: List[str] = comms_group_keys if comms_group_keys else []
        self.comp_group_keys: List[str] = comp_group_keys if comp_group_keys else []

    def rate(self, plan: List[ShardingOption]) -> float:
        comms_data_fwd = defaultdict(lambda: defaultdict(float))
        comms_data_bwd = defaultdict(lambda: defaultdict(float))
        comp_data_fwd = defaultdict(lambda: defaultdict(float))
        comp_data_bwd = defaultdict(lambda: defaultdict(float))
        for so in plan:
            if len(self.comms_group_keys) == 0:
                comms_aggregation_group = ["default"]
            else:
                comms_aggregation_group = [
                    getattr(so, key) for key in self.comms_group_keys
                ]
            if len(self.comp_group_keys) == 0:
                comp_aggregation_group = ["default"]
            else:
                comp_aggregation_group = [
                    getattr(so, key) for key in self.comp_group_keys
                ]
            for shard in so.shards:
                rank = cast(int, shard.rank)
                perf = cast(Perf, shard.perf)
                comms_data_fwd[tuple(comms_aggregation_group)][rank] += perf.fwd_comms
                comms_data_bwd[tuple(comms_aggregation_group)][rank] += perf.bwd_comms
                comp_data_fwd[tuple(comp_aggregation_group)][rank] += perf.fwd_compute
                comp_data_bwd[tuple(comp_aggregation_group)][rank] += perf.bwd_compute

        # Compute the cost by looking at the summing up the max cost across all ranks for each synchronization point
        def _compute_aggregated_cost(
            d: DefaultDict[tuple[str, ...], DefaultDict[int, float]]
        ) -> float:
            return sum(
                {
                    outer_key: max(inner_dict.values())
                    for outer_key, inner_dict in d.items()
                }.values()
            )

        comms_fwd_cost = _compute_aggregated_cost(comms_data_fwd)
        comms_bwd_cost = _compute_aggregated_cost(comms_data_bwd)
        comp_fwd_cost = _compute_aggregated_cost(comp_data_fwd)
        comp_bwd_sum = _compute_aggregated_cost(comp_data_bwd)

        return comms_fwd_cost + comp_fwd_cost + comms_bwd_cost + comp_bwd_sum
