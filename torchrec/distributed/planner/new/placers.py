#!/usr/bin/env python3

import copy
from typing import List, Tuple, cast

from torch.distributed._sharding_spec import EnumerableShardingSpec, ShardMetadata
from torchrec.distributed.planner.new.constants import MAX_SIZE
from torchrec.distributed.planner.new.types import (
    Partitioner,
    Topology,
    ShardingOption,
    Placer,
    RankStack,
    PartitionError,
)
from torchrec.distributed.types import ShardingPlan, ParameterSharding, ShardingType


def _to_sharding_plan(
    sharding_options: List[ShardingOption],
    topology: Topology,
) -> ShardingPlan:
    def _placement(
        compute_device: str,
        rank: int,
        local_size: int,
    ) -> str:
        param_device = compute_device
        if compute_device == "cuda":
            param_device = f"cuda:{rank % local_size}"
        return f"rank:{rank}/{param_device}"

    compute_device = topology.compute_device
    local_size = topology.local_world_size

    plan = {}
    for sharding_option in sharding_options:
        module_plan = plan.get(sharding_option.path, {})
        module_plan[sharding_option.name] = ParameterSharding(
            sharding_spec=None
            if sharding_option.sharding_type == ShardingType.DATA_PARALLEL.value
            else EnumerableShardingSpec(
                [
                    ShardMetadata(
                        shard_lengths=shard.length,
                        shard_offsets=shard.offset,
                        placement=_placement(
                            compute_device, cast(int, shard.rank), local_size
                        ),
                    )
                    for shard in sharding_option.shards
                ]
            ),
            sharding_type=sharding_option.sharding_type,
            compute_kernel=sharding_option.compute_kernel,
            ranks=[cast(int, shard.rank) for shard in sharding_option.shards],
        )
        plan[sharding_option.path] = module_plan
    return ShardingPlan(plan)


class EmbeddingPlacer(Placer):
    def __init__(self, topology: Topology, partitioner: Partitioner) -> None:
        self._topology = topology
        self._partitioner = partitioner
        self._counter = 0

    def run(self, rank_stack: RankStack) -> ShardingPlan:
        sharding_solution = None
        min_cost = MAX_SIZE
        while rank_stack:
            sharding_options = rank_stack.bulk_pop()
            try:
                sharding_candidate, topology_candidate = self._partition(
                    sharding_options
                )
                cost = max([device.cost for device in topology_candidate.devices])
                if cost < min_cost:
                    sharding_solution = sharding_candidate
                    min_cost = cost
            except PartitionError:
                pass
            self._counter += 1
            self._backtrack(rank_stack, sharding_options)

        if sharding_solution:
            return _to_sharding_plan(
                sharding_options=sharding_solution, topology=self._topology
            )
        else:
            raise PartitionError(
                "Unable to find a plan for this model. Possible solutions:\n"
                "  1) Increase the number of devices\n"
                "  2) Reduce the model size\n"
                "  3) Remove planner constraints that might be reducing search space\n"
                f"------ attempted {self._counter} iteration(s))  ------"
            )

    def _partition(
        self, sharding_options: List[ShardingOption]
    ) -> Tuple[List[ShardingOption], Topology]:
        # create a working copy of topology and candidate
        topology_candidate = copy.deepcopy(self._topology)
        sharding_candidate = copy.deepcopy(sharding_options)
        self._partitioner.run(
            sharding_options=sharding_candidate,
            topology=topology_candidate,
        )
        return sharding_candidate, topology_candidate

    def _backtrack(
        self, rank_stack: RankStack, sharding_options: List[ShardingOption]
    ) -> None:
        # attempt to remove sharding option with highest single shard storage cost
        sharding_options.sort(
            key=lambda x: (
                sum([shard.storage.hbm for shard in x.shards]),
                max([shard.storage.hbm for shard in x.shards]),
                sum([shard.storage.ddr for shard in x.shards]),
                max([shard.storage.ddr for shard in x.shards]),
            ),
            reverse=True,
        )
        idx = 0
        for sharding_option in sharding_options:
            if rank_stack.remove(sharding_option):
                break
            idx += 1

        if idx < len(sharding_options):
            del sharding_options[idx]
            rank_stack.bulk_push(sharding_options)
