#!/usr/bin/env python3

import copy
from typing import Optional, List, Tuple, cast

from torch.distributed._sharding_spec import EnumerableShardingSpec, ShardMetadata
from torchrec.distributed.planner.new.constants import MAX_SIZE
from torchrec.distributed.planner.new.types import (
    Partitioner,
    Topology,
    ShardingOption,
    Placer,
    RankStack,
    PartitionError,
    PlacerStats,
    Shard,
    Ranker,
)
from torchrec.distributed.types import ShardingPlan, ParameterSharding, ShardingType


def _merge_shards_by_dim(shards: List[Shard], dim: int) -> List[Shard]:
    # merges shards down to one per rank along dimension.
    # Will recompute shard offsets
    merged_shards = []
    shards = sorted(shards, key=lambda x: x.rank)

    current_rank = -1
    current_shard: Optional[Shard] = None
    current_dim_offset = 0
    for shard in shards:
        if shard.rank != current_rank:
            current_shard = copy.deepcopy(shard)
            current_shard.offset[dim] = current_dim_offset
            merged_shards.append(current_shard)
            current_rank = shard.rank
        else:
            # pyre-ignore [16]
            current_shard.length[dim] += shard.length[dim]
            # pyre-ignore [16]
            current_shard.storage += shard.storage
            # pyre-ignore [16]
            current_shard.cost += shard.cost
        current_dim_offset += shard.length[dim]
    return merged_shards


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
        shards = sharding_option.shards
        sharding_type = sharding_option.sharding_type
        if sharding_type == ShardingType.COLUMN_WISE.value:
            shards = _merge_shards_by_dim(shards, 1)
            if len(shards) == 1:
                sharding_type = ShardingType.TABLE_WISE.value

        module_plan = plan.get(sharding_option.path, {})
        module_plan[sharding_option.name] = ParameterSharding(
            sharding_spec=None
            if sharding_type == ShardingType.DATA_PARALLEL.value
            else EnumerableShardingSpec(
                [
                    ShardMetadata(
                        shard_sizes=shard.length,
                        shard_offsets=shard.offset,
                        placement=_placement(
                            compute_device, cast(int, shard.rank), local_size
                        ),
                    )
                    for shard in shards
                ]
            ),
            sharding_type=sharding_type,
            compute_kernel=sharding_option.compute_kernel,
            ranks=[cast(int, shard.rank) for shard in shards],
        )
        plan[sharding_option.path] = module_plan
    return ShardingPlan(plan)


class EmbeddingPlacer(Placer):
    def __init__(
        self,
        topology: Topology,
        partitioners: List[Partitioner],
        rankers: List[Ranker],
    ) -> None:
        self._topology = topology
        self._partitioners = partitioners
        self._rankers = rankers
        self._sharding_solution: Optional[List[ShardingOption]] = None
        self._topology_solution: Optional[Topology] = None
        self._counter = 0
        self._num_errors = 0

    def run(self, sharding_options: List[ShardingOption]) -> ShardingPlan:
        min_cost = MAX_SIZE
        sharding_solution = None
        topology_solution = None
        for ranker in self._rankers:
            rank_stack = ranker.run(copy.deepcopy(sharding_options))
            cur_sharding_options = rank_stack.bulk_pop()
            while cur_sharding_options:
                for partitioner in self._partitioners:
                    try:
                        sharding_candidate, topology_candidate = self._partition(
                            cur_sharding_options, partitioner
                        )
                        cost = max(
                            [device.cost for device in topology_candidate.devices]
                        )
                        if cost < min_cost:
                            sharding_solution = sharding_candidate
                            topology_solution = topology_candidate
                            min_cost = cost
                    except PartitionError:
                        self._num_errors += 1

                    self._counter += 1
                cur_sharding_options = self._backtrack(rank_stack, cur_sharding_options)

        if sharding_solution:
            self._sharding_solution = sharding_solution
            self._topology_solution = topology_solution
            return _to_sharding_plan(
                sharding_options=sharding_solution, topology=self._topology
            )
        else:
            raise PartitionError(
                "Unable to find a plan for this model. Possible solutions:\n"
                "  1) Increase the number of devices\n"
                "  2) Reduce the model size\n"
                "  3) Reduce batch size\n"
                "  4) Remove planner constraints that might be reducing search space\n"
                f"------ attempted {self._counter} iteration(s) ------\n"
            )

    @property
    def stats(self) -> PlacerStats:
        return PlacerStats(
            num_iterations=self._counter,
            num_errors=self._num_errors,
            topology_solution=self._topology_solution,
            sharding_solution=self._sharding_solution,
        )

    def _partition(
        self, sharding_options: List[ShardingOption], partitioner: Partitioner
    ) -> Tuple[List[ShardingOption], Topology]:
        # create a working copy of topology and candidate
        topology_candidate = copy.deepcopy(self._topology)
        sharding_candidate = copy.deepcopy(sharding_options)
        partitioner.run(
            sharding_options=sharding_candidate,
            topology=topology_candidate,
        )
        return sharding_candidate, topology_candidate

    def _backtrack(
        self, rank_stack: RankStack, sharding_options: List[ShardingOption]
    ) -> List[ShardingOption]:
        # attempt to remove sharding option with highest single shard storage cost
        sharding_options.sort(
            key=lambda x: (
                max([shard.storage.hbm for shard in x.shards]),
                sum([shard.storage.hbm for shard in x.shards]),
                max([shard.storage.ddr for shard in x.shards]),
                sum([shard.storage.ddr for shard in x.shards]),
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
            sharding_options.append(rank_stack.pop())
            return sharding_options

        return []
