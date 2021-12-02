#!/usr/bin/env python3

import copy
from typing import Dict, Optional, List, cast, Union

import torch.distributed as dist
from torch import nn
from torch.distributed._sharding_spec import EnumerableShardingSpec, ShardMetadata
from torchrec.distributed.collective_utils import (
    invoke_on_rank_and_broadcast_result,
)
from torchrec.distributed.planner.new.constants import MAX_SIZE
from torchrec.distributed.planner.new.enumerators import EmbeddingEnumerator
from torchrec.distributed.planner.new.partitioners import GreedyCostPartitioner
from torchrec.distributed.planner.new.perf_models import NoopPerfModel
from torchrec.distributed.planner.new.proposers import GreedyProposer
from torchrec.distributed.planner.new.stats import EmbeddingStats
from torchrec.distributed.planner.new.storage_reservations import (
    FixedPercentageReservation,
)
from torchrec.distributed.planner.new.types import (
    PlannerConstraints,
    InputStats,
    Partitioner,
    Topology,
    Stats,
    Shard,
    ShardingOption,
    StorageReservation,
    Enumerator,
    Proposer,
    PerfModel,
    PlannerError,
)
from torchrec.distributed.types import (
    ShardingPlan,
    ShardingPlanner,
    ModuleSharder,
    ShardingType,
    ParameterSharding,
)


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


class EmbeddingShardingPlanner(ShardingPlanner):
    def __init__(
        self,
        topology: Topology,
        enumerator: Optional[Enumerator] = None,
        storage_reservation: Optional[StorageReservation] = None,
        proposer: Optional[Union[Proposer, List[Proposer]]] = None,
        partitioner: Optional[Partitioner] = None,
        performance_model: Optional[PerfModel] = None,
        stats: Optional[Stats] = None,
        constraints: Optional[Dict[str, PlannerConstraints]] = None,
        input_stats: Optional[Dict[str, InputStats]] = None,
    ) -> None:
        self._topology = topology
        self._input_stats = input_stats

        self._enumerator: Enumerator = (
            enumerator
            if enumerator
            else EmbeddingEnumerator(
                topology=topology,
                constraints=constraints,
                input_stats=input_stats,
            )
        )
        self._storage_reservation: StorageReservation = (
            storage_reservation
            if storage_reservation
            else FixedPercentageReservation(percentage=0.4)
        )
        self._partitioner: Partitioner = (
            partitioner if partitioner else GreedyCostPartitioner()
        )
        if proposer:
            self._proposers: List[Proposer] = (
                [proposer] if not isinstance(proposer, list) else proposer
            )
        else:
            self._proposers = [GreedyProposer(), GreedyProposer(use_depth=False)]
        self._perf_model: PerfModel = (
            performance_model if performance_model else NoopPerfModel(topology=topology)
        )
        self._stats: Stats = stats if stats else EmbeddingStats()
        self._num_proposals: int = 0
        self._num_plans: int = 0

    def collective_plan(
        self,
        module: nn.Module,
        sharders: List[ModuleSharder[nn.Module]],
        pg: dist.ProcessGroup,
    ) -> ShardingPlan:
        """
        Call self.plan(...) on rank 0 and broadcast
        """
        return invoke_on_rank_and_broadcast_result(
            pg,
            0,
            self.plan,
            module,
            sharders,
        )

    def plan(
        self,
        module: nn.Module,
        sharders: List[ModuleSharder[nn.Module]],
    ) -> ShardingPlan:

        best_plan = None
        best_perf_rating = MAX_SIZE

        storage_constraint = self._storage_reservation.reserve(
            topology=self._topology,
            module=module,
            sharders=sharders,
        )

        search_space = self._enumerator.enumerate(
            module=module,
            sharders=sharders,
        )

        for proposer in self._proposers:
            proposer.load(search_space=search_space)

        for proposer in self._proposers:
            proposal = proposer.propose()
            while proposal:
                self._num_proposals += 1
                try:
                    plan = self._partitioner.partition(
                        proposal=proposal,
                        storage_constraint=storage_constraint,
                    )
                    self._num_plans += 1
                    perf_rating = self._perf_model.rate(plan=plan)
                    if perf_rating < best_perf_rating:
                        best_perf_rating = perf_rating
                        best_plan = plan
                    proposer.feedback(
                        partitionable=True, plan=plan, perf_rating=perf_rating
                    )
                except PlannerError:
                    proposer.feedback(partitionable=False)

                proposal = proposer.propose()

        if best_plan:
            sharding_plan = _to_sharding_plan(best_plan, self._topology)

            self._stats.log(
                sharding_plan=sharding_plan,
                topology=self._topology,
                num_proposals=self._num_proposals,
                num_plans=self._num_plans,
                best_plan=best_plan,
                input_stats=self._input_stats,
            )
            return sharding_plan
        else:
            raise PlannerError(
                f"Unable to find a plan for this model are evaluating {self._num_proposals} proposals.\n"
                "Possible solutions:\n"
                "  1) Increase the number of devices\n"
                "  2) Reduce the model size\n"
                "  3) Reduce batch size\n"
                "  4) Remove planner constraints that might be reducing search space\n"
            )
