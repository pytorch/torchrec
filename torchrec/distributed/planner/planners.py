#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from functools import reduce
from time import perf_counter
from typing import cast, Dict, List, Optional, Tuple, Union

import torch

import torch.distributed as dist
from torch import nn
from torchrec.distributed.collective_utils import invoke_on_rank_and_broadcast_result
from torchrec.distributed.comm import get_local_size
from torchrec.distributed.planner.constants import BATCH_SIZE, MAX_SIZE
from torchrec.distributed.planner.enumerators import EmbeddingEnumerator
from torchrec.distributed.planner.partitioners import GreedyPerfPartitioner
from torchrec.distributed.planner.perf_models import NoopPerfModel
from torchrec.distributed.planner.proposers import (
    GreedyProposer,
    GridSearchProposer,
    UniformProposer,
)
from torchrec.distributed.planner.stats import EmbeddingStats
from torchrec.distributed.planner.storage_reservations import (
    HeuristicalStorageReservation,
)
from torchrec.distributed.planner.types import (
    Enumerator,
    ParameterConstraints,
    Partitioner,
    PerfModel,
    PlannerError,
    PlannerErrorType,
    Proposer,
    ShardingOption,
    Stats,
    Storage,
    StorageReservation,
    Topology,
)
from torchrec.distributed.planner.utils import (
    bytes_to_gb,
    reset_shard_rank,
    storage_repr_in_gb,
)
from torchrec.distributed.sharding_plan import get_default_sharders, placement
from torchrec.distributed.types import (
    EmbeddingModuleShardingPlan,
    EnumerableShardingSpec,
    ModuleSharder,
    ParameterSharding,
    ShardingPlan,
    ShardingPlanner,
    ShardingType,
    ShardMetadata,
)


def _to_sharding_plan(
    sharding_options: List[ShardingOption],
    topology: Topology,
) -> ShardingPlan:

    compute_device = topology.compute_device
    local_size = topology.local_world_size

    plan = {}
    for sharding_option in sharding_options:
        shards = sharding_option.shards
        sharding_type = sharding_option.sharding_type

        module_plan = plan.get(sharding_option.path, EmbeddingModuleShardingPlan())
        module_plan[sharding_option.name] = ParameterSharding(
            sharding_spec=None
            if sharding_type == ShardingType.DATA_PARALLEL.value
            else EnumerableShardingSpec(
                [
                    ShardMetadata(
                        shard_sizes=shard.size,
                        shard_offsets=shard.offset,
                        placement=placement(
                            compute_device, cast(int, shard.rank), local_size
                        ),
                    )
                    for shard in shards
                ]
            ),
            sharding_type=sharding_type,
            compute_kernel=sharding_option.compute_kernel,
            ranks=[cast(int, shard.rank) for shard in shards],
            cache_params=sharding_option.cache_params,
            enforce_hbm=sharding_option.enforce_hbm,
            stochastic_rounding=sharding_option.stochastic_rounding,
            bounds_check_mode=sharding_option.bounds_check_mode,
        )
        plan[sharding_option.path] = module_plan
    return ShardingPlan(plan)


class EmbeddingShardingPlanner(ShardingPlanner):
    """
    Provides an optimized sharding plan for a given module with shardable parameters
    according to the provided sharders, topology, and constraints.
    """

    def __init__(
        self,
        topology: Optional[Topology] = None,
        batch_size: Optional[int] = None,
        enumerator: Optional[Enumerator] = None,
        storage_reservation: Optional[StorageReservation] = None,
        proposer: Optional[Union[Proposer, List[Proposer]]] = None,
        partitioner: Optional[Partitioner] = None,
        performance_model: Optional[PerfModel] = None,
        stats: Optional[Union[Stats, List[Stats]]] = None,
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
        debug: bool = True,
    ) -> None:
        if topology is None:
            topology = Topology(
                local_world_size=get_local_size(),
                world_size=dist.get_world_size(),
                compute_device="cuda" if torch.cuda.is_available() else "cpu",
            )
        self._topology: Topology = topology
        self._batch_size: int = batch_size if batch_size else BATCH_SIZE
        self._constraints = constraints
        self._enumerator: Enumerator = (
            enumerator
            if enumerator
            else EmbeddingEnumerator(
                topology=topology,
                batch_size=self._batch_size,
                constraints=constraints,
            )
        )
        self._storage_reservation: StorageReservation = (
            storage_reservation
            if storage_reservation
            else HeuristicalStorageReservation(percentage=0.15)
        )
        self._partitioner: Partitioner = (
            partitioner if partitioner else GreedyPerfPartitioner()
        )
        if proposer:
            self._proposers: List[Proposer] = (
                [proposer] if not isinstance(proposer, list) else proposer
            )
        else:
            self._proposers = [
                GridSearchProposer(),
                GreedyProposer(),
                GreedyProposer(use_depth=False),
                UniformProposer(),
            ]
        self._perf_model: PerfModel = (
            performance_model if performance_model else NoopPerfModel(topology=topology)
        )

        if stats is not None:
            self._stats: List[Stats] = [stats] if not isinstance(stats, list) else stats
        else:
            self._stats = [EmbeddingStats()]

        self._debug = debug
        self._num_proposals: int = 0
        self._num_plans: int = 0
        self._best_plan: Optional[List[ShardingOption]] = None

    def collective_plan(
        self,
        module: nn.Module,
        sharders: Optional[List[ModuleSharder[nn.Module]]] = None,
        pg: Optional[dist.ProcessGroup] = dist.GroupMember.WORLD,
    ) -> ShardingPlan:
        """
        Call self.plan(...) on rank 0 and broadcast
        """
        assert pg is not None, "Process group is not initialized"
        if sharders is None:
            sharders = get_default_sharders()
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

        self._num_proposals = 0
        self._num_plans = 0
        start_time = perf_counter()
        best_plan = None
        lowest_storage = Storage(MAX_SIZE, MAX_SIZE)
        last_planner_error: Optional[PlannerError] = None
        last_proposal: List[ShardingOption] = []
        best_perf_rating = MAX_SIZE

        storage_constraint: Topology = self._storage_reservation.reserve(
            topology=self._topology,
            batch_size=self._batch_size,
            module=module,
            sharders=sharders,
            constraints=self._constraints,
        )

        search_space = self._enumerator.enumerate(
            module=module,
            sharders=sharders,
        )
        if not search_space:
            # No shardable parameters
            return ShardingPlan({})

        proposal_cache: Dict[
            Tuple[int, ...],
            Tuple[bool, Optional[List[ShardingOption]], Optional[float]],
        ] = {}

        for proposer in self._proposers:
            proposer.load(search_space=search_space, enumerator=self._enumerator)

        for proposer in self._proposers:
            proposal = proposer.propose()

            while proposal:
                proposal_key = tuple(sorted(map(hash, proposal)))
                if proposal_key in proposal_cache:
                    partitionable, plan, perf_rating = proposal_cache[proposal_key]
                    proposer.feedback(
                        partitionable=partitionable,
                        plan=plan,
                        perf_rating=perf_rating,
                        storage_constraint=storage_constraint,
                    )
                    proposal = proposer.propose()
                    continue

                self._num_proposals += 1
                try:
                    # plan is just proposal where shard.rank is populated
                    plan = self._partitioner.partition(
                        proposal=proposal,
                        storage_constraint=storage_constraint,
                    )
                    self._num_plans += 1
                    perf_rating = self._perf_model.rate(plan=plan)
                    if perf_rating < best_perf_rating:
                        best_perf_rating = perf_rating
                        best_plan = copy.deepcopy(plan)
                    proposal_cache[proposal_key] = (True, plan, perf_rating)
                    proposer.feedback(
                        partitionable=True,
                        plan=plan,
                        perf_rating=perf_rating,
                        storage_constraint=storage_constraint,
                    )
                except PlannerError as planner_error:
                    last_planner_error = planner_error
                    # shallow copy of the proposal
                    last_proposal: List[ShardingOption] = copy.copy(proposal)
                    current_storage = cast(
                        Storage,
                        reduce(
                            lambda x, y: x + y,
                            [
                                shard.storage
                                for option in proposal
                                for shard in option.shards
                            ],
                        ),
                    )
                    if current_storage < lowest_storage:
                        lowest_storage = current_storage
                    proposal_cache[proposal_key] = (False, None, None)
                    proposer.feedback(
                        partitionable=False, storage_constraint=storage_constraint
                    )

                # clear shard.rank for each sharding_option
                reset_shard_rank(proposal)
                proposal = proposer.propose()

        if best_plan:
            self._best_plan = best_plan
            sharding_plan = _to_sharding_plan(best_plan, self._topology)

            end_time = perf_counter()
            for stats in self._stats:
                stats.log(
                    sharding_plan=sharding_plan,
                    topology=self._topology,
                    batch_size=self._batch_size,
                    storage_reservation=self._storage_reservation,
                    num_proposals=self._num_proposals,
                    num_plans=self._num_plans,
                    run_time=end_time - start_time,
                    best_plan=best_plan,
                    constraints=self._constraints,
                    sharders=sharders,
                    debug=self._debug,
                )
            return sharding_plan
        else:
            global_storage_capacity = reduce(
                lambda x, y: x + y,
                [device.storage for device in self._topology.devices],
            )
            global_storage_constraints = reduce(
                lambda x, y: x + y,
                [device.storage for device in storage_constraint.devices],
            )
            storage_reservation_solution = (
                (
                    f"\n\t  Storage reservation percentage: {self._storage_reservation._percentage}, "
                    f"\n\t  Per rank reservation for dense storage: {storage_repr_in_gb(self._storage_reservation._dense_storage)}, "
                    f"\n\t  Per rank reservation for kjt storage: {storage_repr_in_gb(self._storage_reservation._kjt_storage)}, "  # pyre-ignore[16]
                )
                if isinstance(self._storage_reservation, HeuristicalStorageReservation)
                else f"\n\t  Storage reservation percentage: {self._storage_reservation._percentage}, "  # pyre-ignore[16]
            )
            no_plan_solution = (
                f"Planner evaluated {self._num_proposals} proposals."
                "\nPossible solutions:"
                f"\n  1) Increase the number of devices ({self._topology.world_size})"
                f"\n  2) Reduce the model size ("
                f"\n\t  Global storage: {round(bytes_to_gb(global_storage_capacity.hbm), 3)} GB, "
                f"\n\t  Per rank hardware memory: {storage_repr_in_gb(self._topology.devices[0].storage)}, "
                f"{storage_reservation_solution}"
                f"\n\t  Global storage available for model parallel: {storage_repr_in_gb(global_storage_constraints)}, "
                f"\n\t  Global storage requirement for model parallel: {storage_repr_in_gb(lowest_storage)})"
                f"\n  3) Reduce local batch size ({self._batch_size})"
                "\n  4) Remove planner constraints that might be reducing search space or available storage\n"
            )
            last_planner_error_info = f"Last planner error: \n\t{last_planner_error}\n"

            # printout stats for no plan situation
            end_time = perf_counter()
            sharding_plan = ShardingPlan(plan={})
            # force all shards to have rank= -1
            for sharding_option in last_proposal:
                for shard in sharding_option.shards:
                    shard.rank = -1

            for stats in self._stats:
                stats.log(
                    sharding_plan=sharding_plan,
                    topology=self._topology,
                    batch_size=self._batch_size,
                    storage_reservation=self._storage_reservation,
                    num_proposals=self._num_proposals,
                    num_plans=self._num_plans,
                    run_time=end_time - start_time,
                    best_plan=last_proposal,
                    constraints=self._constraints,
                    sharders=sharders,
                    debug=self._debug,
                )

            if not lowest_storage.fits_in(global_storage_constraints):
                raise PlannerError(
                    error_type=PlannerErrorType.INSUFFICIENT_STORAGE,
                    message="Unable to find a plan for this model because of insufficient storage. \n"
                    + no_plan_solution
                    + last_planner_error_info,
                )
            else:
                raise PlannerError(
                    error_type=PlannerErrorType.STRICT_CONSTRAINTS,
                    message="Unable to find a plan for this model because of the strict constraints. \n"
                    + no_plan_solution
                    + last_planner_error_info,
                )
