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

import torch.distributed as dist
from torch import nn
from torchrec.distributed.collective_utils import invoke_on_rank_and_broadcast_result
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
from torchrec.distributed.sharding_plan import placement
from torchrec.distributed.types import (
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

        module_plan = plan.get(sharding_option.path, {})
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
        topology: Topology,
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
        self._topology = topology
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

        if stats:
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

        self._num_proposals = 0
        self._num_plans = 0
        start_time = perf_counter()
        best_plan = None
        lowest_storage = Storage(MAX_SIZE, MAX_SIZE)
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
            proposer.load(search_space=search_space)

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
                    )
                    proposal = proposer.propose()
                    continue

                self._num_proposals += 1
                try:
                    plan = self._partitioner.partition(
                        proposal=copy.deepcopy(proposal),
                        storage_constraint=storage_constraint,
                    )
                    self._num_plans += 1
                    perf_rating = self._perf_model.rate(plan=plan)
                    if perf_rating < best_perf_rating:
                        best_perf_rating = perf_rating
                        best_plan = plan
                    proposal_cache[proposal_key] = (True, plan, perf_rating)
                    proposer.feedback(
                        partitionable=True, plan=plan, perf_rating=perf_rating
                    )
                except PlannerError:
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
                    proposer.feedback(partitionable=False)

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
            no_plan_solution = (
                f"Planner evaluated {self._num_proposals} proposals."
                "\nPossible solutions:"
                f"\n  1) Increase the number of devices ({self._topology.world_size})"
                f"\n  2) Reduce the model size ("
                f"\n\t  Global storage: {global_storage_capacity.hbm}, "
                f"\n\t  Available for model parallel: {global_storage_constraints},"
                f"\n\t  Requirement for model parallel: {lowest_storage})"
                f"\n  3) Reduce local batch size ({self._batch_size})"
                "\n  4) Remove planner constraints that might be reducing search space or available storage\n"
            )
            if not lowest_storage.fits_in(global_storage_constraints):
                raise PlannerError(
                    error_type=PlannerErrorType.INSUFFICIENT_STORAGE,
                    message="Unable to find a plan for this model because of insufficient storage. \n"
                    + no_plan_solution,
                )
            else:
                raise PlannerError(
                    error_type=PlannerErrorType.STRICT_CONSTRAINTS,
                    message="Unable to find a plan for this model because of the strict constraints. \n"
                    + no_plan_solution,
                )
