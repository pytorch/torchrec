#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
from functools import reduce
from time import perf_counter
from typing import Callable, cast, Dict, List, Optional, Tuple, Union

import torch

import torch.distributed as dist
from torch import nn
from torchrec.distributed.collective_utils import invoke_on_rank_and_broadcast_result
from torchrec.distributed.comm import get_local_size
from torchrec.distributed.planner.constants import BATCH_SIZE, MAX_SIZE
from torchrec.distributed.planner.enumerators import EmbeddingEnumerator
from torchrec.distributed.planner.partitioners import (
    GreedyPerfPartitioner,
    MemoryBalancedPartitioner,
)
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
from torchrec.distributed.utils import none_throws


def to_sharding_plan(
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
            sharding_spec=(
                None
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
                )
            ),
            sharding_type=sharding_type,
            compute_kernel=sharding_option.compute_kernel,
            ranks=[cast(int, shard.rank) for shard in shards],
            cache_params=sharding_option.cache_params,
            enforce_hbm=sharding_option.enforce_hbm,
            stochastic_rounding=sharding_option.stochastic_rounding,
            bounds_check_mode=sharding_option.bounds_check_mode,
            output_dtype=sharding_option.output_dtype,
            key_value_params=sharding_option.key_value_params,
        )
        plan[sharding_option.path] = module_plan
    return ShardingPlan(plan)


def _merge_plans(best_plans: List[ShardingPlan]) -> ShardingPlan:
    if len(best_plans) == 1:
        return best_plans[0]
    else:
        merged_plan = ShardingPlan({})
        for plan in best_plans:
            for name, ps in plan.plan.items():
                ps = cast(EmbeddingModuleShardingPlan, ps)
                if name not in merged_plan.plan:
                    merged_plan.plan[name] = ps
                else:
                    for k, v in ps.items():
                        cur_plan = cast(
                            EmbeddingModuleShardingPlan, merged_plan.plan[name]
                        )
                        if k not in cur_plan:
                            cur_plan[k] = v
                        else:
                            raise PlannerError(
                                "table can not be sharded between two device group"
                            )

        return merged_plan


class EmbeddingShardingPlanner(ShardingPlanner):
    """
    Provides an optimized sharding plan for a given module with shardable parameters
    according to the provided sharders, topology, and constraints.

    Args:
        topology (Optional[Topology]): the topology of the current process group.
        batch_size (Optional[int]): the batch size of the model.
        enumerator (Optional[Enumerator]): the enumerator to use
        storage_reservation (Optional[StorageReservation]): the storage reservation to use
        proposer (Optional[Union[Proposer, List[Proposer]]]): the proposer(s) to use
        partitioner (Optional[Partitioner]): the partitioner to use
        performance_model (Optional[PerfModel]): the performance model to use
        stats (Optional[Union[Stats, List[Stats]]]): the stats to use
        constraints (Optional[Dict[str, ParameterConstraints]]): per table constraints
            for sharding.
        debug (bool): whether to print debug information.

    Example::

        ebc = EmbeddingBagCollection(tables=eb_configs, device=torch.device("meta"))
        planner = EmbeddingShardingPlanner()
        plan = planner.plan(
            module=ebc,
            sharders=[EmbeddingBagCollectionSharder()],
        )

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
        callbacks: Optional[
            List[Callable[[List[ShardingOption]], List[ShardingOption]]]
        ] = None,
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
        self._callbacks: List[
            Callable[[List[ShardingOption]], List[ShardingOption]]
        ] = ([] if callbacks is None else callbacks)

    def collective_plan(
        self,
        module: nn.Module,
        sharders: Optional[List[ModuleSharder[nn.Module]]] = None,
        pg: Optional[dist.ProcessGroup] = None,
    ) -> ShardingPlan:
        """
        Call self.plan(...) on rank 0 and broadcast

        Args:
            module (nn.Module): the module to shard.
            sharders (Optional[List[ModuleSharder[nn.Module]]]): the sharders to use for sharding
            pg (Optional[dist.ProcessGroup]): the process group to use for collective operations

        Returns:
            ShardingPlan: the sharding plan for the module.
        """
        if pg is None:
            assert dist.is_initialized(), (
                "The default process group is not yet initialized. "
                "Please call torch.distributed.init_process_group() first before invoking this. "
                "If you are not within a distributed environment, use the single rank version plan() instead."
            )
            pg = none_throws(dist.GroupMember.WORLD)

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
        """
        Provides an optimized sharding plan for a given module with shardable parameters
        according to the provided sharders, topology, and constraints.

        Args:
            module (nn.Module): the module to shard.
            sharders (List[ModuleSharder[nn.Module]]): the sharders to use for sharding.

        Returns:
            ShardingPlan: the sharding plan for the module.
        """
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
                    proposal_cache[proposal_key] = (False, proposal, None)
                    proposer.feedback(
                        partitionable=False,
                        plan=proposal,
                        storage_constraint=storage_constraint,
                    )

                # clear shard.rank for each sharding_option
                reset_shard_rank(proposal)
                proposal = proposer.propose()

        if best_plan:
            for callback in self._callbacks:
                best_plan = callback(best_plan)

            self._best_plan = best_plan
            sharding_plan = to_sharding_plan(best_plan, self._topology)

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


class HeteroEmbeddingShardingPlanner(ShardingPlanner):
    """
    Provides an optimized sharding plan for a given module with shardable parameters
    according to the provided sharders, topology, and constraints.
    """

    def __init__(
        self,
        topology_groups: Optional[Dict[str, Topology]] = None,
        batch_size: Optional[int] = None,
        enumerators: Optional[Dict[str, Enumerator]] = None,
        storage_reservations: Optional[Dict[str, StorageReservation]] = None,
        proposers: Optional[Dict[str, Union[Proposer, List[Proposer]]]] = None,
        partitioners: Optional[Dict[str, Partitioner]] = None,
        performance_models: Optional[Dict[str, PerfModel]] = None,
        stats: Optional[Dict[str, Union[Stats, List[Stats]]]] = None,
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
        debug: bool = True,
        callbacks: Optional[
            List[Callable[[List[ShardingOption]], List[ShardingOption]]]
        ] = None,
    ) -> None:
        default_device = "cuda" if torch.cuda.is_available() else "cpu"
        if topology_groups is None:
            topology_groups = {
                default_device: Topology(
                    local_world_size=get_local_size(),
                    world_size=dist.get_world_size(),
                    compute_device=default_device,
                )
            }
        self._topology_groups: Dict[str, Topology] = topology_groups
        self._batch_size: int = batch_size if batch_size else BATCH_SIZE
        self._constraints = constraints
        # pyre-ignore
        self._enumerators: Dict[str, Enumerator] = (
            enumerators
            if enumerators
            else {
                group: EmbeddingEnumerator(
                    topology=self._topology_groups[group],
                    batch_size=self._batch_size,
                    constraints=constraints,
                    use_exact_enumerate_order=True,
                )
                for group in self._topology_groups.keys()
            }
        )
        # pyre-ignore
        self._storage_reservations: Dict[str, StorageReservation] = (
            storage_reservations
            if storage_reservations
            else {
                group: HeuristicalStorageReservation(percentage=0.15)
                for group in self._topology_groups.keys()
            }
        )

        # pyre-ignore
        self._partitioners: Dict[str, Partitioner] = (
            partitioners
            if partitioners
            else {
                group: MemoryBalancedPartitioner()
                for group in self._topology_groups.keys()
            }
        )

        if proposers:
            # pyre-ignore
            self._proposers: Dict[str, List[Proposer]] = proposers
        else:
            # pyre-ignore
            self._proposers = {
                group: [
                    GridSearchProposer(),
                    GreedyProposer(),
                    GreedyProposer(use_depth=False),
                    UniformProposer(),
                ]
                for group in self._topology_groups.keys()
            }

        # pyre-ignore
        self._perf_models: Dict[str, PerfModel] = (
            performance_models
            if performance_models
            else {
                group: NoopPerfModel(topology=self._topology_groups[group])
                for group in self._topology_groups
            }
        )

        self._stats: Dict[str, List[Stats]] = {}

        if stats is not None:
            # pyre-ignore [8]
            self._stats = stats
        else:
            # pyre-ignore [8]
            self._stats = {
                group: [EmbeddingStats()] for group in self._topology_groups.keys()
            }

        self._debug = debug
        self._num_proposals: int = 0
        self._num_plans: int = 0
        self._best_plan: Optional[List[ShardingOption]] = None
        self._callbacks: List[
            Callable[[List[ShardingOption]], List[ShardingOption]]
        ] = ([] if callbacks is None else callbacks)

    def collective_plan(
        self,
        module: nn.Module,
        sharders: Optional[List[ModuleSharder[nn.Module]]] = None,
        pg: Optional[dist.ProcessGroup] = dist.GroupMember.WORLD,
    ) -> ShardingPlan:
        """
        Call self.plan(...) on rank 0 and broadcast
        """
        if pg is None:
            assert dist.is_initialized(), (
                "The default process group is not yet initialized. "
                "Please call torch.distributed.init_process_group() first before invoking this. "
                "If you are not within a distributed environment, use the single rank version plan() instead."
            )
            pg = none_throws(dist.GroupMember.WORLD)
        assert len(self._topology_groups) == 1, "Only single topology is supported"

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
        best_plans: List[ShardingPlan] = []
        for group, topology in self._topology_groups.items():
            self._num_proposals = 0
            self._num_plans = 0
            start_time = perf_counter()
            best_plan = None
            lowest_storage = Storage(MAX_SIZE, MAX_SIZE)
            last_planner_error: Optional[PlannerError] = None
            last_proposal: List[ShardingOption] = []
            best_perf_rating = MAX_SIZE

            storage_constraint: Topology = self._storage_reservations[group].reserve(
                topology=topology,
                batch_size=self._batch_size,
                module=module,
                sharders=sharders,
                constraints=self._constraints,
            )

            search_space = self._enumerators[group].enumerate(
                module=module,
                sharders=sharders,
            )

            # filter by device group
            search_space = [
                s_o
                for s_o in search_space
                # pyre-ignore [16]
                if self._constraints[s_o.name].device_group == group
            ]

            if not search_space:
                # No shardable parameters
                continue

            proposal_cache: Dict[
                Tuple[int, ...],
                Tuple[bool, Optional[List[ShardingOption]], Optional[float]],
            ] = {}

            for proposer in self._proposers[group]:
                proposer.load(
                    search_space=search_space, enumerator=self._enumerators[group]
                )

            for proposer in self._proposers[group]:
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
                        plan = self._partitioners[group].partition(
                            proposal=proposal,
                            storage_constraint=storage_constraint,
                        )
                        self._num_plans += 1
                        perf_rating = self._perf_models[group].rate(plan=plan)
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
                        proposal_cache[proposal_key] = (False, proposal, None)
                        proposer.feedback(
                            partitionable=False,
                            plan=proposal,
                            storage_constraint=storage_constraint,
                        )

                    # clear shard.rank for each sharding_option
                    reset_shard_rank(proposal)
                    proposal = proposer.propose()

            if best_plan:
                for callback in self._callbacks:
                    best_plan = callback(best_plan)

                self._best_plan = best_plan
                sharding_plan = to_sharding_plan(
                    best_plan, self._topology_groups[group]
                )
                best_plans.append(sharding_plan)

                end_time = perf_counter()
                for stats in self._stats[group]:
                    stats.log(
                        sharding_plan=sharding_plan,
                        topology=self._topology_groups[group],
                        batch_size=self._batch_size,
                        storage_reservation=self._storage_reservations[group],
                        num_proposals=self._num_proposals,
                        num_plans=self._num_plans,
                        run_time=end_time - start_time,
                        best_plan=best_plan,
                        constraints=self._constraints,
                        sharders=sharders,
                        debug=self._debug,
                    )
            else:
                global_storage_capacity = reduce(
                    lambda x, y: x + y,
                    [device.storage for device in self._topology_groups[group].devices],
                )
                global_storage_constraints = reduce(
                    lambda x, y: x + y,
                    [device.storage for device in storage_constraint.devices],
                )
                storage_reservation_solution = (
                    (
                        # pyre-ignore [16]
                        f"\n\t  Storage reservation percentage: {self._storage_reservations[group]._percentage}, "
                        f"\n\t  Per rank reservation for dense storage: {storage_repr_in_gb(self._storage_reservations[group]._dense_storage)}, "
                        f"\n\t  Per rank reservation for kjt storage: {storage_repr_in_gb(self._storage_reservations[group]._kjt_storage)}, "  # pyre-ignore[16]
                    )
                    if isinstance(
                        self._storage_reservations[group], HeuristicalStorageReservation
                    )
                    else f"\n\t  Storage reservation percentage: {self._storage_reservations[group]._percentage}, "
                )
                no_plan_solution = (
                    f"Planner evaluated {self._num_proposals} proposals for device group {group}."
                    "\nPossible solutions:"
                    f"\n  1) Increase the number of devices ({self._topology_groups[group].world_size})"
                    f"\n  2) Reduce the model size ("
                    f"\n\t  Global storage: {round(bytes_to_gb(global_storage_capacity.hbm), 3)} GB, "
                    f"\n\t  Per rank hardware memory: {storage_repr_in_gb(self._topology_groups[group].devices[0].storage)}, "
                    f"{storage_reservation_solution}"
                    f"\n\t  Global storage available for model parallel: {storage_repr_in_gb(global_storage_constraints)}, "
                    f"\n\t  Global storage requirement for model parallel: {storage_repr_in_gb(lowest_storage)})"
                    f"\n  3) Reduce local batch size ({self._batch_size})"
                    "\n  4) Remove planner constraints that might be reducing search space or available storage\n"
                )
                last_planner_error_info = (
                    f"Last planner error: \n\t{last_planner_error}\n"
                )

                # printout stats for no plan situation
                end_time = perf_counter()
                sharding_plan = ShardingPlan(plan={})
                # force all shards to have rank= -1
                for sharding_option in last_proposal:
                    for shard in sharding_option.shards:
                        shard.rank = -1

                for stats in self._stats[group]:
                    stats.log(
                        sharding_plan=sharding_plan,
                        topology=self._topology_groups[group],
                        batch_size=self._batch_size,
                        storage_reservation=self._storage_reservations[group],
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
                        message=f"Unable to find a plan for this model in device_group {group}  because of insufficient storage. \n"
                        + no_plan_solution
                        + last_planner_error_info,
                    )
                else:
                    raise PlannerError(
                        error_type=PlannerErrorType.STRICT_CONSTRAINTS,
                        message=f"Unable to find a plan for this model in device_group {group} because of the strict constraints. \n"
                        + no_plan_solution
                        + last_planner_error_info,
                    )

        return _merge_plans(best_plans)
