#!/usr/bin/env python3

import heapq
import logging
from collections import deque
from typing import Dict, Optional, List, Callable, Tuple, Any

import torch
import torch.distributed as dist
from torch import nn
from torchrec.distributed.collective_utils import (
    invoke_on_rank_and_broadcast_result,
)
from torchrec.distributed.comm import get_local_size
from torchrec.distributed.planner.cost_functions import (
    cost_func_compute_based,
)
from torchrec.distributed.planner.types import (
    ShardingOption,
    ParameterInfo,
    ParameterHints,
    ParameterInputStats,
    CostInput,
    Topology,
    ParamSortKey,
)
from torchrec.distributed.planner.utils import (
    sharder_name,
    get_topology,
    is_enough_storage,
    allocate_param,
    deallocate_param,
    param_sort_key,
    to_plan,
    bytes_to_gb,
)
from torchrec.distributed.types import (
    ShardingPlan,
    ShardingPlanner,
    ModuleSharder,
    ShardingType,
    ParameterSharding,
)


logger: logging.Logger = logging.getLogger(__name__)


class EmbeddingShardingPlanner(ShardingPlanner):
    def __init__(
        self,
        pg: dist.ProcessGroup,
        device: torch.device,
        hints: Optional[Dict[str, ParameterHints]] = None,
        input_stats: Optional[Dict[str, ParameterInputStats]] = None,
        storage: Optional[Dict[str, int]] = None,
        cost_functions: Optional[List[Callable[[CostInput], int]]] = None,
    ) -> None:
        self._world_size: int = dist.get_world_size(pg)
        self._local_size: int = get_local_size()
        self._hints: Dict[str, ParameterHints] = hints if hints else {}
        self._input_stats: Dict[str, ParameterInputStats] = (
            input_stats if input_stats else {}
        )
        self._pg = pg
        self._device = device

        if cost_functions is None:
            self._cost_functions: List[Callable[[CostInput], int]] = [
                cost_func_compute_based
            ]
        else:
            self._cost_functions = cost_functions

        self._topology: Topology = get_topology(pg, device, storage)
        self._counter: int = 1

    def collective_plan(
        self,
        module: nn.Module,
        sharders: List[ModuleSharder[nn.Module]],
    ) -> ShardingPlan:
        """
        Call self.plan(...) on rank 0 and broadcast
        """
        return invoke_on_rank_and_broadcast_result(
            self._pg,
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
        Algorithm

        Each parameter has a set of sharding options, ordered in terms of compute cost (lowest to highest)

        Using the first sharding option for each parameter, the planner attempts to place the parameters in a
        greedy fashion by placing the highest compute cost parameter remaining on the lowest total cost device.

        In event that planner hits a global storage constraint, the planner with remove the sharding option of
        the parameter with the highest storage cost; and retry same greedy approach.  Typically removing a
        sharding option with a high storage cost will reduce storage cost but increase compute cost for a given
        parameter.

        If no solution found using this approach, planner will fail.  This search is not exhaustive,
        so it does not mean a solution is not possible.

        """
        param_infos = self._get_param_infos(
            module=module,
            sharders=sharders,
        )
        unplaced_param_infos: List[Tuple[ParamSortKey, ParameterInfo]] = [
            (param_sort_key(param_info, self._world_size), param_info)
            for param_info in param_infos
        ]
        placed_param_infos: List[Tuple[ParamSortKey, ParameterInfo]] = []

        heapq.heapify(unplaced_param_infos)
        while unplaced_param_infos:
            if not self._place(unplaced_param_infos, placed_param_infos):
                self._counter += 1
                self._backtrack(unplaced_param_infos, placed_param_infos)

        sharding_plan = to_plan(
            param_infos,
            self._device,
            self._world_size,
            self._local_size,
        )
        self._log_stats(
            sharding_plan=sharding_plan,
            param_infos=param_infos,
        )
        return sharding_plan

    def _log_stats(
        self, sharding_plan: ShardingPlan, param_infos: List[ParameterInfo]
    ) -> None:
        """
        Builds Stats, then logs out results
        """

        # Data structures to read and store statistics
        shard_by_fqn = {
            module_name + "." + param_name: value
            for module_name, param_dict in sharding_plan.plan.items()
            for param_name, value in param_dict.items()
        }
        stats: Dict[int, Dict[str, Any]] = {
            rank: {"type": {}, "pooling_factor": 0.0, "emb_dims": 0}
            for rank in range(self._world_size)
        }

        # Populate stats table
        for param_info in param_infos:
            name = param_info.name
            fqn = param_info.fqn
            shard: ParameterSharding = shard_by_fqn[fqn]
            input_stats = self._input_stats.get(name, None)
            ranks = list(range(self._world_size))
            pooling_factor = [
                sum(input_stats.mean)
                if input_stats and isinstance(input_stats.mean, list)
                else 0.0
            ]
            emb_dims = [param_info.param.shape[1]]
            if shard.sharding_type == ShardingType.ROW_WISE.value:
                pooling_factor = [pooling_factor[0] / self._world_size] * len(ranks)
                emb_dims = emb_dims * len(ranks)
            elif shard.sharding_type == ShardingType.TABLE_ROW_WISE.value:
                # pyre-ignore [16]
                host_id = shard.ranks[0] // self._local_size
                ranks = list(
                    range(host_id * self._local_size, (host_id + 1) * self._local_size)
                )
                pooling_factor = [pooling_factor[0] / self._local_size] * len(ranks)
                emb_dims = emb_dims * len(ranks)
            elif shard.sharding_type == ShardingType.COLUMN_WISE.value:
                ranks = shard.ranks
                emb_dims = [
                    shard.shard_lengths[1]
                    # pyre-ignore [16]
                    for shard in shard.sharding_spec.shards
                ]
                # pyre-ignore [6]
                pooling_factor = pooling_factor * len(ranks)
            elif shard.sharding_type == ShardingType.TABLE_WISE.value:
                ranks = shard.ranks
            else:  # DATA PARALLEL
                emb_dims = emb_dims * len(ranks)
                pooling_factor = pooling_factor * len(ranks)

            # pyre-ignore [6]
            for i, rank in enumerate(ranks):
                count = stats[rank]["type"].get(shard.sharding_type, 0)
                stats[rank]["type"][shard.sharding_type] = count + 1
                stats[rank]["pooling_factor"] += pooling_factor[i]
                stats[rank]["emb_dims"] += emb_dims[i]

        # Log out results
        logger.info(f"------ {self.__class__.__name__} Statistics ------")
        for rank in range(self._world_size):
            host = self._topology.get_host(rank)
            device = self._topology.get_device(rank)
            logger.info(
                f"  Rank {rank} -- "
                f"HBM/DDR: {bytes_to_gb(device.hbm.capacity - device.hbm.free):.1f}/"
                f"{bytes_to_gb(host.ddr.capacity - host.ddr.free):.1f}, "
                f"Cost: {device.total_cost}, "
                f"Mean Pooling: {int(stats[rank]['pooling_factor'])}, "
                f"Emb Dims: {stats[rank]['emb_dims']}, "
                f"Shards: {stats[rank]['type']}"
            )
        logger.info(
            f"------ Executed {self._counter} iteration(s) to find a solution ------"
        )

    def _place(
        self,
        unplaced_param_infos: List[Tuple[ParamSortKey, ParameterInfo]],
        placed_param_infos: List[Tuple[ParamSortKey, ParameterInfo]],
    ) -> bool:
        """
        Places parameters until all parameters are placed, or a storage contraint is hit
        """
        candidate_devices = [
            self._topology.get_device(rank) for rank in range(self._world_size)
        ]
        heapq.heapify(candidate_devices)
        sort_key, param_info = heapq.heappop(unplaced_param_infos)
        sharding_option = param_info.sharding_options[0]

        is_placed = False
        if sharding_option.sharding_type == ShardingType.TABLE_WISE.value:
            constrained_devices = []
            ranks = []
            while candidate_devices:
                candidate_device = heapq.heappop(candidate_devices)
                if is_enough_storage(sharding_option, self._topology, candidate_device):
                    ranks.append(candidate_device.rank)
                    sharding_option.ranks = ranks
                    allocate_param(sharding_option, self._topology)
                    heapq.heappush(candidate_devices, candidate_device)
                    heapq.heappush(
                        placed_param_infos,
                        (
                            param_sort_key(param_info, self._world_size, "storage"),
                            param_info,
                        ),
                    )
                    is_placed = True
                    break
                constrained_devices.append(candidate_device)

            for constrained_device in constrained_devices:
                heapq.heappush(candidate_devices, constrained_device)
        elif sharding_option.sharding_type == ShardingType.COLUMN_WISE.value:
            constrained_devices = []
            ranks = []
            while candidate_devices:
                candidate_device = heapq.heappop(candidate_devices)
                if is_enough_storage(sharding_option, self._topology, candidate_device):
                    ranks.append(candidate_device.rank)
                    sharding_option.ranks = ranks
                    allocate_param(sharding_option, self._topology)
                    heapq.heappush(candidate_devices, candidate_device)
                    if len(ranks) == sharding_option._num_col_wise_shards:
                        heapq.heappush(
                            placed_param_infos,
                            (
                                param_sort_key(param_info, self._world_size, "storage"),
                                param_info,
                            ),
                        )
                        is_placed = True
                        break
                constrained_devices.append(candidate_device)

            for constrained_device in constrained_devices:
                heapq.heappush(candidate_devices, constrained_device)
        elif sharding_option.sharding_type == ShardingType.TABLE_ROW_WISE.value:
            num_hosts = len(self._topology.hosts)
            devices_per_host = len(self._topology.hosts[0].devices)
            candidate_hosts = [0] * num_hosts
            constrained_devices = []
            ranks = []
            while candidate_devices:
                candidate_device = heapq.heappop(candidate_devices)
                host_idx, _ = self._topology.host_and_device_by_rank[
                    candidate_device.rank
                ]
                candidate_hosts[host_idx] += 1
                if candidate_hosts[host_idx] == devices_per_host and is_enough_storage(
                    sharding_option, self._topology, candidate_device
                ):
                    ranks.append(candidate_device.rank)
                    sharding_option.ranks = ranks
                    allocate_param(sharding_option, self._topology)
                    heapq.heappush(
                        placed_param_infos,
                        (
                            param_sort_key(param_info, self._world_size, "storage"),
                            param_info,
                        ),
                    )
                    heapq.heappush(candidate_devices, candidate_device)
                    is_placed = True
                    break
                constrained_devices.append(candidate_device)

            for constrained_device in constrained_devices:
                heapq.heappush(candidate_devices, constrained_device)

        elif sharding_option.sharding_type in [
            ShardingType.DATA_PARALLEL.value,
            ShardingType.ROW_WISE.value,
        ]:
            if is_enough_storage(sharding_option, self._topology):
                sharding_option.ranks = None
                allocate_param(sharding_option, self._topology)
                heapq.heappush(
                    placed_param_infos,
                    (
                        param_sort_key(param_info, self._world_size, "storage"),
                        param_info,
                    ),
                )
                is_placed = True
        else:
            raise ValueError(
                f"{self.__class__.__name__} does not support {sharding_option.sharding_type}"
            )

        if not is_placed:
            heapq.heappush(unplaced_param_infos, (sort_key, param_info))

        return is_placed

    def _backtrack(
        self,
        unplaced_param_infos: List[Tuple[ParamSortKey, ParameterInfo]],
        placed_param_infos: List[Tuple[ParamSortKey, ParameterInfo]],
    ) -> None:
        """
        Called when the planner hits a storage constraint.  A single sharding option is discarded,
        and then reset state such that _place method can recalled.  If no there are no available
        sharding options to discard an error will be raised.
        """

        is_option_discarded = False
        _, param_info = heapq.heappop(unplaced_param_infos)

        # Temporarily place param_info into solution set
        heapq.heappush(
            placed_param_infos,
            (
                param_sort_key(param_info, self._world_size, "storage"),
                param_info,
            ),
        )
        while placed_param_infos:
            (_, placed_param_info) = heapq.heappop(placed_param_infos)

            # Deallocate in necessary
            if placed_param_info is not param_info:
                deallocate_param(placed_param_info.sharding_options[0], self._topology)

            # Discard sharding option from first parameter with more than one sharding option
            if len(placed_param_info.sharding_options) > 1 and not is_option_discarded:
                placed_param_info.sharding_options.popleft()
                is_option_discarded = True
            heapq.heappush(
                unplaced_param_infos,
                (
                    param_sort_key(placed_param_info, self._world_size),
                    placed_param_info,
                ),
            )
        if not is_option_discarded:
            raise RuntimeError(
                f"------ {self.__class__.__name__} is unable to find a plan for model. ------\n"
                "Try: \n"
                "  1) Increasing the number of devices\n"
                "  2) Reducing the model size\n"
                "  3) Removing sharding hints that may reduce search space\n"
                f"------ attempted {self._counter} iteration(s))  ------"
            )

    def _get_param_infos(
        self,
        module: nn.Module,
        sharders: List[ModuleSharder[nn.Module]],
    ) -> List[ParameterInfo]:
        sharder_map: Dict[str, ModuleSharder[nn.Module]] = {
            sharder_name(sharder.module_type): sharder for sharder in sharders
        }
        param_infos: List[ParameterInfo] = []

        for child_path, child_module in module.named_modules():
            sharder_key = sharder_name(type(child_module))
            sharder = sharder_map.get(sharder_key, None)
            if not sharder:
                continue

            for name, param in sharder.shardable_parameters(child_module).items():
                sharding_options = []
                for sharding_type in self._filter_sharding_types(
                    name, sharder.sharding_types
                ):
                    num_col_wise_shards, shard_size = self._get_num_col_wise_shards(
                        name, param, sharding_type
                    )
                    for compute_kernel in self._filter_compute_kernels(
                        name, sharder.compute_kernels(sharding_type, self._device)
                    ):
                        cost_input = CostInput(
                            param=param,
                            device=self._device,
                            compute_kernel=compute_kernel,
                            sharding_type=sharding_type,
                            input_stats=self._input_stats.get(name, None),
                        )
                        cost = sum(
                            [
                                cost_function(cost_input)
                                for cost_function in self._cost_functions
                            ]
                        )
                        sharding_options.append(
                            ShardingOption(
                                cost=cost,
                                sharding_type=sharding_type,
                                compute_kernel=compute_kernel,
                                storage_usage=sharder.storage_usage(
                                    param, self._device, compute_kernel
                                ),
                                _num_col_wise_shards=num_col_wise_shards,
                                col_wise_shard_dim=shard_size,
                            )
                        )
                param_infos.append(
                    ParameterInfo(
                        param=param,
                        name=name,
                        prefix=child_path,
                        sharding_options=deque(sorted(sharding_options)),
                    )
                )
        return param_infos

    def _filter_sharding_types(self, name: str, sharding_types: List[str]) -> List[str]:
        hint = self._hints.get(name, None)
        if not hint or not hint.sharding_types:
            return sharding_types
        sharding_types = list(
            set(hint.sharding_types).intersection(set(sharding_types))
        )
        if not sharding_types:
            raise RuntimeError(
                f"No available sharding types after applying hints for {name}"
            )
        return sharding_types

    def _filter_compute_kernels(
        self, name: str, compute_kernels: List[str]
    ) -> List[str]:
        hint = self._hints.get(name, None)
        if not hint or not hint.compute_kernels:
            return compute_kernels
        compute_kernels = list(
            set(hint.compute_kernels).intersection(set(compute_kernels))
        )
        if not compute_kernels:
            raise RuntimeError(
                f"No available compute kernels after applying hints for {name}"
            )
        return compute_kernels

    def _get_num_col_wise_shards(
        self, name: str, param: torch.Tensor, sharding_type: str
    ) -> Tuple[Optional[int], Optional[int]]:
        num_col_wise_shards = None
        col_wise_shard_dim = None
        if sharding_type == ShardingType.COLUMN_WISE.value:
            _hint = self._hints.get(name, None)
            col_wise_shard_dim_hint = (
                None if _hint is None else _hint.col_wise_shard_dim
            )
            col_wise_shard_dim = (
                col_wise_shard_dim_hint
                if col_wise_shard_dim_hint is not None
                else param.shape[1]
            )
            # column-wise shard the weights
            num_col_wise_shards, residual = divmod(param.shape[1], col_wise_shard_dim)
            if residual > 0:
                num_col_wise_shards += 1
        elif sharding_type == ShardingType.TABLE_WISE.value:
            num_col_wise_shards = 1
        return num_col_wise_shards, col_wise_shard_dim
