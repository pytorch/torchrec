#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import heapq
import itertools
import logging
from dataclasses import dataclass
from enum import Enum
from typing import cast, Dict, List, Optional

from torchrec.distributed.planner.perf_models import NoopPerfModel

from torchrec.distributed.planner.types import (
    DeviceHardware,
    PartitionByType,
    Partitioner,
    Perf,
    PerfModel,
    PlannerError,
    PlannerErrorType,
    ShardingOption,
    Storage,
    Topology,
)
from torchrec.distributed.planner.utils import bytes_to_gb, reset_shard_rank
from torchrec.distributed.types import ShardingType

logger: logging.Logger = logging.getLogger(__name__)


def _sort_devices_by_perf(
    devices: List[List[DeviceHardware]],
) -> List[List[DeviceHardware]]:
    def _get_perf_sum(device_list: List[DeviceHardware]) -> float:
        perf = 0
        for device in device_list:
            perf += device.perf.total
        return perf

    return sorted(devices, key=_get_perf_sum)


def _get_uniform_sharding_options(
    sharding_options: List[ShardingOption],
) -> List[ShardingOption]:
    uniform_sharding_options: List[ShardingOption] = []
    for sharding_option in sharding_options:
        if sharding_option.partition_by == PartitionByType.UNIFORM.value:
            uniform_sharding_options.append(sharding_option)
    return uniform_sharding_options


@dataclass
class ShardingOptionGroup:
    sharding_options: List[ShardingOption]
    storage_sum: Storage
    perf_sum: float
    param_count: int


class SortBy(Enum):
    STORAGE = "storage"
    PERF = "perf"


def _group_and_sort_non_uniform_sharding_options(
    sharding_options: List[ShardingOption],
    sort_by: SortBy = SortBy.STORAGE,
    balance_modules: bool = False,
) -> List[ShardingOptionGroup]:

    # count modules by name
    param_count: Dict[str, int] = {}
    for sharding_option in sharding_options:
        path = sharding_option.path
        if path not in param_count:
            param_count[path] = 0
        param_count[path] += 1
    logger.debug(f"param_count is {param_count}")

    sharding_option_groups_by_dependency = {}
    for sharding_option in sharding_options:
        if sharding_option.partition_by == PartitionByType.UNIFORM.value:
            continue

        group_key = sharding_option.dependency or sharding_option.fqn
        if group_key not in sharding_option_groups_by_dependency:
            sharding_option_groups_by_dependency[group_key] = ShardingOptionGroup(
                [sharding_option],
                sharding_option.total_storage,
                sharding_option.total_perf,
                # negative value to indicate that smaller modules should be sorted first
                param_count=-param_count[sharding_option.path],
            )
        else:
            sharding_option_groups_by_dependency[group_key].sharding_options.append(
                sharding_option
            )
            sharding_option_groups_by_dependency[
                group_key
            ].storage_sum += sharding_option.total_storage
            sharding_option_groups_by_dependency[
                group_key
            ].perf_sum += sharding_option.total_perf

    sharding_option_groups = list(sharding_option_groups_by_dependency.values())

    sort_by_attributes: List[str] = []
    if balance_modules:
        sort_by_attributes.append("param_count")

    if sort_by == SortBy.STORAGE:
        sort_by_attributes.append("storage_sum")
    elif sort_by == SortBy.PERF:
        sort_by_attributes.append("perf_sum")
    else:
        raise RuntimeError(f"Unexpected sort_by: {sort_by}")

    sharding_option_groups.sort(
        key=lambda group: [getattr(group, attr) for attr in sort_by_attributes],
        reverse=True,
    )

    return sharding_option_groups


@dataclass
class OrderedDeviceHardware:
    device: DeviceHardware
    local_world_size: int

    def __lt__(self, other: "OrderedDeviceHardware") -> bool:
        # Use local rank as a tie breaker to ensure that we don't overload a single
        # host's DDR limit.
        return (
            self.device.perf.total,
            self.device.rank % self.local_world_size,
            self.device.rank,
        ) < (
            other.device.perf.total,
            other.device.rank % self.local_world_size,
            other.device.rank,
        )


class GreedyPerfPartitioner(Partitioner):
    """Greedy Partitioner.

    Args:
        sort_by (SortBy): Sort sharding options by storage or perf in
            descending order (i.e., large tables will be placed first).
        balance_modules (bool): Whether to sort by modules first, where
            smaller modules will be sorted first. In effect, this will place
            tables in each module in a balanced way.
    """

    def __init__(
        self, sort_by: SortBy = SortBy.STORAGE, balance_modules: bool = False
    ) -> None:
        self._sort_by = sort_by
        self._balance_modules = balance_modules

    def partition(
        self,
        proposal: List[ShardingOption],
        storage_constraint: Topology,
    ) -> List[ShardingOption]:
        """
        Places sharding options on topology based on each sharding option's
        `partition_by` attribute.
        The topology, storage, and perfs are updated at the end of the placement.

        Args:
            proposal (List[ShardingOption]): list of populated sharding options.
            storage_constraint (Topology): device topology.

        Returns:
            List[ShardingOption]: list of sharding options for selected plan.

        Example::

            sharding_options = [
                    ShardingOption(partition_by="uniform",
                            shards=[
                                Shards(storage=1, perf=1),
                                Shards(storage=1, perf=1),
                            ]),
                    ShardingOption(partition_by="uniform",
                            shards=[
                                Shards(storage=2, perf=2),
                                Shards(storage=2, perf=2),
                            ]),
                    ShardingOption(partition_by="device",
                            shards=[
                                Shards(storage=3, perf=3),
                                Shards(storage=3, perf=3),
                            ])
                    ShardingOption(partition_by="device",
                            shards=[
                                Shards(storage=4, perf=4),
                                Shards(storage=4, perf=4),
                            ]),
                ]
            topology = Topology(world_size=2)

            # First [sharding_options[0] and sharding_options[1]] will be placed on the
            # topology with the uniform strategy, resulting in

            topology.devices[0].perf.total = (1,2)
            topology.devices[1].perf.total = (1,2)

            # Finally sharding_options[2] and sharding_options[3]] will be placed on the
            # topology with the device strategy (see docstring of `partition_by_device` for
            # more details).

            topology.devices[0].perf.total = (1,2) + (3,4)
            topology.devices[1].perf.total = (1,2) + (3,4)

            # The topology updates are done after the end of all the placements (the other
            # in the example is just for clarity).
        """

        _topology: Topology = copy.deepcopy(storage_constraint)
        minheap_devices: Optional[List[OrderedDeviceHardware]] = None
        _host_level_devices = self._get_host_level_devices(_topology)

        # first partition the uniform sharding options (RW & DP)
        uniform_sharding_options = _get_uniform_sharding_options(proposal)
        self._uniform_partition(uniform_sharding_options, _topology.devices)

        # group the rest sharding options by colocation type (co-host, co-device, none)
        # and sort the groups by storage in reverse order
        sharding_option_groups = _group_and_sort_non_uniform_sharding_options(
            proposal, sort_by=self._sort_by, balance_modules=self._balance_modules
        )

        for sharding_option_group in sharding_option_groups:
            if (
                sharding_option_group.sharding_options[0].partition_by
                == PartitionByType.MULTI_HOST.value
            ):
                self._multi_hosts_partition(sharding_option_group, _host_level_devices)
                # _multi_hosts_partition invalidates minheap_devices, force rebuild before using
                minheap_devices = None

            elif (
                sharding_option_group.sharding_options[0].partition_by
                == PartitionByType.HOST.value
            ):
                self._cohost_partition(sharding_option_group, _host_level_devices)
                # _cohost_partition invalidates minheap_devices, force rebuild before using
                minheap_devices = None
            elif (
                sharding_option_group.sharding_options[0].partition_by
                == PartitionByType.DEVICE.value
            ):
                if minheap_devices is None:
                    minheap_devices = self._establish_minheap(
                        _topology.devices, _topology.local_world_size
                    )
                assert (
                    len(sharding_option_group.sharding_options) == 1
                ), f"Unexpected length for sharding options: {len(sharding_option_group.sharding_options)}"
                self._device_partition(
                    sharding_option_group.sharding_options[0],
                    minheap_devices,
                )
            else:
                raise RuntimeError(
                    f"Unexpected sharding option group {sharding_option_group}"
                )
        # pyre-ignore [16]: `GreedyPerfPartitioner` has no attribute `_topology`.
        self._topology: Topology = _topology
        return proposal

    @classmethod
    def _establish_minheap(
        cls, devices: List[DeviceHardware], local_world_size: int
    ) -> List[OrderedDeviceHardware]:
        minheap_devices = [
            OrderedDeviceHardware(device, local_world_size) for device in devices
        ]
        heapq.heapify(minheap_devices)
        return minheap_devices

    @classmethod
    def _device_partition(
        cls,
        sharding_option: ShardingOption,
        minheap_devices: List[OrderedDeviceHardware],
        bulk_heapify_threshold: float = 0.25,
    ) -> None:
        pushlimit = len(minheap_devices) * bulk_heapify_threshold
        for shard in sharding_option.shards:
            tmp_heap = []
            while minheap_devices:
                ordered_device = minheap_devices[0]
                device = ordered_device.device
                storage = cast(Storage, shard.storage)
                if storage.fits_in(device.storage):
                    shard.rank = device.rank
                    device.storage -= cast(Storage, shard.storage)
                    device.perf += cast(Perf, shard.perf)
                    heapq.heapreplace(minheap_devices, ordered_device)
                    break
                else:
                    heapq.heappop(minheap_devices)
                    tmp_heap.append(ordered_device)
            else:
                raise PlannerError(
                    error_type=PlannerErrorType.PARTITION,
                    message=(
                        f"Device partition failed. Couldn't find a rank for shard {shard} of table {sharding_option.name}, "
                        f"largest device storage: {max(ordered_device.device.storage for ordered_device in tmp_heap)}"
                    ),
                )
            if tmp_heap:
                # restablish minheap
                if len(tmp_heap) <= pushlimit:
                    for ordered_device in tmp_heap:
                        heapq.heappush(minheap_devices, ordered_device)
                else:
                    minheap_devices.extend(tmp_heap)
                    heapq.heapify(minheap_devices)

    @classmethod
    def _multi_hosts_partition(
        cls,
        sharding_option_group: ShardingOptionGroup,
        _host_level_devices: List[List[DeviceHardware]],
    ) -> None:
        """
        Partition shards on multiple hosts. This is a greedy algorithm trying to complete partitioning on multiple hosts (sorted by perf).
        First we do columnwise sharding among hosts, then tablewise-rowwise sharding within each host.
        There're two cases depends on the number of hosts needed to partition shards.

        Case one: `num_host_to_allocate >= len(sorted_host_level_devices)`
            We'll try to partition only once. Hosts might be selected multiple times in a circular manner.
            E.g, we have 3 hosts and `num_host_to_allocate` = 4. We sort all devices on host level. The devices of hosts [0, 1, 2, 0] will be selected for uniform partitioning.
            We'll update device information if success, otherwise raise a `PlannerError`.

        Case two: `num_host_to_allocate < len(sorted_host_level_devices)`
            We'll try to partition with hosts `[host_index, host_index + num_host_to_allocate]` iteratively with host_index incremented by 1 each time.
            1) We sort all devices on host level. Set `host_index` = 0
            2) We select hosts`[host_index, host_index + num_host_to_allocate]` if indexes are within range.
            3) We do uniform partitioning over all devices of the selected hosts. If we cannot partition, then we increase `host_index` by 1 and go to 2); Otherwise we go to 4)
            4) Update device information if success, otherwise raise a `PlannerError`.

        Keyword arguments:
        sharding_option_group -- grouped sharding options
        _host_level_devices -- devices

        Example::
            sharding_option_group.sharding_options = [
                    ShardingOption(partition_by="multi_host",
                            shards=[
                                Shards(storage=1, perf=1),
                                Shards(storage=1, perf=1),
                                Shards(storage=1, perf=1),
                                Shards(storage=1, perf=1),
                            ]),
                ]
            topology = Topology(world_size=6, local_world_size=2)

            # sharding_options[0] will be placed on host 1 and host 2 with the multi_hosts strategy, resulting in

            topology.devices[0].perf.total = (1,1)
            topology.devices[1].perf.total = (1,1)
            topology.devices[2].perf.total = (1,1)
            topology.devices[3].perf.total = (1,1)
            topology.devices[4].perf.total = (0,0)
            topology.devices[5].perf.total = (0,0)

        """
        # TODO: for now assume just one option for multi_hosts.
        if len(sharding_option_group.sharding_options) != 1:
            raise PlannerError(
                error_type=PlannerErrorType.PARTITION,
                message=f"Unexpected length for sharding options: {len(sharding_option_group.sharding_options)}. Length needs to be 1",
            )
        num_shards = sharding_option_group.sharding_options[0].num_shards

        if _host_level_devices is None:
            raise PlannerError(
                error_type=PlannerErrorType.PARTITION,
                message="host level devices is None",
            )

        local_world_size = len(_host_level_devices[0])
        num_host_to_allocate, remainder = divmod(num_shards, local_world_size)

        if remainder > 0:
            raise PlannerError(
                error_type=PlannerErrorType.PARTITION,
                message=f"Grid Sharding is unable to place shards equally over hosts without overlapping. {num_shards=} % {local_world_size=} != 0",
            )

        sorted_host_level_devices = _sort_devices_by_perf(_host_level_devices)
        host_index = 0
        all_hosts_used = False
        while True:
            if num_host_to_allocate >= len(sorted_host_level_devices):
                # case one: we need to use all hosts
                all_hosts_used = True
                devices = []
                for i in range(num_host_to_allocate):
                    devices.extend(
                        sorted_host_level_devices[i % len(sorted_host_level_devices)]
                    )
            else:
                # case two: we can use some hosts
                devices = list(
                    itertools.chain(
                        *sorted_host_level_devices[
                            host_index : host_index + num_host_to_allocate
                        ]
                    )
                )
            host_index += 1  # shift to next host
            host_devices = copy.deepcopy(devices)
            success = True
            sharding_option = sharding_option_group.sharding_options[0]
            try:
                if sharding_option.sharding_type == ShardingType.GRID_SHARD.value:
                    cls._uniform_partition([sharding_option], host_devices)
                else:
                    raise PlannerError(
                        error_type=PlannerErrorType.PARTITION,
                        message=f"unexpected multi_host sharding type: {sharding_option.sharding_type}",
                    )
            except PlannerError:
                success = False
            if success:
                # successfully found some hosts and partitioned on these hosts
                # need to update the devices
                for device, host_device in zip(devices, host_devices):
                    # check that devices and host_devices are in the same order
                    if device.rank != host_device.rank:
                        raise PlannerError(
                            error_type=PlannerErrorType.PARTITION,
                            message=f"device rank {device.rank} is not the same as device_copy rank {host_device.rank}",
                        )
                    device.storage = host_device.storage
                    device.perf = host_device.perf
                return

            if (
                host_index + num_host_to_allocate > len(sorted_host_level_devices)
            ) or all_hosts_used:
                break
        raise PlannerError(
            error_type=PlannerErrorType.PARTITION,
            message=f"can't find hosts for sharding option group {sharding_option_group}",
        )

    @classmethod
    def _cohost_partition(
        cls,
        sharding_option_group: ShardingOptionGroup,
        _host_level_devices: List[List[DeviceHardware]],
    ) -> None:
        sorted_host_level_devices = _sort_devices_by_perf(_host_level_devices)
        for devices in sorted_host_level_devices:
            host_devices = copy.deepcopy(devices)
            host_storage = Storage(hbm=0, ddr=0)
            for device in host_devices:
                host_storage += device.storage
            if not sharding_option_group.storage_sum.fits_in(host_storage):
                continue

            success = True
            minheap_devices: Optional[List[OrderedDeviceHardware]] = None
            for sharding_option in sharding_option_group.sharding_options:
                try:
                    if (
                        sharding_option.sharding_type
                        == ShardingType.TABLE_ROW_WISE.value
                    ):
                        cls._uniform_partition([sharding_option], host_devices)
                        # _uniform_partition invalidates minheap_devices, force rebuild
                        # before using
                        minheap_devices = None
                    elif (
                        sharding_option.sharding_type
                        == ShardingType.TABLE_COLUMN_WISE.value
                    ):
                        if minheap_devices is None:
                            minheap_devices = cls._establish_minheap(
                                host_devices, len(host_devices)
                            )
                        cls._device_partition(sharding_option, minheap_devices)
                    else:
                        raise PlannerError(
                            error_type=PlannerErrorType.PARTITION,
                            message=f"unexpected cohost sharding type: {sharding_option.sharding_type}",
                        )
                except PlannerError:
                    success = False
                    break
            if success:
                # successfully found a host and partitioned on that host
                # need to update the devices
                # resorting host_devices before copying data back
                host_devices.sort(key=lambda device: device.rank)
                for device, device_copy in zip(devices, host_devices):
                    device.storage = device_copy.storage
                    device.perf = device_copy.perf
                return
        raise PlannerError(
            error_type=PlannerErrorType.PARTITION,
            message=f"can't find a host for sharding option group {sharding_option_group}",
        )

    @classmethod
    def _get_host_level_devices(cls, _topology: Topology) -> List[List[DeviceHardware]]:
        num_hosts: int = _topology.world_size // _topology.local_world_size
        host_level_devices: List[List[DeviceHardware]] = []
        for i in range(num_hosts):
            devices_in_host = _topology.devices[
                i * _topology.local_world_size : (i + 1) * _topology.local_world_size
            ]
            host_level_devices.append(devices_in_host)
        return host_level_devices

    @classmethod
    def _uniform_partition(
        cls, sharding_options: List[ShardingOption], devices: List[DeviceHardware]
    ) -> None:
        for sharding_option in sharding_options:
            if sharding_option.num_shards != len(devices):
                raise PlannerError(
                    error_type=PlannerErrorType.PARTITION,
                    message=f"For a uniform partition, the number of shards ({sharding_option.num_shards}) must equal the number of devices ({len(devices)})",
                )
            for i in range(len(devices)):
                storage_needed = cast(Storage, sharding_option.shards[i].storage)
                if not storage_needed.fits_in(devices[i].storage):
                    raise PlannerError(
                        error_type=PlannerErrorType.PARTITION,
                        message=f"Shard of size {storage_needed} bytes does not fit on any rank. Device memory cap: {devices[i].storage}.",
                    )
                else:
                    sharding_option.shards[i].rank = devices[i].rank
                    devices[i].storage -= storage_needed
                    devices[i].perf += cast(Perf, sharding_option.shards[i].perf)


class MemoryBalancedPartitioner(Partitioner):
    """Memory balanced Partitioner.

    Args:
        max_search_count (int): Maximum number of times to call the
            GreedyPartitioner.
        tolerance (float): The maximum acceptable difference between the
            original plan and the new plan. If tolerance is 1, that means a new
            plan will be rejected if its perf is 200% of the original plan
            (i.e., the plan is 100% worse).
        balance_modules (bool): Whether to sort by modules first, where
            smaller modules will be sorted first. In effect, this will place
            tables in each module in a balanced way.
    """

    def __init__(
        self,
        max_search_count: int = 10,
        tolerance: float = 0.02,
        balance_modules: bool = False,
    ) -> None:
        self._max_search_count: int = max_search_count
        self._tolerance: float = tolerance
        self._balance_modules: bool = balance_modules

    def partition(
        self,
        proposal: List[ShardingOption],
        storage_constraint: Topology,
    ) -> List[ShardingOption]:
        """
        Repeatedly calls the GreedyPerfPartitioner to find a plan with perf
        within the tolerance of the original plan that uses the least amount
        of memory.
        """
        _perf_model: PerfModel = NoopPerfModel(storage_constraint)
        _partitioner = GreedyPerfPartitioner(
            sort_by=SortBy.PERF, balance_modules=self._balance_modules
        )
        # copying storage_constraint, since we modify it in place
        _topology: Topology = copy.deepcopy(storage_constraint)

        # set up default plan to fall back on
        default_plan = _partitioner.partition(proposal, _topology)
        default_plan = copy.deepcopy(default_plan)
        original_plan_perf = _perf_model.rate(default_plan)

        max_hbm_per_device: int = _topology.devices[0].storage.hbm
        logger.info(
            f"Default plan uses {round(bytes_to_gb(max_hbm_per_device), 3)} GB per device."
        )

        hbm_requirement: int = 0
        for sharding_option in proposal:
            for shard in sharding_option.shards:
                if shard.storage is not None:
                    hbm_requirement += shard.storage.hbm
        min_hbm_per_device: int = int(hbm_requirement / _topology.world_size)
        logger.info(
            "Searching in the range (min_hbm_per_device, max_hbm_per_device): "
            f"({round(bytes_to_gb(min_hbm_per_device), 3)}, "
            f"{round(bytes_to_gb(max_hbm_per_device), 3)})"
        )

        # binary search with (min, max] setting
        search_count = 0
        while (
            search_count < self._max_search_count
            and min_hbm_per_device + 10 * 1024**2 < max_hbm_per_device  # 10MB
        ):
            search_count += 1
            reset_shard_rank(proposal)
            mid_hbm_per_device: int = (max_hbm_per_device + min_hbm_per_device) // 2
            set_hbm_per_device(_topology, mid_hbm_per_device)
            try:
                new_plan = _partitioner.partition(proposal, _topology)
                new_plan_perf = _perf_model.rate(new_plan)
                perf_diff = (
                    (new_plan_perf - original_plan_perf) / original_plan_perf
                    if original_plan_perf
                    else 100
                )
                if new_plan_perf > original_plan_perf * (1 + self._tolerance):
                    # the new plan is worse than the original one
                    logger.info(
                        f"Found a plan with {round(bytes_to_gb(mid_hbm_per_device), 3)} "
                        f"GB per device for embedding tables, "
                        f"but its perf is {round(perf_diff * 100, 3)}% worse than the original plan, "
                        f"which exceeds the {self._tolerance * 100}% tolerance."
                    )
                    min_hbm_per_device = mid_hbm_per_device
                else:
                    # the new plan is better than original one
                    if perf_diff > 0:
                        perf_diff_str = (
                            f"{round((perf_diff) * 100, 3)}% worse than the original plan, "
                            f"which is within the {self._tolerance * 100}% tolerance."
                        )
                    else:
                        perf_diff_str = f"{round((perf_diff) * 100, 3)}% better than the original plan."
                    logger.info(
                        f"Found a more memory-balanced plan with {round(bytes_to_gb(mid_hbm_per_device), 3)} "
                        f"GB per device for embedding tables. The new plan is {perf_diff_str}"
                    )
                    default_plan = copy.deepcopy(new_plan)
                    max_hbm_per_device = mid_hbm_per_device
            except PlannerError:
                logger.info(
                    f"Couldn't find a plan with {round(bytes_to_gb(max_hbm_per_device), 3)} "
                    f"GB per device for embedding tables."
                )
                min_hbm_per_device = mid_hbm_per_device

        return default_plan


def set_hbm_per_device(storage_constraint: Topology, hbm_per_device: int) -> None:
    for device in storage_constraint.devices:
        device.storage.hbm = hbm_per_device
