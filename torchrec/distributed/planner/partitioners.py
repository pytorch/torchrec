#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from dataclasses import dataclass
from typing import List, cast

from torchrec.distributed.planner.types import (
    Partitioner,
    Topology,
    ShardingOption,
    Storage,
    PartitionByType,
    PlannerError,
    DeviceHardware,
)
from torchrec.distributed.types import ShardingType


def _sort_devices_by_perf(
    devices: List[List[DeviceHardware]],
) -> List[List[DeviceHardware]]:
    def _get_perf_sum(device_list: List[DeviceHardware]) -> float:
        perf = 0
        for device in device_list:
            perf += device.perf
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


def _group_and_sort_non_uniform_sharding_options(
    sharding_options: List[ShardingOption],
) -> List[ShardingOptionGroup]:
    sharding_option_groups_by_dependency = {}
    for sharding_option in sharding_options:
        if sharding_option.partition_by == PartitionByType.UNIFORM.value:
            continue

        group_key = sharding_option.dependency or sharding_option.fqn
        if group_key not in sharding_option_groups_by_dependency:
            sharding_option_groups_by_dependency[group_key] = ShardingOptionGroup(
                [sharding_option], sharding_option.total_storage
            )
        else:
            sharding_option_groups_by_dependency[group_key].sharding_options.append(
                sharding_option
            )
            sharding_option_groups_by_dependency[
                group_key
            ].storage_sum += sharding_option.total_storage
    sharding_option_groups = list(sharding_option_groups_by_dependency.values())

    sharding_option_groups.sort(key=lambda group: group.storage_sum, reverse=True)
    return sharding_option_groups


class GreedyPerfPartitioner(Partitioner):
    """
    Greedy Partitioner
    """

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

            topology.devices[0].perf = (1,2)
            topology.devices[1].perf = (1,2)

            # Finally sharding_options[2] and sharding_options[3]] will be placed on the
            # topology with the device strategy (see docstring of `partition_by_device` for
            # more details).

            topology.devices[0].perf = (1,2) + (3,4)
            topology.devices[1].perf = (1,2) + (3,4)

            # The topology updates are done after the end of all the placements (the other
            # in the example is just for clarity).
        """

        # pyre-ignore [16]: `GreedyPerfPartitioner` has no attribute `_topology`.
        self._topology: Topology = copy.deepcopy(storage_constraint)
        plan = copy.deepcopy(proposal)
        # pyre-ignore [16]
        self._host_level_devices = self._get_host_level_devices()

        # first partition the uniform sharding options (RW & DP)
        uniform_sharding_options = _get_uniform_sharding_options(plan)
        self._uniform_partition(uniform_sharding_options, self._topology.devices)

        # group the rest sharding options by colocation type (co-host, co-device, none)
        # and sort the groups by storage in reverse order
        sharding_option_groups = _group_and_sort_non_uniform_sharding_options(plan)

        for sharding_option_group in sharding_option_groups:
            if (
                sharding_option_group.sharding_options[0].partition_by
                == PartitionByType.HOST.value
            ):
                self._cohost_partition(sharding_option_group)
            elif (
                sharding_option_group.sharding_options[0].partition_by
                == PartitionByType.DEVICE.value
            ):
                assert (
                    len(sharding_option_group.sharding_options) == 1
                ), f"Unexpected length for sharding options: {len(sharding_option_group.sharding_options)}"
                self._device_partition(
                    sharding_option_group.sharding_options[0], self._topology.devices
                )
            else:
                raise RuntimeError(
                    f"Unexpected sharding option group {sharding_option_group}"
                )
        return plan

    def _device_partition(
        self, sharding_option: ShardingOption, devices: List[DeviceHardware]
    ) -> None:
        for shard in sharding_option.shards:
            sorted_devices = sorted(devices, key=lambda device: device.perf)
            success = False
            for device in sorted_devices:
                if device.storage >= shard.storage:
                    shard.rank = device.rank
                    device.storage -= cast(Storage, shard.storage)
                    device.perf += cast(float, shard.perf)
                    success = True
                    break
            if not success:
                raise PlannerError(
                    f"Device partition failed. Couldn't find a rank for shard {shard}, devices: {devices}"
                )

    def _cohost_partition(self, sharding_option_group: ShardingOptionGroup) -> None:
        # pyre-ignore [16]
        sorted_host_level_devices = _sort_devices_by_perf(self._host_level_devices)
        for devices in sorted_host_level_devices:
            host_devices = copy.deepcopy(devices)
            host_storage = Storage(hbm=0, ddr=0)
            for device in host_devices:
                host_storage += device.storage
            if host_storage < sharding_option_group.storage_sum:
                continue

            success = True
            for sharding_option in sharding_option_group.sharding_options:
                try:
                    if (
                        sharding_option.sharding_type
                        == ShardingType.TABLE_ROW_WISE.value
                    ):
                        self._uniform_partition([sharding_option], host_devices)
                    elif (
                        sharding_option.sharding_type
                        == ShardingType.TABLE_COLUMN_WISE.value
                    ):
                        self._device_partition(sharding_option, host_devices)
                    else:
                        raise RuntimeError(
                            f"unexpected cohost sharding type: {sharding_option.sharding_type}"
                        )
                except PlannerError:
                    success = False
                    break
            if success:
                # successfully found a host and partitioned on that host
                # need to update the devices
                for device, device_copy in zip(devices, host_devices):
                    device.storage = device_copy.storage
                    device.perf = device_copy.perf
                return
        raise PlannerError(
            f"can't find a host for sharding option group {sharding_option_group}"
        )

    def _get_host_level_devices(self) -> List[List[DeviceHardware]]:
        # pyre-ignore [16]
        num_hosts: int = self._topology.world_size // self._topology.local_world_size
        host_level_devices: List[List[DeviceHardware]] = []
        for i in range(num_hosts):
            devices_in_host = self._topology.devices[
                i
                * self._topology.local_world_size : (i + 1)
                * self._topology.local_world_size
            ]
            host_level_devices.append(devices_in_host)
        return host_level_devices

    def _uniform_partition(
        self, sharding_options: List[ShardingOption], devices: List[DeviceHardware]
    ) -> None:
        for sharding_option in sharding_options:
            if sharding_option.num_shards != len(devices):
                raise RuntimeError(
                    f"For a uniform partition, the number of shards ({sharding_option.num_shards}) must equal the number of devices ({len(devices)})"
                )
            for i in range(len(devices)):
                storage_needed = cast(Storage, sharding_option.shards[i].storage)
                if storage_needed > devices[i].storage:
                    raise PlannerError(
                        f"Shard of size {storage_needed} bytes does not fit on any rank. Device memory cap: {devices[i].storage}."
                    )
                else:
                    sharding_option.shards[i].rank = devices[i].rank
                    devices[i].storage -= storage_needed
                    devices[i].perf += cast(float, sharding_option.shards[i].perf)
