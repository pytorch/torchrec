#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import List, Tuple, Optional, Dict, cast

from torchrec.distributed.planner.constants import MAX_SIZE
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


def greedy_partition(
    num_partitions: int,
    sharding_options: List[ShardingOption],
    shard_idxes: Optional[List[Tuple[int, int]]] = None,
    partition_sums: Optional[List[float]] = None,
    mem_cap: Optional[List[Storage]] = None,
) -> List[List[Tuple[int, int]]]:
    """
    Divides indices among `num_partitions` partitions in a greedy fashion based on perf
    weights associated with each [option_idx, shard_idx].

    Returns:
        List[List[Tuple[int, int]]]: list of indices of (option_idx, shard_idx) that should be allocated to each partition.

    Example::

        sharding_options = [
            [0,1,2,3] with perfs [10,20,30,40]
            [0,1] with perfs [200,300]
        ]
        # with num_partitions=3

        # The final output would be:
        [
            partition_0 = [(1,1)], with a perf of 300
            partition_1 = [(1,0)], with a perf of 200
            partition_2 = [(0,0),(0,1),(0,2),(0,3)], with a perf of 100 (10+20+30+40)
        ]
    """

    if shard_idxes is None:
        shard_idxes = []
        for option_idx, sharding_option in enumerate(sharding_options):
            for shard_idx in range(sharding_option.num_shards):
                shard_idxes.append((option_idx, shard_idx))

    def _to_comparable(order_shard_idx: Tuple[int, int]) -> Tuple[float, Storage]:
        sharding_option: ShardingOption = sharding_options[order_shard_idx[0]]
        return (
            cast(float, sharding_option.shards[order_shard_idx[1]].perf),
            cast(Storage, sharding_option.shards[order_shard_idx[1]].storage),
        )

    # A correct implementation of the greedy algorithm processes items in descending
    # value order. Here, we sort in ascending order, but we'll pop items in descending
    # order below.
    sorted_shard_idxes = sorted(
        shard_idxes, key=lambda order_shard_idx: _to_comparable(order_shard_idx)
    )

    partitions = [[] for p in range(num_partitions)]
    if partition_sums is None:
        partition_sums = [0.0] * num_partitions

    partition_size_sums = [Storage(hbm=0, ddr=0) for _ in range(num_partitions)]

    if mem_cap is None:
        mem_cap = [Storage(hbm=MAX_SIZE, ddr=MAX_SIZE) for _ in range(num_partitions)]

    assert len(partition_size_sums) == len(
        mem_cap
    ), "partition_size_sums and mem_cap must have the same dimensions"

    """
    Successively add remaining pairs to the partition with the minimum sum.
    """
    while sorted_shard_idxes:
        # Remove values from largest to smallest so the algorithm is correct.
        option_idx, shard_idx = sorted_shard_idxes.pop()
        storage_size = cast(
            Storage, sharding_options[option_idx].shards[shard_idx].storage
        )
        perf = cast(float, sharding_options[option_idx].shards[shard_idx].perf)

        min_sum = MAX_SIZE
        min_partition_idx = -1
        for partition_idx in range(num_partitions):
            partition_mem_cap: Storage = mem_cap[partition_idx]
            partition_size_sum: Storage = partition_size_sums[partition_idx]
            if (
                partition_mem_cap.hbm >= partition_size_sum.hbm + storage_size.hbm
            ) and (partition_mem_cap.ddr >= partition_size_sum.ddr + storage_size.ddr):
                if partition_sums[partition_idx] < min_sum:
                    min_sum = partition_sums[partition_idx]
                    min_partition_idx = partition_idx

        if min_partition_idx == -1:
            raise PlannerError(
                f"Table of size {storage_size} GB cannot be added to any rank. partition_size_sums: {partition_size_sums}. mem_cap: {mem_cap}."
            )

        partitions[min_partition_idx].append((option_idx, shard_idx))

        partition_size_sums[min_partition_idx] += storage_size
        partition_sums[min_partition_idx] += perf

    return partitions


def uniform_partition(
    num_partitions: int,
    sharding_options: List[ShardingOption],
    mem_cap: List[Storage],
    shard_idxes: Optional[List[Tuple[int, int]]] = None,
) -> List[List[Tuple[int, int]]]:
    """
    Assigns one shard to each rank.

    Example::

        sharding_options = [
            [0,1,2,3],
            [0,1,2,3],
        ]
        # with num_partitions=4

        # The final output would be:
        [
            partition_0 = [(0,0),(1,0)]
            partition_1 = [(0,1),(1,1)]
            partition_2 = [(0,2),(1,2)]
            partition_3 = [(0,3),(1,3)]
        ]
    """

    partition_size_sums = [Storage(hbm=0, ddr=0) for _ in range(num_partitions)]

    if shard_idxes is None:
        shard_idxes = []
        for option_idx, sharding_option in enumerate(sharding_options):
            for shard_idx in range(sharding_option.num_shards):
                shard_idxes.append((option_idx, shard_idx))

    partitions: List[List[Tuple[int, int]]] = [[] for _ in range(num_partitions)]
    for option_idx, shard_idx in shard_idxes:
        storage_size = cast(
            Storage, sharding_options[option_idx].shards[shard_idx].storage
        )
        if partition_size_sums[shard_idx] + storage_size > mem_cap[shard_idx]:
            raise PlannerError(
                f"Table of size {storage_size} GB cannot be added to any rank. partition_size_sums: {partition_size_sums}. mem_cap: {mem_cap}."
            )
        partition_size_sums[shard_idx] += storage_size
        partitions[shard_idx].append((option_idx, shard_idx))

    return partitions


def _group_sharding_options(
    sharding_options: List[ShardingOption],
) -> Dict[str, List[ShardingOption]]:
    partition_by_groups = {}
    for sharding_option in sharding_options:
        if sharding_option.partition_by not in partition_by_groups:
            partition_by_groups[sharding_option.partition_by] = []
        partition_by_groups[sharding_option.partition_by].append(sharding_option)
    return partition_by_groups


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
        Topology storage and perfs are updated at the end of the placement.

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

        grouped_sharding_options = _group_sharding_options(plan)

        if PartitionByType.UNIFORM.value in grouped_sharding_options:
            self._partition_by_uniform(
                grouped_sharding_options[PartitionByType.UNIFORM.value]
            )
        if PartitionByType.HOST.value in grouped_sharding_options:
            self._partition_by_host(
                grouped_sharding_options[PartitionByType.HOST.value]
            )
        if PartitionByType.DEVICE.value in grouped_sharding_options:
            self._partition_by_device(
                grouped_sharding_options[PartitionByType.DEVICE.value]
            )
        return plan

    def _partition_by_uniform(self, sharding_options: List[ShardingOption]) -> None:
        partitions = uniform_partition(
            # pyre-ignore [16]: `GreedyPerfPartitioner` has no attribute `_topology`.
            num_partitions=self._topology.world_size,
            sharding_options=sharding_options,
            mem_cap=[device.storage for device in self._topology.devices],
        )
        self._update_shards(partitions, sharding_options)

    def _partition_by_device(self, sharding_options: List[ShardingOption]) -> None:
        # pyre-ignore [16]: `GreedyPerfPartitioner` has no attribute `_topology`.
        partition_sums = [float(device.perf) for device in self._topology.devices]
        mem_cap: List[Storage] = [device.storage for device in self._topology.devices]
        partitions = greedy_partition(
            num_partitions=self._topology.world_size,
            sharding_options=sharding_options,
            partition_sums=partition_sums,
            mem_cap=mem_cap,
        )
        self._update_shards(partitions, sharding_options)

    def _partition_by_host(self, sharding_options: List[ShardingOption]) -> None:
        # pyre-ignore [16]: `GreedyPerfPartitioner` has no attribute `_topology`.
        num_hosts: int = self._topology.world_size // self._topology.local_world_size

        host_level_devices: Dict[int, List[DeviceHardware]] = {}
        for i in range(num_hosts):
            devices_in_host = self._topology.devices[
                i
                * self._topology.local_world_size : (i + 1)
                * self._topology.local_world_size
            ]
            host_level_devices[i] = devices_in_host

        self._uniform_partition_by_host(
            sharding_options=sharding_options,
            host_level_devices=host_level_devices,
        )
        self._greedy_partition_by_host(
            sharding_options=sharding_options,
            host_level_devices=host_level_devices,
        )

    def _uniform_partition_by_host(
        self,
        sharding_options: List[ShardingOption],
        host_level_devices: Dict[int, List[DeviceHardware]],
    ) -> None:
        shard_idxes = []
        for option_idx, _ in enumerate(sharding_options):
            if (
                _base_partition_by(sharding_options[option_idx].sharding_type)
                == PartitionByType.UNIFORM.value
            ):
                # only take the first shard from each sharding option. We can infer the rest
                shard_idxes.append((option_idx, 0))
        if not shard_idxes:
            return

        mem_cap: List[Storage] = []
        partition_sums = []

        for _host, devices in host_level_devices.items():
            # mem_cap of a host is the min of the storage of all devices on that host for uniform case.
            mem_cap.append(min([device.storage for device in devices]))
            # perf of a host is the max across all of its devices for uniform case. Typically this should be zero at entry point.
            partition_sums.append(max([float(device.perf) for device in devices]))

        host_level_partitions: List[List[Tuple[int, int]]] = greedy_partition(
            num_partitions=len(host_level_devices),
            sharding_options=sharding_options,
            shard_idxes=shard_idxes,
            partition_sums=partition_sums,
            mem_cap=mem_cap,
        )
        # pyre-ignore [16]: `GreedyPerfPartitioner` has no attribute `_topology`.
        partitions: List[List[Tuple[int, int]]] = [[] for _ in self._topology.devices]

        for host_idx, host_partition in enumerate(host_level_partitions):
            self._uniform_device_level_partition(
                partitions=partitions,
                sharding_options=sharding_options,
                option_idxes=[option_idx for option_idx, _ in host_partition],
                host_level_devices=host_level_devices[host_idx],
                host_idx=host_idx,
            )

        self._update_shards(partitions, sharding_options)

    def _greedy_partition_by_host(
        self,
        sharding_options: List[ShardingOption],
        host_level_devices: Dict[int, List[DeviceHardware]],
    ) -> None:
        shard_idxes = []
        for option_idx, _ in enumerate(sharding_options):
            if (
                _base_partition_by(sharding_options[option_idx].sharding_type)
                == PartitionByType.DEVICE.value
            ):
                # only take the first shard from each sharding option. We can infer the rest
                shard_idxes.append((option_idx, 0))
        if not shard_idxes:
            return

        mem_cap: List[Storage] = []
        partition_sums = []

        for _host, devices in host_level_devices.items():
            # mem_cap of a host is the sum of the storage of all devices on that host for greedy case.
            storage_sum = Storage(hbm=0, ddr=0)
            for device in devices:
                storage_sum += device.storage
            mem_cap.append(storage_sum)
            # perf of a host is the min across all of its devices for greedy case. Typically this should be zero at entry point.
            partition_sums.append(min([float(device.perf) for device in devices]))

        host_level_partitions: List[List[Tuple[int, int]]] = greedy_partition(
            num_partitions=len(host_level_devices),
            sharding_options=sharding_options,
            shard_idxes=shard_idxes,
            partition_sums=partition_sums,
            mem_cap=mem_cap,
        )
        # pyre-ignore [16]: `GreedyPerfPartitioner` has no attribute `_topology`.
        partitions: List[List[Tuple[int, int]]] = [[] for _ in self._topology.devices]

        for host_idx, host_partition in enumerate(host_level_partitions):
            self._greedy_device_level_partition(
                partitions=partitions,
                sharding_options=sharding_options,
                option_idxes=[option_idx for option_idx, _ in host_partition],
                host_level_devices=host_level_devices[host_idx],
                host_idx=host_idx,
            )

        self._update_shards(partitions, sharding_options)

    def _uniform_device_level_partition(
        self,
        partitions: List[List[Tuple[int, int]]],
        sharding_options: List[ShardingOption],
        option_idxes: List[int],
        host_level_devices: List[DeviceHardware],
        host_idx: int,
    ) -> None:
        shard_idxes = []
        for option_idx in option_idxes:
            for shard_idx in range(sharding_options[option_idx].num_shards):
                shard_idxes.append((option_idx, shard_idx))

        device_level_partitions: List[List[Tuple[int, int]]] = uniform_partition(
            # pyre-ignore [16]: `GreedyPerfPartitioner` has no attribute `_topology`.
            num_partitions=self._topology.local_world_size,
            sharding_options=sharding_options,
            mem_cap=[device.storage for device in host_level_devices],
            shard_idxes=shard_idxes,
        )

        for device_idx, device_partition in enumerate(device_level_partitions):
            for option_idx, shard_idx in device_partition:
                partitions[
                    self._topology.local_world_size * host_idx + device_idx
                ].append((option_idx, shard_idx))

    def _greedy_device_level_partition(
        self,
        partitions: List[List[Tuple[int, int]]],
        sharding_options: List[ShardingOption],
        option_idxes: List[int],
        host_level_devices: List[DeviceHardware],
        host_idx: int,
    ) -> None:
        shard_idxes = []
        for option_idx in option_idxes:
            for shard_idx in range(sharding_options[option_idx].num_shards):
                shard_idxes.append((option_idx, shard_idx))

        device_level_partitions: List[List[Tuple[int, int]]] = greedy_partition(
            # pyre-ignore [16]: `GreedyPerfPartitioner` has no attribute `_topology`.
            num_partitions=self._topology.local_world_size,
            sharding_options=sharding_options,
            shard_idxes=shard_idxes,
            partition_sums=[float(device.perf) for device in host_level_devices],
            mem_cap=[device.storage for device in host_level_devices],
        )

        for device_idx, device_partition in enumerate(device_level_partitions):
            for option_idx, shard_idx in device_partition:
                partitions[
                    self._topology.local_world_size * host_idx + device_idx
                ].append((option_idx, shard_idx))

    def _update_shards(
        self,
        partitions: List[List[Tuple[int, int]]],
        sharding_options: List[ShardingOption],
    ) -> None:
        """
        Updates the ranks of the shards as well as device perfs.
        """
        for partition_idx, partition in enumerate(partitions):
            for [option_idx, shard_idx] in partition:
                sharding_options[option_idx].shards[shard_idx].rank = partition_idx
                # pyre-ignore [16]: `GreedyPerfPartitioner` has no attribute `_topology`.
                self._topology.devices[partition_idx].storage -= (
                    sharding_options[option_idx].shards[shard_idx].storage
                )
                self._topology.devices[partition_idx].perf += (
                    sharding_options[option_idx].shards[shard_idx].perf
                )


def _base_partition_by(sharding_type: str) -> str:
    if sharding_type == ShardingType.TABLE_ROW_WISE.value:
        return PartitionByType.UNIFORM.value
    elif sharding_type == ShardingType.TABLE_COLUMN_WISE.value:
        return PartitionByType.DEVICE.value
    else:
        raise ValueError(
            f"Sharding type provided must have a partition_by value of HOST: {sharding_type}"
        )
