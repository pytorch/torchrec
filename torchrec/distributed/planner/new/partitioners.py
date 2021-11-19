#!/usr/bin/env python3

from typing import List, Tuple, Optional, Dict, cast

from torchrec.distributed.planner.new.constants import MAX_SIZE
from torchrec.distributed.planner.new.types import (
    Partitioner,
    Topology,
    ShardingOption,
    Storage,
    PartitionByType,
    PartitionError,
)


def greedy_partition(
    num_partitions: int,
    sharding_options: List[ShardingOption],
    shard_idxes: Optional[List[Tuple[int, int]]] = None,
    partition_sums: Optional[List[float]] = None,
    mem_cap: Optional[List[Storage]] = None,
) -> List[List[Tuple[int, int]]]:
    """
    Divides indexes among `num_parititions` partitions in a greedy
    fashion based on cost weights associated with each [option_idx, shard_idx].
    Returns a list of indices of (option_idx, shard_idx) that should be allocated to each partition

    For example if we have sharding_options = [
        [0,1,2,3] with costs [10,20,30,40]
        [0,1] with costs [200,300]
    ] with num_partitions=3

    The final output would be
    [
       partition_0 = [(1,1)], with a cost of 300
       partition_1 = [(1,0)], with a cost of 200
       partition_2 = [(0,0),(0,1),(0,2)], with a cost of 100 (10+20+30+40)
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
            cast(float, sharding_option.shards[order_shard_idx[1]].cost),
            sharding_option.shards[order_shard_idx[1]].storage,
        )

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
    Successively add remaining pairs to the partition with the
    minimum sum.
    """
    while sorted_shard_idxes:
        option_idx, shard_idx = sorted_shard_idxes.pop()
        storage_size = sharding_options[option_idx].shards[shard_idx].storage
        cost = cast(float, sharding_options[option_idx].shards[shard_idx].cost)

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
            raise PartitionError(
                f"Table of size {storage_size}GB cannot be added to any rank. partition_size_sums: {partition_size_sums}. mem_cap: {mem_cap}."
            )

        partitions[min_partition_idx].append((option_idx, shard_idx))

        partition_size_sums[min_partition_idx] += storage_size
        partition_sums[min_partition_idx] += cost

    return partitions


def uniform_partition(
    num_partitions: int,
    sharding_options: List[ShardingOption],
    mem_cap: List[Storage],
) -> List[List[Tuple[int, int]]]:
    """
        We assign one shard to each rank. For example, For example if we have sharding_options = [
            [0,1,2,3],
            [0,1,2,3],
    ] with num_partitions=4
    The final output would be
    [
       partition_0 = [(0,0),(1,0)]
       partition_1 = [(0,1),(1,1)]
       partition_2 = [(0,2),(1,2)]
       partition_3 = [(0,3),(1,3)]
    ]
    """

    shard_idxes: List[Tuple[int, int]] = []
    partition_size_sums = [Storage(hbm=0, ddr=0) for _ in range(num_partitions)]

    for option_idx, sharding_option in enumerate(sharding_options):
        for shard_idx in range(sharding_option.num_shards):
            shard_idxes.append((option_idx, shard_idx))

    partitions: List[List[Tuple[int, int]]] = [[] for _ in range(num_partitions)]
    for option_idx, shard_idx in shard_idxes:
        storage_size = sharding_options[option_idx].shards[shard_idx].storage
        if partition_size_sums[shard_idx] + storage_size > mem_cap[shard_idx]:
            raise PartitionError(
                f"Table of size {storage_size}GB cannot be added to any rank. partition_size_sums: {partition_size_sums}. mem_cap: {mem_cap}."
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


class GreedyCostPartitioner(Partitioner):
    """
    Greedy Partitioner
    """

    def run(
        self,
        sharding_options: List[ShardingOption],
        topology: Topology,
    ) -> None:
        """
        Places sharding options on topology based on each sharding option's partition_by attribute.
        Topology storage and costs are updated at the end of the placement.

        Args:
            sharding_options (List[ShardingOption]): list of populated sharding options.
            topology (Topology): device topology.

        Returns:
            None.

        Example:

        sharding_options = [
                            ShardingOption(partition_by="uniform",
                                    shards=[
                                        Shards(storage=1, cost=1),
                                        Shards(storage=1, cost=1),
                                    ]),
                            ShardingOption(partition_by="uniform",
                                    shards=[
                                        Shards(storage=2, cost=2),
                                        Shards(storage=2, cost=2),
                                    ]),
                            ShardingOption(partition_by="device",
                                    shards=[
                                        Shards(storage=3, cost=3),
                                        Shards(storage=3, cost=3),
                                    ])
                            ShardingOption(partition_by="device",
                                    shards=[
                                        Shards(storage=4, cost=4),
                                        Shards(storage=4, cost=4),
                                    ])
                            ]
        topology = Topology(world_size=2)

        First [sharding_options[0],sharding_options[1]] will be placed on topology with the uniform strategy, resulting in

        topology.devices[0].cost = (1,1)
        topology.devices[1].cost = (2,2)

        Finally sharding_options[2],sharding_options[3]] will be placed on topology with the device strategy (see doc string of partition_by_device for more details).

        topology.devices[0].cost = (1,1) + (4,4)
        topology.devices[1].cost = (2,2) + (3,3)

        The topology updates are actually done after the end of all the placements (the othering in the example is just for clarity).
        """

        # pyre-ignore[16]
        self._topology = topology
        grouped_sharding_options = _group_sharding_options(sharding_options)

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

    def _partition_by_uniform(self, sharding_options: List[ShardingOption]) -> None:
        partitions = uniform_partition(
            # pyre-ignore [16]: `GreedyCostPartitioner` has no attribute `_topology`.
            num_partitions=self._topology.world_size,
            sharding_options=sharding_options,
            mem_cap=[device.storage for device in self._topology.devices],
        )
        self._update_shards(partitions, sharding_options)

    def _partition_by_device(self, sharding_options: List[ShardingOption]) -> None:
        # pyre-ignore [16]: `GreedyCostPartitioner` has no attribute `_topology`.
        partition_sums = [float(device.cost) for device in self._topology.devices]
        mem_cap: List[Storage] = [device.storage for device in self._topology.devices]
        partitions = greedy_partition(
            num_partitions=self._topology.world_size,
            sharding_options=sharding_options,
            partition_sums=partition_sums,
            mem_cap=mem_cap,
        )
        self._update_shards(partitions, sharding_options)

    def _partition_by_host(self, sharding_options: List[ShardingOption]) -> None:
        # pyre-ignore [16]: `GreedyCostPartitioner` has no attribute `_topology`.
        num_hosts: int = self._topology.world_size // self._topology.local_world_size
        mem_cap: List[Storage] = []
        partition_sums = []

        shard_idxes = []
        for option_idx, _ in enumerate(sharding_options):
            # only take the first shard from each sharding option. We can infer the rest
            shard_idxes.append((option_idx, 0))

        for i in range(num_hosts):
            devices_in_host = self._topology.devices[
                i
                * self._topology.local_world_size : (i + 1)
                * self._topology.local_world_size
            ]

            # mem_cap of a host is the min of the storage of all devies on that host
            mem_cap.append(min([device.storage for device in devices_in_host]))
            # Cost of a host is the sum across all of its devices. Typically this should be zero at entry point.
            partition_sums.append(
                max([float(device.cost) for device in devices_in_host])
            )

        host_level_partitions: List[List[Tuple[int, int]]] = greedy_partition(
            num_partitions=num_hosts,
            sharding_options=sharding_options,
            shard_idxes=shard_idxes,
            partition_sums=partition_sums,
            mem_cap=mem_cap,
        )
        partitions: List[List[Tuple[int, int]]] = [[] for _ in self._topology.devices]
        for host_idx, host_partition in enumerate(host_level_partitions):
            for [option_idx, shard_idx] in host_partition:
                # each shard is placed on one device
                # host+idx + offset is the device within that host
                for offset in range(sharding_options[option_idx].num_shards):
                    partitions[
                        self._topology.local_world_size * host_idx + offset
                    ].append((option_idx, shard_idx + offset))
        self._update_shards(partitions, sharding_options)

    def _update_shards(
        self,
        partitions: List[List[Tuple[int, int]]],
        sharding_options: List[ShardingOption],
    ) -> None:

        # here we update the ranks of the shards as well as device costs
        for partition_idx, partition in enumerate(partitions):
            for [option_idx, shard_idx] in partition:
                sharding_options[option_idx].shards[shard_idx].rank = partition_idx
                # pyre-ignore [16]: `GreedyCostPartitioner` has no attribute `_topology`.
                self._topology.devices[partition_idx].storage -= (
                    sharding_options[option_idx].shards[shard_idx].storage
                )
                self._topology.devices[partition_idx].cost += (
                    sharding_options[option_idx].shards[shard_idx].cost
                )
