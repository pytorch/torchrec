#!/usr/bin/env python3

import math
from typing import Any, Type, Dict, Optional, List, cast, Tuple

import torch
import torch.distributed as dist
from torchrec.distributed.comm import get_local_size, get_num_groups
from torchrec.distributed.planner.parameter_sharding import ParameterShardingFactory
from torchrec.distributed.planner.types import (
    ShardingOption,
    Topology,
    DeviceInfo,
    HostInfo,
    ParameterInfo,
    Storage,
    ParamSortKey,
)
from torchrec.distributed.types import (
    ParameterStorage,
    ShardingPlan,
    ShardingType,
)

MAX_DDR_STORAGE: int = 4 * 1024 * 1024 * 1024 * 1024  # 4 TB
MIN_DIM: int = 32

SHARDING_PREFERENCE: Dict[str, int] = {
    ShardingType.DATA_PARALLEL.value: 0,
    ShardingType.TABLE_WISE.value: 1,
    ShardingType.TABLE_ROW_WISE.value: 2,
    ShardingType.ROW_WISE.value: 3,
    ShardingType.COLUMN_WISE.value: 4,
}


def gb_to_bytes(gb: int) -> int:
    return gb * 1024 * 1024 * 1024


# pyre-ignore[2]
def sharder_name(t: Type[Any]) -> str:
    return t.__module__ + "." + t.__name__


def is_enough_storage(
    sharding_option: ShardingOption,
    topology: Topology,
    device: Optional[DeviceInfo] = None,
) -> bool:
    storage = sharding_option.storage_usage
    device_ranks = range(topology.world_size)
    host_ranks = range(len(topology.hosts))
    if sharding_option.sharding_type == ShardingType.DATA_PARALLEL.value:
        pass
    elif sharding_option.sharding_type == ShardingType.ROW_WISE.value:
        storage = {
            k: math.ceil(v / len(device_ranks if k == "hbm" else host_ranks))
            for k, v in storage.items()
        }
    elif sharding_option.sharding_type == ShardingType.TABLE_ROW_WISE.value:
        assert (
            device is not None
        ), "Sharding option must have a device for TWRW storage calcuation"
        device_ranks = [
            device.rank for device in topology.get_host(device.rank).devices
        ]
        host_ranks = [topology.host_and_device_by_rank[device.rank][0]]
        storage = {
            k: math.ceil(v / len(device_ranks if k == "hbm" else host_ranks))
            for k, v in storage.items()
        }
    elif sharding_option.sharding_type == ShardingType.TABLE_WISE.value:
        assert (
            device is not None
        ), "Sharding option must have a device for TW storage calcuation"
        device_ranks = [device.rank]
        host_ranks = [topology.host_and_device_by_rank[device.rank][0]]
    elif sharding_option.sharding_type == ShardingType.COLUMN_WISE.value:
        assert (
            device is not None
        ), "Sharding option must have a device for CW storage calcuation"
        device_ranks = [device.rank]
        host_ranks = [topology.host_and_device_by_rank[device.rank][0]]
        storage = {
            # pyre-fixme[58]
            k: math.ceil(v / sharding_option.shards_count)
            for k, v in storage.items()
        }
    else:
        raise ValueError(f"unsupported sharding_type {sharding_option.sharding_type}")
    for storage_type, storage_usage in storage.items():
        if storage_type == ParameterStorage.HBM.value:
            for device_rank in device_ranks:
                if topology.get_device(device_rank).hbm.free < storage_usage:
                    return False
        elif storage_type == ParameterStorage.DDR.value:
            for host_rank in host_ranks:
                if topology.get_host(host_rank).ddr.free < storage_usage:
                    return False
        elif storage_type == ParameterStorage.SSD.value:
            for host_rank in host_ranks:
                if topology.get_host(host_rank).ssd.free < storage_usage:
                    return False
        else:
            raise ValueError(f"Unknown ParameterStorage type {storage_type}")
    return True


def allocate_param(
    sharding_option: ShardingOption, topology: Topology, is_deallocation: bool = False
) -> None:
    """
    Reduces relevant free storage in toplogy based on sharding option

    Setting is_deallocation=True will do inverse (free up storage)
    """
    storage = sharding_option.storage_usage
    device_ranks = range(topology.world_size)
    host_ranks = range(len(topology.hosts))
    if sharding_option.sharding_type == ShardingType.DATA_PARALLEL.value:
        pass
    elif sharding_option.sharding_type == ShardingType.ROW_WISE.value:
        storage = {
            k: math.ceil(v / len(device_ranks if k == "hbm" else host_ranks))
            for k, v in storage.items()
        }
    elif sharding_option.sharding_type == ShardingType.TABLE_ROW_WISE.value:
        assert (
            sharding_option.ranks is not None
        ), "Sharding option must have a device for TWRW storage calcuation"
        device_ranks = [
            device.rank
            # pyre-fixme[22]: The cast is redundant.
            for device in topology.get_host(cast(int, sharding_option.ranks[0])).devices
        ]
        host_ranks = [
            # pyre-fixme[22]: The cast is redundant.
            topology.host_and_device_by_rank[cast(int, sharding_option.ranks[0])][0]
        ]
        storage = {
            k: math.ceil(v / len(device_ranks if k == "hbm" else host_ranks))
            for k, v in storage.items()
        }
    elif sharding_option.sharding_type == ShardingType.TABLE_WISE.value:
        assert (
            sharding_option.ranks is not None
        ), "Sharding option must have a device for TW storage calcuation"
        # pyre-fixme[22]: The cast is redundant.
        device_ranks = [cast(int, sharding_option.ranks[0])]
        # pyre-fixme[16]
        host_ranks = [topology.host_and_device_by_rank[sharding_option.ranks[0]][0]]
    elif sharding_option.sharding_type == ShardingType.COLUMN_WISE.value:
        assert (
            sharding_option.ranks is not None
        ), "Sharding option must have at least one device for CW storage calcuation"
        # for col-wise sharding, we allocate one shard at a time
        device_ranks = [sharding_option.ranks[-1]]
        host_ranks = [topology.host_and_device_by_rank[sharding_option.ranks[-1]][0]]
        storage = {
            # pyre-fixme[58]
            k: math.ceil(v / sharding_option.shards_count)
            for k, v in storage.items()
        }
    else:
        raise ValueError(f"unsupported sharding_type {sharding_option.sharding_type}")

    for storage_type, storage_usage in storage.items():
        if is_deallocation:
            storage_usage = -storage_usage
        if storage_type == ParameterStorage.HBM.value:
            for device_rank in device_ranks:
                topology.get_device(device_rank).hbm.free -= storage_usage
        elif storage_type == ParameterStorage.DDR.value:
            for host_rank in host_ranks:
                topology.get_host(host_rank).ddr.free -= storage_usage
        elif storage_type == ParameterStorage.SSD.value:
            for host_rank in host_ranks:
                topology.get_host(host_rank).ssd.free -= storage_usage
        else:
            raise ValueError(f"Unknown ParameterStorage type {storage_type}")

    for device_rank in device_ranks:
        cost = -sharding_option.cost if is_deallocation else sharding_option.cost
        topology.get_device(device_rank).total_cost += cost


def deallocate_param(
    sharding_option: ShardingOption,
    topology: Topology,
) -> None:
    allocate_param(sharding_option, topology, is_deallocation=True)


def param_sort_key(
    parameter_info: ParameterInfo, world_size: int, sort_by: str = "compute"
) -> ParamSortKey:
    sharding_option = parameter_info.sharding_options[0]
    compute_cost = sharding_option.cost
    storage_cost = sum(sharding_option.storage_usage.values())
    if sharding_option.sharding_type == ShardingType.DATA_PARALLEL.value:
        storage_cost *= world_size
    sharding_preference = SHARDING_PREFERENCE[
        parameter_info.sharding_options[0].sharding_type
    ]
    return ParamSortKey(
        compute_cost=compute_cost,
        storage_cost=storage_cost,
        sharding_cost=sharding_preference,
        fqn=parameter_info.fqn,
        sort_by=sort_by,
    )


def to_plan(
    parameter_infos: List[ParameterInfo],
    world_size: int,
    local_size: Optional[int],
) -> ShardingPlan:
    plan = {}
    for parameter_info in parameter_infos:
        shards = plan.get(parameter_info.prefix, {})
        shards[parameter_info.name] = ParameterShardingFactory.shard_parameters(
            param_info=parameter_info,
            world_size=world_size,
            local_size=local_size,
        )
        plan[parameter_info.prefix] = shards
    return ShardingPlan(plan)


def _get_storage(
    device: torch.device, storage_in_gb: Optional[Dict[str, int]]
) -> Dict[str, int]:
    if storage_in_gb is None:
        storage_in_gb = {}

    hbm = storage_in_gb.get("hbm", None)
    if hbm is None and device.type == "cuda":
        hbm = torch.cuda.get_device_properties(device).total_memory
    elif hbm is None:
        hbm = 0
    else:
        hbm = gb_to_bytes(hbm)

    ddr = storage_in_gb.get("ddr", None)
    if ddr is None:
        ddr = MAX_DDR_STORAGE

    ssd = gb_to_bytes(storage_in_gb.get("ssd", 0))

    return {
        "hbm": hbm,
        "ddr": ddr,
        "ssd": ssd,
    }


def get_topology(
    pg: dist.ProcessGroup,
    device: torch.device,
    storage_in_gb: Optional[Dict[str, int]],
) -> Topology:
    world_size = dist.get_world_size(pg)
    devices_per_host = get_local_size()
    num_hosts = get_num_groups()
    compute_device = device.type
    storage = _get_storage(device, storage_in_gb)
    topology = Topology(
        hosts=[
            HostInfo(
                devices=[
                    DeviceInfo(
                        rank=rank,
                        compute_device=compute_device,
                        hbm=Storage(
                            capacity=storage["hbm"],
                            free=storage["hbm"],
                        ),
                    )
                    for rank in range(
                        num_host * devices_per_host,
                        min(world_size, (num_host + 1) * devices_per_host),
                    )
                ],
                ddr=Storage(
                    capacity=storage["ddr"],
                    free=storage["ddr"],
                ),
                ssd=Storage(
                    capacity=storage["ssd"],
                    free=storage["ssd"],
                ),
            )
            for num_host in range(num_hosts)
        ],
        world_size=world_size,
    )
    for i, host in enumerate(topology.hosts):
        for j, device in enumerate(host.devices):
            topology.host_and_device_by_rank[device.rank] = (i, j)
    return topology
