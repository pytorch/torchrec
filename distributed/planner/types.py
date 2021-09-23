#!/usr/bin/env python3

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Deque, Tuple

import torch
from torchrec.distributed.types import ParameterStorage


@dataclass
class ParameterHints:
    """
    Stores user provided hints around
    sharding types and compute kernels
    """

    sharding_types: Optional[List[str]] = None
    compute_kernels: Optional[List[str]] = None
    col_wise_shard_dim: Optional[int] = None


@dataclass
class ParameterInputStats:
    """
    Stores statistics around input data for
    a given parameter
    """

    mean: Optional[List[float]] = None
    std: Optional[List[float]] = None


@dataclass
class Storage:
    capacity: int = 0
    free: int = 0


@dataclass
class DeviceInfo:
    rank: int
    compute_device: str = "cpu"
    total_cost: int = 0
    # Device level storage
    hbm: Storage = field(default_factory=Storage)

    def __lt__(self, other: "DeviceInfo") -> bool:
        return (self.total_cost, -self.hbm.free, self.rank) < (
            other.total_cost,
            -other.hbm.free,
            other.rank,
        )


@dataclass
class HostInfo:
    devices: List[DeviceInfo]
    # Host level storage
    ddr: Storage = field(default_factory=Storage)


@dataclass
class Topology:
    hosts: List[HostInfo]
    world_size: int
    host_and_device_by_rank: Dict[int, Tuple[int, int]] = field(default_factory=dict)

    def get_host(self, rank: int) -> HostInfo:
        host_idx, _ = self.host_and_device_by_rank[rank]
        return self.hosts[host_idx]

    def get_device(self, rank: int) -> DeviceInfo:
        host_idx, device_idx = self.host_and_device_by_rank[rank]
        return self.hosts[host_idx].devices[device_idx]


@dataclass
class ShardingOption:
    sharding_type: str
    compute_kernel: str
    storage_usage: Dict[str, int]
    cost: int = 0
    ranks: Optional[List[int]] = None
    _num_col_wise_shards: Optional[int] = None
    col_wise_shard_dim: Optional[int] = None

    def __lt__(self, other: "ShardingOption") -> bool:
        """
        Sharding option with lowest cost is preferable
        If cost same, pick option with lowest (HBM, DDR, SDD) usage
        """
        return (
            self.cost,
            self.storage_usage.get(ParameterStorage.HBM.value, 0),
            self.storage_usage.get(ParameterStorage.DDR.value, 0),
        ) < (
            other.cost,
            other.storage_usage.get(ParameterStorage.HBM.value, 0),
            other.storage_usage.get(ParameterStorage.DDR.value, 0),
        )


@dataclass
class CostInput:
    param: torch.Tensor
    device: torch.device
    compute_kernel: str
    sharding_type: str
    input_stats: Optional[ParameterInputStats]


@dataclass
class ParameterInfo:
    param: torch.Tensor
    name: str
    prefix: str
    sharding_options: Deque[ShardingOption]

    @property
    def fqn(self) -> str:
        return self.name + "." + self.prefix


@dataclass
class ParamSortKey:
    compute_cost: int
    storage_cost: int
    sharding_cost: int
    fqn: str
    sort_by: str = "compute"

    def __lt__(self, other: "ParamSortKey") -> bool:
        if self.sort_by == "compute":
            return self._lt_compute_cost(other)
        elif self.sort_by == "storage":
            return self._lt_storage_cost(other)
        else:
            raise ValueError(f"Invalid sort_by value {self.sort_by}")

    def _lt_compute_cost(self, other: "ParamSortKey") -> bool:
        return (
            -self.compute_cost,
            -self.storage_cost,
            self.sharding_cost,
            self.fqn,
        ) < (-other.compute_cost, -other.storage_cost, other.sharding_cost, other.fqn)

    def _lt_storage_cost(self, other: "ParamSortKey") -> bool:
        return (
            -self.storage_cost,
            self.sharding_cost,
            -self.compute_cost,
            self.fqn,
        ) < (-other.storage_cost, other.sharding_cost, -other.compute_cost, other.fqn)
