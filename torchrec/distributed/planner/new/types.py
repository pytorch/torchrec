#!/usr/bin/env python3

from __future__ import annotations

import abc
from dataclasses import field, dataclass
from enum import Enum
from typing import Optional, List, Dict, Tuple

import torch
from torch import nn
from torchrec.distributed.planner.new.constants import (
    CROSS_NODE_BANDWIDTH,
    INTRA_NODE_BANDWIDTH,
    HBM_CAP,
    DDR_CAP,
    POOLING_FACTOR,
    BATCH_SIZE,
)
from torchrec.distributed.types import ModuleSharder, ShardingPlan

# ---- TOPOLOGY ---- #


@dataclass(repr=True, order=True, eq=True)
class Storage:
    hbm: int
    ddr: int

    def __add__(self, new: Storage) -> Storage:
        return Storage(
            hbm=self.hbm + new.hbm,
            ddr=self.ddr + new.ddr,
        )

    def __sub__(self, new: Storage) -> Storage:
        return Storage(
            hbm=self.hbm - new.hbm,
            ddr=self.ddr - new.ddr,
        )


@dataclass
class DeviceHardware:
    rank: int
    storage: Storage
    cost: int = 0


class Topology:
    def __init__(
        self,
        world_size: int,
        compute_device: str,
        hbm_cap: Optional[int] = None,
        ddr_cap: int = DDR_CAP,
        local_world_size: Optional[int] = None,
        intra_host_bw: int = INTRA_NODE_BANDWIDTH,
        inter_host_bw: int = CROSS_NODE_BANDWIDTH,
        batch_size: int = BATCH_SIZE,
    ) -> None:
        # validate input
        assert compute_device in [
            "cpu",
            "cuda",
        ], f"unsupported compute device {compute_device}"

        self._compute_device = compute_device
        self._world_size = world_size

        hbm_per_device = 0
        if self._compute_device == "cuda":
            hbm_per_device = hbm_cap if hbm_cap else HBM_CAP

        self._devices: List[DeviceHardware] = []
        for rank in range(world_size):
            self._devices.append(
                DeviceHardware(
                    rank=rank,
                    storage=Storage(hbm=hbm_per_device, ddr=ddr_cap),
                )
            )

        self._local_world_size: int = (
            local_world_size if local_world_size else world_size
        )
        self._intra_host_bw = intra_host_bw
        self._inter_host_bw = inter_host_bw
        self._batch_size = batch_size

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def compute_device(self) -> str:
        return self._compute_device

    @property
    def devices(self) -> List[DeviceHardware]:
        return self._devices

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def local_world_size(self) -> int:
        return self._local_world_size

    @property
    def intra_host_bw(self) -> int:
        return self._intra_host_bw

    @property
    def inter_host_bw(self) -> int:
        return self._inter_host_bw

    def __repr__(self) -> str:
        topology_repr: str = f"world_size={self._world_size} \n"
        topology_repr += f"compute_device={self._compute_device}\n"
        topology_repr += "devices=\n"
        for idx, device in enumerate(self._devices):
            topology_repr += f"\tdevice {idx} {device}\n"
        topology_repr += f"local_world_size={self._local_world_size} \n"
        topology_repr += f"intra_host_bw={self._intra_host_bw} \n"
        topology_repr += f"inter_host_bw={self._inter_host_bw} \n"
        return topology_repr


# ---- INPUT / OUTPUT ----- #


@dataclass
class Shard:
    length: List[int]
    offset: List[int]
    storage: Storage
    cost: Optional[float] = None
    rank: Optional[int] = None


@dataclass
class ShardingOption:
    name: str
    tensor: torch.Tensor
    module: Tuple[str, nn.Module]
    upstream_modules: List[Tuple[str, nn.Module]]
    downstream_modules: List[Tuple[str, nn.Module]]
    input_lengths: List[float]
    batch_size: int  # per single device
    sharding_type: str
    partition_by: str  # {DEVICE, HOST, UNIFORM}
    compute_kernel: str
    cost: Optional[float] = None  # main ranker value
    # relevant to planner output, must be populated if sharding option
    # part of final solution
    shards: List[Shard] = field(default_factory=list)

    @property
    def fqn(self) -> str:
        return self.module[0] + "." + self.name

    @property
    def path(self) -> str:
        return self.module[0]

    @property
    def num_shards(self) -> int:
        return len(self.shards)

    @property
    def num_inputs(self) -> int:
        return len(self.input_lengths)


class PartitionByType(Enum):
    """
    Well-known partition types
    """

    # Partitioning based on device
    DEVICE = "device"
    # Partitioning based on host
    HOST = "host"
    # Uniform, (ie. fixed layout)
    UNIFORM = "uniform"


@dataclass
class PlannerConstraints:
    """
    Stores user provided constraints around
    sharding types, compute kernels and partitioning
    """

    sharding_types: Optional[List[str]] = None
    compute_kernels: Optional[List[str]] = None
    min_partition: Optional[int] = None  # CW sharding
    caching_ratio: Optional[float] = None  # UVM caching


@dataclass
class InputStats:
    """
    Stores statistics around input data for
    a given tensor
    """

    pooling_factors: List[float] = field(default_factory=lambda: [POOLING_FACTOR])


class PartitionError(Exception):
    ...


@dataclass
class PlacerStats:
    num_iterations: int
    num_errors: int
    topology_solution: Optional[Topology]
    sharding_solution: Optional[List[ShardingOption]]


# ---- PLANNER COMPONENTS ---- #


class Enumerator(abc.ABC):
    """
    Generate all relevant sharding options for give nn.Module,
    input stats and user constraints
    """

    @abc.abstractmethod
    def __init__(
        self,
        topology: Topology,
        constraints: Optional[Dict[str, PlannerConstraints]] = None,
        input_stats: Optional[Dict[str, InputStats]] = None,
    ) -> None:
        ...

    @abc.abstractmethod
    def run(
        self, module: nn.Module, sharders: List[ModuleSharder[nn.Module]]
    ) -> List[ShardingOption]:
        ...


class Calculator(abc.ABC):
    """
    calc costs, requires fully specificed sharding option (ie. ranks/lengths)
    """

    @abc.abstractmethod
    def __init__(self, topology: Topology) -> None:
        ...

    @abc.abstractmethod
    def run(self, sharding_options: List[ShardingOption]) -> None:
        # actual costs
        ...


class RankStack(abc.ABC):
    """
    "Stack"-like interface to manage complexity of providing
    next sharding option for placer
    """

    @abc.abstractmethod
    def pop(self) -> ShardingOption:
        # pop next sharding option, no more than one sharding option per tensor
        # should be returned
        ...

    @abc.abstractmethod
    def push(self, sharding_option: ShardingOption) -> None:
        # push back shading_option, rerank as necessary
        ...

    @abc.abstractmethod
    def remove(self, sharding_option: ShardingOption) -> bool:
        # remove a given sharding_option from consideration
        ...

    @abc.abstractmethod
    def bulk_pop(self) -> List[ShardingOption]:
        # pop any remaining sharing options
        ...

    @abc.abstractmethod
    def bulk_push(self, sharding_options: List[ShardingOption]) -> None:
        # push a list of sharding options
        ...


class Ranker(abc.ABC):
    """
    Given a calculator, topology and sharding options, populate a
    RankStack and return it
    """

    @abc.abstractmethod
    def run(self, sharding_options: List[ShardingOption]) -> RankStack:
        ...


class Partitioner(abc.ABC):
    """
    Parition

    Today we have multiple stratigies ie.
    (Greedy, BLDM, Linear)
    """

    @abc.abstractmethod
    def run(
        self,
        sharding_options: List[ShardingOption],
        topology: Topology,
    ) -> None:
        # modifies sharding_options and topology in-place
        ...


class Placer(abc.ABC):
    """
    Controls actual placement via:
    1) calls to rank stack
    2) calling into partitioners
    3) final ShardingOptions
    4) determining stopping conditions
    """

    @abc.abstractmethod
    def __init__(
        self,
        topology: Topology,
        partitioners: Optional[List[Partitioner]] = None,
        rankers: Optional[List[Ranker]] = None,
    ) -> None:
        ...

    @abc.abstractmethod
    def run(self, sharding_options: List[ShardingOption]) -> ShardingPlan:
        ...

    @property
    @abc.abstractmethod
    def stats(self) -> PlacerStats:
        ...


class Stats(abc.ABC):
    """
    Log statistics related to the sharding plan
    """

    @abc.abstractmethod
    def run(
        self,
        sharding_plan: ShardingPlan,
        topology: Topology,
        placer_stats: PlacerStats,
        input_stats: Optional[Dict[str, InputStats]] = None,
    ) -> None:
        ...
