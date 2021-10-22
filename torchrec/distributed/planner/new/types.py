#!/usr/bin/env python3

import abc
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import torch
from torch import nn
from torchrec.distributed.planner.new.constants import HBM_CAP_DEFAULT, DDR_CAP_DEFAULT
from torchrec.distributed.types import ModuleSharder

# ---- TOPOLOGY ---- #


@dataclass
class Storage:
    # In bytes
    hbm: int
    ddr: int


@dataclass
class DeviceHardware:
    rank: int
    storage_capacity: Storage
    storage_remaining: Storage
    cost: int = 0


class Topology:
    def __init__(
        self,
        world_size: int,
        compute_device: str,
        hbm_cap: Optional[int] = None,
        ddr_cap: Optional[int] = None,
    ) -> None:
        # validate input
        assert compute_device in [
            "cpu",
            "cuda",
        ], f"unsupported compute device {compute_device}"

        self._compute_device = compute_device
        self._world_size = world_size

        ddr_per_device = ddr_cap if ddr_cap else DDR_CAP_DEFAULT
        hbm_per_device = 0
        if self._compute_device == "cuda":
            hbm_per_device = hbm_cap if hbm_cap else HBM_CAP_DEFAULT

        self._devices: List[DeviceHardware] = []
        for rank in range(world_size):
            self._devices.append(
                DeviceHardware(
                    rank=rank,
                    storage_capacity=Storage(hbm=hbm_per_device, ddr=ddr_per_device),
                    storage_remaining=Storage(hbm=hbm_per_device, ddr=ddr_per_device),
                )
            )

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
        return self._world_size


# ---- INPUT / OUTPUT ----- #


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
    # main cost/ranker value
    cost: Optional[float] = None

    # relevant to planner output, must be populated if sharding option
    # part of final solution
    shard_lengths: Optional[List[List[int]]] = None  # from enumerator
    shard_offsets: Optional[List[List[int]]] = None  # from enumerator
    shard_storage: Optional[List[Storage]] = None  # from enumerator
    shard_costs: Optional[List[float]] = None  # from cost calculator
    shard_ranks: Optional[List[int]] = None  # from placer

    @property
    def fqn(self) -> str:
        return self.module[0] + "." + self.name

    @property
    def num_of_shards(self) -> int:
        return len(self.shard_lengths) if self.shard_lengths else 0

    @property
    def num_of_inputs(self) -> int:
        return len(self.input_lengths)


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

    pooling_factors: List[float] = [1.0]


# ---- PLANNER COMPONENTS ---- #


class PlannerComponent(abc.ABC):
    """
    Base Class
    """


class Enumerator(PlannerComponent):
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
        batch_size: int = 512,
    ) -> None:
        ...

    @abc.abstractmethod
    def run(
        self, module: nn.Module, sharders: List[ModuleSharder[nn.Module]]
    ) -> List[ShardingOption]:
        ...


class CostCalc(PlannerComponent):
    """
    calc costs, requires fully specificed sharding option (ie. ranks/lengths)
    """

    @abc.abstractmethod
    def __init__(self, topology: Topology) -> None:
        ...

    @abc.abstractmethod
    def run(self, sharding_option: ShardingOption) -> None:
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


class Ranker(PlannerComponent):
    """
    Given a calculator, topology and sharding options, populate a
    RankStack and return it
    """

    @abc.abstractmethod
    def __init__(self, calculator: CostCalc, topology: Topology) -> None:
        ...

    @abc.abstractmethod
    def run(self, sharding_options: List[ShardingOption]) -> RankStack:
        ...


class Partitioner(PlannerComponent):
    """
    Parition

    Today we have multiple stratigies ie.
    (Greedy, BLDM, Linear)
    """

    @abc.abstractmethod
    def run(self, sharding_options: List[ShardingOption], toplogy: Topology) -> None:
        # modifies sharding_options and topology in-place
        ...


class Placer(PlannerComponent):
    """
    Controls actual placement via:
    1) calls to rank stack
    2) calling into partitioners
    3) final ShardingOptions
    4) determining stopping conditions
    """

    @abc.abstractmethod
    def __init__(
        self, topology: Topology, partitioner: Optional[Dict[str, Partitioner]]
    ) -> None:
        ...

    @abc.abstractmethod
    def run(self, rank_stack: RankStack) -> Dict[str, ShardingOption]:
        ...

    @abc.abstractmethod
    def stats(self) -> None:
        # reports stats as a side effect
        ...
