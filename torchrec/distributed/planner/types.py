#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
from dataclasses import field, dataclass
from enum import Enum
from typing import Optional, List, Dict, Tuple, Union

import torch
from torch import nn
from torchrec.distributed.planner.constants import (
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
    """
    Representation of the storage capacities of a hardware used in training.
    """

    hbm: int
    ddr: int

    def __add__(self, other: "Storage") -> "Storage":
        return Storage(
            hbm=self.hbm + other.hbm,
            ddr=self.ddr + other.ddr,
        )

    def __sub__(self, other: "Storage") -> "Storage":
        return Storage(
            hbm=self.hbm - other.hbm,
            ddr=self.ddr - other.ddr,
        )

    def __hash__(self) -> int:
        return hash((self.hbm, self.ddr))


@dataclass
class DeviceHardware:
    """
    Representation of a device in a process group. 'perf' is an estimation of network,
    CPU, and storage usages.
    """

    rank: int
    storage: Storage
    perf: int = 0


class Topology:
    def __init__(
        self,
        world_size: int,
        compute_device: str,
        hbm_cap: Optional[int] = None,
        ddr_cap: int = DDR_CAP,
        local_world_size: Optional[int] = None,
        intra_host_bw: float = INTRA_NODE_BANDWIDTH,
        inter_host_bw: float = CROSS_NODE_BANDWIDTH,
        batch_size: int = BATCH_SIZE,
    ) -> None:
        """
        Representation of a network of devices in a cluster.
        """
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
    def intra_host_bw(self) -> float:
        return self._intra_host_bw

    @property
    def inter_host_bw(self) -> float:
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
    """
    Representation of a subset of an embedding table. 'size' and 'offset' fully
    determine the tensors in the shard. 'storage' is an estimation of how much it takes
    to store the shard with an estimation 'perf'.
    """

    size: List[int]
    offset: List[int]
    storage: Optional[Storage] = None
    perf: Optional[float] = None
    rank: Optional[int] = None

    def __hash__(self) -> int:
        return hash(
            (
                tuple(self.size),
                tuple(self.offset),
                self.storage,
                self.perf,
                self.rank,
            )
        )


@dataclass
class ShardingOption:
    """
    One way of sharding an embedding table.
    """

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

    def __hash__(self) -> int:
        return hash(
            (
                self.fqn,
                self.sharding_type,
                self.compute_kernel,
                tuple(self.shards),
            )
        )


class PartitionByType(Enum):
    """
    Well-known partition types.
    """

    # Partitioning based on device
    DEVICE = "device"
    # Partitioning based on host
    HOST = "host"
    # Uniform, (ie. fixed layout)
    UNIFORM = "uniform"


@dataclass
class ParameterConstraints:
    """
    Stores user provided constraints around the sharding plan.
    """

    sharding_types: Optional[List[str]] = None
    compute_kernels: Optional[List[str]] = None
    min_partition: Optional[int] = None  # CW sharding
    caching_ratio: Optional[float] = None  # UVM caching
    pooling_factors: List[float] = field(
        default_factory=lambda: [POOLING_FACTOR]
    )  # Embedding Tables


class PlannerError(Exception):
    ...


# ---- PLANNER COMPONENTS ---- #


class StorageReservation(abc.ABC):
    """
    Reserves storage space for non-sharded parts of the model.
    """

    @abc.abstractmethod
    def reserve(
        self,
        topology: Topology,
        module: nn.Module,
        sharders: List[ModuleSharder[nn.Module]],
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
    ) -> Topology:
        ...


class PerfModel(abc.ABC):
    @abc.abstractmethod
    def rate(self, plan: List[ShardingOption]) -> float:
        ...


class ShardEstimator(abc.ABC):
    """
    Estimates shard perf or storage, requires fully specified sharding options.
    """

    @abc.abstractmethod
    def __init__(
        self,
        topology: Topology,
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
    ) -> None:
        ...

    @abc.abstractmethod
    def estimate(
        self,
        sharding_options: List[ShardingOption],
        sharder_map: Optional[Dict[str, ModuleSharder[nn.Module]]] = None,
    ) -> None:
        # update sharding_options with per shard estimate in-place
        ...


class Enumerator(abc.ABC):
    """
    Generates all relevant sharding options for given topology, constraints, nn.Module,
    and sharders.
    """

    @abc.abstractmethod
    def __init__(
        self,
        topology: Topology,
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
        estimator: Optional[Union[ShardEstimator, List[ShardEstimator]]] = None,
    ) -> None:
        ...

    @abc.abstractmethod
    def enumerate(
        self,
        module: nn.Module,
        sharders: List[ModuleSharder[nn.Module]],
    ) -> List[ShardingOption]:
        """
        See class description.
        """
        ...


class Proposer(abc.ABC):
    """
    Prosposes complete lists of sharding options which can be parititioned to generate a
    plan.
    """

    @abc.abstractmethod
    def load(
        self,
        search_space: List[ShardingOption],
    ) -> None:
        ...

    @abc.abstractmethod
    def feedback(
        self,
        partitionable: bool,
        plan: Optional[List[ShardingOption]] = None,
        perf_rating: Optional[float] = None,
    ) -> None:
        ...

    @abc.abstractmethod
    def propose(self) -> Optional[List[ShardingOption]]:
        ...


class Partitioner(abc.ABC):
    """
    Partitions shards.

    Today we have multiple strategies ie. (Greedy, BLDM, Linear).
    """

    @abc.abstractmethod
    def partition(
        self,
        proposal: List[ShardingOption],
        storage_constraint: Topology,
    ) -> List[ShardingOption]:
        # modifies sharding_options and topology in-place
        ...


class Stats(abc.ABC):
    """
    Logs statistics related to the sharding plan.
    """

    @abc.abstractmethod
    def log(
        self,
        sharding_plan: ShardingPlan,
        topology: Topology,
        num_proposals: int,
        num_plans: int,
        best_plan: List[ShardingOption],
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
        debug: bool = False,
    ) -> None:
        """
        See class description
        """
        ...
