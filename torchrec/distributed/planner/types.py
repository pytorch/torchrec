#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import cast, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torchrec.distributed.planner.constants import (
    BATCH_SIZE,
    CROSS_NODE_BANDWIDTH,
    DDR_CAP,
    HBM_CAP,
    INTRA_NODE_BANDWIDTH,
    POOLING_FACTOR,
)
from torchrec.distributed.types import ModuleSharder, ShardingPlan
from torchrec.modules.embedding_modules import EmbeddingCollectionInterface

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

    def fits_in(self, other: "Storage") -> bool:
        return self.hbm <= other.hbm and self.ddr <= other.ddr


@dataclass
class DeviceHardware:
    """
    Representation of a device in a process group. 'perf' is an estimation of network,
    CPU, and storage usages.
    """

    rank: int
    storage: Storage
    perf: float = 0


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


class ShardingOption:
    """
    One way of sharding an embedding table.
    """

    def __init__(
        self,
        name: str,
        tensor: torch.Tensor,
        module: Tuple[str, nn.Module],
        input_lengths: List[float],
        batch_size: int,
        sharding_type: str,
        partition_by: str,
        compute_kernel: str,
        shards: List[Shard],
        dependency: Optional[str] = None,
    ) -> None:
        self.name = name
        self._tensor = tensor
        self._module = module
        self.input_lengths = input_lengths
        self.batch_size = batch_size
        self.sharding_type = sharding_type
        self.partition_by = partition_by
        self.compute_kernel = compute_kernel
        # relevant to planner output, must be populated if sharding option
        # part of final solution
        self.shards = shards
        self.dependency = dependency

    @property
    def tensor(self) -> torch.Tensor:
        return self._tensor

    @property
    def module(self) -> Tuple[str, nn.Module]:
        return self._module

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

    @property
    def total_storage(self) -> Storage:
        storage: Storage = Storage(hbm=0, ddr=0)
        for shard in self.shards:
            storage += cast(Storage, shard.storage)
        return storage

    @property
    def is_pooled(self) -> bool:
        if isinstance(self.module[1], EmbeddingCollectionInterface):
            return False
        for name, module in self.module[1].named_modules():
            if self.name in name:
                if isinstance(module, EmbeddingCollectionInterface):
                    return False
        return True

    def __hash__(self) -> int:
        return hash(
            (
                self.fqn,
                self.sharding_type,
                self.compute_kernel,
                tuple(self.shards),
            )
        )

    def __deepcopy__(
        self, memo: Optional[Dict[int, "ShardingOption"]]
    ) -> "ShardingOption":
        cls = self.__class__
        result = cls.__new__(cls)
        for k, v in self.__dict__.items():
            if k in ["_tensor", "_module"]:
                setattr(result, k, v)
            else:
                setattr(result, k, deepcopy(v, memo))
        return result


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

    If provided, `pooling_factors`, `num_poolings`, and `batch_sizes` must match in
    length, as per sample.
    """

    sharding_types: Optional[List[str]] = None
    compute_kernels: Optional[List[str]] = None
    min_partition: Optional[int] = None  # CW sharding
    pooling_factors: List[float] = field(
        default_factory=lambda: [POOLING_FACTOR]
    )  # average number of embedding lookups required per sample
    num_poolings: Optional[List[float]] = None  # number of poolings per sample in batch
    batch_sizes: Optional[List[int]] = None  # batch size per input feature
    is_weighted: bool = False


class PlannerErrorType(Enum):
    """
    Classify PlannerError based on the following cases.
    """

    INSUFFICIENT_STORAGE = "insufficient_storage"
    STRICT_CONSTRAINTS = "strict_constraints"
    PARTITION = "partition"
    OTHER = "other"


class PlannerError(Exception):
    def __init__(
        self,
        message: str,
        error_type: PlannerErrorType = PlannerErrorType.OTHER,
    ) -> None:
        self.error_type = error_type
        super().__init__(message)


# ---- PLANNER COMPONENTS ---- #


class StorageReservation(abc.ABC):
    """
    Reserves storage space for non-sharded parts of the model.
    """

    @abc.abstractmethod
    def reserve(
        self,
        topology: Topology,
        batch_size: int,
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
        batch_size: int = BATCH_SIZE,
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
        batch_size: int,
        storage_reservation: StorageReservation,
        num_proposals: int,
        num_plans: int,
        run_time: float,
        best_plan: List[ShardingOption],
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
        debug: bool = False,
    ) -> None:
        """
        See class description
        """
        ...
