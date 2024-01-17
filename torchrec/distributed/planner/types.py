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
    BWD_COMPUTE_MULTIPLIER,
    CROSS_NODE_BANDWIDTH,
    DDR_CAP,
    DDR_MEM_BW,
    HBM_CAP,
    HBM_MEM_BW,
    INTRA_NODE_BANDWIDTH,
    POOLING_FACTOR,
)
from torchrec.distributed.types import (
    BoundsCheckMode,
    CacheParams,
    ModuleSharder,
    ShardingPlan,
)
from torchrec.modules.embedding_modules import EmbeddingCollectionInterface
from torchrec.modules.mc_embedding_modules import ManagedCollisionEmbeddingCollection

# ---- Perf ---- #


@dataclass(repr=True, eq=True)
class Perf:
    """
    Representation of the breakdown of the perf estimate a single shard of an
    embedding table.
    """

    fwd_compute: float
    fwd_comms: float
    bwd_compute: float
    bwd_comms: float
    prefetch_compute: float = 0.0

    @property
    def total(self) -> float:
        # When using embedding offload, there is a prefetch compute component. This
        # prefetch can overlap with fwd_compute + fwd_comm and dense fwd (some of it
        # overlaps with fwd_compute) and dense bwd. (fwd_compute and bwd_compute are
        # embedding fwd/bwd, nothing to do with dense). Only when prefetch is longer
        # than fwd_compute + dense_fwd + dense_bwd it will block bwd_compute. However,
        # we don't have an effective way to estimate dense fwd/bwd at this point, so our
        # cost model is too simplistic.  Instead prefetch is always considered blocking.
        #
        # Also note, measuring prefetch blocking can only be done after partitioning,
        # here are only have the per shard estimates.
        #
        # However adding a per-shard prefetch component to the cost model does have the
        # benefit that 1) it enables the ScaleupProposer to explore the trade off
        # between increasing cache sizes vs more difficult bin-packing constraints. 2)
        # it helps balance the prefetch compute across the ranks.
        return (
            self.fwd_compute
            + self.bwd_compute
            + self.fwd_comms
            + self.bwd_comms
            + self.prefetch_compute
        )

    def __add__(self, other: "Perf") -> "Perf":
        return Perf(
            fwd_compute=self.fwd_compute + other.fwd_compute,
            fwd_comms=self.fwd_comms + other.fwd_comms,
            bwd_compute=self.bwd_compute + other.bwd_compute,
            bwd_comms=self.bwd_comms + other.bwd_comms,
            prefetch_compute=self.prefetch_compute + other.prefetch_compute,
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.fwd_compute,
                self.fwd_comms,
                self.bwd_compute,
                self.bwd_comms,
                self.prefetch_compute,
            )
        )


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
    perf: Perf


class Topology:
    def __init__(
        self,
        world_size: int,
        compute_device: str,
        hbm_cap: Optional[int] = None,
        ddr_cap: Optional[int] = None,
        local_world_size: Optional[int] = None,
        hbm_mem_bw: float = HBM_MEM_BW,
        ddr_mem_bw: float = DDR_MEM_BW,
        intra_host_bw: float = INTRA_NODE_BANDWIDTH,
        inter_host_bw: float = CROSS_NODE_BANDWIDTH,
        bwd_compute_multiplier: float = BWD_COMPUTE_MULTIPLIER,
    ) -> None:
        """
        Representation of a network of devices in a cluster.
        """
        # validate input
        assert compute_device in [
            "cpu",
            "cuda",
            "mtia",
        ], f"unsupported compute device {compute_device}"

        self._compute_device = compute_device
        self._world_size = world_size

        hbm_per_device = 0
        if self._compute_device == "cuda":
            hbm_per_device = hbm_cap if hbm_cap else HBM_CAP
        ddr_cap = ddr_cap if ddr_cap else DDR_CAP

        self._devices: List[DeviceHardware] = []
        for rank in range(world_size):
            self._devices.append(
                DeviceHardware(
                    rank=rank,
                    storage=Storage(hbm=hbm_per_device, ddr=ddr_cap),
                    perf=Perf(fwd_compute=0, fwd_comms=0, bwd_compute=0, bwd_comms=0),
                )
            )

        self._local_world_size: int = (
            local_world_size if local_world_size else world_size
        )
        self._hbm_mem_bw = hbm_mem_bw
        self._ddr_mem_bw = ddr_mem_bw
        self._intra_host_bw = intra_host_bw
        self._inter_host_bw = inter_host_bw
        self._bwd_compute_multiplier = bwd_compute_multiplier

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
    def hbm_mem_bw(self) -> float:
        return self._hbm_mem_bw

    @property
    def ddr_mem_bw(self) -> float:
        return self._ddr_mem_bw

    @property
    def intra_host_bw(self) -> float:
        return self._intra_host_bw

    @property
    def inter_host_bw(self) -> float:
        return self._inter_host_bw

    @property
    def bwd_compute_multiplier(self) -> float:
        return self._bwd_compute_multiplier

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
    perf: Optional[Perf] = None
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
        cache_params: Optional[CacheParams] = None,
        enforce_hbm: Optional[bool] = None,
        stochastic_rounding: Optional[bool] = None,
        bounds_check_mode: Optional[BoundsCheckMode] = None,
        dependency: Optional[str] = None,
        is_pooled: Optional[bool] = None,
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
        self.cache_params = cache_params
        self.enforce_hbm = enforce_hbm
        self.stochastic_rounding = stochastic_rounding
        self.bounds_check_mode = bounds_check_mode
        self.dependency = dependency
        self._is_pooled = is_pooled
        self.is_weighted: Optional[bool] = None

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
    def cache_load_factor(self) -> Optional[float]:
        if self.cache_params is not None:
            return self.cache_params.load_factor
        return None

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
    def total_perf(self) -> float:
        perf: float = 0
        for shard in self.shards:
            # pyre-ignore: Undefined attribute [16]
            perf += shard.perf.total
        return perf

    @property
    def is_pooled(self) -> bool:
        if self._is_pooled is None:
            self._is_pooled = ShardingOption.module_pooled(self.module[1], self.name)
        return self._is_pooled

    @staticmethod
    def module_pooled(module: nn.Module, sharding_option_name: str) -> bool:
        """Determine if module pools output (e.g. EmbeddingBag) or uses unpooled/sequential output."""
        if isinstance(module, EmbeddingCollectionInterface) or isinstance(
            module, ManagedCollisionEmbeddingCollection
        ):
            return False

        for submodule in module.modules():
            if isinstance(submodule, EmbeddingCollectionInterface) or isinstance(
                module, ManagedCollisionEmbeddingCollection
            ):
                for name, _ in submodule.named_parameters():
                    if sharding_option_name in name:
                        return False

        return True

    def __hash__(self) -> int:
        return hash(
            (
                self.fqn,
                self.sharding_type,
                self.compute_kernel,
                tuple(self.shards),
                self.cache_params,
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
    cache_params: Optional[CacheParams] = None
    enforce_hbm: Optional[bool] = None
    stochastic_rounding: Optional[bool] = None
    bounds_check_mode: Optional[BoundsCheckMode] = None


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

    @abc.abstractmethod
    def populate_estimates(self, sharding_options: List[ShardingOption]) -> None:
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
        enumerator: Optional[Enumerator] = None,
    ) -> None:
        ...

    @abc.abstractmethod
    def feedback(
        self,
        partitionable: bool,
        plan: Optional[List[ShardingOption]] = None,
        perf_rating: Optional[float] = None,
        storage_constraint: Optional[Topology] = None,
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
        sharders: Optional[List[ModuleSharder[nn.Module]]] = None,
        debug: bool = False,
    ) -> None:
        """
        See class description
        """
        ...
