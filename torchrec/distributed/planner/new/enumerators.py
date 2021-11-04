#!/usr/bin/env python3

import math
from typing import Tuple, Optional, Dict, List

import torch
from torch import nn
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.planner.new.constants import (
    DEFAULT_CW_DIM,
    DEFAULT_POOLING_FACTOR,
    BIGINT_DTYPE,
    DEFAULT_CACHING_RATIO,
)
from torchrec.distributed.planner.new.types import (
    PlannerConstraints,
    InputStats,
    Enumerator,
    ShardingOption,
    Shard,
    Storage,
    Topology,
    PartitionByType,
)
from torchrec.distributed.planner.utils import sharder_name
from torchrec.distributed.types import ModuleSharder, ShardingType


class ShardingEnumerator(Enumerator):
    def __init__(
        self,
        topology: Topology,
        constraints: Optional[Dict[str, PlannerConstraints]] = None,
        input_stats: Optional[Dict[str, InputStats]] = None,
        batch_size: int = 512,
    ) -> None:
        self._compute_device: str = topology.compute_device
        self._world_size: int = topology.world_size
        self._local_world_size: int = topology.local_world_size
        self._constraints = constraints
        self._input_stats = input_stats
        self._batch_size = batch_size

    def run(
        self, module: nn.Module, sharders: List[ModuleSharder[nn.Module]]
    ) -> List[ShardingOption]:
        sharder_map: Dict[str, ModuleSharder[nn.Module]] = {
            sharder_name(sharder.module_type): sharder for sharder in sharders
        }
        sharding_options: List[ShardingOption] = []

        for child_path, child_module in module.named_modules():
            sharder_key = sharder_name(type(child_module))
            sharder = sharder_map.get(sharder_key, None)
            if not sharder:
                continue

            for name, param in sharder.shardable_parameters(child_module).items():
                fqn = self._fqn(child_path, name)
                for sharding_type in self._filter_sharding_types(
                    fqn, sharder.sharding_types(self._compute_device)
                ):
                    for compute_kernel in self._filter_compute_kernels(
                        fqn,
                        sharder.compute_kernels(sharding_type, self._compute_device),
                    ):

                        col_wise_shard_dim = (
                            self._constraints[fqn].min_partition
                            if self._constraints and self._constraints.get(fqn)
                            else None
                        )
                        shard_lengths, shard_offsets = get_shard_lengths_and_offsets(
                            param,
                            self._world_size,
                            self._local_world_size,
                            sharding_type,
                            col_wise_shard_dim,
                        )
                        input_lengths = self._get_input_lengths(fqn)
                        caching_ratio = (
                            self._constraints[fqn].caching_ratio
                            if self._constraints and self._constraints.get(fqn)
                            else None
                        )
                        shard_storages = get_shard_storages(
                            sharder=sharder,
                            sharding_type=sharding_type,
                            tensor=param,
                            compute_device=self._compute_device,
                            compute_kernel=compute_kernel,
                            shard_lengths=shard_lengths,
                            batch_size=self._batch_size,
                            world_size=self._world_size,
                            local_world_size=self._local_world_size,
                            input_lengths=input_lengths,
                            caching_ratio=caching_ratio
                            if caching_ratio
                            else DEFAULT_CACHING_RATIO,
                        )
                        sharding_options.append(
                            ShardingOption(
                                name=name,
                                tensor=param,
                                module=(fqn, child_module),
                                upstream_modules=[],
                                downstream_modules=[],
                                input_lengths=input_lengths,
                                batch_size=self._batch_size,
                                compute_kernel=compute_kernel,
                                sharding_type=sharding_type,
                                partition_by=get_partition_by_type(sharding_type),
                                shards=[
                                    Shard(length=length, offset=offset, storage=storage)
                                    for length, offset, storage in zip(
                                        shard_lengths, shard_offsets, shard_storages
                                    )
                                ],
                            )
                        )

        return sharding_options

    def _fqn(self, path: str, name: str) -> str:
        return path + "." + name

    def _filter_sharding_types(self, fqn: str, sharding_types: List[str]) -> List[str]:
        if not self._constraints or not self._constraints.get(fqn):
            return sharding_types
        constraints: PlannerConstraints = self._constraints[fqn]
        if not constraints.sharding_types:
            return sharding_types
        constrained_sharding_types: List[str] = constraints.sharding_types

        sharding_types = list(set(constrained_sharding_types) & set(sharding_types))

        if not sharding_types:
            raise RuntimeError(
                f"No available sharding types after applying user provided constraints for {fqn}"
            )
        return sharding_types

    def _filter_compute_kernels(
        self, fqn: str, compute_kernels: List[str]
    ) -> List[str]:
        if not self._constraints or not self._constraints.get(fqn):
            return compute_kernels
        constraints: PlannerConstraints = self._constraints[fqn]
        if not constraints.compute_kernels:
            return compute_kernels
        constrained_compute_kernels: List[str] = constraints.compute_kernels

        compute_kernels = list(set(constrained_compute_kernels) & set(compute_kernels))

        if not compute_kernels:
            raise RuntimeError(
                f"No available compute kernels after applying user provided constraints for {fqn}"
            )
        return compute_kernels

    def _get_input_lengths(self, fqn: str) -> List[float]:
        return (
            self._input_stats[fqn].pooling_factors
            if self._input_stats
            else [DEFAULT_POOLING_FACTOR]
        )


def get_partition_by_type(sharding_type: str) -> str:
    device_sharding_types = {
        ShardingType.TABLE_WISE.value,
        ShardingType.COLUMN_WISE.value,
    }
    host_sharding_types = {ShardingType.TABLE_ROW_WISE.value}
    uniform_sharding_types = {
        ShardingType.ROW_WISE.value,
        ShardingType.DATA_PARALLEL.value,
    }

    if sharding_type in device_sharding_types:
        return PartitionByType.DEVICE.value
    elif sharding_type in host_sharding_types:
        return PartitionByType.HOST.value
    elif sharding_type in uniform_sharding_types:
        return PartitionByType.UNIFORM.value

    raise ValueError(f"Unrecognized sharding type provided: {sharding_type}")


def get_shard_lengths_and_offsets(
    tensor: torch.Tensor,
    world_size: int,
    local_world_size: int,
    sharding_type: str,
    col_wise_shard_dim: Optional[int] = None,
) -> Tuple[List[List[int]], List[List[int]]]:
    (rows, columns) = tensor.shape

    if sharding_type == ShardingType.DATA_PARALLEL.value:
        return [[rows, columns]] * world_size, [[0, 0]] * world_size
    elif sharding_type == ShardingType.TABLE_WISE.value:
        return [[rows, columns]], [[0, 0]]
    elif sharding_type == ShardingType.COLUMN_WISE.value:
        return _get_cw_shard_lengths_and_offsets(columns, rows, col_wise_shard_dim)
    elif sharding_type == ShardingType.ROW_WISE.value:
        return _get_rw_shard_lengths_and_offsets(rows, world_size, columns)
    elif sharding_type == ShardingType.TABLE_ROW_WISE.value:
        return _get_rw_shard_lengths_and_offsets(rows, local_world_size, columns)

    raise ValueError(f"Unrecognized sharding type provided: {sharding_type}")


def _get_rw_shard_lengths_and_offsets(
    hash_size: int, num_devices: int, columns: int
) -> Tuple[List[List[int]], List[List[int]]]:
    # Set prefix of shard_lengths to be  ceil(hash_size/num_devices). For exmaple
    # if hash_size = 10, num_devices = 3, we will allocate the rows as
    # 3,3,3,1 (rather than 3,3,2,2). This is due to implementation in RWSharding that sets
    # block_size_lists to be ceil. The balanced way is harder to support on GPU. For more details
    # see https://fb.quip.com/xbgbAchCTOL0
    # Also consider the example of hash_size = 5, num_devices = 4. The expected rows per rank is
    # [2,2,1,0].
    num_devices: int = min(num_devices, hash_size)

    block_size: int = math.ceil(hash_size / num_devices)
    last_rank: int = hash_size // block_size
    last_block_size: int = hash_size - block_size * last_rank
    shard_lengths: List[List[int]] = []

    for rank in range(num_devices):
        if rank < last_rank:
            local_row: int = block_size
        elif rank == last_rank:
            local_row: int = last_block_size
        else:
            local_row: int = 0
        shard_lengths.append([local_row, columns])
    shard_offsets = [[0, 0]]

    for i in range(num_devices - 1):
        shard_offsets.append([shard_lengths[i][0] + shard_offsets[i][0], 0])

    return shard_lengths, shard_offsets


def _get_cw_shard_lengths_and_offsets(
    hash_size: int,
    rows: int,
    col_wise_shard_dim: Optional[int] = None,
) -> Tuple[List[List[int]], List[List[int]]]:
    block_size: int = min(
        col_wise_shard_dim if col_wise_shard_dim else DEFAULT_CW_DIM, hash_size
    )
    num_col_wise_shards, residual = divmod(hash_size, block_size)

    shard_lengths: List[List[int]] = [[rows, block_size]] * (num_col_wise_shards - 1)
    shard_lengths.append([rows, block_size + residual])

    shard_offsets: List[List[int]] = [
        [0, block_size * rank] for rank in range(num_col_wise_shards)
    ]
    return shard_lengths, shard_offsets


def get_shard_storages(
    sharder: ModuleSharder[nn.Module],
    sharding_type: str,
    tensor: torch.Tensor,
    compute_device: str,
    compute_kernel: str,
    shard_lengths: List[List[int]],
    batch_size: int,
    world_size: int,
    local_world_size: int,
    input_lengths: List[float],
    caching_ratio: float,
) -> List[Storage]:
    input_data_type_size = BIGINT_DTYPE
    output_data_type_size = tensor.element_size()

    input_sizes, output_sizes = _get_shard_io_sizes(
        sharding_type=sharding_type,
        batch_size=batch_size,
        world_size=world_size,
        local_world_size=local_world_size,
        input_lengths=input_lengths,
        emb_dim=tensor.shape[1],
        shard_lengths=shard_lengths,
        input_data_type_size=input_data_type_size,
        output_data_type_size=output_data_type_size,
    )

    tensor_storage = sharder.storage_usage(tensor, compute_device, compute_kernel)
    hbm_storage: int = tensor_storage.get("hbm", 0)
    ddr_storage: int = tensor_storage.get("ddr", 0)

    if compute_kernel == EmbeddingComputeKernel.BATCHED_FUSED_UVM_CACHING.value:
        hbm_storage = round(ddr_storage * caching_ratio)
        ddr_storage = ddr_storage - hbm_storage

    hbm_specific_sizes: List[int] = _get_storage_specific_sizes(
        storage=hbm_storage,
        shape=tensor.shape,
        shard_lengths=shard_lengths,
        sharding_type=sharding_type,
        compute_kernel=compute_kernel,
        on_device=compute_device == "cuda",
        input_sizes=input_sizes,
        input_data_type_size=input_data_type_size,
        output_data_type_size=output_data_type_size,
    )
    ddr_specific_sizes: List[int] = _get_storage_specific_sizes(
        storage=ddr_storage,
        shape=tensor.shape,
        shard_lengths=shard_lengths,
        sharding_type=sharding_type,
        compute_kernel=compute_kernel,
        on_device=compute_device == "cpu",
        input_sizes=input_sizes,
        input_data_type_size=input_data_type_size,
        output_data_type_size=output_data_type_size,
    )

    hbm_sizes: List[int] = [
        input_size + output_size + hbm_specific_size if compute_device == "cuda" else 0
        for input_size, output_size, hbm_specific_size in zip(
            input_sizes,
            output_sizes,
            hbm_specific_sizes,
        )
    ]
    ddr_sizes: List[int] = [
        input_size + output_size + ddr_specific_size
        if compute_device == "cpu"
        else ddr_specific_size
        for input_size, output_size, ddr_specific_size in zip(
            input_sizes,
            output_sizes,
            ddr_specific_sizes,
        )
    ]

    return [
        Storage(
            hbm=hbm_size,
            ddr=ddr_size,
        )
        for hbm_size, ddr_size in zip(hbm_sizes, ddr_sizes)
    ]


def _get_shard_io_sizes(
    sharding_type: str,
    batch_size: int,
    world_size: int,
    local_world_size: int,
    input_lengths: List[float],
    emb_dim: int,
    shard_lengths: List[List[int]],
    input_data_type_size: int,
    output_data_type_size: int,
) -> Tuple[List[int], List[int]]:
    if sharding_type == ShardingType.DATA_PARALLEL.value:
        return _get_dp_shard_io_sizes(
            batch_size=batch_size,
            input_lengths=input_lengths,
            emb_dim=emb_dim,
            num_shards=len(shard_lengths),
            input_data_type_size=input_data_type_size,
            output_data_type_size=output_data_type_size,
        )
    elif sharding_type == ShardingType.TABLE_WISE.value:
        return _get_tw_shard_io_sizes(
            batch_size=batch_size,
            world_size=world_size,
            input_lengths=input_lengths,
            emb_dim=emb_dim,
            input_data_type_size=input_data_type_size,
            output_data_type_size=output_data_type_size,
        )
    elif sharding_type == ShardingType.COLUMN_WISE.value:
        return _get_cw_shard_io_sizes(
            batch_size=batch_size,
            world_size=world_size,
            input_lengths=input_lengths,
            shard_lengths=shard_lengths,
            input_data_type_size=input_data_type_size,
            output_data_type_size=output_data_type_size,
        )
    elif sharding_type == ShardingType.ROW_WISE.value:
        return _get_rw_shard_io_sizes(
            batch_size=batch_size,
            world_size=world_size,
            input_lengths=input_lengths,
            shard_lengths=shard_lengths,
            input_data_type_size=input_data_type_size,
            output_data_type_size=output_data_type_size,
        )
    elif sharding_type == ShardingType.TABLE_ROW_WISE.value:
        return _get_twrw_shard_io_sizes(
            batch_size=batch_size,
            world_size=world_size,
            local_world_size=local_world_size,
            input_lengths=input_lengths,
            shard_lengths=shard_lengths,
            input_data_type_size=input_data_type_size,
            output_data_type_size=output_data_type_size,
        )
    else:
        raise ValueError(f"Unrecognized sharding type provided: {sharding_type}")


def _get_dp_shard_io_sizes(
    batch_size: int,
    input_lengths: List[float],
    emb_dim: int,
    num_shards: int,
    input_data_type_size: int,
    output_data_type_size: int,
) -> Tuple[List[int], List[int]]:
    input_sizes: List[int] = [
        # pyre-ignore
        math.ceil(batch_size * sum(input_lengths) * input_data_type_size)
    ] * num_shards

    output_sizes: List[int] = [
        batch_size * emb_dim * len(input_lengths) * output_data_type_size
    ] * num_shards

    return input_sizes, output_sizes


def _get_tw_shard_io_sizes(
    batch_size: int,
    world_size: int,
    input_lengths: List[float],
    emb_dim: int,
    input_data_type_size: int,
    output_data_type_size: int,
) -> Tuple[List[int], List[int]]:
    input_sizes: List[int] = [
        # pyre-ignore
        math.ceil(batch_size * world_size * sum(input_lengths) * input_data_type_size)
    ]

    output_sizes: List[int] = [
        batch_size * world_size * emb_dim * len(input_lengths) * output_data_type_size
    ]

    return input_sizes, output_sizes


def _get_cw_shard_io_sizes(
    batch_size: int,
    world_size: int,
    input_lengths: List[float],
    shard_lengths: List[List[int]],
    input_data_type_size: int,
    output_data_type_size: int,
) -> Tuple[List[int], List[int]]:
    input_sizes: List[int] = [
        # pyre-ignore
        math.ceil(batch_size * world_size * sum(input_lengths) * input_data_type_size)
    ] * len(shard_lengths)

    output_sizes: List[int] = [
        (
            batch_size
            * world_size
            * shard_lengths[i][1]
            * len(input_lengths)
            * output_data_type_size
        )
        for i in range(len(shard_lengths))
    ]

    return input_sizes, output_sizes


def _get_rw_shard_io_sizes(
    batch_size: int,
    world_size: int,
    input_lengths: List[float],
    shard_lengths: List[List[int]],
    input_data_type_size: int,
    output_data_type_size: int,
) -> Tuple[List[int], List[int]]:
    input_sizes: List[int] = [
        math.ceil(
            batch_size
            * world_size
            # pyre-ignore
            * sum(input_lengths)
            / world_size
            * input_data_type_size
        )
    ] * len(shard_lengths)

    output_sizes: List[int] = [
        (
            batch_size
            * world_size
            * shard_lengths[i][1]
            * len(input_lengths)
            * output_data_type_size
        )
        for i in range(len(shard_lengths))
    ]

    return input_sizes, output_sizes


def _get_twrw_shard_io_sizes(
    batch_size: int,
    world_size: int,
    local_world_size: int,
    input_lengths: List[float],
    shard_lengths: List[List[int]],
    input_data_type_size: int,
    output_data_type_size: int,
) -> Tuple[List[int], List[int]]:
    input_sizes: List[int] = [
        math.ceil(
            batch_size
            * world_size
            # pyre-ignore
            * sum(input_lengths)
            / local_world_size
            * input_data_type_size
        )
    ] * len(shard_lengths)

    output_sizes: List[int] = [
        (
            batch_size
            * world_size
            * shard_lengths[i][1]
            * len(input_lengths)
            * output_data_type_size
        )
        for i in range(len(shard_lengths))
    ]

    return input_sizes, output_sizes


def _get_storage_specific_sizes(
    storage: int,
    shape: torch.Size,
    shard_lengths: List[List[int]],
    sharding_type: str,
    compute_kernel: str,
    on_device: bool,
    input_sizes: List[int],
    input_data_type_size: int,
    output_data_type_size: int,
) -> List[int]:
    tensor_sizes: List[int] = [
        math.ceil(storage * math.prod(length) / math.prod(shape))
        if sharding_type != ShardingType.DATA_PARALLEL.value
        else storage
        for length in shard_lengths
    ]

    gradient_sizes: List[int] = tensor_sizes
    if compute_kernel == EmbeddingComputeKernel.SPARSE.value and on_device:
        gradient_sizes = [
            math.ceil(
                input_size
                * shard_length[1]
                * output_data_type_size
                / input_data_type_size
            )
            for input_size, shard_length in zip(input_sizes, shard_lengths)
        ]

    optimizer_sizes: List[int] = [
        tensor_size * 2 if sharding_type == ShardingType.DATA_PARALLEL.value else 0
        for tensor_size in tensor_sizes
    ]

    return [
        tensor_size + gradient_size + optimizer_size
        for tensor_size, gradient_size, optimizer_size in zip(
            tensor_sizes, gradient_sizes, optimizer_sizes
        )
    ]
