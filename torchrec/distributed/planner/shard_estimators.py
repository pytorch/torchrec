#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.planner.constants import (
    BATCHED_COPY_PERF_FACTOR,
    BIGINT_DTYPE,
    BWD_COMPUTE_MULTIPLIER,
    CROSS_NODE_BANDWIDTH,
    DP_ELEMENTWISE_KERNELS_PERF_FACTOR,
    FULL_BLOCK_EMB_DIM,
    HALF_BLOCK_PENALTY,
    INTRA_NODE_BANDWIDTH,
    kernel_bw_lookup,
    POOLING_FACTOR,
    QUARTER_BLOCK_PENALTY,
    UVM_CACHING_RATIO,
    WEIGHTED_KERNEL_MULTIPLIER,
)
from torchrec.distributed.planner.types import (
    ParameterConstraints,
    PlannerError,
    ShardEstimator,
    ShardingOption,
    Storage,
    Topology,
)
from torchrec.distributed.planner.utils import prod, sharder_name
from torchrec.distributed.types import ModuleSharder, ShardingType


class EmbeddingPerfEstimator(ShardEstimator):
    """
    Embedding Wall Time Perf Estimator
    """

    def __init__(
        self,
        topology: Topology,
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
    ) -> None:
        self._topology = topology
        self._constraints = constraints

    def estimate(
        self,
        sharding_options: List[ShardingOption],
        sharder_map: Optional[Dict[str, ModuleSharder[nn.Module]]] = None,
    ) -> None:
        for sharding_option in sharding_options:
            caching_ratio = (
                self._constraints[sharding_option.name].caching_ratio
                if self._constraints and self._constraints.get(sharding_option.name)
                else None
            )
            shard_perfs = perf_func_emb_wall_time(
                shard_sizes=[shard.size for shard in sharding_option.shards],
                compute_kernel=sharding_option.compute_kernel,
                compute_device=self._topology.compute_device,
                sharding_type=sharding_option.sharding_type,
                batch_size=sharding_option.batch_size,
                world_size=self._topology.world_size,
                local_world_size=self._topology.local_world_size,
                input_lengths=sharding_option.input_lengths,
                input_data_type_size=BIGINT_DTYPE,
                output_data_type_size=sharding_option.tensor.element_size(),
                bw_intra_host=getattr(
                    self._topology, "intra_host_bw", INTRA_NODE_BANDWIDTH
                ),
                bw_inter_host=getattr(
                    self._topology, "inter_host_bw", CROSS_NODE_BANDWIDTH
                ),
                is_pooled=sharding_option.is_pooled,
                is_weighted=True
                if getattr(sharding_option.module[1], "is_weighted", False)
                else False,
                has_feature_processor=True
                if getattr(sharding_option.module[1], "feature_processor", None)
                else False,
                caching_ratio=caching_ratio,
            )

            for shard, perf in zip(sharding_option.shards, shard_perfs):
                shard.perf = perf


def perf_func_emb_wall_time(
    shard_sizes: List[List[int]],
    compute_kernel: str,
    compute_device: str,
    sharding_type: str,
    batch_size: int,
    world_size: int,
    local_world_size: int,
    input_lengths: List[float],
    input_data_type_size: float,
    output_data_type_size: float,
    bw_intra_host: float,
    bw_inter_host: float,
    is_pooled: bool,
    is_weighted: bool = False,
    has_feature_processor: bool = False,
    caching_ratio: Optional[float] = None,
) -> List[float]:
    """
    Attempts to model perfs as a function of relative wall times.

    Args:
        shard_sizes (List[List[int]]): the list of (local_rows, local_cols) of each
            shard.
        compute_kernel (str): compute kernel.
        compute_device (str): compute device.
        sharding_type (str): tw, rw, cw, twrw, dp.
        batch_size (int): the size of each batch.
        world_size (int): the number of devices for all hosts.
        local_world_size (int): the number of the device for each host.
        input_lengths (List[float]): the list of the average number of lookups of each
            input query feature.
        input_data_type_size (float): the data type size of the distributed
            data_parallel input.
        output_data_type_size (float): the data type size of the distributed
            data_parallel output.
        bw_intra_host (float): the bandwidth within a single host like multiple threads.
        bw_inter_host (float): the bandwidth between two hosts like multiple machines.
        is_pooled (bool): True if embedding output is pooled (ie. EmbeddingBag), False
            if unpooled/sequential (ie. Embedding).
        is_weighted (bool = False): if the module is an EBC and is weighted, typically
            signifying an id score list feature.
        has_feature_processor (bool = False): if the module has a feature processor.
        caching_ratio (Optional[float] = None): cache ratio to determine the bandwidth
            of device.

    Returns:
        List[float]: the list of perf for each shard.
    """

    shard_perfs = []
    B = 1.0 * world_size * batch_size  # global batch size
    device_bw = kernel_bw_lookup(compute_device, compute_kernel, caching_ratio)
    if device_bw is None:
        raise PlannerError(
            f"No kernel bandwidth exists for this combo of compute device: {compute_device}, compute kernel: {compute_kernel}"
        )

    for hash_size, emb_dim in shard_sizes:
        if (
            sharding_type == ShardingType.TABLE_WISE.value
            or sharding_type == ShardingType.COLUMN_WISE.value
            or sharding_type == ShardingType.TABLE_COLUMN_WISE.value
        ):
            shard_perf = _get_tw_sharding_perf(
                global_batch_size=B,
                world_size=world_size,
                local_world_size=local_world_size,
                input_lengths=input_lengths,
                emb_dim=emb_dim,
                input_data_type_size=input_data_type_size,
                output_data_type_size=output_data_type_size,
                device_bw=device_bw,
                bw_inter_host=bw_inter_host,
                bw_intra_host=bw_intra_host,
                is_pooled=is_pooled,
                is_weighted=is_weighted,
                has_feature_processor=has_feature_processor,
            )
        elif sharding_type == ShardingType.ROW_WISE.value:
            shard_perf = _get_rw_sharding_perf(
                global_batch_size=B,
                world_size=world_size,
                local_world_size=local_world_size,
                input_lengths=input_lengths,
                emb_dim=emb_dim,
                input_data_type_size=input_data_type_size,
                output_data_type_size=output_data_type_size,
                device_bw=device_bw,
                bw_inter_host=bw_inter_host,
                bw_intra_host=bw_intra_host,
                is_pooled=is_pooled,
                is_weighted=is_weighted,
                has_feature_processor=has_feature_processor,
            )
        elif sharding_type == ShardingType.TABLE_ROW_WISE.value:
            shard_perf = _get_twrw_sharding_perf(
                global_batch_size=B,
                world_size=world_size,
                local_world_size=local_world_size,
                input_lengths=input_lengths,
                emb_dim=emb_dim,
                input_data_type_size=input_data_type_size,
                output_data_type_size=output_data_type_size,
                device_bw=device_bw,
                bw_inter_host=bw_inter_host,
                bw_intra_host=bw_intra_host,
                is_pooled=is_pooled,
                is_weighted=is_weighted,
                has_feature_processor=has_feature_processor,
            )
        elif sharding_type == ShardingType.DATA_PARALLEL.value:
            shard_perf = _get_dp_sharding_perf(
                local_batch_size=batch_size,
                world_size=world_size,
                local_world_size=local_world_size,
                input_lengths=input_lengths,
                grad_num_elem=hash_size * emb_dim,
                emb_dim=emb_dim,
                input_data_type_size=output_data_type_size,
                output_data_type_size=output_data_type_size,
                device_bw=device_bw,
                bw_inter_host=bw_inter_host,
                is_pooled=is_pooled,
                is_weighted=is_weighted,
                has_feature_processor=has_feature_processor,
            )
        else:
            raise ValueError(
                f"Unrecognized or unsupported sharding type provided: {sharding_type}"
            )
        shard_perfs.append(shard_perf)

    return shard_perfs


def _get_tw_sharding_perf(
    global_batch_size: float,
    world_size: int,
    local_world_size: int,
    input_lengths: List[float],
    emb_dim: int,
    input_data_type_size: float,
    output_data_type_size: float,
    device_bw: float,
    bw_inter_host: float,
    bw_intra_host: float,
    is_pooled: bool,
    is_weighted: bool = False,
    has_feature_processor: bool = False,
) -> float:
    input_read_size = math.ceil(
        global_batch_size * sum(input_lengths) * input_data_type_size
    )
    if is_weighted or has_feature_processor:
        input_read_size *= 2
    # minimum embedding dim is set to 32 due to kernel usage
    embedding_lookup_size = (
        global_batch_size
        * sum(input_lengths)
        * max(emb_dim, 32)
        * output_data_type_size
    )
    num_outputs = len(input_lengths) if is_pooled else float(sum(input_lengths))
    output_write_size = (
        global_batch_size * emb_dim * num_outputs * output_data_type_size
    )
    # embedding dim below 128 will reduce kernel efficency
    block_usage_penalty = 1
    if emb_dim < FULL_BLOCK_EMB_DIM:
        if emb_dim >= 64:
            block_usage_penalty = HALF_BLOCK_PENALTY
        else:  # emb_dim >= 32
            block_usage_penalty = QUARTER_BLOCK_PENALTY
    comms_bw = bw_inter_host if world_size > local_world_size else bw_intra_host

    fwd_comms = output_write_size / comms_bw

    fwd_compute = (
        (input_read_size + embedding_lookup_size + output_write_size)
        * block_usage_penalty
        / device_bw
    )

    bwd_comms = fwd_comms

    bwd_grad_indice_weights_kernel = (
        fwd_compute * WEIGHTED_KERNEL_MULTIPLIER
        if is_weighted or has_feature_processor
        else 0
    )

    # includes fused optimizers
    bwd_compute = fwd_compute * BWD_COMPUTE_MULTIPLIER

    # in order of model parallel execution, starting with:
    # BWD DP -> BWD MP ... FWD MP -> FWD DP
    return (
        bwd_comms
        + bwd_grad_indice_weights_kernel
        + bwd_compute
        + fwd_compute
        + fwd_comms
    )


def _get_rw_sharding_perf(
    global_batch_size: float,
    world_size: int,
    local_world_size: int,
    input_lengths: List[float],
    emb_dim: int,
    input_data_type_size: float,
    output_data_type_size: float,
    device_bw: float,
    bw_inter_host: float,
    bw_intra_host: float,
    is_pooled: bool,
    is_weighted: bool = False,
    has_feature_processor: bool = False,
) -> float:
    input_read_size = math.ceil(
        global_batch_size * sum(input_lengths) / world_size * input_data_type_size
    )
    if is_weighted or has_feature_processor:
        input_read_size *= 2
    embedding_lookup_size = (
        global_batch_size
        * sum(input_lengths)
        / world_size
        * emb_dim
        * output_data_type_size
    )
    num_outputs = (
        len(input_lengths) if is_pooled else float(sum(input_lengths) / world_size)
    )
    output_write_size = (
        global_batch_size * emb_dim * num_outputs * output_data_type_size
    )
    comms_bw = bw_inter_host if world_size > local_world_size else bw_intra_host

    fwd_comms = output_write_size / comms_bw

    fwd_compute = (
        input_read_size + embedding_lookup_size + output_write_size
    ) / device_bw

    bwd_comms = fwd_comms

    bwd_batched_copy = output_write_size * BATCHED_COPY_PERF_FACTOR / device_bw

    bwd_grad_indice_weights_kernel = (
        fwd_compute * WEIGHTED_KERNEL_MULTIPLIER
        if is_weighted or has_feature_processor
        else 0
    )

    bwd_compute = fwd_compute * BWD_COMPUTE_MULTIPLIER

    return (
        bwd_comms
        + bwd_batched_copy
        + bwd_grad_indice_weights_kernel
        + bwd_compute
        + fwd_compute
        + fwd_comms
    )


def _get_twrw_sharding_perf(
    global_batch_size: float,
    world_size: int,
    local_world_size: int,
    input_lengths: List[float],
    emb_dim: int,
    input_data_type_size: float,
    output_data_type_size: float,
    device_bw: float,
    bw_inter_host: float,
    bw_intra_host: float,
    is_pooled: bool,
    is_weighted: bool = False,
    has_feature_processor: bool = False,
) -> float:
    input_read_size = math.ceil(
        global_batch_size * sum(input_lengths) / local_world_size * input_data_type_size
    )
    if is_weighted or has_feature_processor:
        input_read_size *= 2
    embedding_lookup_size = (
        global_batch_size
        * sum(input_lengths)
        / local_world_size
        * emb_dim
        * output_data_type_size
    )
    num_outputs = (
        len(input_lengths)
        if is_pooled
        else float(sum(input_lengths) / local_world_size)
    )
    output_write_size = (
        global_batch_size * emb_dim * num_outputs * output_data_type_size
    )

    fwd_comms = output_write_size / bw_intra_host

    if world_size > local_world_size:
        fwd_comms += output_write_size * (local_world_size / world_size) / bw_inter_host

    fwd_compute = (
        input_read_size + embedding_lookup_size + output_write_size
    ) / device_bw

    bwd_comms = fwd_comms

    bwd_grad_indice_weights_kernel = (
        fwd_compute * WEIGHTED_KERNEL_MULTIPLIER
        if is_weighted or has_feature_processor
        else 0
    )

    bwd_batched_copy = output_write_size * BATCHED_COPY_PERF_FACTOR / device_bw

    bwd_compute = fwd_compute * BWD_COMPUTE_MULTIPLIER

    return (
        bwd_comms
        + bwd_batched_copy
        + bwd_grad_indice_weights_kernel
        + bwd_compute
        + fwd_compute
        + fwd_comms
    )


def _get_dp_sharding_perf(
    local_batch_size: float,
    world_size: int,
    local_world_size: int,
    input_lengths: List[float],
    grad_num_elem: int,
    emb_dim: int,
    input_data_type_size: float,
    output_data_type_size: float,
    device_bw: float,
    bw_inter_host: float,
    is_pooled: bool,
    is_weighted: bool = False,
    has_feature_processor: bool = False,
) -> float:
    input_read_size = math.ceil(
        local_batch_size * sum(input_lengths) * input_data_type_size
    )
    if is_weighted or has_feature_processor:
        input_read_size *= 2
    embedding_lookup_size = (
        local_batch_size * sum(input_lengths) * emb_dim * output_data_type_size
    )
    num_outputs = len(input_lengths) if is_pooled else float(sum(input_lengths))
    output_write_size = local_batch_size * emb_dim * num_outputs * output_data_type_size
    table_size = grad_num_elem * output_data_type_size

    fwd_compute = (
        input_read_size + embedding_lookup_size + output_write_size
    ) / device_bw

    num_nodes = min(world_size / local_world_size, 2)
    # all-reduce data transfer: https://images.nvidia.com/events/sc15/pdfs/NCCL-Woolley.pdf
    all_reduce = (
        table_size
        * (2 * num_nodes - 1)
        / num_nodes
        / (bw_inter_host * local_world_size)  # 1 NIC per GPU
    )
    # inter host communication constraint
    if world_size > 2 * local_world_size:
        all_reduce *= 2

    # SGD + Fill + BUnary
    optimizer_kernels = table_size * DP_ELEMENTWISE_KERNELS_PERF_FACTOR / device_bw

    bwd_compute = fwd_compute * BWD_COMPUTE_MULTIPLIER

    bwd_grad_indice_weights_kernel = (
        fwd_compute * WEIGHTED_KERNEL_MULTIPLIER
        if is_weighted or has_feature_processor
        else 0
    )

    return (
        all_reduce
        + optimizer_kernels
        + bwd_grad_indice_weights_kernel
        + bwd_compute
        + fwd_compute
    )


class EmbeddingStorageEstimator(ShardEstimator):
    """
    Embedding Storage Usage Estimator
    """

    def __init__(
        self,
        topology: Topology,
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
    ) -> None:
        self._topology = topology
        self._constraints = constraints

    def estimate(
        self,
        sharding_options: List[ShardingOption],
        sharder_map: Optional[Dict[str, ModuleSharder[nn.Module]]] = None,
    ) -> None:
        if not sharder_map:
            raise ValueError("sharder map not provided for storage estimator")

        for sharding_option in sharding_options:
            sharder_key = sharder_name(type(sharding_option.module[1]))
            sharder = sharder_map[sharder_key]

            input_lengths = (
                self._constraints[sharding_option.name].pooling_factors
                if self._constraints and self._constraints.get(sharding_option.name)
                else [POOLING_FACTOR]
            )

            caching_ratio = (
                self._constraints[sharding_option.name].caching_ratio
                if self._constraints and self._constraints.get(sharding_option.name)
                else None
            )

            shard_storages = calculate_shard_storages(
                sharder=sharder,
                sharding_type=sharding_option.sharding_type,
                tensor=sharding_option.tensor,
                compute_device=self._topology.compute_device,
                compute_kernel=sharding_option.compute_kernel,
                shard_sizes=[shard.size for shard in sharding_option.shards],
                batch_size=self._topology.batch_size,
                world_size=self._topology.world_size,
                local_world_size=self._topology.local_world_size,
                input_lengths=input_lengths,
                caching_ratio=caching_ratio if caching_ratio else UVM_CACHING_RATIO,
                is_pooled=sharding_option.is_pooled,
            )

            for shard, storage in zip(sharding_option.shards, shard_storages):
                shard.storage = storage


def calculate_shard_storages(
    sharder: ModuleSharder[nn.Module],
    sharding_type: str,
    tensor: torch.Tensor,
    compute_device: str,
    compute_kernel: str,
    shard_sizes: List[List[int]],
    batch_size: int,
    world_size: int,
    local_world_size: int,
    input_lengths: List[float],
    caching_ratio: float,
    is_pooled: bool,
) -> List[Storage]:
    """
    Calculates estimated storage sizes for each sharded tensor, comprised of input,
    output, tensor, gradient, and optimizer sizes.

    Args:
        sharder (ModuleSharder[nn.Module]): sharder for module that supports sharding.
        sharding_type (str): provided ShardingType value.
        tensor (torch.Tensor): tensor to be sharded.
        compute_device (str): compute device to be used.
        compute_kernel (str): compute kernel to be used.
        shard_sizes (List[List[int]]): list of dimensions of each sharded tensor.
        batch_size (int): batch size to be used.
        world_size (int): total number of devices in topology.
        local_world_size (int): total number of devices in host group topology.
        input_lengths (List[float]): average input lengths synonymous with pooling
            factors.
        caching_ratio (float): ratio of HBM to DDR memory for UVM caching.
        is_pooled (bool): True if embedding output is pooled (ie. EmbeddingBag), False
            if unpooled/sequential (ie. Embedding).

    Returns:
        List[Storage]: storage object for each device in topology.
    """

    input_data_type_size = BIGINT_DTYPE
    output_data_type_size = tensor.element_size()

    input_sizes, output_sizes = _calculate_shard_io_sizes(
        sharding_type=sharding_type,
        batch_size=batch_size,
        world_size=world_size,
        local_world_size=local_world_size,
        input_lengths=input_lengths,
        emb_dim=tensor.shape[1],
        shard_sizes=shard_sizes,
        input_data_type_size=input_data_type_size,
        output_data_type_size=output_data_type_size,
        is_pooled=is_pooled,
    )

    tensor_storage = sharder.storage_usage(tensor, compute_device, compute_kernel)
    hbm_storage: int = tensor_storage.get("hbm", 0)
    ddr_storage: int = tensor_storage.get("ddr", 0)

    if compute_kernel == EmbeddingComputeKernel.BATCHED_FUSED_UVM_CACHING.value:
        hbm_storage = round(ddr_storage * caching_ratio)

    hbm_specific_sizes: List[int] = _calculate_storage_specific_sizes(
        storage=hbm_storage,
        shape=tensor.shape,
        shard_sizes=shard_sizes,
        sharding_type=sharding_type,
        compute_kernel=compute_kernel,
    )
    ddr_specific_sizes: List[int] = _calculate_storage_specific_sizes(
        storage=ddr_storage,
        shape=tensor.shape,
        shard_sizes=shard_sizes,
        sharding_type=sharding_type,
        compute_kernel=compute_kernel,
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


def _calculate_shard_io_sizes(
    sharding_type: str,
    batch_size: int,
    world_size: int,
    local_world_size: int,
    input_lengths: List[float],
    emb_dim: int,
    shard_sizes: List[List[int]],
    input_data_type_size: int,
    output_data_type_size: int,
    is_pooled: bool,
) -> Tuple[List[int], List[int]]:
    if sharding_type == ShardingType.DATA_PARALLEL.value:
        return _calculate_dp_shard_io_sizes(
            batch_size=batch_size,
            input_lengths=input_lengths,
            emb_dim=emb_dim,
            num_shards=len(shard_sizes),
            input_data_type_size=input_data_type_size,
            output_data_type_size=output_data_type_size,
            is_pooled=is_pooled,
        )
    elif sharding_type == ShardingType.TABLE_WISE.value:
        return _calculate_tw_shard_io_sizes(
            batch_size=batch_size,
            world_size=world_size,
            input_lengths=input_lengths,
            emb_dim=emb_dim,
            input_data_type_size=input_data_type_size,
            output_data_type_size=output_data_type_size,
            is_pooled=is_pooled,
        )
    elif sharding_type == ShardingType.COLUMN_WISE.value:
        return _calculate_cw_shard_io_sizes(
            batch_size=batch_size,
            world_size=world_size,
            input_lengths=input_lengths,
            shard_sizes=shard_sizes,
            input_data_type_size=input_data_type_size,
            output_data_type_size=output_data_type_size,
            is_pooled=is_pooled,
        )
    elif sharding_type == ShardingType.ROW_WISE.value:
        return _calculate_rw_shard_io_sizes(
            batch_size=batch_size,
            world_size=world_size,
            input_lengths=input_lengths,
            shard_sizes=shard_sizes,
            input_data_type_size=input_data_type_size,
            output_data_type_size=output_data_type_size,
            is_pooled=is_pooled,
        )
    elif sharding_type == ShardingType.TABLE_ROW_WISE.value:
        return _calculate_twrw_shard_io_sizes(
            batch_size=batch_size,
            world_size=world_size,
            local_world_size=local_world_size,
            input_lengths=input_lengths,
            shard_sizes=shard_sizes,
            input_data_type_size=input_data_type_size,
            output_data_type_size=output_data_type_size,
            is_pooled=is_pooled,
        )
    elif sharding_type == ShardingType.TABLE_COLUMN_WISE.value:
        return _calculate_cw_shard_io_sizes(
            batch_size=batch_size,
            world_size=world_size,
            input_lengths=input_lengths,
            shard_sizes=shard_sizes,
            input_data_type_size=input_data_type_size,
            output_data_type_size=output_data_type_size,
            is_pooled=is_pooled,
        )
    else:
        raise ValueError(
            f"Unrecognized or unsupported sharding type provided: {sharding_type}"
        )


def _calculate_dp_shard_io_sizes(
    batch_size: int,
    input_lengths: List[float],
    emb_dim: int,
    num_shards: int,
    input_data_type_size: int,
    output_data_type_size: int,
    is_pooled: bool,
) -> Tuple[List[int], List[int]]:
    input_sizes: List[int] = [
        # pyre-ignore[58]
        math.ceil(batch_size * sum(input_lengths) * input_data_type_size)
    ] * num_shards

    num_outputs = len(input_lengths) if is_pooled else float(sum(input_lengths))
    output_sizes: List[int] = [
        math.ceil(batch_size * emb_dim * num_outputs * output_data_type_size)
    ] * num_shards

    return input_sizes, output_sizes


def _calculate_tw_shard_io_sizes(
    batch_size: int,
    world_size: int,
    input_lengths: List[float],
    emb_dim: int,
    input_data_type_size: int,
    output_data_type_size: int,
    is_pooled: bool,
) -> Tuple[List[int], List[int]]:
    input_sizes: List[int] = [
        # pyre-ignore[58]
        math.ceil(batch_size * world_size * sum(input_lengths) * input_data_type_size)
    ]

    num_outputs = len(input_lengths) if is_pooled else float(sum(input_lengths))
    output_sizes: List[int] = [
        math.ceil(
            batch_size * world_size * emb_dim * num_outputs * output_data_type_size
        )
    ]

    return input_sizes, output_sizes


def _calculate_cw_shard_io_sizes(
    batch_size: int,
    world_size: int,
    input_lengths: List[float],
    shard_sizes: List[List[int]],
    input_data_type_size: int,
    output_data_type_size: int,
    is_pooled: bool,
) -> Tuple[List[int], List[int]]:
    input_sizes: List[int] = [
        # pyre-ignore[58]
        math.ceil(batch_size * world_size * sum(input_lengths) * input_data_type_size)
    ] * len(shard_sizes)

    num_outputs = len(input_lengths) if is_pooled else float(sum(input_lengths))
    output_sizes: List[int] = [
        math.ceil(
            batch_size
            * world_size
            * shard_sizes[i][1]
            * num_outputs
            * output_data_type_size
        )
        for i in range(len(shard_sizes))
    ]

    return input_sizes, output_sizes


def _calculate_rw_shard_io_sizes(
    batch_size: int,
    world_size: int,
    input_lengths: List[float],
    shard_sizes: List[List[int]],
    input_data_type_size: int,
    output_data_type_size: int,
    is_pooled: bool,
) -> Tuple[List[int], List[int]]:
    input_sizes: List[int] = [
        math.ceil(
            batch_size
            * world_size
            # pyre-ignore[58]
            * sum(input_lengths)
            / world_size
            * input_data_type_size
        )
        if prod(shard) != 0
        else 0
        for shard in shard_sizes
    ]

    num_outputs = (
        len(input_lengths) if is_pooled else float(sum(input_lengths) / world_size)
    )
    output_sizes: List[int] = [
        math.ceil(
            batch_size
            * world_size
            * shard_sizes[i][1]
            * num_outputs
            * output_data_type_size
        )
        if prod(shard) != 0
        else 0
        for i, shard in enumerate(shard_sizes)
    ]

    return input_sizes, output_sizes


def _calculate_twrw_shard_io_sizes(
    batch_size: int,
    world_size: int,
    local_world_size: int,
    input_lengths: List[float],
    shard_sizes: List[List[int]],
    input_data_type_size: int,
    output_data_type_size: int,
    is_pooled: bool,
) -> Tuple[List[int], List[int]]:
    input_sizes: List[int] = [
        math.ceil(
            batch_size
            * world_size
            # pyre-ignore[58]
            * sum(input_lengths)
            / local_world_size
            * input_data_type_size
        )
        if prod(shard) != 0
        else 0
        for shard in shard_sizes
    ]

    num_outputs = (
        len(input_lengths)
        if is_pooled
        else float(sum(input_lengths) / local_world_size)
    )
    output_sizes: List[int] = [
        math.ceil(
            batch_size
            * world_size
            * shard_sizes[i][1]
            * num_outputs
            * output_data_type_size
        )
        if prod(shard) != 0
        else 0
        for i, shard in enumerate(shard_sizes)
    ]

    return input_sizes, output_sizes


def _calculate_storage_specific_sizes(
    storage: int,
    shape: torch.Size,
    shard_sizes: List[List[int]],
    sharding_type: str,
    compute_kernel: str,
) -> List[int]:
    tensor_sizes: List[int] = [
        math.ceil(storage * prod(size) / prod(shape))
        if sharding_type != ShardingType.DATA_PARALLEL.value
        else storage
        for size in shard_sizes
    ]

    optimizer_sizes: List[int] = [
        tensor_size * 2
        if compute_kernel
        in {
            EmbeddingComputeKernel.DENSE.value,
            EmbeddingComputeKernel.SPARSE.value,
            EmbeddingComputeKernel.BATCHED_DENSE.value,
        }
        else 0
        for tensor_size in tensor_sizes
    ]

    return [
        tensor_size + optimizer_size
        for tensor_size, optimizer_size in zip(tensor_sizes, optimizer_sizes)
    ]
