#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import cast, Dict, List, Optional, Tuple, Type

import torch
import torchrec.optim as trec_optim
from torch import nn
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.planner.constants import (
    BATCHED_COPY_PERF_FACTOR,
    BIGINT_DTYPE,
    BWD_COMPUTE_MULTIPLIER,
    DP_ELEMENTWISE_KERNELS_PERF_FACTOR,
    FULL_BLOCK_EMB_DIM,
    HALF_BLOCK_PENALTY,
    kernel_bw_lookup,
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

from torchrec.modules.embedding_modules import EmbeddingBagCollectionInterface


class EmbeddingPerfEstimator(ShardEstimator):
    """
    Embedding Wall Time Perf Estimator
    """

    def __init__(
        self,
        topology: Topology,
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
        is_inference: bool = False,
    ) -> None:
        self._topology = topology
        self._constraints = constraints
        self._is_inference = is_inference

    def estimate(
        self,
        sharding_options: List[ShardingOption],
        sharder_map: Optional[Dict[str, ModuleSharder[nn.Module]]] = None,
    ) -> None:
        if not sharder_map:
            assert not sharding_options, "sharder_map not provided for sharding_options"
            return

        for sharding_option in sharding_options:
            sharder_key = sharder_name(type(sharding_option.module[1]))
            sharder = sharder_map[sharder_key]
            caching_ratio = (
                sharder.fused_params.get("cache_load_factor")  # pyre-ignore[16]
                if hasattr(sharder, "fused_params") and sharder.fused_params
                else None
            )
            num_poolings = (
                cast(List[float], self._constraints[sharding_option.name].num_poolings)
                if self._constraints
                and self._constraints.get(sharding_option.name)
                and self._constraints[sharding_option.name].num_poolings
                else [1.0] * sharding_option.num_inputs
            )
            batch_sizes = (
                cast(List[int], self._constraints[sharding_option.name].batch_sizes)
                if self._constraints
                and self._constraints.get(sharding_option.name)
                and self._constraints[sharding_option.name].batch_sizes
                else [sharding_option.batch_size] * sharding_option.num_inputs
            )

            assert (
                len(sharding_option.input_lengths)
                == len(num_poolings)
                == len(batch_sizes)
            ), "Provided `pooling_factors`, `num_poolings`, and `batch_sizes` constraints must match."

            module = sharding_option.module[1]

            # TODO remove this hack once feature processor is disaggregated
            has_feature_processor = (
                True if getattr(module, "feature_processor", None) else False
            )

            if isinstance(module, EmbeddingBagCollectionInterface):
                is_weighted = module.is_weighted()
            elif (
                self._constraints
                and self._constraints.get(sharding_option.name)
                and self._constraints[sharding_option.name].is_weighted
            ):
                is_weighted = self._constraints[sharding_option.name].is_weighted
            else:
                is_weighted = False

            shard_perfs = perf_func_emb_wall_time(
                shard_sizes=[shard.size for shard in sharding_option.shards],
                compute_kernel=sharding_option.compute_kernel,
                compute_device=self._topology.compute_device,
                sharding_type=sharding_option.sharding_type,
                batch_sizes=batch_sizes,
                world_size=self._topology.world_size,
                local_world_size=self._topology.local_world_size,
                input_lengths=sharding_option.input_lengths,
                input_data_type_size=BIGINT_DTYPE,
                output_data_type_size=sharding_option.tensor.element_size(),
                num_poolings=num_poolings,
                hbm_mem_bw=self._topology.hbm_mem_bw,
                ddr_mem_bw=self._topology.ddr_mem_bw,
                intra_host_bw=self._topology.intra_host_bw,
                inter_host_bw=self._topology.inter_host_bw,
                is_pooled=sharding_option.is_pooled,
                is_weighted=is_weighted,
                is_inference=self._is_inference,
                has_feature_processor=has_feature_processor,
                caching_ratio=caching_ratio,
            )

            for shard, perf in zip(sharding_option.shards, shard_perfs):
                shard.perf = perf


def perf_func_emb_wall_time(
    shard_sizes: List[List[int]],
    compute_kernel: str,
    compute_device: str,
    sharding_type: str,
    batch_sizes: List[int],
    world_size: int,
    local_world_size: int,
    input_lengths: List[float],
    input_data_type_size: float,
    output_data_type_size: float,
    num_poolings: List[float],
    hbm_mem_bw: float,
    ddr_mem_bw: float,
    intra_host_bw: float,
    inter_host_bw: float,
    is_pooled: bool,
    is_weighted: bool = False,
    has_feature_processor: bool = False,
    caching_ratio: Optional[float] = None,
    is_inference: bool = False,
) -> List[float]:
    """
    Attempts to model perfs as a function of relative wall times.

    Args:
        shard_sizes (List[List[int]]): the list of (local_rows, local_cols) of each
            shard.
        compute_kernel (str): compute kernel.
        compute_device (str): compute device.
        sharding_type (str): tw, rw, cw, twrw, dp.
        batch_sizes (List[int]): batch size for each input feature.
        world_size (int): the number of devices for all hosts.
        local_world_size (int): the number of the device for each host.
        input_lengths (List[float]): the list of the average number of lookups of each
            input query feature.
        input_data_type_size (float): the data type size of the distributed
            data_parallel input.
        output_data_type_size (float): the data type size of the distributed
            data_parallel output.
        num_poolings (List[float]): number of poolings per sample, typically 1.0.
        hbm_mem_bw (float): the bandwidth of the device HBM.
        ddr_mem_bw (float): the bandwidth of the system DDR memory.
        intra_host_bw (float): the bandwidth within a single host like multiple threads.
        inter_host_bw (float): the bandwidth between two hosts like multiple machines.
        is_pooled (bool): True if embedding output is pooled (ie. `EmbeddingBag`), False
            if unpooled/sequential (ie. `Embedding`).
        is_weighted (bool = False): if the module is an EBC and is weighted, typically
            signifying an id score list feature.
        is_inference (bool = False): if planning for inference.
        has_feature_processor (bool = False): if the module has a feature processor.
        caching_ratio (Optional[float] = None): cache ratio to determine the bandwidth
            of device.

    Returns:
        List[float]: the list of perf for each shard.
    """

    shard_perfs = []
    device_bw = kernel_bw_lookup(
        compute_device, compute_kernel, hbm_mem_bw, ddr_mem_bw, caching_ratio
    )
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
                batch_sizes=batch_sizes,
                world_size=world_size,
                local_world_size=local_world_size,
                input_lengths=input_lengths,
                emb_dim=emb_dim,
                input_data_type_size=input_data_type_size,
                output_data_type_size=output_data_type_size,
                num_poolings=num_poolings,
                device_bw=device_bw,
                inter_host_bw=inter_host_bw,
                intra_host_bw=intra_host_bw,
                is_pooled=is_pooled,
                is_weighted=is_weighted,
                is_inference=is_inference,
                has_feature_processor=has_feature_processor,
            )
        elif sharding_type == ShardingType.ROW_WISE.value:
            shard_perf = _get_rw_sharding_perf(
                batch_sizes=batch_sizes,
                world_size=world_size,
                local_world_size=local_world_size,
                input_lengths=input_lengths,
                emb_dim=emb_dim,
                input_data_type_size=input_data_type_size,
                output_data_type_size=output_data_type_size,
                num_poolings=num_poolings,
                device_bw=device_bw,
                inter_host_bw=inter_host_bw,
                intra_host_bw=intra_host_bw,
                is_pooled=is_pooled,
                is_weighted=is_weighted,
                has_feature_processor=has_feature_processor,
            )
        elif sharding_type == ShardingType.TABLE_ROW_WISE.value:
            shard_perf = _get_twrw_sharding_perf(
                batch_sizes=batch_sizes,
                world_size=world_size,
                local_world_size=local_world_size,
                input_lengths=input_lengths,
                emb_dim=emb_dim,
                input_data_type_size=input_data_type_size,
                output_data_type_size=output_data_type_size,
                num_poolings=num_poolings,
                device_bw=device_bw,
                inter_host_bw=inter_host_bw,
                intra_host_bw=intra_host_bw,
                is_pooled=is_pooled,
                is_weighted=is_weighted,
                has_feature_processor=has_feature_processor,
            )
        elif sharding_type == ShardingType.DATA_PARALLEL.value:
            shard_perf = _get_dp_sharding_perf(
                batch_sizes=batch_sizes,
                world_size=world_size,
                local_world_size=local_world_size,
                input_lengths=input_lengths,
                grad_num_elem=hash_size * emb_dim,
                emb_dim=emb_dim,
                input_data_type_size=output_data_type_size,
                output_data_type_size=output_data_type_size,
                num_poolings=num_poolings,
                device_bw=device_bw,
                inter_host_bw=inter_host_bw,
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
    batch_sizes: List[int],
    world_size: int,
    local_world_size: int,
    input_lengths: List[float],
    emb_dim: int,
    input_data_type_size: float,
    output_data_type_size: float,
    num_poolings: List[float],
    device_bw: float,
    inter_host_bw: float,
    intra_host_bw: float,
    is_pooled: bool,
    is_weighted: bool = False,
    is_inference: bool = False,
    has_feature_processor: bool = False,
) -> float:
    batch_inputs = sum(
        [x * y * z for x, y, z in zip(input_lengths, num_poolings, batch_sizes)]
    )
    batch_outputs = (
        sum([x * y for x, y in zip(num_poolings, batch_sizes)])
        if is_pooled
        else batch_inputs
    )

    input_read_size = math.ceil(batch_inputs * world_size * input_data_type_size)
    if is_weighted or has_feature_processor:
        input_read_size *= 2

    # minimum embedding dim is set to 32 due to kernel usage
    embedding_lookup_size = (
        batch_inputs * world_size * max(emb_dim, 32) * output_data_type_size
    )

    output_write_size = batch_outputs * world_size * emb_dim * output_data_type_size

    # embedding dim below 128 will reduce kernel efficency
    block_usage_penalty = 1
    if emb_dim < FULL_BLOCK_EMB_DIM:
        if emb_dim >= 64:
            block_usage_penalty = HALF_BLOCK_PENALTY
        else:  # emb_dim >= 32
            block_usage_penalty = QUARTER_BLOCK_PENALTY

    comms_bw = inter_host_bw if world_size > local_world_size else intra_host_bw
    fwd_comms = output_write_size / comms_bw

    fwd_compute = (
        (input_read_size + embedding_lookup_size + output_write_size)
        * block_usage_penalty
        / device_bw
    )
    if is_inference:
        # only consider forward compute and comms for inference
        return fwd_compute + fwd_comms

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
    batch_sizes: List[int],
    world_size: int,
    local_world_size: int,
    input_lengths: List[float],
    emb_dim: int,
    input_data_type_size: float,
    output_data_type_size: float,
    num_poolings: List[float],
    device_bw: float,
    inter_host_bw: float,
    intra_host_bw: float,
    is_pooled: bool,
    is_weighted: bool = False,
    has_feature_processor: bool = False,
) -> float:
    batch_inputs = (
        sum([x * y * z for x, y, z in zip(input_lengths, num_poolings, batch_sizes)])
        / world_size
    )
    batch_outputs = (
        sum([x * y for x, y in zip(num_poolings, batch_sizes)])
        if is_pooled
        else batch_inputs
    )

    input_read_size = math.ceil(batch_inputs * world_size * input_data_type_size)
    if is_weighted or has_feature_processor:
        input_read_size *= 2

    embedding_lookup_size = batch_inputs * world_size * emb_dim * output_data_type_size

    output_write_size = batch_outputs * world_size * emb_dim * output_data_type_size

    comms_bw = inter_host_bw if world_size > local_world_size else intra_host_bw
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
    batch_sizes: List[int],
    world_size: int,
    local_world_size: int,
    input_lengths: List[float],
    emb_dim: int,
    input_data_type_size: float,
    output_data_type_size: float,
    num_poolings: List[float],
    device_bw: float,
    inter_host_bw: float,
    intra_host_bw: float,
    is_pooled: bool,
    is_weighted: bool = False,
    has_feature_processor: bool = False,
) -> float:
    batch_inputs = (
        sum([x * y * z for x, y, z in zip(input_lengths, num_poolings, batch_sizes)])
        / local_world_size
    )
    batch_outputs = (
        sum([x * y for x, y in zip(num_poolings, batch_sizes)])
        if is_pooled
        else batch_inputs
    )

    input_read_size = math.ceil(batch_inputs * world_size * input_data_type_size)
    if is_weighted or has_feature_processor:
        input_read_size *= 2

    embedding_lookup_size = batch_inputs * world_size * emb_dim * output_data_type_size

    output_write_size = batch_outputs * world_size * emb_dim * output_data_type_size

    fwd_comms = output_write_size / intra_host_bw

    if world_size > local_world_size:
        fwd_comms += output_write_size * (local_world_size / world_size) / inter_host_bw

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
    batch_sizes: List[int],
    world_size: int,
    local_world_size: int,
    input_lengths: List[float],
    grad_num_elem: int,
    emb_dim: int,
    input_data_type_size: float,
    output_data_type_size: float,
    num_poolings: List[float],
    device_bw: float,
    inter_host_bw: float,
    is_pooled: bool,
    is_weighted: bool = False,
    has_feature_processor: bool = False,
) -> float:
    batch_inputs = sum(
        [x * y * z for x, y, z in zip(input_lengths, num_poolings, batch_sizes)]
    )
    batch_outputs = (
        sum([x * y for x, y in zip(num_poolings, batch_sizes)])
        if is_pooled
        else batch_inputs
    )

    input_read_size = math.ceil(batch_inputs * input_data_type_size)
    if is_weighted or has_feature_processor:
        input_read_size *= 2

    embedding_lookup_size = batch_inputs * emb_dim * output_data_type_size

    output_write_size = batch_outputs * emb_dim * output_data_type_size
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
        / (inter_host_bw * local_world_size)  # 1 NIC per GPU
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
            assert not sharding_options, "sharder_map not provided for sharding_options"
            return

        for sharding_option in sharding_options:
            sharder_key = sharder_name(type(sharding_option.module[1]))
            sharder = sharder_map[sharder_key]
            caching_ratio = (
                sharder.fused_params.get("cache_load_factor")  # pyre-ignore[16]
                if hasattr(sharder, "fused_params") and sharder.fused_params
                else None
            )
            num_poolings = (
                cast(List[float], self._constraints[sharding_option.name].num_poolings)
                if self._constraints
                and self._constraints.get(sharding_option.name)
                and self._constraints[sharding_option.name].num_poolings
                else [1.0] * sharding_option.num_inputs
            )
            assert len(num_poolings) == sharding_option.num_inputs
            batch_sizes = (
                cast(List[int], self._constraints[sharding_option.name].batch_sizes)
                if self._constraints
                and self._constraints.get(sharding_option.name)
                and self._constraints[sharding_option.name].batch_sizes
                else [sharding_option.batch_size] * sharding_option.num_inputs
            )

            shard_storages = calculate_shard_storages(
                sharder=sharder,
                sharding_type=sharding_option.sharding_type,
                tensor=sharding_option.tensor,
                compute_device=self._topology.compute_device,
                compute_kernel=sharding_option.compute_kernel,
                shard_sizes=[shard.size for shard in sharding_option.shards],
                batch_sizes=batch_sizes,
                world_size=self._topology.world_size,
                local_world_size=self._topology.local_world_size,
                input_lengths=sharding_option.input_lengths,
                num_poolings=num_poolings,
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
    batch_sizes: List[int],
    world_size: int,
    local_world_size: int,
    input_lengths: List[float],
    num_poolings: List[float],
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
        batch_sizes (List[int]): batch size for each input feature.
        world_size (int): total number of devices in topology.
        local_world_size (int): total number of devices in host group topology.
        input_lengths (List[float]): average input lengths synonymous with pooling
            factors.
        num_poolings (List[float]): average number of poolings per sample
            (typically 1.0).
        caching_ratio (float): ratio of HBM to DDR memory for UVM caching.
        is_pooled (bool): True if embedding output is pooled (ie. `EmbeddingBag`), False
            if unpooled/sequential (ie. `Embedding`).

    Returns:
        List[Storage]: storage object for each device in topology.
    """

    input_data_type_size = BIGINT_DTYPE
    output_data_type_size = tensor.element_size()

    input_sizes, output_sizes = _calculate_shard_io_sizes(
        sharding_type=sharding_type,
        batch_sizes=batch_sizes,
        world_size=world_size,
        local_world_size=local_world_size,
        input_lengths=input_lengths,
        emb_dim=tensor.shape[1],
        shard_sizes=shard_sizes,
        input_data_type_size=input_data_type_size,
        output_data_type_size=output_data_type_size,
        num_poolings=num_poolings,
        is_pooled=is_pooled,
    )

    tensor_storage = sharder.storage_usage(tensor, compute_device, compute_kernel)
    hbm_storage: int = tensor_storage.get("hbm", 0)
    ddr_storage: int = tensor_storage.get("ddr", 0)

    if compute_kernel in {
        EmbeddingComputeKernel.FUSED_UVM_CACHING.value,
        EmbeddingComputeKernel.QUANT_UVM_CACHING.value,
    }:
        hbm_storage = round(ddr_storage * caching_ratio)

    optimizer_class = getattr(tensor, "_optimizer_class", None)

    hbm_specific_sizes: List[int] = _calculate_storage_specific_sizes(
        storage=hbm_storage,
        shape=tensor.shape,
        shard_sizes=shard_sizes,
        sharding_type=sharding_type,
        optimizer_class=optimizer_class,
    )
    ddr_specific_sizes: List[int] = _calculate_storage_specific_sizes(
        storage=ddr_storage,
        shape=tensor.shape,
        shard_sizes=shard_sizes,
        sharding_type=sharding_type,
        optimizer_class=optimizer_class,
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
    batch_sizes: List[int],
    world_size: int,
    local_world_size: int,
    input_lengths: List[float],
    emb_dim: int,
    shard_sizes: List[List[int]],
    input_data_type_size: int,
    output_data_type_size: int,
    num_poolings: List[float],
    is_pooled: bool,
) -> Tuple[List[int], List[int]]:
    if sharding_type == ShardingType.DATA_PARALLEL.value:
        return _calculate_dp_shard_io_sizes(
            batch_sizes=batch_sizes,
            input_lengths=input_lengths,
            emb_dim=emb_dim,
            num_shards=len(shard_sizes),
            input_data_type_size=input_data_type_size,
            output_data_type_size=output_data_type_size,
            num_poolings=num_poolings,
            is_pooled=is_pooled,
        )
    elif sharding_type == ShardingType.TABLE_WISE.value:
        return _calculate_tw_shard_io_sizes(
            batch_sizes=batch_sizes,
            world_size=world_size,
            input_lengths=input_lengths,
            emb_dim=emb_dim,
            input_data_type_size=input_data_type_size,
            output_data_type_size=output_data_type_size,
            num_poolings=num_poolings,
            is_pooled=is_pooled,
        )
    elif sharding_type in {
        ShardingType.COLUMN_WISE.value,
        ShardingType.TABLE_COLUMN_WISE.value,
    }:
        return _calculate_cw_shard_io_sizes(
            batch_sizes=batch_sizes,
            world_size=world_size,
            input_lengths=input_lengths,
            shard_sizes=shard_sizes,
            input_data_type_size=input_data_type_size,
            output_data_type_size=output_data_type_size,
            num_poolings=num_poolings,
            is_pooled=is_pooled,
        )
    elif sharding_type == ShardingType.ROW_WISE.value:
        return _calculate_rw_shard_io_sizes(
            batch_sizes=batch_sizes,
            world_size=world_size,
            input_lengths=input_lengths,
            shard_sizes=shard_sizes,
            input_data_type_size=input_data_type_size,
            output_data_type_size=output_data_type_size,
            num_poolings=num_poolings,
            is_pooled=is_pooled,
        )
    elif sharding_type == ShardingType.TABLE_ROW_WISE.value:
        return _calculate_twrw_shard_io_sizes(
            batch_sizes=batch_sizes,
            world_size=world_size,
            local_world_size=local_world_size,
            input_lengths=input_lengths,
            shard_sizes=shard_sizes,
            input_data_type_size=input_data_type_size,
            output_data_type_size=output_data_type_size,
            num_poolings=num_poolings,
            is_pooled=is_pooled,
        )
    else:
        raise ValueError(
            f"Unrecognized or unsupported sharding type provided: {sharding_type}"
        )


def _calculate_dp_shard_io_sizes(
    batch_sizes: List[int],
    input_lengths: List[float],
    emb_dim: int,
    num_shards: int,
    input_data_type_size: int,
    output_data_type_size: int,
    num_poolings: List[float],
    is_pooled: bool,
) -> Tuple[List[int], List[int]]:
    batch_inputs = sum(
        [x * y * z for x, y, z in zip(input_lengths, num_poolings, batch_sizes)]
    )
    batch_outputs = (
        sum([x * y for x, y in zip(num_poolings, batch_sizes)])
        if is_pooled
        else batch_inputs
    )

    input_sizes = [math.ceil(batch_inputs * input_data_type_size)] * num_shards
    output_sizes = [
        math.ceil(batch_outputs * emb_dim * output_data_type_size)
    ] * num_shards

    return input_sizes, output_sizes


def _calculate_tw_shard_io_sizes(
    batch_sizes: List[int],
    world_size: int,
    input_lengths: List[float],
    emb_dim: int,
    input_data_type_size: int,
    output_data_type_size: int,
    num_poolings: List[float],
    is_pooled: bool,
) -> Tuple[List[int], List[int]]:
    batch_inputs = sum(
        [x * y * z for x, y, z in zip(input_lengths, num_poolings, batch_sizes)]
    )
    batch_outputs = (
        sum([x * y for x, y in zip(num_poolings, batch_sizes)])
        if is_pooled
        else batch_inputs
    )

    input_sizes = [math.ceil(batch_inputs * world_size * input_data_type_size)]
    output_sizes = [
        math.ceil(batch_outputs * world_size * emb_dim * output_data_type_size)
    ]

    return input_sizes, output_sizes


def _calculate_cw_shard_io_sizes(
    batch_sizes: List[int],
    world_size: int,
    input_lengths: List[float],
    shard_sizes: List[List[int]],
    input_data_type_size: int,
    output_data_type_size: int,
    num_poolings: List[float],
    is_pooled: bool,
) -> Tuple[List[int], List[int]]:
    batch_inputs = sum(
        [x * y * z for x, y, z in zip(input_lengths, num_poolings, batch_sizes)]
    )
    batch_outputs = (
        sum([x * y for x, y in zip(num_poolings, batch_sizes)])
        if is_pooled
        else batch_inputs
    )

    input_sizes = [math.ceil(batch_inputs * world_size * input_data_type_size)] * len(
        shard_sizes
    )
    output_sizes = [
        math.ceil(
            batch_outputs * world_size * shard_sizes[i][1] * output_data_type_size
        )
        for i in range(len(shard_sizes))
    ]

    return input_sizes, output_sizes


def _calculate_rw_shard_io_sizes(
    batch_sizes: List[int],
    world_size: int,
    input_lengths: List[float],
    shard_sizes: List[List[int]],
    input_data_type_size: int,
    output_data_type_size: int,
    num_poolings: List[float],
    is_pooled: bool,
) -> Tuple[List[int], List[int]]:
    batch_inputs = (
        sum([x * y * z for x, y, z in zip(input_lengths, num_poolings, batch_sizes)])
        / world_size
    )
    batch_outputs = (
        sum([x * y for x, y in zip(num_poolings, batch_sizes)])
        if is_pooled
        else batch_inputs
    )

    input_sizes = [
        math.ceil(batch_inputs * world_size * input_data_type_size)
        if prod(shard) != 0
        else 0
        for shard in shard_sizes
    ]
    output_sizes = [
        math.ceil(
            batch_outputs * world_size * shard_sizes[i][1] * output_data_type_size
        )
        if prod(shard) != 0
        else 0
        for i, shard in enumerate(shard_sizes)
    ]

    return input_sizes, output_sizes


def _calculate_twrw_shard_io_sizes(
    batch_sizes: List[int],
    world_size: int,
    local_world_size: int,
    input_lengths: List[float],
    shard_sizes: List[List[int]],
    input_data_type_size: int,
    output_data_type_size: int,
    num_poolings: List[float],
    is_pooled: bool,
) -> Tuple[List[int], List[int]]:
    batch_inputs = (
        sum([x * y * z for x, y, z in zip(input_lengths, num_poolings, batch_sizes)])
        / local_world_size
    )
    batch_outputs = (
        sum([x * y for x, y in zip(num_poolings, batch_sizes)])
        if is_pooled
        else batch_inputs
    )

    input_sizes = [
        math.ceil(batch_inputs * world_size * input_data_type_size)
        if prod(shard) != 0
        else 0
        for shard in shard_sizes
    ]
    output_sizes = [
        math.ceil(
            batch_outputs * world_size * shard_sizes[i][1] * output_data_type_size
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
    optimizer_class: Optional[Type[torch.optim.Optimizer]] = None,
) -> List[int]:
    tensor_sizes: List[int] = [
        math.ceil(storage * prod(size) / prod(shape))
        if sharding_type != ShardingType.DATA_PARALLEL.value
        else storage
        for size in shard_sizes
    ]
    optimizer_multipler: float = _get_optimizer_multipler(optimizer_class, shape)

    optimizer_sizes: List[int] = [
        math.ceil(tensor_size * optimizer_multipler) for tensor_size in tensor_sizes
    ]

    return [
        tensor_size + optimizer_size
        for tensor_size, optimizer_size in zip(tensor_sizes, optimizer_sizes)
    ]


def _get_optimizer_multipler(
    optimizer_class: Optional[Type[torch.optim.Optimizer]],
    shape: torch.Size,
) -> float:
    if not optimizer_class:
        return 0.0
    if optimizer_class in [torch.optim.SGD, trec_optim.SGD]:
        return 0
    elif optimizer_class in [torch.optim.Adam, trec_optim.Adam]:
        return 2
    elif optimizer_class == trec_optim.RowWiseAdagrad:
        return 1 / shape[-1]
    else:
        return 1
