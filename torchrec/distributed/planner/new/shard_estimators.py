#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, Optional, Tuple, List

import torch
from torch import nn
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.planner.new.constants import (
    BIGINT_DTYPE,
    INTRA_NODE_BANDWIDTH,
    CROSS_NODE_BANDWIDTH,
    kernel_bw_lookup,
    POOLING_FACTOR,
    CACHING_RATIO,
)
from torchrec.distributed.planner.new.types import (
    PlannerConstraints,
    ShardEstimator,
    Topology,
    ShardingOption,
    Storage,
    InputStats,
    PlannerError,
)
from torchrec.distributed.planner.utils import sharder_name
from torchrec.distributed.types import ModuleSharder, ShardingType


class EmbeddingPerfEstimator(ShardEstimator):
    """
    Embedding Wall Time Perf Estimator
    """

    def __init__(
        self,
        topology: Topology,
        constraints: Optional[Dict[str, PlannerConstraints]] = None,
        input_stats: Optional[Dict[str, InputStats]] = None,
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
            shard_costs = cost_func_emb_wall_time(
                shard_lengths=[shard.length for shard in sharding_option.shards],
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
                has_input_dist=True if sharding_option.upstream_modules else False,
                has_output_dist=False if sharding_option.downstream_modules else True,
                caching_ratio=caching_ratio,
            )

            for shard, cost in zip(sharding_option.shards, shard_costs):
                shard.cost = cost


def cost_func_emb_wall_time(
    shard_lengths: List[List[int]],
    compute_kernel: str,
    compute_device: str,
    sharding_type: str,
    batch_size: int,
    world_size: int,
    local_world_size: int,
    input_lengths: List[float],
    input_data_type_size: float,
    output_data_type_size: float,
    bw_intra_host: int,
    bw_inter_host: int,
    has_input_dist: bool = True,
    has_output_dist: bool = True,
    caching_ratio: Optional[float] = None,
) -> List[float]:
    """
    Attempts to model costs as a function of relative wall times.
    Only models forward costs (ignores backward costs).
    The computation cost estimation is based on EmbeddingBagCollectionSharder
    (pooledEmbedding).

    Args:
        shard_lengths (List[List[int]]): the list of (local_rows, local_cols) pf each shard.
        compute_kernel (str): comput kernel.
        compute_device (str): compute device.
        sharding_type (str): tw, rw, cw, twrw, dp.
        batch_size (int): the size of each batch.
        world_size (int): the number of devices for all hosts.
        local_world_size (int): the number of the device for each host.
        input_lengths (List[float]): the list of the average number of lookups of each input query feature.
        input_data_type_size (float): the data type size of the distributed data_parallel input.
        output_data_type_size (float): the data type size of the distributed data_parallel output.
        bw_intra_host (int): the bandwidth within the single host like multiple threads.
        bw_inter_host (int): the bandwidth between two hosts like multiple machines.
        has_input_dist (bool = True): if we need input distributed.
        has_output_dist (bool = True): if we need output distributed.
        caching_ratio (Optional[float] = None): cache ratio to determine the bandwidth of device.

    Returns:
        List[float]: the list of cost for each shard.
    """
    shard_costs = []
    B = 1.0 * world_size * batch_size  # global batch size
    device_bw = kernel_bw_lookup(compute_device, compute_kernel, caching_ratio)
    if device_bw is None:
        raise PlannerError(
            f"No kernel BW exists for this combo of compute device: {compute_device}, compute kernel: {compute_kernel}"
        )

    for hash_size, emb_dim in shard_lengths:

        if sharding_type is ShardingType.TABLE_WISE.value:
            input_cost, compute_cost, output_cost = _get_tw_sharding_cost(
                B,
                world_size,
                input_lengths,
                emb_dim,
                input_data_type_size,
                output_data_type_size,
                device_bw,
                bw_inter_host,
            )
        elif sharding_type is ShardingType.COLUMN_WISE.value:
            input_cost, compute_cost, output_cost = _get_cw_sharding_cost(
                B,
                world_size,
                input_lengths,
                emb_dim,
                input_data_type_size,
                output_data_type_size,
                device_bw,
                bw_inter_host,
            )
        elif sharding_type is ShardingType.ROW_WISE.value:
            input_cost, compute_cost, output_cost = _get_rw_sharding_cost(
                B,
                world_size,
                input_lengths,
                emb_dim,
                input_data_type_size,
                output_data_type_size,
                device_bw,
                bw_inter_host,
            )
        elif sharding_type is ShardingType.TABLE_ROW_WISE.value:
            input_cost, compute_cost, output_cost = _get_twrw_sharding_cost(
                B,
                world_size,
                local_world_size,
                input_lengths,
                emb_dim,
                input_data_type_size,
                output_data_type_size,
                device_bw,
                bw_inter_host,
                bw_intra_host,
            )
        elif sharding_type is ShardingType.DATA_PARALLEL.value:
            input_cost, compute_cost, output_cost = _get_dp_sharding_cost(
                batch_size,
                input_lengths,
                hash_size * emb_dim,
                bw_inter_host,
                emb_dim,
                output_data_type_size,
                device_bw,
            )
        else:
            raise RuntimeError(f"Unexpected sharding type: {sharding_type}")

        shard_cost = 0
        shard_cost += input_cost if has_input_dist else 0
        shard_cost += compute_cost
        shard_cost += output_cost if has_output_dist else 0
        shard_costs.append(shard_cost)

    return shard_costs


def _get_tw_sharding_cost(
    global_batch_size: float,
    world_size: int,
    input_lengths: List[float],
    emb_dim: int,
    input_data_type_size: float,
    output_data_type_size: float,
    device_bw: float,
    bw_inter_host: int,
) -> Tuple[float, float, float]:
    input_cost = (
        global_batch_size * sum(input_lengths) * input_data_type_size / bw_inter_host
    )
    compute_cost = (
        global_batch_size
        * sum(input_lengths)
        * emb_dim
        * output_data_type_size
        / device_bw
    )
    output_cost = (
        global_batch_size
        * emb_dim
        * len(input_lengths)
        * output_data_type_size
        / bw_inter_host
    )
    return (input_cost, compute_cost, output_cost)


def _get_cw_sharding_cost(
    global_batch_size: float,
    world_size: int,
    input_lengths: List[float],
    emb_dim: int,
    input_data_type_size: float,
    output_data_type_size: float,
    device_bw: float,
    bw_inter_host: int,
) -> Tuple[float, float, float]:
    input_cost = (
        global_batch_size * sum(input_lengths) * input_data_type_size / bw_inter_host
    )
    compute_cost = (
        global_batch_size
        * sum(input_lengths)
        * emb_dim
        * output_data_type_size
        / device_bw
    )
    output_cost = (
        global_batch_size
        * emb_dim
        * len(input_lengths)
        * output_data_type_size
        / bw_inter_host
    )
    return (input_cost, compute_cost, output_cost)


def _get_rw_sharding_cost(
    global_batch_size: float,
    world_size: int,
    input_lengths: List[float],
    emb_dim: int,
    input_data_type_size: float,
    output_data_type_size: float,
    device_bw: float,
    bw_inter_host: int,
) -> Tuple[float, float, float]:
    input_cost = (
        global_batch_size
        * sum(input_lengths)
        / world_size
        * input_data_type_size
        / bw_inter_host
    )
    compute_cost = (
        global_batch_size
        * sum(input_lengths)
        / world_size
        * emb_dim
        * output_data_type_size
        / device_bw
    )
    output_cost = (
        global_batch_size
        * emb_dim
        * len(input_lengths)
        * output_data_type_size
        / bw_inter_host
    )
    return (input_cost, compute_cost, output_cost)


def _get_twrw_sharding_cost(
    global_batch_size: float,
    world_size: int,
    local_world_size: int,
    input_lengths: List[float],
    emb_dim: int,
    input_data_type_size: float,
    output_data_type_size: float,
    device_bw: float,
    bw_inter_host: int,
    bw_intra_host: int,
) -> Tuple[float, float, float]:
    input_cost = (
        global_batch_size
        * sum(input_lengths)
        / local_world_size
        * input_data_type_size
        / bw_inter_host
    )
    compute_cost = (
        global_batch_size
        * sum(input_lengths)
        / local_world_size
        * emb_dim
        * output_data_type_size
        / device_bw
    )
    output_cost = (
        global_batch_size
        * emb_dim
        * len(input_lengths)
        * output_data_type_size
        / bw_intra_host
        + global_batch_size
        * emb_dim
        * len(input_lengths)
        * output_data_type_size
        * (local_world_size / world_size)
        / bw_inter_host
    )
    return (input_cost, compute_cost, output_cost)


def _get_dp_sharding_cost(
    batch_size: float,
    input_lengths: List[float],
    grad_num_elem: int,
    bw_inter_host: int,
    emb_dim: int,
    output_data_type_size: float,
    device_bw: float,
) -> Tuple[float, float, float]:
    input_cost = 0
    compute_cost = (
        batch_size * sum(input_lengths) * emb_dim * output_data_type_size / device_bw
    )
    # TODO: this is allreduce cost, better separated out as backward cost
    output_cost = grad_num_elem * output_data_type_size / bw_inter_host
    return (input_cost, compute_cost, output_cost)


class EmbeddingStorageEstimator(ShardEstimator):
    """
    Embedding Storage Usage Estimator
    """

    def __init__(
        self,
        topology: Topology,
        constraints: Optional[Dict[str, PlannerConstraints]] = None,
        input_stats: Optional[Dict[str, InputStats]] = None,
    ) -> None:
        self._topology = topology
        self._constraints = constraints
        self._input_stats = input_stats

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
                self._input_stats[sharding_option.name].pooling_factors
                if self._input_stats and self._input_stats.get(sharding_option.name)
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
                shard_lengths=[shard.length for shard in sharding_option.shards],
                batch_size=self._topology.batch_size,
                world_size=self._topology.world_size,
                local_world_size=self._topology.local_world_size,
                input_lengths=input_lengths,
                caching_ratio=caching_ratio if caching_ratio else CACHING_RATIO,
            )

            for shard, storage in zip(sharding_option.shards, shard_storages):
                shard.storage = storage


def calculate_shard_storages(
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
    """
    Calculates estimated storage sizes for each sharded tensor, comprised of input,
    output, tensor, gradient, and optimizer sizes.

    Args:
        sharder (ModuleSharder[nn.Module]): sharder for module that supports sharding.
        sharding_type (str): provided ShardingType value.
        tensor (torch.Tensor): tensor to be sharded.
        compute_device (str): compute device to be used.
        compute_kernel (str): compute kernel to be used.
        shard_lengths (List[List[int]]): list of dimensions of each sharded tensor.
        batch_size (int): batch size to be used.
        world_size (int): total number of devices in topology.
        local_world_size (int): total number of devices in host group topology.
        input_lengths (List[float]): average input lengths synonymous with pooling
            factors.
        caching_ratio (float): ratio of HBM to DDR memory for UVM caching.

    Returns:
        List[Storage]: storage object for each device in topology

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

    hbm_specific_sizes: List[int] = _calculate_storage_specific_sizes(
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
    ddr_specific_sizes: List[int] = _calculate_storage_specific_sizes(
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


def _calculate_shard_io_sizes(
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
        return _calculate_dp_shard_io_sizes(
            batch_size=batch_size,
            input_lengths=input_lengths,
            emb_dim=emb_dim,
            num_shards=len(shard_lengths),
            input_data_type_size=input_data_type_size,
            output_data_type_size=output_data_type_size,
        )
    elif sharding_type == ShardingType.TABLE_WISE.value:
        return _calculate_tw_shard_io_sizes(
            batch_size=batch_size,
            world_size=world_size,
            input_lengths=input_lengths,
            emb_dim=emb_dim,
            input_data_type_size=input_data_type_size,
            output_data_type_size=output_data_type_size,
        )
    elif sharding_type == ShardingType.COLUMN_WISE.value:
        return _calculate_cw_shard_io_sizes(
            batch_size=batch_size,
            world_size=world_size,
            input_lengths=input_lengths,
            shard_lengths=shard_lengths,
            input_data_type_size=input_data_type_size,
            output_data_type_size=output_data_type_size,
        )
    elif sharding_type == ShardingType.ROW_WISE.value:
        return _calculate_rw_shard_io_sizes(
            batch_size=batch_size,
            world_size=world_size,
            input_lengths=input_lengths,
            shard_lengths=shard_lengths,
            input_data_type_size=input_data_type_size,
            output_data_type_size=output_data_type_size,
        )
    elif sharding_type == ShardingType.TABLE_ROW_WISE.value:
        return _calculate_twrw_shard_io_sizes(
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


def _calculate_dp_shard_io_sizes(
    batch_size: int,
    input_lengths: List[float],
    emb_dim: int,
    num_shards: int,
    input_data_type_size: int,
    output_data_type_size: int,
) -> Tuple[List[int], List[int]]:
    input_sizes: List[int] = [
        # pyre-ignore[58]
        math.ceil(batch_size * sum(input_lengths) * input_data_type_size)
    ] * num_shards

    output_sizes: List[int] = [
        batch_size * emb_dim * len(input_lengths) * output_data_type_size
    ] * num_shards

    return input_sizes, output_sizes


def _calculate_tw_shard_io_sizes(
    batch_size: int,
    world_size: int,
    input_lengths: List[float],
    emb_dim: int,
    input_data_type_size: int,
    output_data_type_size: int,
) -> Tuple[List[int], List[int]]:
    input_sizes: List[int] = [
        # pyre-ignore[58]
        math.ceil(batch_size * world_size * sum(input_lengths) * input_data_type_size)
    ]

    output_sizes: List[int] = [
        batch_size * world_size * emb_dim * len(input_lengths) * output_data_type_size
    ]

    return input_sizes, output_sizes


def _calculate_cw_shard_io_sizes(
    batch_size: int,
    world_size: int,
    input_lengths: List[float],
    shard_lengths: List[List[int]],
    input_data_type_size: int,
    output_data_type_size: int,
) -> Tuple[List[int], List[int]]:
    input_sizes: List[int] = [
        # pyre-ignore[58]
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


def _calculate_rw_shard_io_sizes(
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
            # pyre-ignore[58]
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


def _calculate_twrw_shard_io_sizes(
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
            # pyre-ignore[58]
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


def _calculate_storage_specific_sizes(
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
