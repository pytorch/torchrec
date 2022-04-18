#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Tuple, Optional, Dict, List, Union

import torch
from torch import nn
from torchrec.distributed.planner.constants import (
    MIN_CW_DIM,
    POOLING_FACTOR,
)
from torchrec.distributed.planner.shard_estimators import (
    EmbeddingPerfEstimator,
    EmbeddingStorageEstimator,
)
from torchrec.distributed.planner.types import (
    ParameterConstraints,
    Enumerator,
    ShardingOption,
    Shard,
    Topology,
    PartitionByType,
    ShardEstimator,
)
from torchrec.distributed.planner.utils import sharder_name
from torchrec.distributed.types import ModuleSharder, ShardingType


class EmbeddingEnumerator(Enumerator):
    """
    Generates embedding sharding options for given `nn.Module`, considering user provided
    constraints.

    Args:
        topology (Topology): device topology.
        constraints (Optional[Dict[str, ParameterConstraints]]): dict of parameter names
            to provided ParameterConstraints.
    """

    def __init__(
        self,
        topology: Topology,
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
        estimator: Optional[Union[ShardEstimator, List[ShardEstimator]]] = None,
    ) -> None:
        self._compute_device: str = topology.compute_device
        self._world_size: int = topology.world_size
        self._local_world_size: int = topology.local_world_size
        self._constraints = constraints
        self._batch_size: int = topology.batch_size

        if estimator:
            self._estimators: List[ShardEstimator] = (
                [estimator] if not isinstance(estimator, list) else estimator
            )
        else:
            self._estimators: List[ShardEstimator] = [
                EmbeddingPerfEstimator(topology=topology, constraints=constraints),
                EmbeddingStorageEstimator(topology=topology, constraints=constraints),
            ]

    def enumerate(
        self,
        module: nn.Module,
        sharders: List[ModuleSharder[nn.Module]],
    ) -> List[ShardingOption]:
        """
        Generates relevant sharding options given module and sharders.

        Args:
            module (nn.Module): module to be sharded.
            sharders (List[ModuleSharder[nn.Module]]): provided sharders for module.

        Returns:
            List[ShardingOption]: valid sharding options with values populated.
        """

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
                for sharding_type in self._filter_sharding_types(
                    name, sharder.sharding_types(self._compute_device)
                ):
                    for compute_kernel in self._filter_compute_kernels(
                        name,
                        sharder.compute_kernels(sharding_type, self._compute_device),
                    ):
                        input_lengths = (
                            self._constraints[name].pooling_factors
                            if self._constraints and self._constraints.get(name)
                            else [POOLING_FACTOR]
                        )
                        col_wise_shard_dim = (
                            self._constraints[name].min_partition
                            if self._constraints and self._constraints.get(name)
                            else None
                        )
                        (
                            shard_sizes,
                            shard_offsets,
                        ) = calculate_shard_sizes_and_offsets(
                            tensor=param,
                            world_size=self._world_size,
                            local_world_size=self._local_world_size,
                            sharding_type=sharding_type,
                            col_wise_shard_dim=col_wise_shard_dim,
                        )
                        sharding_options.append(
                            ShardingOption(
                                name=name,
                                tensor=param,
                                module=(child_path, child_module),
                                upstream_modules=[],
                                downstream_modules=[],
                                input_lengths=input_lengths,
                                batch_size=self._batch_size,
                                compute_kernel=compute_kernel,
                                sharding_type=sharding_type,
                                partition_by=get_partition_by_type(sharding_type),
                                shards=[
                                    Shard(size=size, offset=offset)
                                    for size, offset in zip(shard_sizes, shard_offsets)
                                ],
                            )
                        )

        for estimator in self._estimators:
            estimator.estimate(sharding_options, sharder_map)

        return sharding_options

    def _filter_sharding_types(self, name: str, sharding_types: List[str]) -> List[str]:
        if not self._constraints or not self._constraints.get(name):
            return sharding_types
        constraints: ParameterConstraints = self._constraints[name]
        if not constraints.sharding_types:
            return sharding_types
        constrained_sharding_types: List[str] = constraints.sharding_types

        sharding_types = list(set(constrained_sharding_types) & set(sharding_types))

        if not sharding_types:
            raise RuntimeError(
                f"No available sharding types after applying user provided constraints for {name}"
            )
        return sharding_types

    def _filter_compute_kernels(
        self, name: str, compute_kernels: List[str]
    ) -> List[str]:
        if not self._constraints or not self._constraints.get(name):
            return compute_kernels
        constraints: ParameterConstraints = self._constraints[name]
        if not constraints.compute_kernels:
            return compute_kernels
        constrained_compute_kernels: List[str] = constraints.compute_kernels

        compute_kernels = list(set(constrained_compute_kernels) & set(compute_kernels))

        if not compute_kernels:
            raise RuntimeError(
                f"No available compute kernels after applying user provided constraints for {name}"
            )
        return compute_kernels


def get_partition_by_type(sharding_type: str) -> str:
    """
    Gets corresponding partition by type for provided sharding type.

    Args:
        sharding_type (str): sharding type string.

    Returns:
        str: the corresponding `PartitionByType` value.
    """

    device_sharding_types = {
        ShardingType.TABLE_WISE.value,
        ShardingType.COLUMN_WISE.value,
    }
    host_sharding_types = {
        ShardingType.TABLE_ROW_WISE.value,
        ShardingType.TABLE_COLUMN_WISE.value,
    }
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

    raise ValueError(
        f"Unrecognized or unsupported sharding type provided: {sharding_type}"
    )


def calculate_shard_sizes_and_offsets(
    tensor: torch.Tensor,
    world_size: int,
    local_world_size: int,
    sharding_type: str,
    col_wise_shard_dim: Optional[int] = None,
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Calculates sizes and offsets for tensor sharded according to provided sharding type.

    Args:
        tensor (torch.Tensor): tensor to be sharded.
        world_size (int): total number of devices in topology.
        local_world_size (int): total number of devices in host group topology.
        sharding_type (str): provided ShardingType value.
        col_wise_shard_dim (Optional[int]): dimension for column wise sharding split.

    Returns:
        Tuple[List[List[int]], List[List[int]]]: shard sizes, represented as a list of the dimensions of the sharded tensor on each device, and shard offsets, represented as a list of coordinates of placement on each device.

    Raises:
        ValueError: If `sharding_type` is not a valid ShardingType.
    """

    (rows, columns) = tensor.shape

    if sharding_type == ShardingType.DATA_PARALLEL.value:
        return [[rows, columns]] * world_size, [[0, 0]] * world_size
    elif sharding_type == ShardingType.TABLE_WISE.value:
        return [[rows, columns]], [[0, 0]]
    elif sharding_type == ShardingType.ROW_WISE.value:
        return _calculate_rw_shard_sizes_and_offsets(rows, world_size, columns)
    elif sharding_type == ShardingType.TABLE_ROW_WISE.value:
        return _calculate_rw_shard_sizes_and_offsets(rows, local_world_size, columns)
    elif (
        sharding_type == ShardingType.COLUMN_WISE.value
        or sharding_type == ShardingType.TABLE_COLUMN_WISE.value
    ):
        return _calculate_cw_shard_sizes_and_offsets(columns, rows, col_wise_shard_dim)

    raise ValueError(
        f"Unrecognized or unsupported sharding type provided: {sharding_type}"
    )


def _calculate_rw_shard_sizes_and_offsets(
    hash_size: int, num_devices: int, columns: int
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Sets prefix of shard_sizes to be ceil(hash_size/num_devices).

    For example if hash_size = 10, num_devices = 3, we will allocate the rows as 3,3,3,1
    (rather than 3,3,2,2).
    This is due to implementation in RW sharding that sets block_size_lists to be ceil.
    The balanced way is harder to support on GPU.
    For more details see https://fb.quip.com/xbgbAchCTOL0

    Also consider the example of hash_size = 5, num_devices = 4. The expected rows per
    rank is [2,2,1,0].
    """

    block_size: int = math.ceil(hash_size / num_devices)
    last_rank: int = hash_size // block_size
    last_block_size: int = hash_size - block_size * last_rank
    shard_sizes: List[List[int]] = []

    for rank in range(num_devices):
        if rank < last_rank:
            local_row: int = block_size
        elif rank == last_rank:
            local_row: int = last_block_size
        else:
            local_row: int = 0
        shard_sizes.append([local_row, columns])
    shard_offsets = [[0, 0]]

    for i in range(num_devices - 1):
        shard_offsets.append([shard_sizes[i][0] + shard_offsets[i][0], 0])

    return shard_sizes, shard_offsets


def _calculate_cw_shard_sizes_and_offsets(
    hash_size: int,
    rows: int,
    col_wise_shard_dim: Optional[int] = None,
) -> Tuple[List[List[int]], List[List[int]]]:
    block_size: int = min(
        col_wise_shard_dim if col_wise_shard_dim else MIN_CW_DIM, hash_size
    )
    num_col_wise_shards, residual = divmod(hash_size, block_size)

    shard_sizes: List[List[int]] = [[rows, block_size]] * (num_col_wise_shards - 1)
    shard_sizes.append([rows, block_size + residual])

    shard_offsets: List[List[int]] = [
        [0, block_size * rank] for rank in range(num_col_wise_shards)
    ]
    return shard_sizes, shard_offsets
