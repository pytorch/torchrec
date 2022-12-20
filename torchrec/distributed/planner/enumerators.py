#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, List, Optional, Union

from torch import nn
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.planner.constants import POOLING_FACTOR
from torchrec.distributed.planner.shard_estimators import (
    EmbeddingPerfEstimator,
    EmbeddingStorageEstimator,
)
from torchrec.distributed.planner.types import (
    Enumerator,
    ParameterConstraints,
    PartitionByType,
    Shard,
    ShardEstimator,
    ShardingOption,
    Topology,
)
from torchrec.distributed.planner.utils import sharder_name
from torchrec.distributed.sharding_plan import calculate_shard_sizes_and_offsets
from torchrec.distributed.types import ModuleSharder, ShardingType
from torchrec.modules.embedding_tower import EmbeddingTower, EmbeddingTowerCollection


logger: logging.Logger = logging.getLogger(__name__)


class EmbeddingEnumerator(Enumerator):
    """
    Generates embedding sharding options for given `nn.Module`, considering user provided
    constraints.

    Args:
        topology (Topology): device topology.
        batch_size (int): batch size.
        constraints (Optional[Dict[str, ParameterConstraints]]): dict of parameter names
            to provided ParameterConstraints.
    """

    def __init__(
        self,
        topology: Topology,
        batch_size: int,
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
        estimator: Optional[Union[ShardEstimator, List[ShardEstimator]]] = None,
    ) -> None:
        self._compute_device: str = topology.compute_device
        self._world_size: int = topology.world_size
        self._local_world_size: int = topology.local_world_size
        self._batch_size: int = batch_size
        self._constraints = constraints

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

        named_modules_queue = [("", module)]
        while named_modules_queue:
            child_path, child_module = named_modules_queue.pop()
            sharder_key = sharder_name(type(child_module))
            sharder = sharder_map.get(sharder_key, None)
            if not sharder:
                for n, m in child_module.named_children():
                    if child_path != "":
                        named_modules_queue.append((child_path + "." + n, m))
                    else:
                        named_modules_queue.append((n, m))
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
                        dependency = None
                        if isinstance(child_module, EmbeddingTower):
                            dependency = child_path
                        elif isinstance(child_module, EmbeddingTowerCollection):
                            tower_index = _get_tower_index(name, child_module)
                            dependency = child_path + ".tower_" + str(tower_index)
                        sharding_options.append(
                            ShardingOption(
                                name=name,
                                tensor=param,
                                module=(child_path, child_module),
                                input_lengths=input_lengths,
                                batch_size=self._batch_size,
                                compute_kernel=compute_kernel,
                                sharding_type=sharding_type,
                                partition_by=get_partition_by_type(sharding_type),
                                shards=[
                                    Shard(size=size, offset=offset)
                                    for size, offset in zip(shard_sizes, shard_offsets)
                                ],
                                dependency=dependency,
                            )
                        )
                if not sharding_options:
                    raise RuntimeError(
                        "No available sharding type and compute kernel combination "
                        f"after applying user provided constraints for {name}"
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
            logger.warn(
                f"No available sharding types after applying user provided constraints for {name}"
            )
        return sharding_types

    def _filter_compute_kernels(
        self,
        name: str,
        compute_kernels: List[str],
    ) -> List[str]:

        if not self._constraints or not self._constraints.get(name):
            filtered_compute_kernels = compute_kernels
        else:
            constraints: ParameterConstraints = self._constraints[name]
            if not constraints.compute_kernels:
                filtered_compute_kernels = compute_kernels
            else:
                constrained_compute_kernels: List[str] = constraints.compute_kernels
                filtered_compute_kernels = list(
                    set(constrained_compute_kernels) & set(compute_kernels)
                )

        if EmbeddingComputeKernel.DENSE.value in filtered_compute_kernels:
            if (
                EmbeddingComputeKernel.FUSED.value in filtered_compute_kernels
            ):  # always false for data_parallel
                filtered_compute_kernels.remove(EmbeddingComputeKernel.DENSE.value)

        if not filtered_compute_kernels:
            logger.warn(
                f"No available compute kernels after applying user provided constraints for {name}"
            )
        return filtered_compute_kernels


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


def _get_tower_index(name: str, child_module: EmbeddingTowerCollection) -> int:
    for i, tower in enumerate(child_module.towers):
        for n, m in tower.named_modules():
            if isinstance(m, nn.Embedding) or isinstance(m, nn.EmbeddingBag):
                table_name = n.split(".")[-1]
                if name == table_name:
                    return i
    raise RuntimeError(
        f"couldn't get the tower index for table {name}, tower collection: {child_module}"
    )
