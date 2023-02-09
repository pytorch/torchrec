#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
import math
from typing import Dict, List, Optional, Set, Tuple

from torch import nn
from torchrec.distributed.planner.constants import BIGINT_DTYPE, POOLING_FACTOR
from torchrec.distributed.planner.types import (
    ParameterConstraints,
    Storage,
    StorageReservation,
    Topology,
)
from torchrec.distributed.planner.utils import sharder_name
from torchrec.distributed.types import ModuleSharder


logger: logging.Logger = logging.getLogger(__name__)


def _get_module_size(module: nn.Module, multiplier: float) -> int:
    parameters_size = sum(
        [
            multiplier * parameter.element_size() * parameter.nelement()
            for parameter in module.parameters()
        ]
    )

    buffers_size = sum(
        [buffer.element_size() * buffer.nelement() for buffer in module.buffers()]
    )

    return round(parameters_size + buffers_size)


def _get_dense_tensor_size(
    module: nn.Module,
    shardable_modules: Set[nn.Module],
    multiplier: float = 6.0,
) -> int:
    dense_tensor_size = _get_module_size(module, multiplier) - sum(
        [
            _get_module_size(shardable_module, multiplier)
            for shardable_module in shardable_modules
        ]
    )
    return dense_tensor_size


def _reserve_dense_storage(
    topology: Topology,
    module: nn.Module,
    shardable_modules: Set[nn.Module],
    multiplier: float,
    dense_tensor_estimate: Optional[int] = None,
) -> Storage:

    dense_tensor_size = _get_dense_tensor_size(module, shardable_modules, multiplier)
    if dense_tensor_estimate:
        logger.info(
            f"We override default dense tensor estimate ({dense_tensor_size} bytes) "
            f"with user-provided dense tensor estimate ({dense_tensor_estimate} bytes)."
        )
        dense_tensor_size = dense_tensor_estimate

    dense_tensor_storage = Storage(
        hbm=dense_tensor_size if topology.compute_device == "cuda" else 0,
        ddr=dense_tensor_size if topology.compute_device == "cpu" else 0,
    )

    for device in topology.devices:
        device.storage -= dense_tensor_storage

    return dense_tensor_storage


def _reserve_kjt_storage(
    topology: Topology,
    batch_size: int,
    input_lengths: List[float],
    input_data_type_size: int,
    multiplier: int,
) -> Storage:
    kjt_size = (
        math.ceil(float(batch_size) * sum(input_lengths) * float(input_data_type_size))
        * multiplier
    )

    kjt_storage = Storage(
        hbm=kjt_size if topology.compute_device == "cuda" else 0,
        ddr=kjt_size if topology.compute_device == "cpu" else 0,
    )

    for device in topology.devices:
        device.storage -= kjt_storage

    return kjt_storage


def _reserve_storage_percentage(topology: Topology, percent: float) -> None:
    for device in topology.devices:
        device.storage.hbm = int((1 - percent) * device.storage.hbm)


def _get_input_lengths_and_shardable_parameters(
    module: nn.Module,
    sharders: List[ModuleSharder[nn.Module]],
    constraints: Optional[Dict[str, ParameterConstraints]] = None,
) -> Tuple[List[float], Set[nn.Module]]:
    sharder_map: Dict[str, ModuleSharder[nn.Module]] = {
        sharder_name(sharder.module_type): sharder for sharder in sharders
    }
    input_lengths: List[float] = []
    shardable_modules: Set[nn.Module] = set()

    def populate_shardable_modules(
        module: nn.Module,
    ) -> None:
        sharder_key = sharder_name(type(module))
        sharder = sharder_map.get(sharder_key)

        if not sharder:
            for _child_name, child in module.named_children():
                populate_shardable_modules(child)
        else:
            names = sharder.shardable_parameters(module).keys()
            shardable_modules.add(module)

            input_lengths.extend(
                [
                    sum(constraints[name].pooling_factors)
                    if constraints and constraints.get(name)
                    else POOLING_FACTOR
                    for name in names
                ]
            )

    populate_shardable_modules(module)

    return input_lengths, shardable_modules


class FixedPercentageStorageReservation(StorageReservation):
    def __init__(self, percentage: float) -> None:
        assert percentage >= 0 and percentage <= 1
        self._percentage: float = percentage

    def reserve(
        self,
        topology: Topology,
        batch_size: int,
        module: nn.Module,
        sharders: List[ModuleSharder[nn.Module]],
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
    ) -> Topology:
        reserved_topology = copy.deepcopy(topology)
        _reserve_storage_percentage(reserved_topology, self._percentage)
        return reserved_topology


class HeuristicalStorageReservation(StorageReservation):
    """
    Reserves storage for model to be sharded with heuristical calculation. The storage
    reservation is comprised of dense tensor storage, KJT storage, and an extra
    percentage of total storage.

    Args:
        percentage (float): extra storage percent to reserve that acts as a margin of
            error beyond heuristic calculation of storage.
        parameter_multiplier (float): heuristic multiplier for total parameter storage.
        dense_tensor_estimate (Optional[int]): storage estimate for dense tensors, uses
            default heuristic estimate if not provided.
    """

    def __init__(
        self,
        percentage: float,
        # heuristic: 6 * dense parameter size
        # parameter + optimizer (~2x parameter) + ddp (~3x parameter)
        parameter_multiplier: float = 6.0,
        dense_tensor_estimate: Optional[int] = None,
    ) -> None:
        assert percentage >= 0 and percentage <= 1
        self._percentage: float = percentage
        self._parameter_multiplier = parameter_multiplier
        self._dense_tensor_estimate = dense_tensor_estimate

        self._dense_storage: Optional[Storage] = None
        self._kjt_storage: Optional[Storage] = None

    def reserve(
        self,
        topology: Topology,
        batch_size: int,
        module: nn.Module,
        sharders: List[ModuleSharder[nn.Module]],
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
    ) -> Topology:
        reserved_topology = copy.deepcopy(topology)

        input_lengths, shardable_modules = _get_input_lengths_and_shardable_parameters(
            module, sharders, constraints
        )

        _reserve_storage_percentage(reserved_topology, self._percentage)

        self._dense_storage = _reserve_dense_storage(
            topology=reserved_topology,
            module=module,
            shardable_modules=shardable_modules,
            multiplier=self._parameter_multiplier,
            dense_tensor_estimate=self._dense_tensor_estimate,
        )

        self._kjt_storage = _reserve_kjt_storage(
            topology=reserved_topology,
            batch_size=batch_size,
            input_lengths=input_lengths,
            input_data_type_size=BIGINT_DTYPE,
            # 2 pipelined batches each with 10 internal copies
            multiplier=20,
        )

        return reserved_topology


class InferenceStorageReservation(StorageReservation):
    """
    Reserves storage for model to be sharded for inference. The storage reservation
    is comprised of dense tensor storage, KJT storage, and an extra percentage of total
    storage.

    Args:
        percentage (float): extra storage percentage to reserve that acts as a margin of
            error beyond storage calculation.
    """

    def __init__(
        self,
        percentage: float,
    ) -> None:
        assert percentage >= 0 and percentage <= 1
        self._percentage: float = percentage

        self._dense_storage: Optional[Storage] = None
        self._kjt_storage: Optional[Storage] = None

    def reserve(
        self,
        topology: Topology,
        batch_size: int,
        module: nn.Module,
        sharders: List[ModuleSharder[nn.Module]],
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
    ) -> Topology:
        reserved_topology = copy.deepcopy(topology)

        input_lengths, shardable_modules = _get_input_lengths_and_shardable_parameters(
            module, sharders, constraints
        )

        _reserve_storage_percentage(reserved_topology, self._percentage)

        self._dense_storage = _reserve_dense_storage(
            topology=reserved_topology,
            module=module,
            shardable_modules=shardable_modules,
            multiplier=1,
        )

        self._kjt_storage = _reserve_kjt_storage(
            topology=reserved_topology,
            batch_size=batch_size,
            input_lengths=input_lengths,
            input_data_type_size=BIGINT_DTYPE,
            multiplier=1,
        )

        return reserved_topology
