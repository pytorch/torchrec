#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import math
from typing import Dict, Optional, Set, List

from torch import nn
from torchrec.distributed.planner.constants import BIGINT_DTYPE, POOLING_FACTOR
from torchrec.distributed.planner.types import (
    StorageReservation,
    Topology,
    ParameterConstraints,
    Storage,
)
from torchrec.distributed.planner.utils import sharder_name
from torchrec.distributed.types import ModuleSharder


class FixedPercentageReservation(StorageReservation):
    def __init__(self, percentage: float) -> None:
        assert percentage >= 0 and percentage <= 1
        self._percentage: float = percentage

    def reserve(
        self,
        topology: Topology,
        module: nn.Module,
        sharders: List[ModuleSharder[nn.Module]],
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
    ) -> Topology:
        reserved_topology = copy.deepcopy(topology)
        for device in reserved_topology.devices:
            device.storage.hbm = int((1 - self._percentage) * device.storage.hbm)
            device.storage.ddr = int((1 - self._percentage) * device.storage.ddr)
        return reserved_topology


class HeuristicalStorageReservation(StorageReservation):
    """
    Reserves storage for model to be sharded with heuristical calculation. The storage
    reservation is comprised of unshardable tensor storage, KJT storage, and an extra
    percentage.

    Constructor Args:
        percentage (float): extra storage percentage to reserve that acts as a margin of
            error beyond heuristic calculation of storage.
    """

    def __init__(self, percentage: float) -> None:
        assert percentage >= 0 and percentage <= 1
        self._percentage: float = percentage

    def reserve(
        self,
        topology: Topology,
        module: nn.Module,
        sharders: List[ModuleSharder[nn.Module]],
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
    ) -> Topology:
        reserved_topology = copy.deepcopy(topology)

        sharder_map: Dict[str, ModuleSharder[nn.Module]] = {
            sharder_name(sharder.module_type): sharder for sharder in sharders
        }
        shardable_parameters: Set[nn.Parameter] = set()
        all_input_lengths: List[float] = []

        for child_module in module.modules():
            sharder_key = sharder_name(type(child_module))
            sharder = sharder_map.get(sharder_key)
            if not sharder:
                continue

            names = sharder.shardable_parameters(child_module).keys()
            parameters = sharder.shardable_parameters(child_module).values()

            shardable_parameters.update(parameters)

            all_input_lengths.extend(
                [
                    sum(constraints[name].pooling_factors)
                    if constraints and constraints.get(name)
                    else POOLING_FACTOR
                    for name in names
                ]
            )

        _reserve_unshardable_tensors_storage(
            reserved_topology, module, shardable_parameters
        )

        _reserve_kjt_storage(reserved_topology, all_input_lengths, BIGINT_DTYPE)

        _reserve_storage_percentage(reserved_topology, self._percentage)

        return reserved_topology


def _reserve_unshardable_tensors_storage(
    topology: Topology, module: nn.Module, shardable_parameters: Set[nn.Parameter]
) -> None:
    unshardable_parameters = set(module.parameters()) - shardable_parameters

    unshardable_parameters_size = sum(
        [
            # heuristic: 6 * dense parameter size (https://fburl.com/q8qcxvgx)
            # parameter + optimizer (~2x parameter) + ddp (~3x parameter)
            6 * parameter.element_size() * parameter.nelement()
            for parameter in unshardable_parameters
        ]
    )

    buffers_size = sum(
        [
            buffer.element_size() * buffer.nelement()
            for _, buffer in module.named_buffers()
        ]
    )

    unshardable_tensors_size = unshardable_parameters_size + buffers_size

    unshardable_tensors_storage = Storage(
        hbm=unshardable_tensors_size if topology.compute_device == "cuda" else 0,
        ddr=unshardable_tensors_size if topology.compute_device == "cpu" else 0,
    )

    for device in topology.devices:
        device.storage -= unshardable_tensors_storage


def _reserve_kjt_storage(
    topology: Topology,
    all_input_lengths: List[float],
    input_data_type_size: int,
) -> None:
    kjt_size = (
        math.ceil(
            topology.batch_size
            # pyre-ignore[58]
            * sum(all_input_lengths)
            * input_data_type_size
        )
        * 20  # 2 pipelined batches each with 10 internal copies
    )

    kjt_storage = Storage(
        hbm=kjt_size if topology.compute_device == "cuda" else 0,
        ddr=kjt_size if topology.compute_device == "cpu" else 0,
    )

    for device in topology.devices:
        device.storage -= kjt_storage


def _reserve_storage_percentage(topology: Topology, percent: float) -> None:
    for device in topology.devices:
        device.storage.hbm = int((1 - percent) * device.storage.hbm)
        device.storage.ddr = int((1 - percent) * device.storage.ddr)
