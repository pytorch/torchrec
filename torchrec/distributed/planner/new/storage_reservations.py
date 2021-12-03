#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import List

from torch import nn
from torchrec.distributed.planner.new.types import StorageReservation, Topology
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
    ) -> Topology:
        reserved_topology = copy.deepcopy(topology)
        for device in reserved_topology.devices:
            device.storage.hbm = int((1 - self._percentage) * device.storage.hbm)
            device.storage.ddr = int((1 - self._percentage) * device.storage.ddr)
        return reserved_topology
