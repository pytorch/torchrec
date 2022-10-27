#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod

import torch
from torch import nn


class CopyMixIn:
    @abstractmethod
    def copy(self, device: torch.device) -> nn.Module:
        ...


class ModuleCopyMixin(CopyMixIn):
    """
    A mixin to allow modules to override copy behaviors in DMP.
    """

    def copy(self, device: torch.device) -> nn.Module:
        # pyre-ignore [16]
        return self.to(device)


class ModuleNoCopyMixin(CopyMixIn):
    """
    A mixin to allow modules to override copy behaviors in DMP.
    """

    def copy(self, device: torch.device) -> nn.Module:
        # pyre-ignore [7]
        return self
