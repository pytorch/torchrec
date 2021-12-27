#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
from typing import Any

from torch import optim
from torchrec.optim.keyed import KeyedOptimizer


class FusedOptimizer(KeyedOptimizer, abc.ABC):
    """
    Assumes that weight update is done during backward pass,
    thus step() is a no-op.
    """

    @abc.abstractmethod
    # pyre-ignore [2]
    def step(self, closure: Any = None) -> None:
        ...

    @abc.abstractmethod
    def zero_grad(self, set_to_none: bool = False) -> None:
        ...

    def __repr__(self) -> str:
        return optim.Optimizer.__repr__(self)


class FusedOptimizerModule(abc.ABC):
    """
    Module, which does weight update during backward pass.
    """

    @property
    @abc.abstractmethod
    def fused_optimizer(self) -> KeyedOptimizer:
        ...
