#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum, unique
from typing import Any, List

import torch
from torchrec.optim.keyed import KeyedOptimizer, OptimizerWrapper


@unique
class GradientClipping(Enum):
    NORM = "norm"
    VALUE = "value"
    NONE = "none"


class GradientClippingOptimizer(OptimizerWrapper):
    """
    Clips gradients before doing optimization step.

    Args:
        optimizer (KeyedOptimizer): optimizer to wrap
        clipping (GradientClipping): how to clip gradients
        max_gradient (float): max value for clipping
    """

    def __init__(
        self,
        optimizer: KeyedOptimizer,
        clipping: GradientClipping = GradientClipping.NONE,
        max_gradient: float = 0.1,
    ) -> None:
        super().__init__(optimizer)
        self._clipping = clipping
        self._max_gradient = max_gradient

        self._params: List[torch.Tensor] = []
        for param_group in self.param_groups:
            self._params += list(param_group["params"])

    # pyre-ignore [2]
    def step(self, closure: Any = None) -> None:
        if self._clipping == GradientClipping.NORM:
            torch.nn.utils.clip_grad_norm_(self._params, self._max_gradient)
        elif self._clipping == GradientClipping.VALUE:
            torch.nn.utils.clip_grad_value_(self._params, self._max_gradient)

        super().step(closure)
