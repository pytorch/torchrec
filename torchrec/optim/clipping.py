#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from enum import Enum, unique
from typing import Any, List, Union

import torch

from torch.distributed._tensor.api import DTensor

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
        norm_type (float or str): type of the used p-norm. Can be ``'inf'`` for infinity norm.
    """

    def __init__(
        self,
        optimizer: KeyedOptimizer,
        clipping: GradientClipping = GradientClipping.NONE,
        max_gradient: float = 0.1,
        norm_type: Union[float, str] = 2.0,
    ) -> None:
        super().__init__(optimizer)
        self._clipping = clipping
        self._max_gradient = max_gradient
        self._norm_type = norm_type
        self._check_meta: bool = True

        self._params: List[torch.Tensor] = []
        for param_group in self.param_groups:
            self._params += list(param_group["params"])

        # Convert dtensors to local tensors for performance reason;
        # otherwise, it needs to go thru dtensor dispatch, which is
        # quite slow currently.
        with torch.autograd.profiler.record_function(
            "Dtensors => Tensors in GradientClippingOptimizer::init()"
        ):
            with torch.no_grad():
                # Under no_grad(), p.to_local() will be as cheap as p._local_tensor.
                for i, p in enumerate(self._params):
                    if not isinstance(p, DTensor):
                        continue
                    local_p = p.to_local()
                    if p.grad is None:
                        local_p.grad = None
                    else:
                        # if p is a DTensor, so should be p.grad
                        assert isinstance(p.grad, DTensor)
                        local_p.grad = p.grad.to_local()
                    self._params[i] = local_p

    # pyre-ignore [2]
    def step(self, closure: Any = None) -> None:
        if self._check_meta:
            if any(t.device.type == "meta" for t in self._params):
                # skip gradient clipping and early return
                super().step(closure)
                return
            self._check_meta = False

        if self._clipping == GradientClipping.NORM:
            torch.nn.utils.clip_grad_norm_(
                self._params, self._max_gradient, norm_type=self._norm_type
            )
        elif self._clipping == GradientClipping.VALUE:
            torch.nn.utils.clip_grad_value_(self._params, self._max_gradient)

        super().step(closure)
