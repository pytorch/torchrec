#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections import defaultdict
from enum import Enum, unique
from typing import Any, cast, Dict, List, Union

import torch
import torch.distributed as dist

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
        # Only used if there are DTensor parameters, in which case this dict
        # holds the sharded DTensor parameters and `self._params` holds the
        # replicated tensor parameters
        self._mesh_to_dtensor_params: Dict[dist.DeviceMesh, List[DTensor]] = (
            defaultdict(list)
        )

        for param_group in self.param_groups:
            for param in param_group["params"]:
                if isinstance(param, DTensor):
                    self._mesh_to_dtensor_params[param.device_mesh].append(param)
                else:
                    self._params.append(param)

        if len(self._mesh_to_dtensor_params) == 0:
            return

        if self._clipping == GradientClipping.VALUE:
            # This path is currently not used in any production.
            raise NotImplementedError(
                "clip_grad_value_ for DTensor parameters is not supported yet"
            )

    # pyre-ignore [2]
    def step(self, closure: Any = None) -> None:
        if self._check_meta:
            if any(t.device.type == "meta" for t in self._params):
                # skip gradient clipping and early return
                super().step(closure)
                return
            self._check_meta = False

        if self._clipping == GradientClipping.NORM:
            if len(self._mesh_to_dtensor_params) == 0:
                # No DTensor parameters, so we can use the regular clip_grad_norm_
                torch.nn.utils.clip_grad_norm_(
                    self._params, self._max_gradient, norm_type=self._norm_type
                )
            else:
                # There are DTensor parameters, so we need to use _dist_clip_grad_norm
                world_size = dist.get_world_size()
                for device_mesh, dtensor_params in self._mesh_to_dtensor_params.items():
                    if device_mesh.ndim > 1:
                        # pyre-ignore[16]: `dist.device_mesh.DeviceMesh` has no attribute `_flatten`.
                        process_group = device_mesh._flatten().get_group()
                        # only do global clipping in the nD case
                        if process_group.size() != world_size:
                            continue
                    else:
                        process_group = device_mesh.get_group()
                    sharded_grads = [
                        cast(DTensor, p.grad)._local_tensor
                        for p in dtensor_params
                        if p.grad is not None
                    ]
                    sharded_grads = [grad for grad in sharded_grads if grad.numel() > 0]
                    if sharded_grads:
                        replicated_grads = [
                            p.grad for p in self._params if p.grad is not None
                        ]
                        _dist_clip_grad_norm(
                            sharded_grads,
                            replicated_grads,
                            process_group,
                            self._max_gradient,
                            float(self._norm_type),
                        )
        elif self._clipping == GradientClipping.VALUE:
            torch.nn.utils.clip_grad_value_(self._params, self._max_gradient)

        super().step(closure)


def _dist_clip_grad_norm(
    sharded_grads: List[torch.Tensor],
    replicated_grads: List[torch.Tensor],
    process_group: dist.ProcessGroup,
    max_norm: float,
    norm_type: float = 2.0,
) -> torch.Tensor:
    sharded_grads_bases = _dedup_to_base_tensors(sharded_grads)
    if len(sharded_grads_bases) > 0:
        sharded_grads = sharded_grads_bases
    sharded_norms = torch._foreach_norm(sharded_grads, norm_type)
    local_norm = torch.linalg.vector_norm(torch.stack(sharded_norms), norm_type)
    if replicated_grads:
        replicated_norms = torch._foreach_norm(replicated_grads, norm_type)
        replicated_norm = torch.linalg.vector_norm(
            torch.stack(replicated_norms), norm_type
        )
    else:
        replicated_norm = None

    if norm_type == torch.inf:
        total_norm = local_norm
        dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=process_group)
        if replicated_norm is not None:
            total_norm = torch.maximum(total_norm, replicated_norm)
    else:
        total_norm = local_norm**norm_type
        dist.all_reduce(total_norm, group=process_group)
        if replicated_norm is not None:
            total_norm += replicated_norm**norm_type
        total_norm = total_norm ** (1.0 / norm_type)

    clip_coef = cast(torch.Tensor, max_norm / (total_norm + 1e-6))
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    torch._foreach_mul_(sharded_grads + replicated_grads, clip_coef_clamped)
    return total_norm


def _dedup_to_base_tensors(tensors: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    This is a performance optimization specific to FSDP2. Each gradient tensor
    of the same FSDP module share the same base tensor, so for the total norm
    computation, we can directly use the base tensor to reduce the number of
    tensors to compute norm over.
    """
    seen_base_tensors = set()
    base_tensors = []
    for tensor in tensors:
        base_tensor = tensor._base if tensor._base is not None else tensor
        if base_tensor not in seen_base_tensors:
            seen_base_tensors.add(base_tensor)
            base_tensors.append(base_tensor)
    return base_tensors
