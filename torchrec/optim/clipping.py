#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import logging
from collections import defaultdict
from enum import Enum, unique
from typing import Any, cast, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch.distributed._tensor.api import DTensor

from torchrec.optim.keyed import KeyedOptimizer, OptimizerWrapper

logger: logging.Logger = logging.getLogger()

log_grad_norm: bool = False
use_64bit_grad_norm: bool = False


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
        enable_global_grad_clip (bool): whether to enable global gradient clipping.
        param_to_pgs (Dict[torch.nn.Parameter, List[dist.ProcessGroup]], optional): Mapping of parameters
            to process groups. Used for global gradient clipping in n-D model parallelism case.
            Defaults to None, local gradient clipping is used.
    """

    def __init__(
        self,
        optimizer: KeyedOptimizer,
        clipping: GradientClipping = GradientClipping.NONE,
        max_gradient: float = 0.1,
        norm_type: Union[float, str] = 2.0,
        enable_global_grad_clip: bool = False,
        param_to_pgs: Optional[
            Dict[torch.nn.Parameter, List[dist.ProcessGroup]]
        ] = None,
    ) -> None:
        super().__init__(optimizer)
        self._clipping = clipping
        self._max_gradient = max_gradient
        self._norm_type = norm_type
        self._check_meta: bool = True
        self._enable_global_grad_clip = enable_global_grad_clip
        self._step_num = 0

        # Group parameters by model parallelism process group if global clipping is enabled.
        # Otherwise, all parameters are treated as replicated and will be clipped locally.
        sharded_param_cnt = 0
        self._replicate_params: List[torch.Tensor] = []
        self._sharded_params: Dict[Tuple[dist.ProcessGroup], List[torch.Tensor]] = (
            defaultdict(list)
        )

        for param_group in self.param_groups:
            for param in param_group["params"]:
                if not self._enable_global_grad_clip:
                    self._replicate_params.append(param)
                    continue
                if param_to_pgs is None or len(param_to_pgs) == 0:
                    self._replicate_params.append(param)
                    continue

                # Group parameters by model parallelism process group.
                if param in param_to_pgs and len(param_to_pgs[param]) != 0:
                    self._sharded_params[tuple(param_to_pgs[param])].append(param)
                    sharded_param_cnt += 1
                else:
                    self._replicate_params.append(param)
        logger.info(
            f"Optimizer found {sharded_param_cnt} dist params and {len(self._replicate_params)} replicate params."
        )

        # Sanity check: this path is currently not used in any production.
        if self._clipping == GradientClipping.VALUE:
            if sharded_param_cnt > 0:
                raise NotImplementedError(
                    "clip_grad_value_ for sharded parameters is not supported yet"
                )

    # pyre-ignore [2]
    def step(self, closure: Any = None) -> None:
        if self._check_meta:
            # skip gradient clipping and early return
            if any(t.device.type == "meta" for t in self._replicate_params):
                super().step(closure)
                return
            if any(
                t.device.type == "meta"
                for params in self._sharded_params.values()
                for t in params
            ):
                super().step(closure)
                return
            self._check_meta = False

        if self._clipping == GradientClipping.NORM:
            # No sharded parameters, local gradient clipping == global gradient clipping
            if len(self._sharded_params) == 0:
                replicate_params = [
                    p._local_tensor if isinstance(p, DTensor) else p
                    for p in self._replicate_params
                ]
                torch.nn.utils.clip_grad_norm_(
                    replicate_params,
                    self._max_gradient,
                    norm_type=self._norm_type,
                )
            else:
                self.clip_grad_norm_()

        elif self._clipping == GradientClipping.VALUE:
            torch.nn.utils.clip_grad_value_(self._replicate_params, self._max_gradient)

        super().step(closure)
        self._step_num += 1

    @torch.no_grad()
    def clip_grad_norm_(self) -> Optional[Union[float, torch.Tensor]]:
        """Clip the gradient norm of all parameters."""
        max_norm = self._max_gradient
        norm_type = float(self._norm_type)
        all_grads = []
        total_grad_norm = None

        # Process distributed parameters and gradients
        for pgs, dist_params in self._sharded_params.items():
            sharded_grads = [
                p.grad._local_tensor if isinstance(p.grad, DTensor) else p.grad
                for p in dist_params
                if p.grad is not None and p.grad.numel() > 0
            ]
            if len(sharded_grads) == 0:
                continue
            all_grads.extend(sharded_grads)

            sharded_grad_norm = _batch_cal_norm(
                sharded_grads,
                max_norm,
                norm_type,
                pgs,
            )
            total_grad_norm = (
                sharded_grad_norm
                if total_grad_norm is None
                else (
                    torch.maximum(total_grad_norm, sharded_grad_norm)
                    if self._norm_type == torch.inf
                    else total_grad_norm + sharded_grad_norm
                )
            )

        square_sharded_grad_norm = total_grad_norm if total_grad_norm is not None else 0

        # Process replicated parameters and gradients
        if self._replicate_params:
            replicated_grads = [
                p.grad._local_tensor if isinstance(p.grad, DTensor) else p.grad
                for p in self._replicate_params
                if p.grad is not None and p.grad.numel() > 0
            ]
            all_grads.extend(replicated_grads)

            replicated_grad_norm = _batch_cal_norm(
                replicated_grads,
                max_norm,
                norm_type,
                None,
            )
            total_grad_norm = (
                replicated_grad_norm
                if total_grad_norm is None
                else (
                    torch.maximum(total_grad_norm, replicated_grad_norm)
                    if self._norm_type == torch.inf
                    else total_grad_norm + replicated_grad_norm
                )
            )
            square_replicated_grad_norm = replicated_grad_norm
        else:
            square_replicated_grad_norm = 0

        global log_grad_norm
        if log_grad_norm:
            if total_grad_norm is not None and self._norm_type != torch.inf:
                # pyre-ignore[58]
                grad_norm = total_grad_norm ** (1.0 / norm_type)
            else:
                grad_norm = 0

            rank = dist.get_rank()
            logger.info(
                f"Clipping [rank={rank}, step={self._step_num}]: square_sharded_grad_norm = {square_sharded_grad_norm}, square_replicated_grad_norm = {square_replicated_grad_norm}, total_grad_norm = {grad_norm}"
            )

        # Aggregation
        if total_grad_norm is None:
            return

        if self._norm_type != torch.inf:
            # pyre-ignore [58]: ** is not supported for operand types torch._tensor.Tensor and float.
            total_grad_norm = total_grad_norm ** (1.0 / norm_type)
        # pyre-ignore [58]: / is not supported for operand types float and Union[float, torch._tensor.Tensor].
        clip_coef = cast(torch.Tensor, max_norm / (total_grad_norm + 1e-6))
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        torch._foreach_mul_(all_grads, clip_coef_clamped)
        return total_grad_norm


def _batch_cal_norm(
    grad_list: List[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    process_groups: Optional[Tuple[dist.ProcessGroup]] = None,
) -> torch.Tensor:
    """Helper function that calculates the norm of a list of gradients in batches. If process_groups
    are passed in, the norm will be aggregated across all ranks in the process group.
    """

    global use_64bit_grad_norm
    if use_64bit_grad_norm:
        grad_norms = torch.linalg.vector_norm(
            torch.stack(torch._foreach_norm(grad_list, norm_type, dtype=torch.float64)),
            norm_type,
        )
    else:
        grad_norms = torch.linalg.vector_norm(
            torch.stack(torch._foreach_norm(grad_list, norm_type)),
            norm_type,
        )

    if norm_type == torch.inf:
        if process_groups is not None:
            for pg in process_groups:
                dist.all_reduce(grad_norms, op=dist.ReduceOp.MAX, group=pg)
    else:
        grad_norms = grad_norms**norm_type
        if process_groups is not None:
            for pg in process_groups:
                dist.all_reduce(grad_norms, group=pg)

    if use_64bit_grad_norm:
        grad_norms = grad_norms.to(torch.float32)

    return grad_norms


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
