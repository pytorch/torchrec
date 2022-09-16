#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Iterable, Type

import torch
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)


def apply_overlapped_optimizer(
    optimizer_class: Type[torch.optim.Optimizer],
    params: Iterable[torch.nn.Parameter],
    optimizer_kwargs: Dict[str, Any],
) -> None:
    """
    Upon backwards(), parameters will fire the corresponding optimizer
    Each parameter will have the optimizer_class and optimizer_kwargs attached to
    _optimizer and _optimizer_kwargs.

    Note - gradients for these parameters will be set to None after backwards().
    This means that any other (non applied) optimizer over this parameter will be
    a no-op.

    Args:
        optimizer_class: Type[torch.optim.Optimizer]: Optimizer to apply to parameter
        params: Iterator[nn.Parameter]: parameters to apply optimizer state to
        optimizer_kwargs: Dict[str, Any]: kwargs to pass to optimizer constructor

    Example::
        params_generator = model.parameters()
        param_1 = next(params_generator)
        param_2 = list(params_generator)

        apply_overlapped_optimizer(torch.optim.SGD, [param_1], {"lr": .02})
        apply_overlapped_optimizer(torch.optim.Adam, param_2, {"lr": .04})

        print(param_1._optimizer, param_1._optimizer_kwargs)
        >> torch.optim.SGD, {"lr": .02}
    """

    def _apply_overlapped_optimizer_to_param(param: torch.nn.Parameter) -> None:
        # acc_grad creates a new node in the auto_grad graph that comes after
        # the parameter node. In this node's backwards we call the overlapped
        # optimizer.step(). We cannot do the backwards hook on param because
        # the gradients are not fully ready by then.
        # for more details see https://github.com/pytorch/pytorch/issues/76464
        acc_grad = param.view_as(param).grad_fn.next_functions[0][0]
        optimizer = optimizer_class([param], **optimizer_kwargs)

        if hasattr(param, "_acc_grad") and hasattr(param, "_overlapped_optimizer"):
            raise ValueError(
                # pyre-ignore
                f"{param} already has {param._overlapped_optimizer} applied as an overlapped optimizer. Cannot apply again"
            )

        # The grad accumulator is a weak ref, so we need to keep it
        # alive until the Tensor is alive.
        # Store it on the module to avoid uncollectable ref-cycle
        # pyre-ignore
        param._acc_grad = acc_grad
        param._overlapped_optimizer = optimizer

        # pyre-ignore
        param._optimizer_class = optimizer_class
        # pyre-ignore
        param._optimizer_kwargs = optimizer_kwargs

        # pyre-ignore
        def optimizer_hook(*_unused) -> None:
            # pyre-ignore
            param._overlapped_optimizer.step()
            param.grad = None

        param._acc_grad.register_hook(optimizer_hook)

    for param in params:
        _apply_overlapped_optimizer_to_param(param)


def apply_overlapped_optimizer_to_module(
    optimizer_class: Type[torch.optim.Optimizer],
    module: torch.nn.Module,
    optimizer_kwargs: Dict[str, Any],
) -> None:
    """
    Recursively apply overlapped optimizer to all EmbeddingBagCollection and EmbeddingCollection in the module.
    """
    if isinstance(module, EmbeddingBagCollection):
        for emb_bag_module in module.embedding_bags.values():
            apply_overlapped_optimizer(
                optimizer_class, emb_bag_module.parameters(), optimizer_kwargs
            )
        return
    elif isinstance(module, EmbeddingCollection):
        for emb_module in module.embeddings.values():
            apply_overlapped_optimizer(
                optimizer_class, emb_module.parameters(), optimizer_kwargs
            )
        return

    def apply_children() -> None:
        for _, child_module in module.named_children():
            if isinstance(child_module, EmbeddingBagCollection):
                for emb_bag_module in child_module.embedding_bags.values():
                    apply_overlapped_optimizer(
                        optimizer_class, emb_bag_module.parameters(), optimizer_kwargs
                    )
            elif isinstance(child_module, EmbeddingCollection):
                for emb_module in child_module.embeddings.values():
                    apply_overlapped_optimizer(
                        optimizer_class, emb_module.parameters(), optimizer_kwargs
                    )
            else:
                apply_overlapped_optimizer_to_module(
                    optimizer_class, child_module, optimizer_kwargs
                )

    apply_children()
