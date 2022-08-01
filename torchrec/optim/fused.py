#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
from typing import Any, List, Tuple

from torch import nn, optim
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizer


class FusedOptimizer(KeyedOptimizer, abc.ABC):
    """
    Assumes that weight update is done during backward pass,
    thus step() does not update weights.
    However, it may be used to update learning rates internally.
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


def _get_fused_optimizer_recurse(
    module: nn.Module, path: str, accum: List[Tuple[str, KeyedOptimizer]]
) -> None:
    if isinstance(module, FusedOptimizerModule):
        accum.append((path, module.fused_optimizer()))
    else:
        for name, child in module.named_children():
            new_path = path + "." + name if path else name
            _get_fused_optimizer_recurse(child, new_path, accum)


def get_fused_optimizers(module: nn.Module) -> CombinedOptimizer:
    """
    Returns a CombinedOptimizer that contains all of the fused_optimizers of a given module.

    Example::

    model = DLRM(ebc=EmbeddingBagCollection(tables))
    model = fuse_embedding_optimizer(model, optim_type=torch.optim.SGD, optim_kwargs={"lr":.02})
    fused_opt = get_fused_optimizers(model)

    output = model(kjt)
    output.sum().backward()

    print(fused_opt.state_dict())
    ...
    """
    accum = []
    _get_fused_optimizer_recurse(module, "", accum)
    return CombinedOptimizer(optims=accum)
