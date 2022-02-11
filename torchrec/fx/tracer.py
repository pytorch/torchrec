#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Optional, Dict, Union, Callable

import torch
from torch.fx.node import Argument
from torchrec.distributed.types import NoWait


class Tracer(torch.fx.Tracer):
    """
    Custom FX tracer for torchrec

    See `Torch.FX documentation <https://pytorch.org/docs/stable/fx.html>`_

    We create a custom FX tracer to trace torchrec based models. The custom tracer
    handles python generic types (i.e. NoWait[T], Awaitable[T]) and lower it to
    TorchScript if needed
    """

    def __init__(self) -> None:
        super().__init__()

    # pyre-ignore[2]
    def create_arg(self, a: Any) -> Argument:
        """
        A method to specify the behavior of tracing when preparing values to
        be used as arguments to nodes in the ``Graph``.

        Adds support for the NoWait type in addition to the default tracer

        Args:
            a (Any): The value to be emitted as an ``Argument`` in the ``Graph``.

        Returns:
            Argument: The value ``a`` converted into the appropriate ``Argument``
        """
        if isinstance(a, NoWait):
            return self.create_node(
                "call_function",
                target=NoWait,
                args=self.create_arg((a._obj,)),
                kwargs={},
                type_expr=NoWait,
            )
        return super().create_arg(a)


def symbolic_trace(
    # pyre-ignore[24]
    root: Union[torch.nn.Module, Callable],
    concrete_args: Optional[Dict[str, Any]] = None,
) -> torch.fx.GraphModule:
    """
    Symbolic tracing API

    Given an ``nn.Module`` or function instance ``root``, this function will return a ``GraphModule``
    constructed by recording operations seen while tracing through ``root``.

    ``concrete_args`` allows you to partially specialize your function, whether it's to remove control flow or data structures.

    Args:
        root (Union[torch.nn.Module, Callable]): Module or function to be traced and converted
            into a Graph representation.
        concrete_args (Optional[Dict[str, any]]): Inputs to be partially specialized

    Returns:
        GraphModule: a Module created from the recorded operations from ``root``.
    """

    tracer = Tracer()
    graph = tracer.trace(root, concrete_args)
    return torch.fx.GraphModule(root, graph)
