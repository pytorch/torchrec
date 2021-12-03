#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
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
    NOTE [ Custom FX tracer for torchrec ]

    We create a custom FX tracer to trace torchrec based models. The custom tracer
    right now have several purposes (the list might expand if we have more use cases):
    1. Handling python generic types (i.e. NoWait[T], Awaitable[T]) and lower it to
       TorchScript if needed
    """

    def __init__(self) -> None:
        super().__init__()

    # pyre-ignore[2]
    def create_arg(self, a: Any) -> Argument:
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

    tracer = Tracer()
    graph = tracer.trace(root, concrete_args)
    return torch.fx.GraphModule(root, graph)
