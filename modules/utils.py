#!/usr/bin/env python3

from typing import Callable, Union

import torch


def extract_module_or_tensor_callable(
    module_or_callable: Union[
        Callable[[], torch.nn.Module],
        torch.nn.Module,
        Callable[[torch.Tensor], torch.Tensor],
    ]
) -> Union[torch.nn.Module, Callable[[torch.Tensor], torch.Tensor]]:
    try:
        # pyre-ignore[20]: PositionalOnly call expects argument in position 0
        module = module_or_callable()
        if isinstance(module, torch.nn.Module):
            return module
        else:
            raise ValueError(
                "Expected callable that takes no input to return "
                "a torch.nn.Module, but got: {}".format(type(module))
            )
    except TypeError as e:
        if "required positional argument" in str(e):
            # pyre-ignore[7]: Expected `Union[typing.Callable[[torch.Tensor], torch.Tensor], torch.nn.Module]`
            return module_or_callable
        raise


def get_module_output_dimension(
    module: Union[Callable[[torch.Tensor], torch.Tensor], torch.nn.Module],
    in_features: int,
) -> int:
    input = torch.zeros(1, in_features)
    output = module(input)
    return output.size(-1)
