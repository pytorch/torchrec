#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Callable, Iterable, Tuple, Union

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


def check_module_output_dimension(
    module: Union[Iterable[torch.nn.Module], torch.nn.Module],
    in_features: int,
    out_features: int,
) -> bool:
    """
    Verify that the out_features of a given module or a list of modules matches the
    specified number. If a list of modules or a ModuleList is given, recursively check
    all the submodules.
    """
    if isinstance(module, list) or isinstance(module, torch.nn.ModuleList):
        return all(
            check_module_output_dimension(submodule, in_features, out_features)
            for submodule in module
        )
    else:
        # pyre-fixme[6]: Expected `Union[typing.Callable[[torch.Tensor],
        #  torch.Tensor], torch.nn.Module]` for 1st param but got
        #  `Union[Iterable[torch.nn.Module], torch.nn.Module]`.
        return get_module_output_dimension(module, in_features) == out_features


def init_mlp_weights_xavier_uniform(m: torch.nn.Module) -> None:
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)


def construct_modulelist_from_single_module(
    module: torch.nn.Module, sizes: Tuple[int, ...]
) -> torch.nn.Module:
    """
    Given a single module, construct a (nested) ModuleList of size of sizes by making
    copies of the provided module and reinitializing the Linear layers.
    """
    if len(sizes) == 1:
        return torch.nn.ModuleList(
            [
                copy.deepcopy(module).apply(init_mlp_weights_xavier_uniform)
                for _ in range(sizes[0])
            ]
        )
    else:
        # recursively create nested ModuleList
        return torch.nn.ModuleList(
            [
                construct_modulelist_from_single_module(module, sizes[1:])
                for _ in range(sizes[0])
            ]
        )


def convert_list_of_modules_to_modulelist(
    modules: Iterable[torch.nn.Module], sizes: Tuple[int, ...]
) -> torch.nn.Module:
    assert (
        # pyre-fixme[6]: Expected `Sized` for 1st param but got
        #  `Iterable[torch.nn.Module]`.
        len(modules)
        == sizes[0]
    ), f"the counts of modules ({len(modules)}) do not match with the required counts {sizes}"
    if len(sizes) == 1:
        return torch.nn.ModuleList(modules)
    else:
        # recursively create nested list
        return torch.nn.ModuleList(
            # pyre-fixme[6]: Expected `Iterable[torch.nn.Module]` for 1st param but
            #  got `Module`.
            convert_list_of_modules_to_modulelist(m, sizes[1:])
            for m in modules
        )
