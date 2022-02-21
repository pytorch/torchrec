#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
from typing import Optional, List, Set, Union

import torch
from torchrec.distributed.types import ShardedModule


def append_prefix(prefix: str, name: str) -> str:
    """
    Appends provided prefix to provided name.
    """

    if prefix != "" and name != "":
        return prefix + "." + name
    else:
        return prefix + name


def filter_state_dict(
    state_dict: "OrderedDict[str, torch.Tensor]", name: str
) -> "OrderedDict[str, torch.Tensor]":
    """
    Filters state dict for keys that start with provided name.
    Strips provided name from beginning of key in the resulting state dict.

    Args:
        state_dict (OrderedDict[str, torch.Tensor]): input state dict to filter.
        name (str): name to filter from state dict keys.

    Returns:
        OrderedDict[str, torch.Tensor]: filtered state dict.
    """

    filtered_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith(name):
            # + 1 to length is to remove the '.' after the key
            filtered_state_dict[key[len(name) + 1 :]] = value
    return filtered_state_dict


def _get_unsharded_module_names_helper(
    model: torch.nn.Module,
    path: str,
    unsharded_module_names: Set[str],
) -> bool:
    sharded_children = set()
    for name, child in model.named_children():
        curr_path = path + name
        if isinstance(child, ShardedModule):
            sharded_children.add(name)
        else:
            child_sharded = _get_unsharded_module_names_helper(
                child,
                curr_path + ".",
                unsharded_module_names,
            )
            if child_sharded:
                sharded_children.add(name)

    if len(sharded_children) > 0:
        for name, _ in model.named_children():
            if name not in sharded_children:
                unsharded_module_names.add(path + name)

    return len(sharded_children) > 0


def get_unsharded_module_names(model: torch.nn.Module) -> List[str]:
    """
    Retrieves names of top level modules that do not contain any sharded sub-modules.

    Args:
        model (torch.nn.Module): model to retrieve unsharded module names from.

    Returns:
        List[str]: list of names of modules that don't have sharded sub-modules.
    """

    unsharded_module_names: Set[str] = set()
    _get_unsharded_module_names_helper(
        model,
        "",
        unsharded_module_names,
    )
    return list(unsharded_module_names)


class sharded_model_copy:
    """
    Allows copying of DistributedModelParallel module to a target device.

    Example:
        >>> # Copying model to CPU.

        >>> m = DistributedModelParallel(m)
        >>> with sharded_model_copy("cpu"):
                m_cpu = copy.deepcopy(m)
    """

    def __init__(self, device: Optional[Union[str, int, torch.device]]) -> None:
        self.device = device

    def __enter__(self) -> None:
        # pyre-ignore [16]
        self.t_copy_save_ = torch.Tensor.__deepcopy__
        # pyre-ignore [16]
        self.p_copy_save_ = torch.nn.Parameter.__deepcopy__

        device = self.device

        # pyre-ignore [2, 3, 53]
        def _tensor_copy(tensor, memo):
            if tensor.device != device:
                return tensor.detach().to(device)
            else:
                return tensor.detach().clone()

        # pyre-ignore [2, 3]
        def _no_copy(obj, memo):
            return obj

        _copy_or_not = _tensor_copy if self.device is not None else _no_copy

        # pyre-ignore [2, 3, 53]
        def _param_copy(param, memo):
            return torch.nn.Parameter(_copy_or_not(param, memo))

        # pyre-ignore [16]
        torch.Tensor.__deepcopy__ = _copy_or_not
        torch.nn.Parameter.__deepcopy__ = _param_copy
        torch._C._distributed_c10d.ProcessGroupNCCL.__deepcopy__ = _no_copy
        torch._C._distributed_c10d.ProcessGroupGloo.__deepcopy__ = _no_copy
        torch._C._distributed_c10d.Work.__deepcopy__ = _no_copy
        # pyre-ignore [16]
        torch.cuda.streams.Stream.__deepcopy__ = _no_copy

    # pyre-ignore [2]
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # pyre-ignore [16]
        torch.Tensor.__deepcopy__ = self.t_copy_save_
        # pyre-ignore [16]
        torch.nn.Parameter.__deepcopy__ = self.p_copy_save_
        torch._C._distributed_c10d.ProcessGroupNCCL.__deepcopy__ = None
        torch._C._distributed_c10d.ProcessGroupGloo.__deepcopy__ = None
        torch._C._distributed_c10d.Work.__deepcopy__ = None
        # pyre-ignore [16]
        torch.cuda.streams.Stream.__deepcopy__ = None
