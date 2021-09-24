#!/usr/bin/env python3

from typing import List, Set

import torch
from torchrec.distributed.types import ShardedModule


def append_prefix(prefix: str, name: str) -> str:
    if prefix != "" and name != "":
        return prefix + "." + name
    else:
        return prefix + name


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
    Returns a list of top level modules do not contain any sharded sub modules.
    """
    unsharded_module_names: Set[str] = set()
    _get_unsharded_module_names_helper(
        model,
        "",
        unsharded_module_names,
    )
    return list(unsharded_module_names)
