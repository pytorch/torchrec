#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

#!/usr/bin/env python3

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Type, Union

import torch

from torch import nn
from torch.export import Dim, ShapesCollection
from torch.export.dynamic_shapes import _Dim as DIM
from torchrec.ir.types import SerializerInterface
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


# TODO: Replace the default interface with the python dataclass interface
DEFAULT_SERIALIZER_CLS = SerializerInterface
DYNAMIC_DIMS: Dict[str, int] = defaultdict(int)
logger: logging.Logger = logging.getLogger(__name__)


@torch.library.custom_op("torchrec::ir_custom_op", mutates_args={})
def ir_custom_op_impl(
    tensors: List[Optional[torch.Tensor]], batch_size: int, dim: int
) -> torch.Tensor:
    device = None
    for t in tensors:
        if t is not None:
            device = t.device
            break
    logger.info(f"torch.ops.torchrec.ir_custom_op -> ({batch_size}, {dim}) {device}")
    return torch.empty(batch_size, dim, device=device)


@torch.library.register_fake("torchrec::ir_custom_op")
def ir_custom_op_fake(
    tensors: List[Optional[torch.Tensor]], batch_size: int, dim: int
) -> torch.Tensor:
    device = None
    for t in tensors:
        if t is not None:
            device = t.device
            break
    logger.info(f"ir_custom_op_fake -> ({batch_size}, {dim}) {device}")
    return torch.empty(batch_size, dim, device=device)


def encapsulate_ir_modules(
    module: nn.Module,
    serializer: Type[SerializerInterface] = DEFAULT_SERIALIZER_CLS,
    fqn: str = "",
) -> Tuple[nn.Module, List[str]]:
    """
    Takes a module and encapsulate its embedding modules and serializes them to the module buffer.
    Returns the modified module and a list of fqns that had the buffer added, which is needed for torch.export
    The encapsulation is done by using meta_forward function provided by the serializer
    to replace the module's original forward function.
    """
    preserve_fqns: List[str] = []  # fqns of the serialized modules
    children: List[str] = []  # fqns of the children that need further serialization
    # handle current module, and find the children which need further serialization
    if type(module).__name__ in serializer.module_to_serializer_cls:
        children = serializer.encapsulate_module(module)
        preserve_fqns.append(fqn)
    else:
        # if the module is not of type serializer, then we check all its children
        children = [child for child, _ in module.named_children()]

    # handle child modules recursively
    for child in children:
        submodule = module.get_submodule(child)
        child_fqn = f"{fqn}.{child}" if len(fqn) > 0 else child
        _, fqns = encapsulate_ir_modules(submodule, serializer, child_fqn)
        preserve_fqns.extend(fqns)
    return module, preserve_fqns


def decapsulate_ir_modules(
    module: nn.Module,
    serializer: Type[SerializerInterface] = DEFAULT_SERIALIZER_CLS,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Takes a module and decapsulate its embedding modules by retrieving the buffer.
    Returns the module with restored embedding (sub) modules.
    """
    for child_fqn, child in module.named_children():
        # perform deserialization on the children first, so that we can replace the child module with
        # the deserialized module, and then replace it in the parent
        child = decapsulate_ir_modules(
            module=child, serializer=serializer, device=device
        )
        # replace the child module with deserialized one if applicable
        setattr(module, child_fqn, child)

    # only deserialize if the module has ir_metadata buffer, otherwise return as is
    # we use "ir_metadata" as a convention to identify the deserializable module
    if "ir_metadata" in dict(module.named_buffers()):
        module = serializer.decapsulate_module(module, device)
    return module


def _get_dim(
    x: Union[DIM, str, None],
    s: str,
    min: Optional[int] = None,
    max: Optional[int] = None,
) -> DIM:
    if isinstance(x, DIM):
        return x
    elif isinstance(x, str):
        if x in DYNAMIC_DIMS:
            DYNAMIC_DIMS[x] += 1
            x += str(DYNAMIC_DIMS[x])
        dim = Dim(x, min=min, max=max)
    else:
        DYNAMIC_DIMS[s] += 1
        dim = Dim(s + str(DYNAMIC_DIMS[s]), min=min, max=max)
    return dim


def mark_dynamic_kjt(
    kjt: KeyedJaggedTensor,
    shapes_collection: Optional[ShapesCollection] = None,
    variable_length: bool = False,
    vlen: Optional[Union[DIM, str]] = None,
    llen: Optional[Union[DIM, str]] = None,
    batch_size: Optional[Union[DIM, str]] = None,
) -> ShapesCollection:
    """
    Makes the given KJT dynamic. If it's not variable length, it will only have
    one dynamic dimension, which is the length of the values (and weights).
    If it is variable length, then the lengths and offsets will be dynamic.

    If a shapes collection is provided, it will be updated with the new shapes,
    otherwise a new shapes collection will be created. A passed-in shapes_collection is
    useful if you have multiple KJTs or other dynamic shapes that you want to trace.

    If a dynamic dim/name is provided, it will directly use that dim/name. Otherwise,
    it will use the default name "vlen" for values, and "llen", "lofs" if variable length.
    A passed-in dynamic dim is useful if the dynamic dim is already used in other places.

    Args:
        kjt (KeyedJaggedTensor): The KJT to make dynamic.
        shapes_collection (Optional[ShapesCollection]): The collection to update.
        variable_length (bool): Whether the KJT is variable length.
        vlen (Optional[Union[DIM, str]]): The dynamic length for the values.
        batch_size (Optional[Union[DIM, str]]): The dynamic length for the batch_size.
    """
    global DYNAMIC_DIMS
    if shapes_collection is None:
        shapes_collection = ShapesCollection()
    vlen = _get_dim(vlen, "vlen")
    shapes_collection[kjt._values] = (vlen,)
    if kjt._weights is not None:
        shapes_collection[kjt._weights] = (vlen,)
    if variable_length:
        keys = len(kjt.keys())
        if llen is None or batch_size is None:
            llen = _get_dim(None, "llen", min=keys * 2, max=4294967295)
        elif batch_size is not None:
            batch_size = _get_dim(
                batch_size, "batch_size", max=4294967295 // (keys + 1)
            )
            llen = len(kjt.keys()) * batch_size
        olen = llen + 1
        if kjt._lengths is not None:
            shapes_collection[kjt._lengths] = (llen,)
        if kjt._offsets is not None:
            shapes_collection[kjt._offsets] = (olen,)
    return shapes_collection


def move_to_copy_nodes_to_device(
    unflattened_module: nn.Module,
    device: torch.device,
) -> nn.Module:
    """
    Moves all the copy nodes to the given device.
    """
    for nodes in unflattened_module.graph.nodes:
        if "_to_copy" in nodes.name:
            new_kwargs = {}
            for k, v in nodes.kwargs.items():
                if isinstance(v, torch.device):
                    v = device
                new_kwargs[k] = v
            nodes.kwargs = new_kwargs

    return unflattened_module
