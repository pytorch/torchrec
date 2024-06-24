#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

#!/usr/bin/env python3

from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Tuple, Type, Union

import torch

from torch import nn
from torch.export import Dim, ExportedProgram, ShapesCollection
from torch.export.dynamic_shapes import _Dim as DIM
from torchrec.ir.types import SerializerInterface
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


# TODO: Replace the default interface with the python dataclass interface
DEFAULT_SERIALIZER_CLS = SerializerInterface
DYNAMIC_DIMS: Dict[str, int] = defaultdict(int)


def serialize_embedding_modules(
    model: nn.Module,
    serializer_cls: Type[SerializerInterface] = DEFAULT_SERIALIZER_CLS,
) -> Tuple[nn.Module, List[str]]:
    """
    Takes all the modules that are of type `serializer_cls` and serializes them
    in the given format with a registered buffer to the module.

    Returns the modified module and the list of fqns that had the buffer added.
    """
    # store the fqns of the modules that is serialized, so that the ir_export can preserve
    # the fqns of those modules
    preserve_fqns = []

    # store the fqn of the module that does not require serializing its children,
    # so that we can skip based on the children's fqn
    skip_children: Optional[str] = None
    for fqn, module in model.named_modules():
        if skip_children is not None:
            if fqn.startswith(skip_children):
                # {fqn} is skipped for further serialization
                continue
            else:
                # reset the skip_children to None because fqn is no longer a child
                skip_children = None
        if type(module).__name__ in serializer_cls.module_to_serializer_cls:
            serialized_module = serializer_cls.serialize(module)
            # store the fqn of a serialized module that doesn't not require further serialization of its children
            if not serializer_cls.requires_children(type(module).__name__):
                skip_children = fqn
            module.register_buffer("ir_metadata", serialized_module, persistent=False)
            preserve_fqns.append(fqn)

    return model, preserve_fqns


def _next_or_none(
    generator: Iterator[Tuple[str, nn.Module]]
) -> Tuple[Optional[str], Optional[nn.Module]]:
    try:
        return next(generator)
    except StopIteration:
        return None, None


def _deserialize_node(
    parent_fqn: str,
    fqn: Optional[str],
    module: Optional[nn.Module],
    generator: Iterator[Tuple[str, nn.Module]],
    serializer_cls: Type[SerializerInterface],
    module_type_dict: Dict[str, str],
    fqn_to_new_module: Dict[str, nn.Module],
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, nn.Module], Optional[str], Optional[nn.Module]]:
    """
    returns:
    1. the children of the parent_fqn Dict[relative_fqn -> module]
    2. the next node Optional[fqn], Optional[module], which is not a child of the parent_fqn
    """
    children: Dict[str, nn.Module] = {}
    # we only starts the while loop when the current fqn is a child of the parent_fqn
    # it stops at either the current node is not a child of the parent_fqn or
    # the generator is exhausted
    while fqn is not None and module is not None and fqn.startswith(parent_fqn):
        # the current node is a serialized module, need to deserialize it
        if "ir_metadata" in dict(module.named_buffers()):
            serialized_module = module.get_buffer("ir_metadata")
            if fqn not in module_type_dict:
                raise RuntimeError(
                    f"Cannot find the type of module {fqn} in the exported program"
                )

            # current module's deserialization requires its children
            if serializer_cls.requires_children(module_type_dict[fqn]):
                # set current fqn as the new parent_fqn, and call deserialize_node function
                # recursively to get the children of current_fqn, and the next sibling of current_fqn
                next_fqn, next_module = _next_or_none(generator)
                grand_children, next_fqn, next_module = _deserialize_node(
                    fqn,
                    next_fqn,
                    next_module,
                    generator,
                    serializer_cls,
                    module_type_dict,
                    fqn_to_new_module,
                    device,
                )
                # deserialize the current module with its children
                deserialized_module = serializer_cls.deserialize(
                    serialized_module,
                    module_type_dict[fqn],
                    device=device,
                    children=grand_children,
                )
            else:
                # current module's deserialization doesn't require its children
                # deserialize it first then get the next sibling
                deserialized_module = serializer_cls.deserialize(
                    serialized_module, module_type_dict[fqn], device=device
                )
                next_fqn, next_module = _next_or_none(generator)

            # register the deserialized module
            rel_fqn = fqn[len(parent_fqn) + 1 :] if len(parent_fqn) > 0 else fqn
            children[rel_fqn] = deserialized_module
            fqn_to_new_module[fqn] = deserialized_module

        else:  # current node doesn't require deserialization, move on
            next_fqn, next_module = _next_or_none(generator)
        # move to the next node
        fqn, module = next_fqn, next_module
    return children, fqn, module


def deserialize_embedding_modules(
    ep: ExportedProgram,
    serializer_cls: Type[SerializerInterface] = DEFAULT_SERIALIZER_CLS,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Takes ExportedProgram (IR) and looks for ir_metadata buffer.
    If found, deserializes the buffer and replaces the module with the deserialized
    module.

    Returns the unflattened ExportedProgram with the deserialized modules.
    """
    model = torch.export.unflatten(ep)
    module_type_dict: Dict[str, str] = {}
    for node in ep.graph.nodes:
        if "nn_module_stack" in node.meta:
            for fqn, type_name in node.meta["nn_module_stack"].values():
                # Only get the module type name, not the full type name
                module_type_dict[fqn] = type_name.split(".")[-1]

    fqn_to_new_module: Dict[str, nn.Module] = {}
    generator = model.named_modules()
    fqn, root = _next_or_none(generator)
    _deserialize_node(
        "",
        fqn,
        root,
        generator,
        serializer_cls,
        module_type_dict,
        fqn_to_new_module,
        device,
    )

    for fqn, new_module in fqn_to_new_module.items():
        # handle nested attribute like "x.y.z"
        attrs = fqn.split(".")
        parent = model
        for a in attrs[:-1]:
            parent = getattr(parent, a)
        setattr(parent, attrs[-1], new_module)

    return model


def _get_dim(x: Union[DIM, str, None], s: str, max: Optional[int] = None) -> DIM:
    if isinstance(x, DIM):
        return x
    elif isinstance(x, str):
        if x in DYNAMIC_DIMS:
            DYNAMIC_DIMS[x] += 1
            x += str(DYNAMIC_DIMS[x])
        dim = Dim(x, max=max)
    else:
        DYNAMIC_DIMS[s] += 1
        dim = Dim(s + str(DYNAMIC_DIMS[s]), max=max)
    return dim


def mark_dynamic_kjt(
    kjt: KeyedJaggedTensor,
    shapes_collection: Optional[ShapesCollection] = None,
    variable_length: bool = False,
    vlen: Optional[Union[DIM, str]] = None,
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
        batch_size = _get_dim(batch_size, "batch_size", max=4294967295)
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
