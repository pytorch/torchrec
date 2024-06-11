#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

#!/usr/bin/env python3

from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Type, Union

import torch

from torch import nn
from torch.export import Dim, ExportedProgram, ShapesCollection
from torch.export.dynamic_shapes import _Dim as DIM
from torchrec import KeyedJaggedTensor
from torchrec.ir.types import SerializerInterface


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
    preserve_fqns = []
    for fqn, module in model.named_modules():
        if type(module).__name__ in serializer_cls.module_to_serializer_cls:
            serialized_module = serializer_cls.serialize(module)
            module.register_buffer("ir_metadata", serialized_module, persistent=False)
            preserve_fqns.append(fqn)

    return model, preserve_fqns


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
    module_type_dict = {}
    for node in ep.graph.nodes:
        if "nn_module_stack" in node.meta:
            for fqn, type_name in node.meta["nn_module_stack"].values():
                # Only get the module type name, not the full type name
                module_type_dict[fqn] = type_name.split(".")[-1]

    fqn_to_new_module = {}
    for fqn, module in model.named_modules():
        if "ir_metadata" in dict(module.named_buffers()):
            serialized_module = dict(module.named_buffers())["ir_metadata"]

            if fqn not in module_type_dict:
                raise RuntimeError(
                    f"Cannot find the type of module {fqn} in the exported program"
                )

            deserialized_module = serializer_cls.deserialize(
                serialized_module,
                module_type_dict[fqn],
                device,
            )
            fqn_to_new_module[fqn] = deserialized_module

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
