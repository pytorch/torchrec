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
from typing import Dict, List, Optional, Tuple, Type

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


def get_device(tensors: List[Optional[torch.Tensor]]) -> Optional[torch.device]:
    """
    Returns the device of the first non-None tensor in the list.
    """
    for t in tensors:
        if t is not None:
            return t.device
    return None


@torch.library.custom_op("torchrec::ir_emb_lookup", mutates_args={})
def ir_emb_lookup_impl(
    tensors: List[Optional[torch.Tensor]], batch_size: int, dims: List[int]
) -> List[torch.Tensor]:
    device = get_device(tensors)
    logger.info(f"torch.ops.torchrec.ir_emb_lookup -> ({batch_size}, {dims}) {device}")
    return [torch.empty(batch_size, dim, device=device) for dim in dims]


@torch.library.register_fake("torchrec::ir_emb_lookup")
def ir_emb_lookup_fake(
    tensors: List[Optional[torch.Tensor]], batch_size: int, dims: List[int]
) -> List[torch.Tensor]:
    device = get_device(tensors)
    logger.info(f"ir_emb_lookup_fake -> ({batch_size}, {dims}) {device}")
    return [torch.empty(batch_size, dim, device=device) for dim in dims]


@torch.library.custom_op("torchrec::ir_kt_regroup", mutates_args={})
def ir_kt_regroup_impl(
    tensors: List[Optional[torch.Tensor]], batch_size: int, dims: List[int]
) -> List[torch.Tensor]:
    device = get_device(tensors)
    logger.info(f"torch.ops.torchrec.ir_kt_regroup -> ({batch_size}, {dims}) {device}")
    return [torch.empty(batch_size, dim, device=device) for dim in dims]


@torch.library.register_fake("torchrec::ir_kt_regroup")
def ir_kt_regroup_fake(
    tensors: List[Optional[torch.Tensor]], batch_size: int, dims: List[int]
) -> List[torch.Tensor]:
    device = get_device(tensors)
    logger.info(f"ir_kt_regroup_fake -> ({batch_size}, {dims}) {device}")
    return [torch.empty(batch_size, dim, device=device) for dim in dims]


@torch.library.custom_op("torchrec::ir_dynamic_batch_emb_lookup", mutates_args={})
def ir_dynamic_batch_emb_lookup_impl(
    tensors: List[Optional[torch.Tensor]], batch_size: int, dims: List[int]
) -> List[torch.Tensor]:
    device = get_device(tensors)
    logger.info(
        f"torch.ops.torchrec.ir_dynamic_batch_emb_lookup -> ({batch_size}, {dims}) {device}"
    )
    return [torch.empty(batch_size, dim, device=device) for dim in dims]


@torch.library.register_fake("torchrec::ir_dynamic_batch_emb_lookup")
def ir_dynamic_batch_emb_lookup_fake(
    tensors: List[Optional[torch.Tensor]], batch_dize: int, dims: List[int]
) -> List[torch.Tensor]:
    device = get_device(tensors)
    batch_size = torch.library.get_ctx().new_dynamic_size()
    logger.info(f"ir_dynamic_batch_emb_lookup_fake -> ({batch_size}, {dims}) {device}")
    return [torch.empty(batch_size, dim, device=device) for dim in dims]


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


def _get_dim(name: str, min: Optional[int] = None, max: Optional[int] = None) -> DIM:
    """
    Returns a Dim object with the given name and min/max. If the name is not unique, it will append a suffix to the name.
    """
    dim = f"{name}_{DYNAMIC_DIMS[name]}"
    DYNAMIC_DIMS[name] += 1
    return Dim(dim, min=min, max=max)


def mark_dynamic_kjt(
    kjt: KeyedJaggedTensor,
    shapes_collection: Optional[ShapesCollection] = None,
    variable_length: bool = False,
    vlen: Optional[DIM] = None,
    llen: Optional[DIM] = None,
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
        vlen (Optional[DIM]): The dynamic length for the values. If it's None, it will use the default name "vlen".
        llen (Optional[DIM]): The dynamic length for the lengths, it's only used when variable_length is true. If it's None, it will use the default name "llen".
        batch_size (Optional[DIM]): The dynamic length for the batch_size, it's only used when variable_length and mark_batch_size are both true.
    """

    def _has_dim(t: Optional[torch.Tensor]) -> bool:
        return t is not None and t.dim() > 0

    if shapes_collection is None:
        shapes_collection = ShapesCollection()
    vlen = _get_dim("vlen") if vlen is None else vlen

    if _has_dim(kjt._values):
        if kjt._values.numel() == 0:
            # if the values is empty, we need to set the shape to (2,) to make it compatible with dynamic shape
            # a 0-size dynamic shape will cause error in torch.export.
            # logically when the values is empty, the lengths and offsets should all be zero-value tensors.
            # And this makes the actual values irrelavent to the downstream process.
            kjt._values = torch.ones(
                2, device=kjt._values.device, dtype=kjt._values.dtype
            )
        shapes_collection[kjt._values] = (vlen,)
    if _has_dim(kjt._weights):
        shapes_collection[kjt._weights] = (vlen,)
    if variable_length:
        llen = _get_dim("llen") if llen is None else llen
        if _has_dim(kjt._lengths):
            shapes_collection[kjt._lengths] = (llen,)
        if _has_dim(kjt._offsets):
            shapes_collection[kjt._offsets] = (llen + 1,)
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
