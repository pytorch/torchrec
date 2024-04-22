#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

import threading
from typing import Callable, Dict, List, Optional, Tuple, Type

import torch

from torch import nn
from torch.export.exported_program import ExportedProgram
from torch.library import Library
from torchrec.ir.types import SerializerInterface


lib = Library("custom", "FRAGMENT")


class OpRegistryState:
    """
    State of operator registry.

    We can only register the op schema once. So if we're registering multiple
    times we need a lock and check if they're the same schema
    """

    op_registry_lock = threading.Lock()
    # operator schema: op_name: schema
    op_registry_schema: Dict[str, str] = {}


operator_registry_state = OpRegistryState()

# TODO: Replace the default interface with the python dataclass interface
DEFAULT_SERIALIZER_CLS = SerializerInterface


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
                serialized_module, module_type_dict[fqn]
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


def register_custom_op(
    module: nn.Module, dims: List[int]
) -> Callable[[List[Optional[torch.Tensor]], int], List[torch.Tensor]]:
    """
    Register a customized operator.

    Args:
        module: customized module instance
        dims: output dimensions
    """

    global operator_registry_state

    op_name = f"{type(module).__name__}_{hash(module)}"
    with operator_registry_state.op_registry_lock:
        if op_name in operator_registry_state.op_registry_schema:
            return getattr(torch.ops.custom, op_name)

    def pea_op(
        values: List[Optional[torch.Tensor]],
        batch_size: int,
    ) -> List[torch.Tensor]:
        device = None
        for v in values:
            if v is not None:
                device = v.device
                break
        else:
            raise AssertionError(
                f"Custom op {type(module).__name__} expects at least one "
                "input tensor"
            )

        return [
            torch.empty(
                batch_size,
                dim,
                device=device,
            )
            for dim in dims
        ]

    schema_string = f"{op_name}(Tensor?[] values, int batch_size) -> Tensor[]"
    with operator_registry_state.op_registry_lock:
        if op_name in operator_registry_state.op_registry_schema:
            return getattr(torch.ops.custom, op_name)
        operator_registry_state.op_registry_schema[op_name] = schema_string
        # Register schema
        lib.define(schema_string)

        # Register implementation
        lib.impl(op_name, pea_op, "CPU")
        lib.impl(op_name, pea_op, "CUDA")

        # Register meta formula
        lib.impl(op_name, pea_op, "Meta")

    return getattr(torch.ops.custom, op_name)
