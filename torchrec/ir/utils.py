#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

from typing import Type

from torch import nn
from torchrec.ir.types import SerializerInterface


# TODO: Replace the default interface with the python dataclass interface
DEFAULT_SERIALIZER_CLS = SerializerInterface


def serialize_embedding_modules(
    model: nn.Module,
    serializer_cls: Type[SerializerInterface] = DEFAULT_SERIALIZER_CLS,
) -> nn.Module:
    for _, module in model.named_modules():
        if type(module) in serializer_cls.module_to_serializer_cls:
            serialized_module = serializer_cls.serialize(module)
            module.register_buffer("ir_metadata", serialized_module, persistent=False)

    return model


def deserialize_embedding_modules(
    model: nn.Module,
    serializer_cls: Type[SerializerInterface] = DEFAULT_SERIALIZER_CLS,
) -> nn.Module:
    fqn_to_new_module = {}
    for name, module in model.named_modules():
        if "ir_metadata" in dict(module.named_buffers()):
            serialized_module = dict(module.named_buffers())["ir_metadata"]
            deserialized_module = serializer_cls.deserialize(serialized_module)
            fqn_to_new_module[name] = deserialized_module

    for fqn, new_module in fqn_to_new_module.items():
        setattr(model, fqn, new_module)

    return model
