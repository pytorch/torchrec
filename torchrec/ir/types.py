#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

import abc
from typing import Any

from torch import nn
from torchrec.modules.embedding_modules import EmbeddingBagCollection


class SerializerInterface(abc.ABC):
    """
    Interface for Serializer classes for torch.export IR.
    """

    @classmethod
    @property
    def module_to_serializer_cls(cls):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def serialize(
        cls,
        module: nn.Module,
    ) -> Any:
        # Take the eager embedding module and generate bytes in buffer
        pass

    @classmethod
    @abc.abstractmethod
    def deserialize(cls, input: Any) -> nn.Module:
        # Take the bytes in the buffer and regenerate the eager embedding module
        pass


class EBCJsonSerializer(SerializerInterface):
    @classmethod
    def serialize(
        cls,
        module: nn.Module,
    ) -> str:
        # TODO: Add support for EBC with dataclass
        pass

    @classmethod
    def deserialize(cls, input: str) -> nn.Module:
        # TODO: Add support for EBC with dataclass
        pass


class JsonSerializer(SerializerInterface):
    module_to_serializer_cls = {
        EmbeddingBagCollection: EBCJsonSerializer,
    }

    @classmethod
    def serialize(
        cls,
        module: nn.Module,
    ) -> str:
        # TODO: Add support for dataclass
        pass

    @classmethod
    def deserialize(cls, input: str) -> nn.Module:
        # TODO: Add support for dataclass
        pass
