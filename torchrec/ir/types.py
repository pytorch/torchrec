#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

#!/usr/bin/env python3

import abc
from typing import Any, Dict, Optional, Type

import torch

from torch import nn


class SerializerInterface(abc.ABC):
    """
    Interface for Serializer classes for torch.export IR.
    """

    @classmethod
    @property
    # pyre-ignore [3]: Returning `None` but type `Any` is specified.
    def module_to_serializer_cls(cls) -> Dict[str, Type[Any]]:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    # pyre-ignore [3]: Returning `None` but type `Any` is specified.
    def serialize(
        cls,
        module: nn.Module,
    ) -> Any:
        # Take the eager embedding module and generate bytes in buffer
        pass

    @classmethod
    @abc.abstractmethod
    def deserialize(
        cls,
        # pyre-ignore [2]: Parameter `input` must have a type other than `Any`.
        input: Any,
        typename: str,
        device: Optional[torch.device] = None,
    ) -> nn.Module:
        # Take the bytes in the buffer and regenerate the eager embedding module
        pass
