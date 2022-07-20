#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractproperty

from dataclasses import dataclass
from enum import Enum, unique

from typing import Optional, Type

import torch


@unique
class CommType(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"

    def __str__(self) -> str:
        return self.value


TORCH_HALF_MIN: float = torch.finfo(torch.float16).min
TORCH_HALF_MAX: float = torch.finfo(torch.float16).max

TORCH_BFLOAT16_MIN: float = torch.finfo(torch.bfloat16).min
TORCH_BFLOAT16_MAX: float = torch.finfo(torch.bfloat16).max


class QuantizedCommsCodecIf(ABC):
    @abstractproperty
    def encoder(self) -> Type[torch.autograd.Function]:
        ...

    @abstractproperty
    def decoder(self) -> Type[torch.autograd.Function]:
        ...


@dataclass
class QCommsConfig:
    """
    Quantization configs for the AllToAll and ReduceScatter communication modules used in sharding.
    """

    # Quantization of comm modules in the forward pass
    forward_precision: CommType
    # Quantization of comm modules in the backward pass
    backward_precision: CommType
    # For supported backward precisions (currently FP16), scale the gradient of the decoder and
    # divide the gradient of the encode r by this value. This can provide some
    loss_scale: Optional[float] = None
