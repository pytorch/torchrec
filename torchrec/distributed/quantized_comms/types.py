#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractproperty
from enum import Enum, unique

from typing import Tuple, Type

import torch


@unique
class CommType(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"

    def __str__(self) -> str:
        return self.value


DTYPE_NO_DIM: Tuple[CommType, ...] = (CommType.FP32, CommType.FP16, CommType.BF16)
TORCH_QUANT_DTYPES: Tuple[CommType, ...] = (CommType.FP32, CommType.BF16, CommType.FP16)

TORCH_HALF_MIN: float = torch.finfo(torch.float16).min
TORCH_HALF_MAX: float = torch.finfo(torch.float16).max

# pyre-fixme[5]: Global expression must be annotated.
TORCH_BFLOAT16_MIN = torch.finfo(torch.bfloat16).min
# pyre-fixme[5]: Global expression must be annotated.
TORCH_BFLOAT16_MAX = torch.finfo(torch.bfloat16).max


class QuantizationCodec(ABC):
    @abstractproperty
    def encoder(self) -> Type[torch.autograd.Function]:
        ...

    @abstractproperty
    def decoder(self) -> Type[torch.autograd.Function]:
        ...
