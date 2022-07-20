#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

import logging

import torch

from torchrec.distributed.quantized_comms.types import (
    TORCH_BFLOAT16_MAX,
    TORCH_BFLOAT16_MIN,
    TORCH_HALF_MAX,
    TORCH_HALF_MIN,
)

logger: logging.Logger = logging.getLogger()


def fp32_to_fp16_with_clamp(tensor: torch.Tensor) -> torch.Tensor:
    return torch.clamp(tensor, TORCH_HALF_MIN, TORCH_HALF_MAX).half()


def fp32_to_bf16_with_clamp(tensor: torch.Tensor) -> torch.Tensor:
    return (
        torch.clamp(tensor, TORCH_BFLOAT16_MIN, TORCH_BFLOAT16_MAX)
        .bfloat16()
        .view(torch.float16)
    )
