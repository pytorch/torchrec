#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

import logging
from typing import Optional, Type

import torch

from torchrec.distributed.quantized_comms.types import CommType, QuantizedCommsCodecIf
from torchrec.distributed.quantized_comms.utils import (
    fp32_to_bf16_with_clamp,
    fp32_to_fp16_with_clamp,
)

logger: logging.Logger = logging.getLogger()

try:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
except OSError:
    pass

# OSS
try:
    import fbgemm_gpu  # @manual  # noqa
except ImportError:
    pass


def quantize_tensor(
    input_tensor: torch.Tensor,
    comm_precision: CommType,
) -> torch.Tensor:
    if comm_precision == CommType.FP32:
        return input_tensor
    elif comm_precision == CommType.FP16:
        return fp32_to_fp16_with_clamp(input_tensor)
    elif comm_precision == CommType.BF16:
        return fp32_to_bf16_with_clamp(input_tensor)
    raise ValueError(f"comm_precision={comm_precision} is not supported")


def dequantize_tensor(
    quantized_tensor: torch.Tensor,
    comm_precision: CommType,
) -> torch.Tensor:
    if comm_precision == CommType.FP32:
        if quantized_tensor.dtype != torch.float32:
            raise RuntimeError(
                "tensor dtype is {} while bitwidth is 32.".format(
                    quantized_tensor.dtype
                )
            )
        return quantized_tensor
    elif comm_precision == CommType.FP16:
        if quantized_tensor.dtype != torch.float16:
            raise RuntimeError(
                "tensor dtype is {} while bitwidth is 16.".format(
                    quantized_tensor.dtype
                )
            )
        return quantized_tensor.float()
    elif comm_precision == CommType.BF16:
        if quantized_tensor.dtype != torch.float16:
            raise RuntimeError(
                "tensor dtype is {} while bitwidth is 16, expecting float16.".format(
                    quantized_tensor.dtype
                )
            )
        return quantized_tensor.view(torch.bfloat16).float()
    raise ValueError(f"comm_precision={comm_precision} is not supported")


class QuantizedCommsCodec(QuantizedCommsCodecIf):
    def __init__(
        self,
        fwd_comm_precision: CommType,
        bwd_comm_precision: CommType,
        loss_scale: Optional[float] = None,
    ) -> None:

        logger.info(
            f"Creating QuantizedCommsCodec fwd_comm_precision:{fwd_comm_precision} bwd_comm_precision:{bwd_comm_precision} "
        )

        if loss_scale is not None:
            if bwd_comm_precision not in [CommType.FP16]:
                logger.warning(
                    f"Setting loss scale for bwd_comm_precision={bwd_comm_precision} is not supported. Overriding to None"
                )
                loss_scale = None

        class Encoder(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input_tensor):
                ctx.input_tensor_shape = input_tensor.shape
                return quantize_tensor(input_tensor, fwd_comm_precision)

            @staticmethod
            def backward(ctx, grad_output):
                if loss_scale is not None:
                    grad_output = grad_output / loss_scale
                return dequantize_tensor(grad_output, bwd_comm_precision).view(
                    ctx.input_tensor_shape
                )

        class Decoder(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input_tensor):
                return dequantize_tensor(input_tensor, fwd_comm_precision)

            @staticmethod
            def backward(ctx, grad_output):
                if loss_scale is not None:
                    grad_output = grad_output * loss_scale
                return quantize_tensor(grad_output, bwd_comm_precision)

        self._encoder = Encoder
        self._decoder = Decoder

    @property
    def encoder(self) -> Type[torch.autograd.Function]:
        return self._encoder

    @property
    def decoder(self) -> Type[torch.autograd.Function]:
        return self._decoder
