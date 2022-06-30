#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

import logging
from typing import Optional, Tuple, Type

import torch

from torchrec.distributed.quantized_comms.types import (
    CommType,
    QuantizationCodec,
    TORCH_QUANT_DTYPES,
)
from torchrec.distributed.quantized_comms.utils import (
    fp32_to_bf16_with_clamp,
    fp32_to_fp16_with_clamp,
    measure_fp16_quant_error,
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


class QuantizedSingleDim:
    def __init__(
        self,
        comm_precision: CommType,
        row_dim: int = -1,
        is_fwd: bool = False,
        loss_scale: Optional[float] = None,
        measure_quant_error: bool = False,  # set to True will have perf impact
    ) -> None:
        self.comm_precision = comm_precision

        assert row_dim != -1 or comm_precision in [
            CommType.FP32,
            CommType.FP16,
            CommType.BF16,
        ], "row_dim is -1 only works for FP32, FP16 and BF16"
        self.row_dim = row_dim

        self.row_dim_quant: Optional[int] = None
        self.is_fwd = is_fwd
        self.loss_scale = loss_scale
        self.measure_quant_error = measure_quant_error

    def _get_quantized_all2all_input_tensor(
        self,
        input_tensor: torch.Tensor,
        row_dim: int,
        comm_precision: CommType,
    ) -> Tuple[torch.Tensor, int]:
        # returns quantized input tensor ready to send via all2all
        # TODO: support measure quant error for other precision
        if comm_precision == CommType.FP32:
            return input_tensor, row_dim
        elif comm_precision == CommType.FP16:
            if not self.is_fwd and self.loss_scale is not None:
                input_tensor = input_tensor * self.loss_scale

            if self.measure_quant_error:
                measure_fp16_quant_error(input_tensor)

            return fp32_to_fp16_with_clamp(input_tensor), row_dim
        elif comm_precision == CommType.BF16:
            # pyre-ignore
            return fp32_to_bf16_with_clamp(input_tensor), row_dim

    def _dequantize_tensor(
        self,
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
            if not self.is_fwd and self.loss_scale is not None:
                return quantized_tensor.float() / self.loss_scale
            if quantized_tensor.dtype != torch.float16:
                raise RuntimeError(
                    "tensor dtype is {} while bitwidth is 16.".format(
                        quantized_tensor.dtype
                    )
                )
            return quantized_tensor.float()

        elif comm_precision == CommType.BF16:
            # TODO: enable this part after BFloat16 is supported by Nvidia NCCL, change float16 to bfloat16
            if quantized_tensor.dtype != torch.float16:
                raise RuntimeError(
                    "tensor dtype is {} while bitwidth is 16, expecting float16.".format(
                        quantized_tensor.dtype
                    )
                )
            # pyre-ignore
            return quantized_tensor.view(torch.bfloat16).float()


class QuantizationAll2AllCodec(QuantizationCodec):
    def __init__(
        self,
        fwd_comm_precision: CommType,
        bwd_comm_precision: CommType,
        quantize_dim: int = -1,
        loss_scale: Optional[float] = None,
        measure_quant_error: bool = False,
    ) -> None:
        assert (
            fwd_comm_precision in TORCH_QUANT_DTYPES
            and bwd_comm_precision in TORCH_QUANT_DTYPES
        ), f"Invalid supported types {fwd_comm_precision} {bwd_comm_precision}. Supported types are {TORCH_QUANT_DTYPES}."

        fwd_work = QuantizedSingleDim(
            fwd_comm_precision,
            quantize_dim,
            is_fwd=True,
            loss_scale=loss_scale,
            measure_quant_error=measure_quant_error,
        )
        bwd_work = QuantizedSingleDim(
            bwd_comm_precision,
            quantize_dim,
            is_fwd=False,
            loss_scale=loss_scale,
            measure_quant_error=measure_quant_error,
        )

        logger.info(
            f"Creating QuantizationAll2AllCodec fwd_comm_precision:{fwd_comm_precision} bwd_comm_precision:{bwd_comm_precision} "
            f"loss_scale:{loss_scale} row_dim:{quantize_dim}"
        )

        class Encoder(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input_tensor):
                ctx.input_tensor_shape = input_tensor.shape
                (
                    quantized_input_tensor,
                    fwd_work.row_dim_quant,
                ) = fwd_work._get_quantized_all2all_input_tensor(
                    input_tensor, fwd_work.row_dim, fwd_work.comm_precision
                )
                return quantized_input_tensor

            @staticmethod
            def backward(ctx, grad_output):
                dequantized_grad_output = bwd_work._dequantize_tensor(
                    grad_output, bwd_work.comm_precision
                )
                return dequantized_grad_output.view(ctx.input_tensor_shape)

        class Decoder(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input_tensor):
                dequantized_input_tensor = fwd_work._dequantize_tensor(
                    input_tensor, fwd_work.comm_precision
                )
                return dequantized_input_tensor.view(input_tensor.shape)

            @staticmethod
            def backward(ctx, grad_output):
                (
                    quantized_grad_output,
                    bwd_work.row_dim_quant,
                ) = bwd_work._get_quantized_all2all_input_tensor(
                    grad_output, bwd_work.row_dim, bwd_work.comm_precision
                )
                return quantized_grad_output

        self._encoder = Encoder
        self._decoder = Decoder

    @property
    def encoder(self) -> Type[torch.autograd.Function]:
        return self._encoder

    @property
    def decoder(self) -> Type[torch.autograd.Function]:
        return self._decoder
