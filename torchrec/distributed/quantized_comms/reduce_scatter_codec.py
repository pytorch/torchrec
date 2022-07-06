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

from torchrec.distributed.quantized_comms.types import CommType, QuantizationCodec
from torchrec.distributed.quantized_comms.utils import fp32_to_fp16_with_clamp

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


class QuantizedReduceScatterWorkV1:
    def __init__(
        self,
        comm_precision: CommType,
    ) -> None:
        self.comm_precision = comm_precision

    def _prepare_quantized_tensor(
        self,
        input_tensor: torch.Tensor,
        comm_precision: CommType,
    ) -> torch.Tensor:
        if comm_precision == CommType.FP32:
            return input_tensor

        if comm_precision != CommType.FP16:
            raise NotImplementedError(
                "FP16 is the lowest supported precision currently."
            )

        q_input_tensor = input_tensor.half()
        return q_input_tensor

    def _dequantize_tensor(
        self,
        quantized_output_tensor: torch.Tensor,
        comm_precision: CommType,
    ) -> torch.Tensor:
        if comm_precision == CommType.FP32:
            return quantized_output_tensor
        if comm_precision == CommType.FP16:
            return quantized_output_tensor.float()
        else:
            raise RuntimeError(
                "FP16 is the lowest supported precision currently. Dequantize output list called with comm precision: {}".format(
                    comm_precision
                )
            )


class QuantizedAllGatherWorkV1:
    def __init__(
        self,
        comm_precision: CommType,
        loss_scale: Optional[float] = None,
        measure_quant_error: bool = False,
    ) -> None:
        self.comm_precision = comm_precision
        self.loss_scale = loss_scale
        self.measure_quant_error = measure_quant_error

    def _quantize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.comm_precision == CommType.FP16:
            return fp32_to_fp16_with_clamp(tensor)
        elif self.comm_precision == CommType.BF16:
            input_2d = tensor.view(-1)
            input_2d_quant = torch.ops.fbgemm.FloatToBfloat16Quantized(input_2d)
            return input_2d_quant.view(tensor.shape)
        else:
            raise NotImplementedError(
                "Unsupported precision: {}".format(self.comm_precision)
            )

    def _prepare_quantized_tensor(
        self,
        input_tensor: torch.Tensor,
        comm_precision: CommType,
    ) -> torch.Tensor:
        if comm_precision == CommType.FP32:
            return input_tensor
        if comm_precision not in [CommType.FP16, CommType.BF16]:
            raise NotImplementedError(
                "FP16/BF16 is the lowest supported precision for all_gather."
            )
        this_loss_scale = self.loss_scale
        if this_loss_scale is not None:
            return self._quantize_tensor(this_loss_scale * input_tensor)
        else:
            return self._quantize_tensor(input_tensor)

    def _dequantize_tensor(
        self,
        quantized_output_tensor: torch.Tensor,
        comm_precision: CommType,
    ) -> torch.Tensor:
        tensor_deq = quantized_output_tensor
        if comm_precision == CommType.FP32:
            return quantized_output_tensor
        if comm_precision == CommType.FP16:
            tensor_deq = quantized_output_tensor.float()
        elif comm_precision == CommType.BF16:
            quantized_tensor_2d = quantized_output_tensor.view(-1)
            tensor_deq = torch.ops.fbgemm.Bfloat16QuantizedToFloat(quantized_tensor_2d)
            tensor_deq = tensor_deq.view(quantized_output_tensor.shape)
        else:
            raise RuntimeError(
                "FP16/BF16 are the lowest supported precision "
                "currently. Dequantize output list called with comm "
                "precision: {}".format(comm_precision)
            )
        if self.loss_scale is not None:
            tensor_deq = tensor_deq / self.loss_scale
        return tensor_deq


class QuantizationReduceScatterCodec(QuantizationCodec):
    def __init__(
        self,
        fwd_comm_precision: CommType,
        bwd_comm_precision: CommType,
        loss_scale: Optional[float] = None,
        measure_quant_error: bool = False,
    ) -> None:
        if (fwd_comm_precision not in (CommType.FP32, CommType.FP16)) or (
            bwd_comm_precision not in [CommType.FP32, CommType.FP16, CommType.BF16]
        ):
            raise NotImplementedError(
                "FP16 is the lowest supported precision for the forward "
                "pass and BF16/FP16 are the lowest supported precision for the "
                "backward pass."
            )

        fwd_work = QuantizedReduceScatterWorkV1(
            comm_precision=fwd_comm_precision,
        )
        bwd_work = QuantizedAllGatherWorkV1(
            comm_precision=bwd_comm_precision,
            loss_scale=loss_scale,
            measure_quant_error=measure_quant_error,
        )

        logger.info(
            f"Creating QuantizationReduceScatterCodec fwd_comm_precision:{fwd_comm_precision} bwd_comm_precision:{bwd_comm_precision} "
            f"loss_scale:{loss_scale}"
        )

        class Encoder(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input_tensor):
                quantized_input_tensors = fwd_work._prepare_quantized_tensor(
                    input_tensor, fwd_work.comm_precision
                )
                return quantized_input_tensors

            @staticmethod
            def backward(ctx, grad_output):
                dequantized_grad_output = bwd_work._dequantize_tensor(
                    grad_output, bwd_work.comm_precision
                )
                return dequantized_grad_output

        class Decoder(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input_tensor):
                dequantized_input_tensor = fwd_work._dequantize_tensor(
                    input_tensor, fwd_work.comm_precision
                )
                return dequantized_input_tensor

            @staticmethod
            def backward(ctx, grad_output):
                quantized_grad_output = bwd_work._prepare_quantized_tensor(
                    grad_output, bwd_work.comm_precision
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
