#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import List, Optional

import hypothesis.strategies as st
import numpy as np
import torch
from hypothesis import assume, given, settings
from torchrec.distributed.quantized_comms.reduce_scatter_codec import (
    QuantizationReduceScatterCodec,
)
from torchrec.distributed.quantized_comms.types import CommType


class QuantizedReduceScatterCodecTest(unittest.TestCase):
    @settings(deadline=2000)
    # pyre-ignore
    @given(
        fwd_comm_precision=st.sampled_from([CommType.FP16]),
        bwd_comm_precision=st.sampled_from([CommType.BF16, CommType.FP16]),
        rand_seed=st.integers(0, 65534),
        is_fwd=st.booleans(),
        apply_loss_scale=st.booleans(),
        loss_scale=st.integers(1, 5),
        measure_quant_error=st.booleans(),
        row_size=st.sampled_from([100]),
        col_size=st.integers(4, 256),
    )
    def test_quantizer_reduce_scatter_codec(
        self,
        fwd_comm_precision: CommType,
        bwd_comm_precision: CommType,
        rand_seed: int,
        is_fwd: bool,
        apply_loss_scale: bool,
        loss_scale: Optional[int],
        measure_quant_error: bool,
        row_size: int,
        col_size: int,
    ) -> None:
        assume(col_size % 4 == 0)
        torch.manual_seed(rand_seed)
        if not apply_loss_scale:
            loss_scale = None
        shape = (row_size, col_size)

        quant_codec = QuantizationReduceScatterCodec(
            fwd_comm_precision, bwd_comm_precision, loss_scale=loss_scale
        )

        input_tensor = torch.rand(shape, requires_grad=True)
        # pyre-fixme[16]: `Encoder` has no attribute `apply`.
        quant_tensor = quant_codec.encoder.apply(input_tensor)
        output_tensor = quant_codec.decoder.apply(quant_tensor)

        if is_fwd:
            expected = input_tensor
            actual = output_tensor
        else:
            expected = torch.rand(shape)
            output_tensor.backward(expected)
            actual = input_tensor.grad

        comm_precision = fwd_comm_precision if is_fwd else bwd_comm_precision
        if is_fwd:
            expected_ref = self._compute_codec_rs_reference(
                expected, comm_precision, is_fwd, loss_scale
            )
        else:
            expected_ref = self._compute_codec_ag_reference(
                expected, comm_precision, is_fwd, loss_scale
            )

        np.testing.assert_allclose(expected_ref.detach(), actual.detach())

    def _compute_reference_reduces_scatter(
        self, input_list: List[List[torch.Tensor]], comm_precision: CommType
    ) -> List[torch.Tensor]:
        if comm_precision == CommType.FP32:
            return [input_list[0][0] + input_list[1][0]]
        if comm_precision == CommType.FP16:
            return [(input_list[0][0].half() + input_list[1][0].half()).half()]
        else:
            raise NotImplementedError()

    def _compute_reference_all_gather(
        self,
        input_list: List[torch.Tensor],
        comm_precision: CommType,
        loss_scale: Optional[float],
    ) -> List[List[torch.Tensor]]:
        # adding +10 to test loss scaling is applied properly
        if comm_precision == CommType.FP32:
            return [[input_list[0] + 10]]
        if comm_precision == CommType.FP16:
            if loss_scale is not None:
                scaled_0 = (loss_scale * input_list[0]).half()
                de_scale = (scaled_0 + 10).float() / loss_scale
                return [[de_scale]]
            else:
                return [[(input_list[0].half() + 10).float()]]
        elif comm_precision == CommType.BF16:
            if loss_scale is not None:
                scaled_0 = torch.ops.fbgemm.FloatToBfloat16Quantized(
                    loss_scale * input_list[0]
                )
                de_scale = (
                    torch.ops.fbgemm.Bfloat16QuantizedToFloat(scaled_0 + 10)
                    / loss_scale
                )
                return [[de_scale]]
            else:
                # pyre-fixme[7]: Expected `List[List[torch.Tensor]]` but got implicit return value of `None`.
                return [
                    [
                        torch.ops.fbgemm.Bfloat16QuantizedToFloat(
                            torch.ops.fbgemm.FloatToBfloat16Quantized(input_list[0])
                            + 10
                        )
                    ]
                ]

    def _compute_codec_rs_reference(
        self,
        tensor: torch.Tensor,
        comm_precision: CommType,
        is_fwd: bool,
        loss_scale: Optional[float],
    ) -> torch.Tensor:
        if comm_precision == CommType.FP32:
            return tensor
        elif comm_precision == CommType.FP16:
            # pyre-ignore[7]
            return tensor.half().float()

    def _compute_codec_ag_reference(
        self,
        tensor: torch.Tensor,
        comm_precision: CommType,
        is_fwd: bool,
        loss_scale: Optional[float],
    ) -> torch.Tensor:
        if comm_precision == CommType.FP32:
            return tensor
        elif comm_precision == CommType.FP16:
            if loss_scale is not None:
                return ((tensor * loss_scale).half()).float() / loss_scale
            else:
                return tensor.half().float()
        elif comm_precision == CommType.BF16:
            if loss_scale is not None:
                return (
                    torch.ops.fbgemm.Bfloat16QuantizedToFloat(
                        torch.ops.fbgemm.FloatToBfloat16Quantized(loss_scale * tensor)
                    )
                    / loss_scale
                )
            else:
                # pyre-ignore[7]
                return torch.ops.fbgemm.Bfloat16QuantizedToFloat(
                    torch.ops.fbgemm.FloatToBfloat16Quantized(tensor)
                )
