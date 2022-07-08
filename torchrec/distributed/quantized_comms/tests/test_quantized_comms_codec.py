#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Optional

import hypothesis.strategies as st
import torch
from hypothesis import given, settings
from torchrec.distributed.quantized_comms.quantized_comms_codec import (
    QuantizedCommsCodec,
)
from torchrec.distributed.quantized_comms.types import CommType


class QuantizationCommsCodecTest(unittest.TestCase):
    @settings(deadline=2000)
    # pyre-ignore
    @given(
        fwd_comm_precision=st.sampled_from([CommType.FP16]),
        bwd_comm_precision=st.sampled_from([CommType.FP16, CommType.BF16]),
        loss_scale=st.sampled_from([None, 4.0]),
        rand_seed=st.integers(0, 65534),
    )
    def test_quantized_comms_codec(
        self,
        fwd_comm_precision: CommType,
        bwd_comm_precision: CommType,
        loss_scale: Optional[float],
        rand_seed: int,
    ) -> None:
        row_size = 128
        col_size = 128
        torch.manual_seed(rand_seed)

        shape = (row_size, col_size)

        quant_codec = QuantizedCommsCodec(
            fwd_comm_precision,
            bwd_comm_precision,
            loss_scale=loss_scale,
        )

        input_tensor = torch.rand(shape, requires_grad=True)
        # pyre-ignore
        quant_tensor = quant_codec.encoder.apply(input_tensor)
        output_tensor = quant_codec.decoder.apply(quant_tensor)

        torch.testing.assert_close(
            input_tensor.detach(), output_tensor.detach(), rtol=0.0005, atol=0.0003
        )

        expected_grad = torch.rand(shape)
        output_tensor.backward(expected_grad)
        actual_grad = input_tensor.grad

        self.assertIsNotNone(actual_grad)
        torch.testing.assert_close(
            expected_grad.detach(), actual_grad.detach(), atol=0.003, rtol=0.009
        )
