#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Optional, Tuple

import hypothesis.strategies as st
import torch
from hypothesis import assume, given, settings
from torchrec.distributed.fbgemm_qcomm_codec import (
    CommType,
    get_qcomm_codecs,
    QCommsConfig,
)


class QuantizationCommCodecTest(unittest.TestCase):
    @settings(deadline=2000)
    # pyre-ignore
    @given(
        comm_precisions_loss_scale=st.sampled_from(
            [
                (CommType.FP32, None),
                (CommType.FP16, None),
                (CommType.FP16, 4.0),
                (CommType.BF16, None),
                (CommType.FP8, None),
                (CommType.INT8, None),
            ]
        ),
        row_size=st.integers(4, 256),
        col_size=st.integers(4, 256),
        rand_seed=st.integers(0, 65534),
    )
    def test_quantized_comm_codec(
        self,
        comm_precisions_loss_scale: Tuple[CommType, Optional[float]],
        row_size: int,
        col_size: int,
        rand_seed: int,
    ) -> None:
        (comm_precision, loss_scale) = comm_precisions_loss_scale
        if comm_precision == CommType.FP8:
            assume(col_size % 4 == 0)

        torch.manual_seed(rand_seed)
        shape = (row_size, col_size)

        quant_codec = get_qcomm_codecs(
            QCommsConfig(
                forward_precision=comm_precision,
            )
        )

        input_tensor = torch.rand(shape, requires_grad=True)

        ctx = quant_codec.forward.create_context()
        if comm_precision == CommType.INT8:
            assume(row_size * col_size % ctx.row_dim == 0)
            input_tensor = input_tensor.view(-1)

        quant_tensor = quant_codec.forward.encode(input_tensor, ctx)
        output_tensor = quant_codec.forward.decode(quant_tensor, ctx)

        rtol = 0.005
        atol = 0.005
        if comm_precision == CommType.FP8:
            rtol = 0.05
            atol = 0.05

        torch.testing.assert_close(
            input_tensor.detach(), output_tensor.detach(), rtol=rtol, atol=atol
        )
