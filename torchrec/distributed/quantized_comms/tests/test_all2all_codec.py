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
from hypothesis import assume, given, settings
from torchrec.distributed.quantized_comms.all2all_codec import QuantizationAll2AllCodec
from torchrec.distributed.quantized_comms.types import CommType


class QuantizationAll2AllCodecTest(unittest.TestCase):
    @settings(deadline=2000)
    # pyre-ignore
    @given(
        fwd_comm_precision=st.sampled_from([CommType.BF16, CommType.FP16]),
        bwd_comm_precision=st.sampled_from([CommType.BF16, CommType.FP16]),
        rand_seed=st.integers(0, 65534),
        is_fwd=st.booleans(),
        loss_scale=st.sampled_from([None, 1, 2, 3, 4, 5]),
        measure_quant_error=st.booleans(),
        row_size=st.sampled_from([100]),
        col_size=st.integers(4, 256),
    )
    def test_all2all_codec(
        self,
        fwd_comm_precision: CommType,
        bwd_comm_precision: CommType,
        rand_seed: int,
        is_fwd: bool,
        loss_scale: Optional[int],
        measure_quant_error: bool,
        row_size: int,
        col_size: int,
    ) -> None:
        assume(col_size % 4 == 0)
        row_dim = 4
        torch.manual_seed(rand_seed)

        shape = (row_size, col_size)

        quant_codec = QuantizationAll2AllCodec(
            fwd_comm_precision, bwd_comm_precision, row_dim, loss_scale=loss_scale
        )

        input_tensor = torch.rand(shape, requires_grad=True)
        # pyre-ignore
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
        expected_ref = self._compute_a2a_reference(
            expected, comm_precision, is_fwd, loss_scale
        )
        assert expected_ref is not None

        torch.testing.assert_close(expected_ref.detach(), actual.detach())

    def _compute_a2a_reference(
        self,
        tensor: torch.Tensor,
        comm_precision: CommType,
        is_fwd: bool,
        loss_scale: Optional[float],
    ) -> Optional[torch.Tensor]:
        # adding +10 to test loss scaling is applied properly
        if comm_precision == CommType.FP32:
            return tensor
        elif comm_precision == CommType.FP16:
            if not is_fwd and loss_scale is not None:
                return (tensor * loss_scale).half().float() / loss_scale
            else:
                return tensor.half().float()
        elif comm_precision == CommType.BF16:
            return tensor.bfloat16().view(torch.float16).view(torch.bfloat16).float()
        return None
