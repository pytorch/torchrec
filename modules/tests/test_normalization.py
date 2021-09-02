#!/usr/bin/env python3

import unittest

import hypothesis.strategies as st
import torch
from hypothesis import given, settings
from torchrec.fx import Tracer
from torchrec.modules.normalization import LayerNorm


class TestNormalization(unittest.TestCase):
    # pyre-ignore[56]: Pyre was not able to infer the type of argument
    # `hypothesis.strategies.booleans()` to decorator factory `hypothesis.given`.
    @given(
        elementwise_affine=st.booleans(),
        has_num_channels=st.booleans(),
        num_feature_dims=st.integers(2, 4),
        norm_axis_starting_from_first_feature_dim=st.integers(0, 3),
    )
    @settings(deadline=None)
    def test_layer_norm(
        self,
        elementwise_affine: bool,
        has_num_channels: bool,
        num_feature_dims: int,
        norm_axis_starting_from_first_feature_dim: int,
    ) -> None:
        # either (C, B, F1, F2, ...) or (B, F1, F2, ...)
        first_feature_dim = 2 if has_num_channels else 1

        # the starting axis for normalization
        norm_axis = min(
            first_feature_dim + norm_axis_starting_from_first_feature_dim,
            first_feature_dim + num_feature_dims - 1,
        )

        num_channels = 3 if has_num_channels else None
        m = LayerNorm(
            norm_axis, num_channels=num_channels, elementwise_affine=elementwise_affine
        )

        batch_size = 7
        input_shape = [batch_size] + [
            torch.randint(10, (1,)).int().item()
        ] * num_feature_dims
        if num_channels is not None:
            input_shape = [num_channels] + input_shape
        input = torch.randn(*input_shape)

        output = m(input)

        ref_m = torch.nn.LayerNorm(
            input.shape[norm_axis:], elementwise_affine=False, eps=m.eps
        )
        if elementwise_affine:
            ref_output = ref_m(input) * m.weight + m.bias
        ref_output = ref_m(input)

        self.assertTrue(torch.allclose(output, ref_output))

        gm = torch.fx.GraphModule(m, Tracer().trace(m))
        torch.jit.script(gm)


if __name__ == "__main__":
    unittest.main()
