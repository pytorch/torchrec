#!/usr/bin/env python3

import unittest

import hypothesis.strategies as st
import torch
from hypothesis import given, settings
from torchrec.fx import symbolic_trace
from torchrec.modules.linear import MCLinear


class TestLinear(unittest.TestCase):
    # pyre-ignore[56]: Pyre was not able to infer the type of argument
    # to decorator factory `hypothesis.given`.
    @given(has_bias=st.booleans(), input_num_dims=st.integers(2, 3))
    @settings(deadline=None)
    def test_mclinear_single_channel(self, has_bias: bool, input_num_dims: int) -> None:
        in_features = 5
        out_features = 10
        num_channels = 1
        batch_size = 3

        if input_num_dims == 2:
            batch_dim = 0
            input_tensor_shape = [batch_size, in_features]
        else:
            batch_dim = 1
            input_tensor_shape = [1, batch_size, in_features]
        input_tensor = torch.randn(input_tensor_shape).float()

        mc_linear_module = MCLinear(out_features, num_channels, bias=has_bias)
        # Dry-run with input of a different batch size
        dry_run_batch_size = 1
        assert dry_run_batch_size != batch_size
        mc_linear_module(
            torch.randn_like(input_tensor).narrow(batch_dim, 0, dry_run_batch_size)
        )

        output_tensor = mc_linear_module(input_tensor)

        # Reference implementation
        linear_ref = torch.nn.Linear(in_features, out_features, bias=has_bias)
        with torch.no_grad():
            linear_ref.weight.copy_(
                mc_linear_module.weight.view(in_features, out_features).t()
            )
            if has_bias:
                linear_ref.bias.copy_(mc_linear_module.bias.view(out_features))
        output_tensor_ref = linear_ref(input_tensor).view(1, batch_size, out_features)

        self.assertEqual(list(output_tensor.shape), [1, batch_size, out_features])
        self.assertTrue(torch.allclose(output_tensor, output_tensor_ref))

    # pyre-ignore[56]: Pyre was not able to infer the type of argument
    # to decorator factory `hypothesis.given`.
    @given(has_bias=st.booleans())
    @settings(deadline=None)
    def test_mclinear_multi_channel_2d_input(self, has_bias: bool) -> None:
        in_features = 5
        out_features = 10
        num_channels = 3
        batch_size = 7

        input_tensor = torch.randn(batch_size, in_features).float()

        mc_linear_module = MCLinear(out_features, num_channels, bias=has_bias)
        # Dry-run with input of a different batch size
        dry_run_batch_size = 1
        assert dry_run_batch_size != batch_size
        mc_linear_module(
            torch.randn_like(input_tensor).narrow(0, 0, dry_run_batch_size)
        )

        output_tensor = mc_linear_module(input_tensor)

        # Reference implementation
        linear_refs = [
            torch.nn.Linear(in_features, out_features, bias=has_bias)
            for _ in range(num_channels)
        ]
        with torch.no_grad():
            for i in range(num_channels):
                linear_refs[i].weight.copy_(mc_linear_module.weight[i].t())
                if has_bias:
                    linear_refs[i].bias.copy_(mc_linear_module.bias[i])
        output_tensor_refs = []
        for i in range(num_channels):
            output_tensor_refs.append(linear_refs[i](input_tensor.unsqueeze(1)))
        output_tensor_ref = torch.cat(output_tensor_refs, dim=1).transpose(0, 1)

        self.assertEqual(
            list(output_tensor.shape), [num_channels, batch_size, out_features]
        )
        self.assertTrue(torch.allclose(output_tensor, output_tensor_ref, rtol=1e-05))

    # pyre-ignore[56]: Pyre was not able to infer the type of argument
    # to decorator factory `hypothesis.given`.
    @given(has_bias=st.booleans())
    @settings(deadline=None)
    def test_mclinear_multi_channel_3d_input(self, has_bias: bool) -> None:
        in_features = 5
        out_features = 10
        num_channels = 3
        batch_size = 7

        input_tensor = torch.randn(num_channels, batch_size, in_features).float()

        mc_linear_module = MCLinear(out_features, num_channels, bias=has_bias)
        # Dry-run with input of a different batch size
        dry_run_batch_size = 1
        assert dry_run_batch_size != batch_size
        mc_linear_module(
            torch.randn_like(input_tensor).narrow(1, 0, dry_run_batch_size)
        )

        output_tensor = mc_linear_module(input_tensor)

        # Reference implementation
        linear_refs = [
            torch.nn.Linear(in_features, out_features, bias=has_bias)
            for _ in range(num_channels)
        ]
        with torch.no_grad():
            for i in range(num_channels):
                linear_refs[i].weight.copy_(mc_linear_module.weight[i].t())
                if has_bias:
                    linear_refs[i].bias.copy_(mc_linear_module.bias[i])
        output_tensor_refs = []
        for i in range(num_channels):
            output_tensor_refs.append(
                linear_refs[i](input_tensor[i, :, :].unsqueeze(0))
            )
        output_tensor_ref = torch.cat(output_tensor_refs, dim=0)

        self.assertEqual(
            list(output_tensor.shape), [num_channels, batch_size, out_features]
        )
        self.assertTrue(torch.allclose(output_tensor, output_tensor_ref, rtol=1e-05))

    def test_fx_script_MCLinear(self) -> None:
        m = MCLinear(out_features=5, num_channels=4)

        # Dry-run to initialize lazy module.
        m(torch.randn(4, 1, 3))

        gm = symbolic_trace(m)
        torch.jit.script(gm)


if __name__ == "__main__":
    unittest.main()
