#!/usr/bin/env python3

import copy
import unittest

import torch
from torchrec.fx import Tracer
from torchrec.modules.activation import Swish


class TestActivation(unittest.TestCase):
    def test_swish_takes_float(self) -> None:
        beta = 1.5
        m = Swish(beta)
        input = torch.randn(2, 3, 4)
        output = m(input)
        ref_output = input * torch.sigmoid(beta * input)
        self.assertTrue(torch.allclose(output, ref_output))

    def test_swish_takes_module(self) -> None:
        beta = torch.nn.LayerNorm([3, 4])
        m = Swish(beta)
        input = torch.randn(2, 3, 4)
        output = m(input)
        ref_m = copy.deepcopy(m._beta_fn)
        ref_output = input * torch.sigmoid(ref_m(input))
        self.assertTrue(torch.allclose(output, ref_output))

    def test_swish_takes_func(self) -> None:
        def beta_fn(input: torch.Tensor) -> torch.Tensor:
            return input * 2.2

        m = Swish(beta_fn)
        input = torch.randn(2, 3, 4)
        output = m(input)
        ref_output = input * torch.sigmoid(beta_fn(input))
        self.assertTrue(torch.allclose(output, ref_output))

    def test_fx_script_Swish(self) -> None:
        m = Swish(1.5)

        gm = torch.fx.GraphModule(m, Tracer().trace(m))
        torch.jit.script(gm)


if __name__ == "__main__":
    unittest.main()
