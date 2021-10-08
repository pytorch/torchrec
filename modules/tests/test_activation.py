#!/usr/bin/env python3

import unittest

import torch
from torchrec.fx import Tracer
from torchrec.modules.activation import SwishLayerNorm


class TestActivation(unittest.TestCase):
    def test_swish_takes_float(self) -> None:
        m = SwishLayerNorm([3, 4])
        input = torch.randn(2, 3, 4)
        output = m(input)
        norm = torch.nn.LayerNorm([3, 4])
        ref_output = input * torch.sigmoid(norm(input))
        self.assertTrue(torch.allclose(output, ref_output))

    def test_fx_script_swish(self) -> None:
        m = SwishLayerNorm(10)

        gm = torch.fx.GraphModule(m, Tracer().trace(m))
        torch.jit.script(gm)


if __name__ == "__main__":
    unittest.main()
