#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import List, Optional, Tuple

import torch
import torch.fx
from torch.testing import FileCheck  # @manual
from torchrec.distributed.types import LazyAwaitable
from torchrec.fx import symbolic_trace
from torchrec.sparse.jagged_tensor import (
    JaggedTensor,
    KeyedJaggedTensor,
)


torch.fx.wrap("len")


class TestTracer(unittest.TestCase):
    maxDiff: Optional[int] = None

    def test_jagged_tensor(self) -> None:
        class ModuleCreateAndAccessJaggedTensor(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input: int) -> int:
                features = JaggedTensor(
                    values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
                    weights=torch.Tensor([1.0, 0.5, 1.5, 1.0, 0.5, 1.0, 1.0, 1.5]),
                    offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8]),
                )
                return (
                    features.values().numel()
                    + features.weights().numel()
                    + features.lengths().numel()
                    + features.offsets().numel()
                )

        class ModuleUseJaggedTensorAsInputAndOutput(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input: JaggedTensor) -> JaggedTensor:
                return JaggedTensor(
                    input.values(),
                    input.weights(),
                    lengths=input.lengths(),
                    offsets=input.offsets(),
                )

        class ModuleUseJaggedTensorAsInput(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input: JaggedTensor) -> int:
                return (
                    input.values().numel()
                    + input.weights().numel()
                    + input.lengths().numel()
                    + input.offsets().numel()
                )

        class ModuleUseJaggedTensorAsOutput(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(
                self,
                values: torch.Tensor,
                weights: torch.Tensor,
                lengths: torch.Tensor,
            ) -> JaggedTensor:
                return JaggedTensor(values, weights, lengths)

        # Case 1: JaggedTensor is only used as an output of the root module.
        m = ModuleUseJaggedTensorAsOutput()
        gm = symbolic_trace(m)
        FileCheck().check("JaggedTensor").check("return jagged_tensor").run(gm.code)

        values = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        weights = torch.tensor([1.0, 0.5, 1.5, 1.0, 0.5, 1.0, 1.0, 1.5])
        lengths = torch.tensor([0, 2, 2, 3, 4, 5, 8])

        ref_jt = m(values, weights, lengths)
        traced_jt = gm(values, weights, lengths)

        self.assertTrue(torch.equal(traced_jt.values(), ref_jt.values()))
        self.assertTrue(torch.equal(traced_jt.weights(), ref_jt.weights()))
        self.assertTrue(torch.equal(traced_jt.lengths(), ref_jt.lengths()))

        # Case 2: JaggedTensor is only used as an input of the root module.
        m = ModuleUseJaggedTensorAsInput()
        gm = symbolic_trace(m)
        FileCheck().check("values()").check("numel()").check("weights").check(
            "lengths"
        ).check("offsets").run(gm.code)

        input = JaggedTensor(
            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
            weights=torch.Tensor([1.0, 0.5, 1.5, 1.0, 0.5, 1.0, 1.0, 1.5]),
            offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8]),
        )
        ref_out = m(input)
        traced_out = gm(input)
        self.assertEqual(ref_out, traced_out)

        # Case 3: JaggedTensor is used as both an input and an output of the root module.
        m = ModuleUseJaggedTensorAsInputAndOutput()
        gm = symbolic_trace(m)
        FileCheck().check("values()").check("weights").check("lengths").check(
            "offsets"
        ).check("JaggedTensor").run(gm.code)

        ref_out = m(input)
        traced_out = gm(input)
        self.assertTrue(torch.equal(traced_out.values(), ref_out.values()))
        self.assertTrue(torch.equal(traced_out.weights(), ref_out.weights()))
        self.assertTrue(torch.equal(traced_out.lengths(), ref_out.lengths()))

        # Case 4: JaggedTensor is only used within the root module and not as part of
        # the root module's input/output interface.
        m = ModuleCreateAndAccessJaggedTensor()
        gm = symbolic_trace(m)
        FileCheck().check("return 29").check_not("JaggedTensor").run(gm.code)
        ref_out = m(8)
        traced_out = gm(8)
        self.assertEqual(ref_out, traced_out)

    def test_keyed_jagged_tensor(self) -> None:
        class ModuleCreateAndAccessKeyedJaggedTensor(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input: int) -> int:
                features = KeyedJaggedTensor.from_offsets_sync(
                    values=torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
                    weights=torch.Tensor([1.0, 0.5, 1.5, 1.0, 0.5, 1.0, 1.0, 1.5]),
                    keys=["index_0", "index_1"],
                    offsets=torch.IntTensor([0, 0, 2, 2, 3, 4, 5, 5, 8]),
                )
                return (
                    len(features.keys())
                    + features.values().numel()
                    + features.weights().numel()
                    + features.lengths().numel()
                    + features.offsets().numel()
                )

        class ModuleUseKeyedJaggedTensorAsInputAndOutput(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(
                self, input: KeyedJaggedTensor
            ) -> Tuple[KeyedJaggedTensor, int]:
                output = KeyedJaggedTensor(
                    input.keys(),
                    input.values(),
                    input.weights(),
                    lengths=input.lengths(),
                    offsets=input.offsets(),
                )
                return output, output._stride

        class ModuleUseKeyedJaggedTensorAsInput(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input: KeyedJaggedTensor) -> int:
                return (
                    len(input.keys())
                    + input.values().numel()
                    + input.weights().numel()
                    + input.lengths().numel()
                    + input.offsets().numel()
                )

        class ModuleUseKeyedJaggedTensorAsOutput(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(
                self,
                keys: List[str],
                values: torch.Tensor,
                weights: torch.Tensor,
                lengths: torch.Tensor,
            ) -> Tuple[KeyedJaggedTensor, int]:
                output = KeyedJaggedTensor(keys, values, weights, lengths)
                return output, output._stride

        # Case 1: KeyedJaggedTensor is only used as an output of the root module.
        m = ModuleUseKeyedJaggedTensorAsOutput()
        gm = symbolic_trace(m)
        FileCheck().check("KeyedJaggedTensor").check(
            "return (keyed_jagged_tensor,"
        ).run(gm.code)

        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        weights = torch.Tensor([1.0, 0.5, 1.5, 1.0, 0.5, 1.0, 1.0, 1.5])
        keys = ["index_0", "index_1"]
        lengths = torch.IntTensor([2, 0, 1, 1, 1, 3])

        ref_out = m(keys, values, weights, lengths)
        traced_out = gm(keys, values, weights, lengths)

        self.assertEqual(ref_out[1], traced_out[1])
        self.assertTrue(torch.equal(traced_out[0].offsets(), ref_out[0].offsets()))

        # Case 2: KeyedJaggedTensor is only used as an input of the root module.
        m = ModuleUseKeyedJaggedTensorAsInput()
        gm = symbolic_trace(m)
        FileCheck().check("KeyedJaggedTensor").check("keys()").check("len").check(
            "values()"
        ).check("numel()").run(gm.code)

        input = KeyedJaggedTensor.from_offsets_sync(
            values=torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            weights=torch.Tensor([1.0, 0.5, 1.5, 1.0, 0.5, 1.0, 1.0, 1.5]),
            keys=["index_0", "index_1"],
            offsets=torch.IntTensor([0, 0, 2, 2, 3, 4, 5, 5, 8]),
        )
        ref_out = m(input)
        traced_out = gm(input)
        self.assertEqual(ref_out, traced_out)

        # Case 3: KeyedJaggedTensor is used as both an input and an output of the root module.
        m = ModuleUseKeyedJaggedTensorAsInputAndOutput()
        gm = symbolic_trace(m)
        FileCheck().check("KeyedJaggedTensor").check("keys()").check("values()").check(
            "._stride"
        ).run(gm.code)
        input = KeyedJaggedTensor.from_offsets_sync(
            values=torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            weights=torch.Tensor([1.0, 0.5, 1.5, 1.0, 0.5, 1.0, 1.0, 1.5]),
            keys=["index_0", "index_1"],
            offsets=torch.IntTensor([0, 0, 2, 2, 3, 4, 5, 5, 8]),
        )
        ref_out = m(input)
        traced_out = gm(input)
        self.assertEqual(ref_out[1], traced_out[1])

        # Case 4: KeyedJaggedTensor is only used within the root module and not as part of
        # the root module's input/output interface.
        m = ModuleCreateAndAccessKeyedJaggedTensor()
        gm = symbolic_trace(m)
        FileCheck().check("return 35").check_not("KeyedJaggedTensor").run(gm.code)
        ref_out = m(8)
        traced_out = gm(8)
        self.assertEqual(ref_out, traced_out)

    def test_trace_async_module(self) -> None:
        class NeedWait(LazyAwaitable[torch.Tensor]):
            def __init__(self, obj: torch.Tensor) -> None:
                super().__init__()
                self._obj = obj

            def wait(self) -> torch.Tensor:
                return self._obj + 3

        class MyAsyncModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input) -> LazyAwaitable[torch.Tensor]:
                return NeedWait(input + 2)

        # Test automated LazyAwaitable type `wait()`
        class AutoModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.sparse = MyAsyncModule()

            def forward(self, input: torch.Tensor) -> torch.Tensor:
                return torch.add(self.sparse(input), input * 10)

        auto_model = AutoModel()
        auto_gm = symbolic_trace(auto_model)
        FileCheck().check("+ 2").check("NeedWait").check("* 10").run(auto_gm.code)

        input = torch.randn(3, 4)
        ref_out = auto_model(input)
        traced_out = auto_gm(input)
        self.assertTrue(torch.equal(ref_out, traced_out))
