#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest
from typing import List, Tuple

import torch
from torch.testing import FileCheck
from torchrec.fx import symbolic_trace
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor, KeyedTensor

torch.fx.wrap("len")


class TestJaggedTensor(unittest.TestCase):
    def test_str(self) -> None:
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        j_1d = JaggedTensor(
            values=values,
            lengths=torch.IntTensor([1, 0, 2, 3]),
        )
        jt_str = str(j_1d)
        expected_str = (
            "JaggedTensor({\n    [[1.0], [], [2.0, 3.0], [4.0, 5.0, 6.0]]\n})\n"
        )
        self.assertEqual(expected_str, jt_str)

        values = torch.Tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
        )
        j_2d = JaggedTensor(
            values=values,
            lengths=torch.IntTensor([1, 0, 2, 0]),
        )
        jt_str = str(j_2d)
        expected_str = "JaggedTensor({\n    [[[1.0, 2.0, 3.0]], [], [[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], []]\n})\n"
        self.assertEqual(expected_str, jt_str)

        values = torch.Tensor(
            [
                [[1.0, 2.0, 3.0]],
                [[4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0]],
                [[10.0, 11.0, 12.0]],
            ]
        )
        j_3d = JaggedTensor(
            values=values,
            lengths=torch.IntTensor([1, 0, 2, 0]),
        )
        with self.assertRaises(ValueError):
            jt_str = str(j_3d)

    def test_from_dense_lengths(self) -> None:
        values = torch.Tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
        )
        j0 = JaggedTensor.from_dense_lengths(
            values=values,
            lengths=torch.IntTensor([1, 0, 2, 3]),
        )
        self.assertTrue(torch.equal(j0.lengths(), torch.IntTensor([1, 0, 2, 3])))
        self.assertTrue(
            torch.equal(j0.values(), torch.Tensor([1.0, 7.0, 8.0, 10.0, 11.0, 12.0]))
        )
        self.assertTrue(j0.weights_or_none() is None)

        traced_from_dense_lengths = torch.fx.symbolic_trace(
            JaggedTensor.from_dense_lengths
        )
        traced_j0 = traced_from_dense_lengths(
            values=values,
            lengths=torch.IntTensor([1, 0, 2, 3]),
        )
        self.assertTrue(torch.equal(traced_j0.lengths(), torch.IntTensor([1, 0, 2, 3])))
        self.assertTrue(
            torch.equal(
                traced_j0.values(), torch.Tensor([1.0, 7.0, 8.0, 10.0, 11.0, 12.0])
            )
        )

    def test_from_dense_lengths_weighted(self) -> None:
        values = torch.Tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
        )
        weights = 12.0 - values

        j1 = JaggedTensor.from_dense_lengths(
            values=values,
            lengths=torch.IntTensor([2, 0, 1, 1]),
            weights=weights,
        )
        self.assertTrue(torch.equal(j1.lengths(), torch.IntTensor([2, 0, 1, 1])))
        self.assertTrue(torch.equal(j1.values(), torch.Tensor([1.0, 2.0, 7.0, 10.0])))
        self.assertTrue(torch.equal(j1.weights(), torch.Tensor([11.0, 10.0, 5.0, 2.0])))

        traced_from_dense_lengths = torch.fx.symbolic_trace(
            JaggedTensor.from_dense_lengths
        )
        traced_j1 = traced_from_dense_lengths(
            values=values,
            lengths=torch.IntTensor([2, 0, 1, 1]),
            weights=weights,
        )
        self.assertTrue(torch.equal(traced_j1.lengths(), torch.IntTensor([2, 0, 1, 1])))
        self.assertTrue(
            torch.equal(traced_j1.values(), torch.Tensor([1.0, 2.0, 7.0, 10.0]))
        )
        self.assertTrue(
            torch.equal(traced_j1.weights(), torch.Tensor([11.0, 10.0, 5.0, 2.0]))
        )

    def test_from_dense(self) -> None:
        values = [
            torch.Tensor([1.0]),
            torch.Tensor(),
            torch.Tensor([7.0, 8.0]),
            torch.Tensor([10.0, 11.0, 12.0]),
        ]

        weights = [
            torch.Tensor([1.0]),
            torch.Tensor(),
            torch.Tensor([7.0, 8.0]),
            torch.Tensor([10.0, 11.0, 12.0]),
        ]
        j0 = JaggedTensor.from_dense(
            values=values,
        )
        self.assertTrue(torch.equal(j0.lengths(), torch.IntTensor([1, 0, 2, 3])))
        self.assertTrue(
            torch.equal(j0.values(), torch.Tensor([1.0, 7.0, 8.0, 10.0, 11.0, 12.0]))
        )
        self.assertTrue(j0.weights_or_none() is None)
        j1 = JaggedTensor.from_dense(
            values=values,
            weights=weights,
        )
        self.assertTrue(torch.equal(j1.offsets(), torch.IntTensor([0, 1, 1, 3, 6])))
        self.assertTrue(
            torch.equal(j1.values(), torch.Tensor([1.0, 7.0, 8.0, 10.0, 11.0, 12.0]))
        )
        self.assertTrue(
            torch.equal(j1.weights(), torch.Tensor([1.0, 7.0, 8.0, 10.0, 11.0, 12.0]))
        )

    def test_to_dense(self) -> None:
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        offsets = torch.IntTensor([0, 2, 2, 3, 4, 5, 8])
        jt = JaggedTensor(
            values=values,
            offsets=offsets,
        )
        torch_list = jt.to_dense()
        expected_list = [
            torch.tensor([1.0, 2.0]),
            torch.tensor([]),
            torch.tensor([3.0]),
            torch.tensor([4.0]),
            torch.tensor([5.0]),
            torch.tensor([6.0, 7.0, 8.0]),
        ]
        for t0, expected_t0 in zip(torch_list, expected_list):
            self.assertTrue(torch.equal(t0, expected_t0))

    def test_to_padded_dense(self) -> None:
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).type(
            torch.float64
        )
        offsets = torch.IntTensor([0, 2, 2, 3, 4, 5, 8])
        jt = JaggedTensor(
            values=values,
            offsets=offsets,
        )
        t0 = jt.to_padded_dense()
        self.assertEqual(t0.dtype, torch.float64)
        t0_value = [
            [1.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [6.0, 7.0, 8.0],
        ]
        expected_t0 = torch.tensor(t0_value).type(torch.float64)
        self.assertTrue(torch.equal(t0, expected_t0))

        t1 = jt.to_padded_dense(desired_length=2, padding_value=10.0)
        self.assertEqual(t1.dtype, torch.float64)
        t1_value = [
            [1.0, 2.0],
            [10.0, 10.0],
            [3.0, 10.0],
            [4.0, 10.0],
            [5.0, 10.0],
            [6.0, 7.0],
        ]
        expected_t1 = torch.tensor(t1_value).type(torch.float64)
        self.assertTrue(torch.equal(t1, expected_t1))

        values = torch.Tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
            ]
        ).type(torch.int64)
        jt = JaggedTensor(
            values=values,
            lengths=torch.IntTensor([1, 0, 2, 0]),
        )
        t2 = jt.to_padded_dense(desired_length=3)
        self.assertEqual(t2.dtype, torch.int64)
        t2_value = [
            [
                [1.0, 2.0, 3.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            [
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        ]
        expected_t2 = torch.tensor(t2_value).type(torch.int64)
        self.assertTrue(torch.equal(t2, expected_t2))

    def test_key_lookup(self) -> None:
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        keys = ["index_0", "index_1"]
        offsets = torch.IntTensor([0, 2, 2, 3, 4, 5, 8])

        jag_tensor = KeyedJaggedTensor(
            values=values,
            keys=keys,
            offsets=offsets,
        )
        j0 = jag_tensor["index_0"]
        j1 = jag_tensor["index_1"]

        self.assertTrue(isinstance(j0, JaggedTensor))
        self.assertTrue(torch.equal(j0.lengths(), torch.IntTensor([2, 0, 1])))
        self.assertTrue(torch.equal(j0.values(), torch.Tensor([1.0, 2.0, 3.0])))
        self.assertTrue(torch.equal(j1.lengths(), torch.IntTensor([1, 1, 3])))
        self.assertTrue(
            torch.equal(j1.values(), torch.Tensor([4.0, 5.0, 6.0, 7.0, 8.0]))
        )

    def test_split(self) -> None:
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        keys = ["index_0", "index_1"]
        offsets = torch.IntTensor([0, 2, 2, 3, 4, 5, 8])

        jag_tensor = KeyedJaggedTensor(
            values=values,
            keys=keys,
            offsets=offsets,
        )
        j0, j1 = jag_tensor.split([1, 1])

        self.assertTrue(isinstance(j0, KeyedJaggedTensor))
        self.assertEqual(j0.keys(), ["index_0"])
        self.assertEqual(j1.keys(), ["index_1"])
        self.assertTrue(torch.equal(j0.lengths(), torch.IntTensor([2, 0, 1])))
        self.assertTrue(torch.equal(j0.values(), torch.Tensor([1.0, 2.0, 3.0])))
        self.assertTrue(torch.equal(j1.lengths(), torch.IntTensor([1, 1, 3])))
        self.assertTrue(
            torch.equal(j1.values(), torch.Tensor([4.0, 5.0, 6.0, 7.0, 8.0]))
        )

    def test_length_vs_offset(self) -> None:
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        keys = ["index_0", "index_1"]
        offsets = torch.IntTensor([0, 0, 2, 2, 3, 4, 5, 5, 8])
        lengths = torch.IntTensor([0, 2, 0, 1, 1, 1, 0, 3])

        j_offset = KeyedJaggedTensor.from_offsets_sync(
            values=values,
            keys=keys,
            offsets=offsets,
        )

        j_lens = KeyedJaggedTensor.from_lengths_sync(
            values=values,
            keys=keys,
            lengths=lengths,
        )

        self.assertTrue(torch.equal(j_offset.lengths(), j_lens.lengths()))
        # TODO: T88149179
        self.assertTrue(torch.equal(j_offset.offsets(), j_lens.offsets().int()))

    def test_empty(self) -> None:
        jt = JaggedTensor.empty()

        self.assertTrue(torch.equal(jt.values(), torch.tensor([])))
        self.assertTrue(torch.equal(jt.offsets(), torch.tensor([])))

    def test_2d(self) -> None:
        values = torch.Tensor([[i * 0.5, i * 1.0, i * 1.5] for i in range(1, 4)])
        offsets = torch.IntTensor([0, 2, 2, 3])

        jt = JaggedTensor(
            values=values,
            offsets=offsets,
        )

        self.assertTrue(torch.equal(jt.lengths(), torch.IntTensor([2, 0, 1])))
        self.assertTrue(
            torch.equal(
                jt.values(),
                torch.Tensor(
                    [
                        [0.5, 1.0, 1.5],
                        [1.0, 2.0, 3.0],
                        [1.5, 3.0, 4.5],
                    ],
                ),
            )
        )

    def test_float_lengths_offsets_throws(self) -> None:
        values = torch.rand((7, 3))
        lengths = torch.tensor([3.0, 4.0])
        offsets = torch.tensor([0.0, 3.0, 7.0])

        with self.assertRaises(AssertionError):
            JaggedTensor(values=values, lengths=lengths)
        with self.assertRaises(AssertionError):
            JaggedTensor(values=values, offsets=offsets)

    def test_to(self) -> None:
        j = JaggedTensor(
            offsets=torch.tensor([0, 2, 2, 3]),
            values=torch.tensor([0.5, 1.0, 1.5]),
            weights=torch.tensor([5.0, 10.0, 15.0]),
        )
        j2 = j.to(device=torch.device("cpu"))
        self.assertTrue(torch.equal(j.offsets(), j2.offsets()))
        self.assertTrue(torch.equal(j.lengths(), j2.lengths()))
        self.assertTrue(torch.equal(j.values(), j2.values()))
        self.assertTrue(torch.equal(j.weights(), j2.weights()))

    # pyre-ignore[56]
    @unittest.skipIf(
        torch.cuda.device_count() <= 0,
        "CUDA is not available",
    )
    def test_record_stream(self) -> None:
        j = JaggedTensor(
            offsets=torch.tensor([0, 2, 2, 3]),
            values=torch.tensor([0.5, 1.0, 1.5]),
            weights=torch.tensor([5.0, 10.0, 15.0]),
        ).to(torch.device("cuda"))
        j.record_stream(torch.cuda.current_stream())

    def test_string_basic(self) -> None:
        values = torch.Tensor([1.0])
        offsets = torch.IntTensor([0, 1])

        jag_tensor = JaggedTensor(
            values=values,
            offsets=offsets,
        )

        self.assertEqual(
            str(jag_tensor),
            """\
JaggedTensor({
    [[1.0]]
})
""",
        )

    def test_string_values(self) -> None:
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        offsets = torch.IntTensor([0, 2, 2, 3, 4, 5, 8])

        jag_tensor = JaggedTensor(
            values=values,
            offsets=offsets,
        )

        self.assertEqual(
            str(jag_tensor),
            """\
JaggedTensor({
    [[1.0, 2.0], [], [3.0], [4.0], [5.0], [6.0, 7.0, 8.0]]
})
""",
        )

    def test_string_weights(self) -> None:
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        weights = torch.Tensor([1.0, 0.5, 1.5, 1.0, 0.5, 1.0, 1.0, 1.5])
        offsets = torch.IntTensor([0, 2, 2, 3, 4, 5, 8])

        jag_tensor = JaggedTensor(
            values=values,
            offsets=offsets,
            weights=weights,
        )

        self.assertEqual(
            str(jag_tensor),
            """\
JaggedTensor({
    "values": [[1.0, 2.0], [], [3.0], [4.0], [5.0], [6.0, 7.0, 8.0]],
    "weights": [[1.0, 0.5], [], [1.5], [1.0], [0.5], [1.0, 1.0, 1.5]]
})
""",
        )


class TestJaggedTensorTracing(unittest.TestCase):
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


class TestKeyedJaggedTensor(unittest.TestCase):
    def test_key_lookup(self) -> None:
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        weights = torch.Tensor([1.0, 0.5, 1.5, 1.0, 0.5, 1.0, 1.0, 1.5])
        keys = ["index_0", "index_1"]
        offsets = torch.IntTensor([0, 2, 2, 3, 4, 5, 8])

        jag_tensor = KeyedJaggedTensor(
            values=values,
            keys=keys,
            offsets=offsets,
            weights=weights,
        )
        j0 = jag_tensor["index_0"]
        j1 = jag_tensor["index_1"]

        self.assertTrue(isinstance(j0, JaggedTensor))
        self.assertTrue(torch.equal(j0.lengths(), torch.IntTensor([2, 0, 1])))
        self.assertTrue(torch.equal(j0.weights(), torch.Tensor([1.0, 0.5, 1.5])))
        self.assertTrue(torch.equal(j0.values(), torch.Tensor([1.0, 2.0, 3.0])))
        self.assertTrue(torch.equal(j1.lengths(), torch.IntTensor([1, 1, 3])))
        self.assertTrue(
            torch.equal(j1.weights(), torch.Tensor([1.0, 0.5, 1.0, 1.0, 1.5]))
        )
        self.assertTrue(
            torch.equal(j1.values(), torch.Tensor([4.0, 5.0, 6.0, 7.0, 8.0]))
        )

    def test_to_dict(self) -> None:
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        weights = torch.Tensor([1.0, 0.5, 1.5, 1.0, 0.5, 1.0, 1.0, 1.5])
        keys = ["index_0", "index_1"]
        offsets = torch.IntTensor([0, 2, 2, 3, 4, 5, 8])

        jag_tensor = KeyedJaggedTensor(
            values=values,
            keys=keys,
            offsets=offsets,
            weights=weights,
        )
        jag_tensor_dict = jag_tensor.to_dict()
        j0 = jag_tensor_dict["index_0"]
        j1 = jag_tensor_dict["index_1"]

        self.assertTrue(isinstance(j0, JaggedTensor))
        self.assertTrue(torch.equal(j0.lengths(), torch.IntTensor([2, 0, 1])))
        self.assertTrue(torch.equal(j0.weights(), torch.Tensor([1.0, 0.5, 1.5])))
        self.assertTrue(torch.equal(j0.values(), torch.Tensor([1.0, 2.0, 3.0])))
        self.assertTrue(torch.equal(j1.lengths(), torch.IntTensor([1, 1, 3])))
        self.assertTrue(
            torch.equal(j1.weights(), torch.Tensor([1.0, 0.5, 1.0, 1.0, 1.5]))
        )
        self.assertTrue(
            torch.equal(j1.values(), torch.Tensor([4.0, 5.0, 6.0, 7.0, 8.0]))
        )

    def test_empty(self) -> None:
        keys = ["index_0"]
        values = torch.tensor([])
        lengths = torch.tensor([])
        offsets = torch.tensor([])

        kjt_0 = KeyedJaggedTensor(keys=keys, values=values, lengths=lengths)
        j0 = kjt_0["index_0"]
        self.assertTrue(isinstance(j0, JaggedTensor))
        self.assertTrue(torch.equal(j0.lengths(), torch.Tensor([])))
        self.assertTrue(torch.equal(j0.values(), torch.Tensor([])))

        keys = ["index_1"]
        kjt_1 = KeyedJaggedTensor(keys=keys, values=values, offsets=offsets)
        j1 = kjt_1["index_1"]

        self.assertTrue(isinstance(j1, JaggedTensor))
        self.assertTrue(torch.equal(j1.lengths(), torch.Tensor([])))
        self.assertTrue(torch.equal(j1.values(), torch.Tensor([])))

        combined_kjt = KeyedJaggedTensor.concat([kjt_0, kjt_1])
        j0 = combined_kjt["index_0"]
        j1 = combined_kjt["index_1"]

        self.assertTrue(isinstance(j0, JaggedTensor))
        self.assertTrue(torch.equal(j0.lengths(), torch.Tensor([])))
        self.assertTrue(torch.equal(j0.values(), torch.Tensor([])))
        self.assertTrue(isinstance(j1, JaggedTensor))
        self.assertTrue(torch.equal(j1.lengths(), torch.Tensor([])))
        self.assertTrue(torch.equal(j1.values(), torch.Tensor([])))

        kjt_2 = KeyedJaggedTensor.empty()
        self.assertEqual(kjt_2.to_dict(), {})

    def test_empty_to_dict(self) -> None:
        keys = ["index_0", "index_1"]
        values = torch.tensor([])
        lengths = torch.tensor([[], []])
        length_per_key = [0, 0]

        jag_tensor = KeyedJaggedTensor(
            keys=keys, values=values, lengths=lengths, length_per_key=length_per_key
        )
        jag_tensor_dict = jag_tensor.to_dict()
        j0 = jag_tensor_dict["index_0"]
        j1 = jag_tensor_dict["index_1"]

        self.assertTrue(isinstance(j0, JaggedTensor))
        self.assertTrue(torch.equal(j0.lengths(), torch.Tensor([])))
        self.assertTrue(torch.equal(j0.offsets(), torch.Tensor([])))
        self.assertTrue(torch.equal(j0.values(), torch.Tensor([])))
        self.assertTrue(isinstance(j1, JaggedTensor))
        self.assertTrue(torch.equal(j1.lengths(), torch.Tensor([])))
        self.assertTrue(torch.equal(j1.offsets(), torch.Tensor([])))
        self.assertTrue(torch.equal(j1.values(), torch.Tensor([])))

        jag_tensor = KeyedJaggedTensor.from_lengths_sync(
            keys=keys, values=values, lengths=lengths
        )
        jag_tensor_dict = jag_tensor.to_dict()
        j0 = jag_tensor_dict["index_0"]
        j1 = jag_tensor_dict["index_1"]

        self.assertTrue(isinstance(j0, JaggedTensor))
        self.assertTrue(torch.equal(j0.lengths(), torch.Tensor([])))
        self.assertTrue(torch.equal(j0.offsets(), torch.Tensor([])))
        self.assertTrue(torch.equal(j0.values(), torch.Tensor([])))
        self.assertTrue(isinstance(j1, JaggedTensor))
        self.assertTrue(torch.equal(j1.lengths(), torch.Tensor([])))
        self.assertTrue(torch.equal(j1.offsets(), torch.Tensor([])))
        self.assertTrue(torch.equal(j1.values(), torch.Tensor([])))

    def test_split(self) -> None:
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        weights = torch.Tensor([1.0, 0.5, 1.5, 1.0, 0.5, 1.0, 1.0, 1.5])
        keys = ["index_0", "index_1"]
        offsets = torch.IntTensor([0, 2, 2, 3, 4, 5, 8])

        jag_tensor = KeyedJaggedTensor(
            values=values,
            keys=keys,
            offsets=offsets,
            weights=weights,
        )
        j0, j1 = jag_tensor.split([1, 1])

        self.assertTrue(isinstance(j0, KeyedJaggedTensor))
        self.assertEqual(j0.keys(), ["index_0"])
        self.assertEqual(j1.keys(), ["index_1"])
        self.assertTrue(torch.equal(j0.lengths(), torch.IntTensor([2, 0, 1])))
        self.assertTrue(torch.equal(j0.weights(), torch.Tensor([1.0, 0.5, 1.5])))
        self.assertTrue(torch.equal(j0.values(), torch.Tensor([1.0, 2.0, 3.0])))
        self.assertTrue(torch.equal(j1.lengths(), torch.IntTensor([1, 1, 3])))
        self.assertTrue(
            torch.equal(j1.weights(), torch.Tensor([1.0, 0.5, 1.0, 1.0, 1.5]))
        )
        self.assertTrue(
            torch.equal(j1.values(), torch.Tensor([4.0, 5.0, 6.0, 7.0, 8.0]))
        )

    def test_zero_split(self) -> None:
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        weights = torch.Tensor([1.0, 0.5, 1.5, 1.0, 0.5, 1.0, 1.0, 1.5])
        keys = ["index_0", "index_1"]
        offsets = torch.IntTensor([0, 2, 2, 3, 4, 5, 8])

        jag_tensor = KeyedJaggedTensor(
            values=values,
            keys=keys,
            offsets=offsets,
            weights=weights,
        )
        j0, j1 = jag_tensor.split([0, 2])

        self.assertTrue(isinstance(j0, KeyedJaggedTensor))
        self.assertEqual(j0.keys(), [])
        self.assertTrue(torch.equal(j0.lengths(), torch.IntTensor([])))
        self.assertTrue(torch.equal(j0.weights(), torch.Tensor([])))
        self.assertTrue(torch.equal(j0.values(), torch.Tensor([])))
        self.assertEqual(j0.stride(), 3)

        self.assertEqual(j1.keys(), ["index_0", "index_1"])
        self.assertTrue(torch.equal(j1.lengths(), torch.IntTensor([2, 0, 1, 1, 1, 3])))
        self.assertTrue(torch.equal(j1.weights(), weights))
        self.assertTrue(torch.equal(j1.values(), values))
        self.assertEqual(j0.stride(), 3)

    def test_permute_w_weights(self) -> None:
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        weights = torch.Tensor([1.0, 0.5, 1.5, 1.0, 0.5, 1.0, 1.0, 1.5])
        lengths = torch.IntTensor([0, 2, 0, 1, 1, 1, 0, 3, 0])
        keys = ["index_0", "index_1", "index_2"]

        jag_tensor = KeyedJaggedTensor.from_lengths_sync(
            values=values,
            keys=keys,
            lengths=lengths,
            weights=weights,
        )

        indices = [1, 0, 2]
        permuted_jag_tensor = jag_tensor.permute(indices)
        self.assertEqual(permuted_jag_tensor.keys(), ["index_1", "index_0", "index_2"])
        self.assertEqual(
            permuted_jag_tensor.offset_per_key(),
            [0, 3, 5, 8],
        )
        self.assertTrue(
            torch.equal(
                permuted_jag_tensor.values(),
                torch.Tensor([3.0, 4.0, 5.0, 1.0, 2.0, 6.0, 7.0, 8.0]),
            )
        )
        self.assertTrue(
            torch.equal(
                permuted_jag_tensor.lengths(),
                torch.IntTensor([1, 1, 1, 0, 2, 0, 0, 3, 0]),
            )
        )
        self.assertTrue(
            torch.equal(
                permuted_jag_tensor.weights(),
                torch.Tensor([1.5, 1.0, 0.5, 1.0, 0.5, 1.0, 1.0, 1.5]),
            ),
        )

    def test_permute(self) -> None:
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        lengths = torch.IntTensor([0, 2, 0, 1, 1, 1, 0, 3, 0])
        keys = ["index_0", "index_1", "index_2"]

        jag_tensor = KeyedJaggedTensor.from_lengths_sync(
            values=values,
            keys=keys,
            lengths=lengths,
        )

        indices = [1, 0, 2]
        permuted_jag_tensor = jag_tensor.permute(indices)

        self.assertEqual(permuted_jag_tensor.keys(), ["index_1", "index_0", "index_2"])
        self.assertEqual(
            permuted_jag_tensor.offset_per_key(),
            [0, 3, 5, 8],
        )
        self.assertTrue(
            torch.equal(
                permuted_jag_tensor.values(),
                torch.Tensor([3.0, 4.0, 5.0, 1.0, 2.0, 6.0, 7.0, 8.0]),
            )
        )
        self.assertTrue(
            torch.equal(
                permuted_jag_tensor.lengths(),
                torch.IntTensor([1, 1, 1, 0, 2, 0, 0, 3, 0]),
            )
        )
        self.assertEqual(permuted_jag_tensor.weights_or_none(), None)

    def test_permute_duplicates(self) -> None:
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        lengths = torch.IntTensor([0, 2, 0, 1, 1, 1, 0, 3, 0])
        keys = ["index_0", "index_1", "index_2"]

        jag_tensor = KeyedJaggedTensor.from_lengths_sync(
            values=values,
            keys=keys,
            lengths=lengths,
        )

        indices = [1, 0, 2, 1, 1]
        permuted_jag_tensor = jag_tensor.permute(indices)

        self.assertEqual(
            permuted_jag_tensor.keys(),
            ["index_1", "index_0", "index_2", "index_1", "index_1"],
        )
        self.assertEqual(
            permuted_jag_tensor.offset_per_key(),
            [0, 3, 5, 8, 11, 14],
        )
        self.assertTrue(
            torch.equal(
                permuted_jag_tensor.values(),
                torch.Tensor(
                    [
                        3.0,
                        4.0,
                        5.0,
                        1.0,
                        2.0,
                        6.0,
                        7.0,
                        8.0,
                        3.0,
                        4.0,
                        5.0,
                        3.0,
                        4.0,
                        5.0,
                    ]
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                permuted_jag_tensor.lengths(),
                torch.IntTensor([1, 1, 1, 0, 2, 0, 0, 3, 0, 1, 1, 1, 1, 1, 1]),
            )
        )
        self.assertEqual(permuted_jag_tensor.weights_or_none(), None)

    def test_concat(self) -> None:
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        keys = ["index_0", "index_1", "index_2"]
        lengths = torch.IntTensor([0, 2, 0, 1, 1, 1, 0, 3, 0, 0, 1, 0])

        kjt_expected = KeyedJaggedTensor.from_lengths_sync(
            values=values,
            keys=keys,
            lengths=lengths,
        )
        kjt_actual = KeyedJaggedTensor.concat(
            [
                KeyedJaggedTensor.from_lengths_sync(
                    values=values[:4],
                    keys=keys[:1],
                    lengths=lengths[:4],
                ),
                KeyedJaggedTensor.from_lengths_sync(
                    values=values[4:],
                    keys=keys[1:],
                    lengths=lengths[4:],
                ),
            ],
        )
        self.assertTrue(torch.equal(kjt_expected.lengths(), kjt_actual.lengths()))
        self.assertTrue(torch.equal(kjt_expected.offsets(), kjt_actual.offsets()))
        self.assertTrue(torch.equal(kjt_expected.values(), kjt_actual.values()))
        # pyre-ignore[6]
        self.assertListEqual(kjt_expected._length_per_key, kjt_actual._length_per_key)

    def test_length_vs_offset(self) -> None:
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        weights = torch.Tensor([1.0, 0.5, 1.5, 1.0, 0.5, 1.0, 1.0, 1.5])
        keys = ["index_0", "index_1"]
        offsets = torch.IntTensor([0, 0, 2, 2, 3, 4, 5, 5, 8])
        lengths = torch.IntTensor([0, 2, 0, 1, 1, 1, 0, 3])

        j_offset = KeyedJaggedTensor.from_offsets_sync(
            values=values,
            keys=keys,
            offsets=offsets,
            weights=weights,
        )

        j_lens = KeyedJaggedTensor.from_lengths_sync(
            values=values,
            keys=keys,
            lengths=lengths,
            weights=weights,
        )

        self.assertTrue(torch.equal(j_offset.lengths(), j_lens.lengths()))
        # TO DO: T88149179
        self.assertTrue(torch.equal(j_offset.offsets(), j_lens.offsets().int()))

    def test_2d(self) -> None:
        values = torch.Tensor([[i * 0.5, i * 1.0, i * 1.5] for i in range(1, 9)])
        weights = torch.Tensor([1.0, 0.5, 1.5, 1.0, 0.5, 1.0, 1.0, 1.5])
        keys = ["index_0", "index_1"]
        offsets = torch.IntTensor([0, 2, 2, 3, 4, 5, 8])

        j = KeyedJaggedTensor.from_offsets_sync(
            values=values,
            weights=weights,
            keys=keys,
            offsets=offsets,
        )
        j_0 = j["index_0"]

        self.assertTrue(torch.equal(j_0.lengths(), torch.IntTensor([2, 0, 1])))
        self.assertTrue(
            torch.equal(
                j_0.values(),
                torch.Tensor(
                    [
                        [0.5, 1.0, 1.5],
                        [1.0, 2.0, 3.0],
                        [1.5, 3.0, 4.5],
                    ],
                ),
            )
        )

    def test_float_lengths_offsets_throws(self) -> None:
        values = torch.rand((7, 3))
        keys = ["f1", "f2"]
        # torch.Tensor([3, 4]) also fails
        # pyre-fixme[6]: Expected `Optional[typing.Type[torch._dtype]]` for 2nd
        #  param but got `Type[float]`.
        lengths = torch.tensor([3, 4], dtype=float)
        # pyre-fixme[6]: Expected `Optional[typing.Type[torch._dtype]]` for 2nd
        #  param but got `Type[float]`.
        offsets = torch.tensor([0, 3, 7], dtype=float)

        with self.assertRaises(AssertionError):
            KeyedJaggedTensor.from_lengths_sync(
                keys=keys, values=values, lengths=lengths
            )
        with self.assertRaises(AssertionError):
            KeyedJaggedTensor.from_offsets_sync(
                keys=keys, values=values, offsets=offsets
            )

    def test_scriptable(self) -> None:
        class MyModule(torch.nn.Module):
            def forward(self, input: KeyedJaggedTensor) -> torch.Tensor:
                values = input["any"].values()
                return values

        m = MyModule()
        torch.jit.script(m)

    def test_to(self) -> None:
        j = KeyedJaggedTensor.from_offsets_sync(
            offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8]),
            values=torch.arange(8),
            weights=torch.arange(8 * 10),
            keys=["index_0", "index_1"],
        )
        j2 = j.to(device=torch.device("cpu"))
        self.assertTrue(torch.equal(j.offsets(), j2.offsets()))
        self.assertTrue(torch.equal(j.lengths(), j2.lengths()))
        self.assertTrue(torch.equal(j.values(), j2.values()))
        self.assertTrue(torch.equal(j.weights(), j2.weights()))

    def test_string_none(self) -> None:
        jag_tensor = KeyedJaggedTensor(
            # pyre-fixme[6]: For 1st param expected `List[str]` but got `Tensor`.
            torch.Tensor(),
            # pyre-fixme[6]: For 2nd param expected `Tensor` but got
            #  `List[Variable[_T]]`.
            [],
        )

        self.assertEqual(
            str(jag_tensor),
            """\
KeyedJaggedTensor()
""",
        )

    def test_string_basic(self) -> None:
        values = torch.Tensor([1.0])
        keys = ["key"]
        offsets = torch.IntTensor([0, 1])

        jag_tensor = KeyedJaggedTensor(
            values=values,
            keys=keys,
            offsets=offsets,
        )

        self.assertEqual(
            str(jag_tensor),
            """\
KeyedJaggedTensor({
    "key": [[1.0]]
})
""",
        )

    def test_string_values(self) -> None:
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        keys = ["index_0", "index_1"]
        offsets = torch.IntTensor([0, 2, 2, 3, 4, 5, 8])

        jag_tensor = KeyedJaggedTensor(
            values=values,
            keys=keys,
            offsets=offsets,
        )

        self.assertEqual(
            str(jag_tensor),
            """\
KeyedJaggedTensor({
    "index_0": [[1.0, 2.0], [], [3.0]],
    "index_1": [[4.0], [5.0], [6.0, 7.0, 8.0]]
})
""",
        )

    def test_string_weights(self) -> None:
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        weights = torch.Tensor([1.0, 0.5, 1.5, 1.0, 0.5, 1.0, 1.0, 1.5])
        keys = ["index_0", "index_1"]
        offsets = torch.IntTensor([0, 2, 2, 3, 4, 5, 8])

        jag_tensor = KeyedJaggedTensor(
            values=values,
            keys=keys,
            offsets=offsets,
            weights=weights,
        )

        self.assertEqual(
            str(jag_tensor),
            """\
KeyedJaggedTensor({
    "index_0": {
        "values": [[1.0, 2.0], [], [3.0]],
        "weights": [[1.0, 0.5], [], [1.5]]
    },
    "index_1": {
        "values": [[4.0], [5.0], [6.0, 7.0, 8.0]],
        "weights": [[1.0], [0.5], [1.0, 1.0, 1.5]]
    }
})
""",
        )

    # pyre-ignore[56]
    @unittest.skipIf(
        torch.cuda.device_count() <= 0,
        "CUDA is not available",
    )
    def test_record_stream(self) -> None:
        j = KeyedJaggedTensor.from_offsets_sync(
            offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8]),
            values=torch.arange(8),
            weights=torch.arange(8 * 10),
            keys=["index_0", "index_1"],
        ).to(torch.device("cuda"))
        j.record_stream(torch.cuda.current_stream())


class TestKeyedJaggedTensorScripting(unittest.TestCase):
    def test_scriptable_forward(self) -> None:
        class MyModule(torch.nn.Module):
            def forward(self, input: KeyedJaggedTensor) -> torch.Tensor:
                values = input["any"].values()
                return values

        m = MyModule()
        torch.jit.script(m)

    def test_scriptable_init(self) -> None:
        def create_kjt() -> KeyedJaggedTensor:
            return KeyedJaggedTensor.from_offsets_sync(
                values=torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
                weights=torch.tensor([1.0, 0.5, 1.5, 1.0, 0.5, 1.0, 1.0, 1.5]),
                keys=["index_0", "index_1"],
                offsets=torch.tensor([0, 0, 2, 2, 3, 4, 5, 5, 8], dtype=torch.int32),
            )

        # assert that we can script KJT creation
        torch.jit.script(create_kjt)


class TestKeyedJaggedTensorTracingScripting(unittest.TestCase):
    def test_jit_tracable(self) -> None:
        # This module will simply go through the constructor of the
        # KeyedJaggedTensor to construct it with multiple different batch sizes
        class MyModule(torch.nn.Module):
            def forward(
                self, offsets: torch.Tensor, values: torch.Tensor, weights: torch.Tensor
            ) -> torch.Tensor:
                j = KeyedJaggedTensor.from_offsets_sync(
                    offsets=offsets,
                    values=values,
                    weights=weights,
                    keys=["index_0", "index_1"],
                )
                return j["index_0"].offsets()

        sample_2 = (
            torch.tensor([0, 2, 2]),
            torch.arange(2),
            torch.arange(2 * 10),
        )
        sample_6 = (
            torch.tensor([0, 2, 2, 3, 4, 6, 8]),
            torch.arange(8),
            torch.arange(8 * 10),
        )
        m = MyModule()
        model_eager_traced: torch.jit.ScriptModule = torch.jit.trace(
            m, sample_2, strict=False
        )
        self.assertTrue(
            torch.equal(model_eager_traced(*sample_2), torch.tensor([0, 2]))
        )
        self.assertTrue(
            torch.equal(model_eager_traced(*sample_6), torch.tensor([0, 2, 2, 3]))
        )

    def test_keyed_jagged_tensor_to_dict_jit_tracable(self) -> None:
        # This module will go through the constructor of the
        # KeyedJaggedTensor to construct it with multiple different batch sizes and call to_dict()
        class MyModule(torch.nn.Module):
            def forward(
                self, lengths: torch.Tensor, values: torch.Tensor, weights: torch.Tensor
            ) -> torch.Tensor:
                j = KeyedJaggedTensor.from_lengths_sync(
                    lengths=lengths,
                    values=values,
                    weights=weights,
                    keys=["index_0", "index_1"],
                )

                jt_dict = j.to_dict()
                return jt_dict["index_0"].values()

        sample_2 = (
            torch.tensor([2, 0]),
            torch.arange(2),
            torch.arange(2),
        )
        sample_6 = (
            torch.tensor([2, 0, 1, 1, 2, 2]),
            torch.arange(8),
            torch.arange(8),
        )
        m = MyModule()
        model_eager_traced: torch.jit.ScriptModule = torch.jit.trace(
            m, sample_2, strict=False
        )
        self.assertTrue(
            torch.equal(model_eager_traced(*sample_2), torch.tensor([0, 1]))
        )
        self.assertTrue(
            torch.equal(model_eager_traced(*sample_6), torch.tensor([0, 1, 2]))
        )
        self.assertTrue(torch.equal(m(*sample_6), torch.tensor([0, 1, 2])))

    def test_create_and_access_keyed_jagged_tensor(self) -> None:
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

        # Case 4: KeyedJaggedTensor is only used within the root module and not as part of
        # the root module's input/output interface.
        m = ModuleCreateAndAccessKeyedJaggedTensor()
        gm = symbolic_trace(m)
        FileCheck().check("return 35").check_not("KeyedJaggedTensor").run(gm.code)
        ref_out = m(8)
        traced_out = gm(8)
        self.assertEqual(ref_out, traced_out)
        torch.jit.script(gm)

    def test_use_keyed_jagged_tensor_as_input_and_output(self) -> None:
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
        torch.jit.script(gm)

    def test_use_keyed_jagged_tensor_as_input(self) -> None:
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
        torch.jit.script(gm)

    def test_use_keyed_jagged_tensor_as_output(self) -> None:
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
        torch.jit.script(gm)


class TestKeyedTensor(unittest.TestCase):
    def test_key_lookup(self) -> None:
        tensor_list = [
            torch.Tensor([[1.0, 1.0]]),
            torch.Tensor([[2.0, 2.0], [3.0, 3.0]]),
        ]
        keys = ["dense_0", "dense_1"]
        kt = KeyedTensor.from_tensor_list(keys, tensor_list, cat_dim=0, key_dim=0)
        self.assertEqual(kt.key_dim(), 0)

        self.assertTrue(torch.equal(kt["dense_0"], tensor_list[0]))
        self.assertTrue(torch.equal(kt["dense_1"], tensor_list[1]))

    def test_key_lookup_dim_1(self) -> None:
        tensor_list = [
            torch.tensor([[1.0, 1.0]]).T,
            torch.tensor([[2.0, 2.0], [3.0, 3.0]]).T,
        ]
        keys = ["dense_0", "dense_1"]
        kt = KeyedTensor.from_tensor_list(keys, tensor_list, key_dim=1)
        self.assertEqual(kt.key_dim(), 1)
        self.assertTrue(torch.equal(kt["dense_0"], tensor_list[0]))
        self.assertTrue(torch.equal(kt["dense_1"], tensor_list[1]))

    def test_to_dict(self) -> None:
        tensor_list = [
            torch.Tensor([[1.0, 1.0]]),
            torch.Tensor([[2.0, 2.0], [3.0, 3.0]]),
        ]
        keys = ["dense_0", "dense_1"]
        kt = KeyedTensor.from_tensor_list(keys, tensor_list, cat_dim=0, key_dim=0)
        self.assertEqual(kt.key_dim(), 0)

        d = kt.to_dict()
        for key in keys:
            self.assertTrue(torch.equal(kt[key], d[key]))

    def test_to_dict_dim_1(self) -> None:
        tensor_list = [
            torch.tensor([[1.0, 1.0]]).T,
            torch.tensor([[2.0, 2.0], [3.0, 3.0]]).T,
        ]
        keys = ["dense_0", "dense_1"]
        kt = KeyedTensor.from_tensor_list(keys, tensor_list, key_dim=1)
        self.assertEqual(kt.key_dim(), 1)

        d = kt.to_dict()
        for key in keys:
            self.assertTrue(torch.equal(kt[key], d[key]))

    def test_regroup_single_kt(self) -> None:
        tensor_list = [torch.randn(2, 3) for i in range(5)]
        key_dim = 1
        keys = ["dense_0", "dense_1", "dense_2", "dense_3", "dense_4"]
        kt = KeyedTensor.from_tensor_list(keys, tensor_list, key_dim)
        grouped_tensors = KeyedTensor.regroup(
            [kt], [["dense_0", "dense_4"], ["dense_1", "dense_3"], ["dense_2"]]
        )
        self.assertTrue(
            torch.equal(
                grouped_tensors[0], torch.cat([tensor_list[0], tensor_list[4]], key_dim)
            )
        )
        self.assertTrue(
            torch.equal(
                grouped_tensors[1], torch.cat([tensor_list[1], tensor_list[3]], key_dim)
            )
        )
        self.assertTrue(torch.equal(grouped_tensors[2], tensor_list[2]))

    def test_regroup_multiple_kt(self) -> None:
        key_dim = 1
        tensor_list_1 = [torch.randn(2, 3) for i in range(3)]
        keys_1 = ["dense_0", "dense_1", "dense_2"]
        kt_1 = KeyedTensor.from_tensor_list(keys_1, tensor_list_1, key_dim)
        tensor_list_2 = [torch.randn(2, 3) for i in range(2)]
        keys_2 = ["sparse_0", "sparse_1"]
        kt_2 = KeyedTensor.from_tensor_list(keys_2, tensor_list_2, key_dim)
        grouped_tensors = KeyedTensor.regroup(
            [kt_1, kt_2], [["dense_0", "sparse_1", "dense_2"], ["dense_1", "sparse_0"]]
        )
        self.assertTrue(
            torch.equal(
                grouped_tensors[0],
                torch.cat(
                    [tensor_list_1[0], tensor_list_2[1], tensor_list_1[2]], key_dim
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                grouped_tensors[1],
                torch.cat([tensor_list_1[1], tensor_list_2[0]], key_dim),
            )
        )

    def test_regroup_scriptable(self) -> None:
        class MyModule(torch.nn.Module):
            def forward(
                self, inputs: List[KeyedTensor], groups: List[List[str]]
            ) -> List[torch.Tensor]:
                return KeyedTensor.regroup(inputs, groups)

        m = MyModule()
        torch.jit.script(m)

    def test_regroup_fxable(self) -> None:
        class MyModule(torch.nn.Module):
            def forward(
                self, inputs: List[KeyedTensor], groups: List[List[str]]
            ) -> List[torch.Tensor]:
                return KeyedTensor.regroup(inputs, groups)

        m = MyModule()

        # input
        key_dim = 1
        tensor_list_1 = [torch.randn(2, 3) for i in range(3)]
        keys_1 = ["dense_0", "dense_1", "dense_2"]
        kt_1 = KeyedTensor.from_tensor_list(keys_1, tensor_list_1, key_dim)
        tensor_list_2 = [torch.randn(2, 3) for i in range(2)]
        keys_2 = ["sparse_0", "sparse_1"]
        kt_2 = KeyedTensor.from_tensor_list(keys_2, tensor_list_2, key_dim)
        inputs = [kt_1, kt_2]
        groups = [["dense_0", "sparse_1", "dense_2"], ["dense_1", "sparse_0"]]

        # ensure that symbolic tracing works
        gm = torch.fx.symbolic_trace(m)
        results = m(inputs, groups)
        traced_results = gm(inputs, groups)
        self.assertEqual(len(results), len(traced_results))
        for result, traced_result in zip(results, traced_results):
            self.assertTrue(torch.equal(result, traced_result))

    def test_scriptable(self) -> None:
        class MyModule(torch.nn.Module):
            def forward(self, input: KeyedTensor) -> torch.Tensor:
                values = input["any"].values()
                return values

        m = MyModule()
        torch.jit.script(m)

    def test_string_none(self) -> None:
        jag_tensor = KeyedTensor(
            [],
            [],
            torch.Tensor(),
        )

        self.assertEqual(
            str(jag_tensor),
            """\
KeyedTensor()
""",
        )

    def test_string_basic(self) -> None:
        tensor_list = [
            torch.tensor([[1.0]]),
        ]
        keys = ["key"]
        kt = KeyedTensor.from_tensor_list(keys, tensor_list, key_dim=0)

        self.assertEqual(
            str(kt),
            """\
KeyedTensor({
    "key": [[1.0]]
})
""",
        )

    def test_string_values(self) -> None:
        tensor_list = [
            torch.tensor([[1.0, 1.0]]).T,
            torch.tensor([[2.0, 2.0], [3.0, 3.0]]).T,
        ]
        keys = ["dense_0", "dense_1"]
        kt = KeyedTensor.from_tensor_list(keys, tensor_list)

        self.assertEqual(
            str(kt),
            """\
KeyedTensor({
    "dense_0": [[1.0], [1.0]],
    "dense_1": [[2.0, 3.0], [2.0, 3.0]]
})
""",
        )
