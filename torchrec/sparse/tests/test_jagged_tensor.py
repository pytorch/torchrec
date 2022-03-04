#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest
from typing import List

import torch
from torchrec.sparse.jagged_tensor import (
    JaggedTensor,
    KeyedTensor,
    KeyedJaggedTensor,
)


class TestJaggedTensor(unittest.TestCase):
    def test_from_dense_lengths(self) -> None:
        values = torch.Tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
        )
        weights = 12.0 - values
        j0 = JaggedTensor.from_dense_lengths(
            values=values,
            lengths=torch.IntTensor([1, 0, 2, 3]),
        )
        self.assertTrue(torch.equal(j0.lengths(), torch.IntTensor([1, 0, 2, 3])))
        self.assertTrue(
            torch.equal(j0.values(), torch.Tensor([1.0, 7.0, 8.0, 10.0, 11.0, 12.0]))
        )
        self.assertTrue(j0.weights_or_none() is None)
        j1 = JaggedTensor.from_dense_lengths(
            values=values,
            lengths=torch.IntTensor([2, 0, 1, 1]),
            weights=weights,
        )
        self.assertTrue(torch.equal(j1.lengths(), torch.IntTensor([2, 0, 1, 1])))
        self.assertTrue(torch.equal(j1.values(), torch.Tensor([1.0, 2.0, 7.0, 10.0])))
        self.assertTrue(torch.equal(j1.weights(), torch.Tensor([11.0, 10.0, 5.0, 2.0])))

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
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        offsets = torch.IntTensor([0, 2, 2, 3, 4, 5, 8])
        jt = JaggedTensor(
            values=values,
            offsets=offsets,
        )
        t0 = jt.to_padded_dense()
        t0_value = [
            [0.0, 1.0, 2.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 3.0],
            [0.0, 0.0, 4.0],
            [0.0, 0.0, 5.0],
            [6.0, 7.0, 8.0],
        ]
        expected_t0 = torch.tensor(t0_value)
        self.assertTrue(torch.equal(t0, expected_t0))
        t1 = jt.to_padded_dense(desired_length=2, padding_value=10.0)
        t1_value = [
            [1.0, 2.0],
            [10.0, 10.0],
            [10.0, 3.0],
            [10.0, 4.0],
            [10.0, 5.0],
            [7.0, 8.0],
        ]
        expected_t1 = torch.tensor(t1_value)
        self.assertTrue(torch.equal(t1, expected_t1))
        t2 = jt.to_padded_dense(
            desired_length=2, padding_value=10.0, pad_from_beginning=False
        )
        t2_value = [
            [1.0, 2.0],
            [10.0, 10.0],
            [3.0, 10.0],
            [4.0, 10.0],
            [5.0, 10.0],
            [7.0, 8.0],
        ]
        expected_t2 = torch.tensor(t2_value)
        self.assertTrue(torch.equal(t2, expected_t2))
        t3 = jt.to_padded_dense(
            desired_length=2,
            padding_value=10.0,
            pad_from_beginning=False,
            chop_from_beginning=False,
        )
        t3_value = [
            [1.0, 2.0],
            [10.0, 10.0],
            [3.0, 10.0],
            [4.0, 10.0],
            [5.0, 10.0],
            [6.0, 7.0],
        ]
        expected_t3 = torch.tensor(t3_value)
        self.assertTrue(torch.equal(t3, expected_t3))

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
            a=KeyedJaggedTensor.from_lengths_sync(
                values=values[:4],
                keys=keys[:1],
                lengths=lengths[:4],
            ),
            b=KeyedJaggedTensor.from_lengths_sync(
                values=values[4:],
                keys=keys[1:],
                lengths=lengths[4:],
            ),
        )
        self.assertTrue(torch.equal(kjt_expected.lengths(), kjt_actual.lengths()))
        self.assertTrue(torch.equal(kjt_expected.offsets(), kjt_actual.offsets()))
        self.assertTrue(torch.equal(kjt_expected.values(), kjt_actual.values()))

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
            a=KeyedJaggedTensor.from_lengths_sync(
                values=values[:4],
                keys=keys[:1],
                lengths=lengths[:4],
            ),
            b=KeyedJaggedTensor.from_lengths_sync(
                values=values[4:],
                keys=keys[1:],
                lengths=lengths[4:],
            ),
        )
        self.assertTrue(torch.equal(kjt_expected.lengths(), kjt_actual.lengths()))
        self.assertTrue(torch.equal(kjt_expected.offsets(), kjt_actual.offsets()))
        self.assertTrue(torch.equal(kjt_expected.values(), kjt_actual.values()))

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
            torch.Tensor(),
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
