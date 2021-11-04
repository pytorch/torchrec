#!/usr/bin/env python3

import itertools
import math
import random
import unittest
from typing import List, Optional, Tuple

import numpy as np
import torch
from torchrec.sparse.jagged_tensor import (
    JaggedTensor,
    KeyedTensor,
    KeyedJaggedTensor,
)
from torchrec.sparse.tests.tests_utils import keyed_jagged_tensor_equals


def _compute_translated_lengths(
    row_indices: List[int],
    indices_offsets: List[int],
    lengths_size: int,
    trainers_size: int,
    block_sizes: List[int],
) -> List[int]:
    translated_lengths = [0] * trainers_size * lengths_size

    batch_size = int(lengths_size / len(block_sizes))
    iteration = feature_offset = batch_iteration = 0
    for start_offset, end_offset in zip(indices_offsets, indices_offsets[1:]):
        # iterate all rows that belong to current feature and batch iteration
        for row_idx in row_indices[start_offset:end_offset]:
            # compute the owner of this row
            trainer_offset = int(row_idx / block_sizes[feature_offset])
            # we do not have enough trainers to handle this row
            if trainer_offset >= trainers_size:
                continue
            trainer_lengths_offset = trainer_offset * lengths_size
            # compute the offset in lengths that is local in each trainer
            local_lengths_offset = feature_offset * batch_size + batch_iteration
            # increment the corresponding length in the trainer
            translated_lengths[trainer_lengths_offset + local_lengths_offset] += 1
        # bookkeeping
        iteration += 1
        feature_offset = int(iteration / batch_size)
        batch_iteration = (batch_iteration + 1) % batch_size
    return translated_lengths


def _compute_translated_indices_with_weights(
    translated_lengths: List[int],
    row_indices: List[int],
    indices_offsets: List[int],
    lengths_size: int,
    weights: Optional[List[int]],
    trainers_size: int,
    block_sizes: List[int],
) -> List[Tuple[int, int]]:
    translated_indices_with_weights = [(0, 0)] * len(row_indices)

    translated_indices_offsets = np.cumsum([0] + translated_lengths)
    batch_size = int(lengths_size / len(block_sizes))
    iteration = feature_offset = batch_iteration = 0
    for start_offset, end_offset in zip(indices_offsets, indices_offsets[1:]):
        # iterate all rows that belong to current feature and batch iteration
        # and assign the translated row index to the corresponding offset in output
        for current_offset in range(start_offset, end_offset):
            row_idx = row_indices[current_offset]
            feature_block_size = block_sizes[feature_offset]
            # compute the owner of this row
            trainer_offset = int(row_idx / feature_block_size)
            if trainer_offset >= trainers_size:
                continue
            trainer_lengths_offset = trainer_offset * lengths_size
            # compute the offset in lengths that is local in each trainer
            local_lengths_offset = feature_offset * batch_size + batch_iteration
            # since we know the number of rows belonging to each trainer,
            # we can figure out the corresponding offset in the translated indices list
            # for the current translated index
            translated_indices_offset = translated_indices_offsets[
                trainer_lengths_offset + local_lengths_offset
            ]
            translated_indices_with_weights[translated_indices_offset] = (
                row_idx % feature_block_size,
                weights[current_offset] if weights else 0,
            )
            # the next row that goes to this trainer for this feature and batch
            # combination goes to the next offset
            translated_indices_offsets[
                trainer_lengths_offset + local_lengths_offset
            ] += 1
        # bookkeeping
        iteration += 1
        feature_offset = int(iteration / batch_size)
        batch_iteration = (batch_iteration + 1) % batch_size
    return translated_indices_with_weights


def block_bucketize_ref(
    keyed_jagged_tensor: KeyedJaggedTensor,
    trainers_size: int,
    block_sizes: torch.Tensor,
) -> KeyedJaggedTensor:
    lengths_list = keyed_jagged_tensor.lengths().view(-1).tolist()
    indices_list = keyed_jagged_tensor.values().view(-1).tolist()
    weights_list = (
        keyed_jagged_tensor.weights().view(-1).tolist()
        if keyed_jagged_tensor.weights() is not None
        else None
    )
    block_sizes_list = block_sizes.view(-1).tolist()
    lengths_size = len(lengths_list)

    """
    each element in indices_offsets signifies both the starting offset, in indices_list,
    that corresponds to all rows in a particular feature and batch iteration,
    and the ending offset of the previous feature/batch iteration

    For example:
    given that features_size = 2 and batch_size = 2, an indices_offsets of
    [0,1,4,6,6] signifies that:

    elements in indices_list[0:1] belongs to feature 0 batch 0
    elements in indices_list[1:4] belongs to feature 0 batch 1
    elements in indices_list[4:6] belongs to feature 1 batch 0
    elements in indices_list[6:6] belongs to feature 1 batch 1
    """
    indices_offsets = np.cumsum([0] + lengths_list)

    translated_lengths = _compute_translated_lengths(
        row_indices=indices_list,
        indices_offsets=indices_offsets,
        lengths_size=lengths_size,
        trainers_size=trainers_size,
        block_sizes=block_sizes_list,
    )
    translated_indices_with_weights = _compute_translated_indices_with_weights(
        translated_lengths=translated_lengths,
        row_indices=indices_list,
        indices_offsets=indices_offsets,
        lengths_size=lengths_size,
        weights=weights_list,
        trainers_size=trainers_size,
        block_sizes=block_sizes_list,
    )

    translated_indices = [
        translated_index for translated_index, _ in translated_indices_with_weights
    ]

    translated_weights = [
        translated_weight for _, translated_weight in translated_indices_with_weights
    ]

    expected_keys = [
        f"{key}@bucket_{index}"
        for index in range(trainers_size)
        for key in keyed_jagged_tensor.keys()
    ]

    return KeyedJaggedTensor(
        keys=expected_keys,
        lengths=torch.tensor(
            translated_lengths, dtype=keyed_jagged_tensor.lengths().dtype
        )
        .view(-1)
        .cuda(),
        values=torch.tensor(
            translated_indices, dtype=keyed_jagged_tensor.values().dtype
        ).cuda(),
        weights=torch.tensor(translated_weights).float().cuda()
        if weights_list
        else None,
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

    def test_concat_sync(self) -> None:
        values = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        keys = ["index_0", "index_1", "index_2"]
        lengths = torch.IntTensor([0, 2, 0, 1, 1, 1, 0, 3, 0, 0, 1, 0])

        kjt_expected = KeyedJaggedTensor.from_lengths_sync(
            values=values,
            keys=keys,
            lengths=lengths,
        )
        kjt_actual = KeyedJaggedTensor.concat_sync(
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
        values = torch.Tensor()
        keys = []
        offsets = torch.Tensor()

        KeyedJaggedTensor.from_offsets_sync(values=values, keys=keys, offsets=offsets)

    def test_2d(self) -> None:
        values = torch.Tensor([[i * 0.5, i * 1.0, i * 1.5] for i in range(1, 9)])
        keys = ["index_0", "index_1"]
        offsets = torch.IntTensor([0, 2, 2, 3, 4, 5, 8])

        j = KeyedJaggedTensor.from_offsets_sync(
            values=values,
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

    # pyre-ignore[56]
    @unittest.skipIf(
        torch.cuda.device_count() <= 0,
        "CUDA is not available",
    )
    def test_bucketize(self) -> None:
        index_type = random.choice([torch.int, torch.long])
        offset_type = random.choice([torch.int, torch.long])
        world_size = random.randint(1, 129)
        MAX_NUM_FEATURES = 15
        MAX_BATCH_SIZE = 15
        MAX_LENGTH = 10
        # max number of rows needed for a given feature to have unique row index
        MAX_ROW_COUNT = MAX_LENGTH * MAX_BATCH_SIZE

        num_features = random.randint(2, MAX_NUM_FEATURES)
        batch_size = random.randint(2, MAX_BATCH_SIZE)
        lengths_list = [
            random.randrange(MAX_LENGTH + 1) for _ in range(num_features * batch_size)
        ]
        keys_list = [f"feature_{i}" for i in range(num_features)]
        # for each feature, generate unrepeated row indices
        indices_lists = [
            random.sample(
                range(MAX_ROW_COUNT),
                # number of indices needed is the length sum of all batches for a feature
                sum(
                    lengths_list[
                        feature_offset * batch_size : (feature_offset + 1) * batch_size
                    ]
                ),
            )
            for feature_offset in range(num_features)
        ]
        indices_list = list(itertools.chain(*indices_lists))

        weights_list = [random.randint(1, 100) for _ in range(len(indices_list))]

        # for each feature, calculate the minimum block size needed to
        # distribute all rows to the available trainers
        block_sizes_list = [
            math.ceil((max(feature_indices_list) + 1) / world_size)
            for feature_indices_list in indices_lists
        ]

        kjt = KeyedJaggedTensor(
            keys=keys_list,
            lengths=torch.tensor(lengths_list, dtype=offset_type)
            .view(num_features * batch_size)
            .cuda(),
            values=torch.tensor(indices_list, dtype=index_type).cuda(),
            weights=torch.tensor(weights_list, dtype=torch.float).cuda(),
        )
        """
        each entry in block_sizes identifies how many hashes for each feature goes
        to every rank; we have three featues in `self.features`
        """
        block_sizes = torch.tensor(block_sizes_list, dtype=index_type).cuda()

        block_bucketized_kjt, _ = kjt.bucketize(world_size, block_sizes, False, False)

        expected_block_bucketized_kjt = block_bucketize_ref(
            kjt,
            world_size,
            block_sizes,
        )

        print(f"block_sizes: {block_sizes}")
        print(f"num_features: {num_features}")
        print(f"batch_size: {batch_size}")
        print(f"world_size: {world_size}")
        print(f"KeyedJaggedTensor: {kjt}")
        print(f"block_bucketized KeyedJaggedTensor: {block_bucketized_kjt}")
        print(
            f"expected_block_bucketized KeyedJaggedTensor: {expected_block_bucketized_kjt}"
        )
        self.assertTrue(
            keyed_jagged_tensor_equals(
                block_bucketized_kjt, expected_block_bucketized_kjt
            )
        )

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
