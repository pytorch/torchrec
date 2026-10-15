#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.distributed as dist
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


def tpu_is_available() -> bool:
    try:
        import torch_tpu  # pyrefly: ignore [missing-import]  # noqa: F401
    except ImportError:
        return False
    tpu = getattr(torch, "tpu", None)
    if tpu is None:
        return False
    is_available = getattr(tpu, "is_available", None)
    if callable(is_available):
        return bool(is_available())
    return bool(getattr(tpu, "available", False))


class Permute1DSparseDataTest(unittest.TestCase):
    @classmethod
    def tearDownClass(cls) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_with_duplicates_and_weights(self) -> None:
        kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=["a", "b", "c", "d"],
            lengths=torch.tensor([2, 0, 3, 1], dtype=torch.int32),
            values=torch.tensor([10.0, 11.0, 20.0, 21.0, 22.0, 30.0]),
            weights=torch.arange(6, dtype=torch.float32),
            stride_per_key_per_rank=[[1], [1], [1], [1]],
        )
        output = kjt.permute([2, 0, 2, 3])

        self.assertEqual(output.keys(), ["c", "a", "c", "d"])
        self.assertEqual(output.lengths().tolist(), [3, 2, 3, 1])
        self.assertEqual(
            output.values().tolist(),
            [20.0, 21.0, 22.0, 10.0, 11.0, 20.0, 21.0, 22.0, 30.0],
        )
        self.assertEqual(
            output.weights().tolist(),
            [2.0, 3.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        )

    def test_tpu_forward_backward(self) -> None:
        if not tpu_is_available():
            self.skipTest("requires an attached TPU")
        from torchrec.experimental.torch_tpu.pallas import (  # noqa: F401
            dispatcher,
            impl,
        )

        if not dist.is_initialized():
            dist.init_process_group(backend="tpu_dist")

        device = torch.device("tpu")
        lengths = torch.tensor([2, 0, 3, 1], dtype=torch.int32, device=device)
        permute = torch.tensor([2, 0, 2, 3], dtype=torch.int32, device=device)
        values = torch.tensor(
            [10.0, 11.0, 20.0, 21.0, 22.0, 30.0],
            device=device,
            requires_grad=True,
        )

        kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=["a", "b", "c", "d"],
            lengths=lengths,
            values=values,
            stride_per_key_per_rank=[[1], [1], [1], [1]],
        )
        output = kjt.permute([2, 0, 2, 3], permute)
        torch.testing.assert_close(
            output.lengths().cpu(), torch.tensor([3, 2, 3, 1], dtype=torch.int32)
        )
        torch.testing.assert_close(
            output.values().detach().cpu(),
            torch.tensor([20.0, 21.0, 22.0, 10.0, 11.0, 20.0, 21.0, 22.0, 30.0]),
        )
        self.assertIsNone(output.weights_or_none())

        output.values().sum().backward()
        self.assertIsNotNone(values.grad)
        assert values.grad is not None
        torch.testing.assert_close(
            values.grad.cpu(), torch.tensor([1.0, 1.0, 2.0, 2.0, 2.0, 1.0])
        )

    def _assert_tpu_matches_fbgemm_cpu(
        self,
        lengths: torch.Tensor,
        permute: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        permuted_lengths_sum = int(lengths[permute.long()].sum().item())
        expected_lengths, expected_values, expected_weights = (
            torch.ops.fbgemm.permute_1D_sparse_data(
                permute,
                lengths,
                values,
                None,
                permuted_lengths_sum,
            )
        )

        device = torch.device("tpu")
        actual_lengths, actual_values, actual_weights = (
            torch.ops.fbgemm.permute_1D_sparse_data(
                permute.to(device),
                lengths.to(device),
                values.to(device),
                None,
                permuted_lengths_sum,
            )
        )

        torch.testing.assert_close(actual_lengths.cpu(), expected_lengths)
        torch.testing.assert_close(actual_values.cpu(), expected_values)
        self.assertIsNone(expected_weights)
        self.assertIsNone(actual_weights)

    def test_permute_1d_randomized_matches_fbgemm_cpu(self) -> None:
        if not tpu_is_available():
            self.skipTest("requires an attached TPU")
        from torchrec.experimental.torch_tpu.pallas import (  # noqa: F401
            dispatcher,
            impl,
        )

        if not dist.is_initialized():
            dist.init_process_group(backend="tpu_dist")

        generator = torch.Generator().manual_seed(0)
        for num_segments, max_length, num_outputs in (
            (1, 1, 1),
            (7, 17, 11),
            (19, 33, 23),
        ):
            with self.subTest(
                num_segments=num_segments,
                max_length=max_length,
                num_outputs=num_outputs,
            ):
                lengths = torch.randint(
                    0,
                    max_length + 1,
                    (num_segments,),
                    dtype=torch.int32,
                    generator=generator,
                )
                lengths[0] = 0
                lengths[-1] = max_length
                permute = torch.randint(
                    0,
                    num_segments,
                    (num_outputs,),
                    dtype=torch.int32,
                    generator=generator,
                )
                permute[0] = num_segments - 1
                values = torch.randint(
                    0,
                    1_000_000,
                    (int(lengths.sum().item()),),
                    dtype=torch.int32,
                    generator=generator,
                )
                self._assert_tpu_matches_fbgemm_cpu(lengths, permute, values)

    def test_permute_1d_edge_cases_match_fbgemm_cpu(self) -> None:
        if not tpu_is_available():
            self.skipTest("requires an attached TPU")
        from torchrec.experimental.torch_tpu.pallas import (  # noqa: F401
            dispatcher,
            impl,
        )

        if not dist.is_initialized():
            dist.init_process_group(backend="tpu_dist")

        generator = torch.Generator().manual_seed(1)
        cases = (
            (
                "empty_values",
                torch.tensor([0, 0, 0], dtype=torch.int32),
                torch.tensor([2, 0, 1], dtype=torch.int32),
            ),
            (
                "empty_and_duplicate_segments",
                torch.tensor([0, 1, 0, 17], dtype=torch.int32),
                torch.tensor([3, 0, 3, 2, 1], dtype=torch.int32),
            ),
            (
                "tile_boundaries",
                torch.tensor([1, 15, 16, 17, 31, 32, 33], dtype=torch.int32),
                torch.tensor([6, 0, 3, 2, 5, 1, 4], dtype=torch.int32),
            ),
        )
        for name, lengths, permute in cases:
            with self.subTest(name=name):
                values = torch.randn(
                    int(lengths.sum().item()),
                    dtype=torch.float32,
                    generator=generator,
                )
                self._assert_tpu_matches_fbgemm_cpu(lengths, permute, values)


if __name__ == "__main__":
    unittest.main()
