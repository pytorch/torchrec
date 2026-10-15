#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

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

    def _check_tpu_backward(
        self, lengths: torch.Tensor, permute: torch.Tensor, values: torch.Tensor
    ) -> None:
        if not tpu_is_available():
            self.skipTest("requires an attached TPU")
        from torchrec.experimental.torch_tpu.pallas import dispatcher, impl

        _ = impl
        if not dist.is_initialized():
            dist.init_process_group(backend="tpu_dist")

        output_size = int(lengths[permute.long()].sum().item())
        expected_lengths, expected_values, _ = torch.ops.fbgemm.permute_1D_sparse_data(
            permute, lengths, values.flatten(), None, output_size
        )
        reference_input = values.detach().clone().requires_grad_(True)
        _, reference_output, _ = torch.ops.fbgemm.permute_2D_sparse_data(
            permute, lengths.view(-1, 1), reference_input.flatten(), None, output_size
        )
        generator = torch.Generator().manual_seed(42)
        grad_output = torch.randn(output_size, generator=generator, dtype=values.dtype)
        reference_output.backward(grad_output)

        device = torch.device("tpu")
        actual_input = values.detach().to(device).requires_grad_(True)
        with (
            patch.object(
                dispatcher,
                "_permute_1d_sparse_data",
                wraps=dispatcher._permute_1d_sparse_data,
            ) as forward,
            patch.object(
                dispatcher,
                "_permute_1d_sparse_data_bwd",
                wraps=dispatcher._permute_1d_sparse_data_bwd,
            ) as backward,
        ):
            actual_lengths, actual_values, actual_weights = (
                torch.ops.fbgemm.permute_1D_sparse_data(
                    permute.to(device),
                    lengths.to(device),
                    actual_input.flatten(),
                    None,
                    output_size,
                )
            )
            self.assertTrue(actual_values.requires_grad)
            actual_values.backward(grad_output.to(device))
            forward.assert_called_once()
            backward.assert_called_once()

        torch.testing.assert_close(actual_lengths.cpu(), expected_lengths)
        torch.testing.assert_close(actual_values.detach().cpu(), expected_values)
        self.assertIsNone(actual_weights)
        self.assertIsNotNone(actual_input.grad)
        assert actual_input.grad is not None
        torch.testing.assert_close(actual_input.grad.cpu(), reference_input.grad)

    def test_tpu_backward_cases(self) -> None:
        if not tpu_is_available():
            self.skipTest("requires an attached TPU")
        cases = (
            ("identity", [0, 1, 15, 16, 17, 31, 32, 33], list(range(8))),
            ("reverse", [0, 1, 15, 16, 17, 31, 32, 33], list(reversed(range(8)))),
            ("cycle", [4, 1, 3, 2], [2, 0, 3, 1]),
            ("empty_values", [0, 0, 0], [2, 0, 1]),
            ("no_segments", [], []),
        )
        generator = torch.Generator().manual_seed(0)
        for name, lengths, permute in cases:
            with self.subTest(case=name):
                self._check_tpu_backward(
                    torch.tensor(lengths, dtype=torch.int32),
                    torch.tensor(permute, dtype=torch.int32),
                    torch.randn(sum(lengths), generator=generator),
                )

    def test_tpu_backward_randomized(self) -> None:
        if not tpu_is_available():
            self.skipTest("requires an attached TPU")
        generator = torch.Generator().manual_seed(1)
        for num_segments, max_length in ((7, 17), (131, 65)):
            with self.subTest(num_segments=num_segments):
                lengths = torch.randint(
                    0,
                    max_length + 1,
                    (num_segments,),
                    dtype=torch.int32,
                    generator=generator,
                )
                lengths[0] = 0
                lengths[-1] = max_length
                permute = torch.randperm(
                    num_segments,
                    dtype=torch.int32,
                    generator=generator,
                )
                self._check_tpu_backward(
                    lengths,
                    permute,
                    torch.randn(int(lengths.sum().item()), generator=generator),
                )

    def test_tpu_vbe_backward(self) -> None:
        counts = [[2, 2], [0, 0], [1, 1]]
        dimension = 16
        lengths = [count * dimension for feature in counts for count in feature]
        order = [feature * 2 + rank for rank in range(2) for feature in range(3)]
        values = torch.randn(
            (sum(lengths) // dimension, dimension),
            generator=torch.Generator().manual_seed(3),
        )
        self._check_tpu_backward(
            torch.tensor(lengths, dtype=torch.int32),
            torch.tensor(order, dtype=torch.int32),
            values,
        )

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
        permute = torch.tensor([2, 0, 3, 1], dtype=torch.int32, device=device)
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
        output = kjt.permute([2, 0, 3, 1], permute)
        torch.testing.assert_close(
            output.lengths().cpu(), torch.tensor([3, 2, 1, 0], dtype=torch.int32)
        )
        torch.testing.assert_close(
            output.values().detach().cpu(),
            torch.tensor([20.0, 21.0, 22.0, 10.0, 11.0, 30.0]),
        )
        self.assertIsNone(output.weights_or_none())

        output.values().sum().backward()
        self.assertIsNotNone(values.grad)
        assert values.grad is not None
        torch.testing.assert_close(values.grad.cpu(), torch.ones(6))

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
