#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

import fbgemm_gpu.sparse_ops  # noqa: F401
import torch
import torch.distributed as dist
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class Permute2DSparseDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import torch_tpu  # pyrefly: ignore [missing-import]  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("requires torch_tpu")

        tpu = getattr(torch, "tpu", None)
        if tpu is None:
            raise unittest.SkipTest("requires an attached TPU")
        is_available = getattr(tpu, "is_available", None)
        available = (
            bool(is_available())
            if callable(is_available)
            else bool(getattr(tpu, "available", False))
        )
        if not available:
            raise unittest.SkipTest("requires an attached TPU")

        if not dist.is_initialized():
            dist.init_process_group(backend="tpu_dist")
            cls.addClassCleanup(dist.destroy_process_group)

        from torch_tpu._internal import pallas  # pyrefly: ignore [missing-import]
        from torchrec.experimental.torch_tpu.pallas import permute_2d

        cls._permute = staticmethod(
            pallas.jax_op(
                "torchrec_pallas_test::permute_2d_sparse_data",
                permute_2d.permute_2d_sparse_data_sc,
            )
        )

    def _assert_matches_fbgemm_cpu(
        self,
        lengths: torch.Tensor,
        order: list[int],
        values: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        permute = torch.tensor(order, dtype=torch.int32)
        if values is None:
            values = torch.arange(int(lengths.sum().item()), dtype=torch.int32) + 10
        output_size = int(lengths[permute.long()].sum().item())
        expected_lengths, expected_values, expected_weights = (
            torch.ops.fbgemm.permute_2D_sparse_data(
                permute, lengths, values, None, output_size
            )
        )

        actual_lengths, actual_values = self._permute(
            permute=permute.to("tpu"),
            lengths=lengths.to("tpu"),
            values=values.to("tpu"),
            weights=None,
            permuted_lengths_sum=output_size,
        )
        self.assertEqual(actual_lengths.device.type, "tpu")
        self.assertEqual(actual_values.device.type, "tpu")
        self.assertEqual(actual_lengths.dtype, lengths.dtype)
        self.assertEqual(actual_values.dtype, values.dtype)
        self.assertEqual(tuple(actual_lengths.shape), (len(order), lengths.shape[1]))
        self.assertEqual(tuple(actual_values.shape), (output_size,))
        self.assertIsNone(expected_weights)

        actual_lengths = actual_lengths.cpu()
        actual_values = actual_values.cpu()
        torch.testing.assert_close(actual_lengths, expected_lengths, rtol=0, atol=0)
        torch.testing.assert_close(actual_values, expected_values, rtol=0, atol=0)
        return actual_lengths, actual_values

    def test_kjt_feature_swap_preserves_sample_bags(self) -> None:
        kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=["user", "item"],
            values=torch.tensor([10, 11, 12, 20, 21, 22], dtype=torch.int32),
            lengths=torch.tensor([2, 1, 1, 2], dtype=torch.int32),
        )
        lengths, values = self._assert_matches_fbgemm_cpu(
            kjt.lengths().view(2, kjt.stride()), [1, 0], kjt.values()
        )
        expected = kjt.permute([1, 0])
        torch.testing.assert_close(lengths.flatten(), expected.lengths())
        torch.testing.assert_close(values, expected.values())
        torch.testing.assert_close(
            lengths, torch.tensor([[1, 2], [2, 1]], dtype=torch.int32)
        )
        torch.testing.assert_close(
            values, torch.tensor([20, 21, 22, 10, 11, 12], dtype=torch.int32)
        )

    def test_identity_reverse_subset_and_repeated_features(self) -> None:
        lengths = torch.tensor([[2, 0, 1], [1, 4, 0], [0, 2, 5]], dtype=torch.int32)
        for name, order in (
            ("identity", [0, 1, 2]),
            ("reverse", [2, 1, 0]),
            ("subset", [2, 0]),
            ("single_feature", [1]),
            ("repeat_with_same_feature_count", [1, 1, 0]),
            ("repeat_with_more_features", [2, 0, 2, 1, 2]),
        ):
            with self.subTest(case=name):
                self._assert_matches_fbgemm_cpu(lengths, order)

    def test_sparse_input_all_to_all_recat(self) -> None:
        lengths = torch.tensor([[1, 0], [2, 1], [0, 3], [1, 2]], dtype=torch.int32)
        self._assert_matches_fbgemm_cpu(lengths, [0, 2, 1, 3])

    def test_sequence_embeddings_all_to_all_recat(self) -> None:
        counts = torch.tensor([[1, 2], [0, 1], [3, 0], [1, 1]], dtype=torch.int32)
        dimension = 16
        lengths = counts * dimension
        values = torch.randn(
            int(lengths.sum().item()), generator=torch.Generator().manual_seed(42)
        )
        self._assert_matches_fbgemm_cpu(lengths, [0, 2, 1, 3], values)

    def test_empty_features_bags_and_outputs(self) -> None:
        for name, lengths, order in (
            ("all_empty_bags", torch.zeros((3, 2), dtype=torch.int32), [2, 0, 2]),
            (
                "select_only_empty_feature",
                torch.tensor([[2, 1], [0, 0]], dtype=torch.int32),
                [1, 1],
            ),
            (
                "empty_feature_between_nonempty_features",
                torch.tensor([[2, 0], [0, 0], [0, 3]], dtype=torch.int32),
                [2, 1, 0, 1, 2],
            ),
            ("no_selected_features", torch.ones((3, 2), dtype=torch.int32), []),
            ("no_input_features", torch.empty((0, 2), dtype=torch.int32), []),
            ("no_samples", torch.empty((3, 0), dtype=torch.int32), [2, 0, 2]),
        ):
            with self.subTest(case=name):
                self._assert_matches_fbgemm_cpu(lengths, order)

    def test_feature_copy_tile_boundaries(self) -> None:
        lengths = torch.tensor(
            [[0, 1], [7, 8], [8, 8], [8, 9], [15, 16], [16, 16], [16, 17]],
            dtype=torch.int32,
        )
        self._assert_matches_fbgemm_cpu(lengths, [6, 0, 3, 2, 5, 1, 4, 6])

    def test_randomized_multiple_subcore_rounds(self) -> None:
        generator = torch.Generator().manual_seed(7)
        for num_features, batch_size, num_outputs in (
            (1, 4, 1),
            (7, 3, 11),
            (131, 5, 263),
        ):
            with self.subTest(num_features=num_features, batch_size=batch_size):
                lengths = torch.randint(
                    0,
                    34,
                    (num_features, batch_size),
                    dtype=torch.int32,
                    generator=generator,
                )
                lengths[0, 0] = 0
                lengths[-1, -1] = 33
                order = torch.randint(
                    num_features, (num_outputs,), generator=generator
                ).tolist()
                order[0] = num_features - 1
                for dtype in (torch.int32, torch.float32):
                    with self.subTest(dtype=dtype):
                        values = torch.arange(int(lengths.sum().item()), dtype=dtype)
                        self._assert_matches_fbgemm_cpu(lengths, order, values)

    def test_requires_static_output_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "static permuted_lengths_sum"):
            self._permute(
                permute=torch.tensor([0], dtype=torch.int32, device="tpu"),
                lengths=torch.tensor([[1]], dtype=torch.int32, device="tpu"),
                values=torch.tensor([10], dtype=torch.int32, device="tpu"),
                weights=None,
                permuted_lengths_sum=None,
            )

    def test_dispatcher_uses_pallas(self) -> None:
        from torchrec.experimental.torch_tpu.pallas import dispatcher

        lengths = torch.tensor([[2, 0], [1, 2]], dtype=torch.int64)
        values = torch.tensor([10, 11, 20, 21, 22], dtype=torch.int64)
        permute = torch.tensor([1, 0, 1], dtype=torch.int64)
        expected_lengths, expected_values, _ = torch.ops.fbgemm.permute_2D_sparse_data(
            permute.int(), lengths, values, None, 8
        )
        with patch.object(
            dispatcher,
            "_permute_2d_sparse_data",
            wraps=dispatcher._permute_2d_sparse_data,
        ) as kernel:
            actual_lengths, actual_values, actual_weights = (
                torch.ops.fbgemm.permute_2D_sparse_data(
                    permute.to("tpu"), lengths.to("tpu"), values.to("tpu"), None, 8
                )
            )
            kernel.assert_called_once()
        torch.testing.assert_close(
            actual_lengths.cpu(), expected_lengths.int(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            actual_values.cpu(), expected_values.int(), rtol=0, atol=0
        )
        self.assertIsNone(actual_weights)

    def test_dispatcher_falls_back_for_weights_or_dynamic_size(self) -> None:
        from torchrec.experimental.torch_tpu.pallas import dispatcher

        lengths = torch.tensor([[2, 0], [1, 2]], dtype=torch.int32)
        values = torch.tensor([10, 11, 20, 21, 22], dtype=torch.int32)
        permute = torch.tensor([1, 0, 1], dtype=torch.int32)
        for weights, output_size in (
            (None, None),
            (torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]), 8),
        ):
            with self.subTest(weighted=weights is not None):
                expected = torch.ops.fbgemm.permute_2D_sparse_data(
                    permute, lengths, values, weights, output_size
                )
                with patch.object(dispatcher, "_permute_2d_sparse_data") as kernel:
                    actual = torch.ops.fbgemm.permute_2D_sparse_data(
                        permute.to("tpu"),
                        lengths.to("tpu"),
                        values.to("tpu"),
                        weights.to("tpu") if weights is not None else None,
                        output_size,
                    )
                    kernel.assert_not_called()
                for result, reference in zip(actual, expected):
                    if reference is None:
                        self.assertIsNone(result)
                    else:
                        self.assertEqual(result.device.type, "tpu")
                        torch.testing.assert_close(
                            result.cpu(), reference, rtol=0, atol=0
                        )

    def test_rejects_weights(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "Permuting weights"):
            self._permute(
                permute=torch.tensor([0], dtype=torch.int32, device="tpu"),
                lengths=torch.tensor([[1]], dtype=torch.int32, device="tpu"),
                values=torch.tensor([10], dtype=torch.int32, device="tpu"),
                weights=torch.tensor([1.0], device="tpu"),
                permuted_lengths_sum=1,
            )


if __name__ == "__main__":
    unittest.main()
