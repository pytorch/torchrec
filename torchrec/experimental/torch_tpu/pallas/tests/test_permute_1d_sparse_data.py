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


if __name__ == "__main__":
    unittest.main()
