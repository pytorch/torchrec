#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import fbgemm_gpu.sparse_ops  # noqa: F401
import torch
import torch.distributed as dist


def tpu_is_available() -> bool:
    try:
        import torch_tpu  # pyrefly: ignore [missing-import]  # noqa: F401

        _ = torch_tpu
    except ImportError:
        return False
    tpu = getattr(torch, "tpu", None)
    if tpu is None:
        return False
    is_available = getattr(tpu, "is_available", None)
    if callable(is_available):
        return bool(is_available())
    return bool(getattr(tpu, "available", False))


class BatchIndexSelectTest(unittest.TestCase):

    def test_cpu_backward_impl_matches_autograd(self) -> None:
        inputs = torch.tensor(
            [10, 11, 20, 21, 100, 200, 300, 1, 2, 3, 4, 5, 6],
            dtype=torch.float32,
            requires_grad=True,
        )
        indices = torch.tensor([1, 0, 1, 2, 0, 1, 0, 1, 0], dtype=torch.int64)
        input_num_indices = [3, 3, 3]
        input_rows = [2, 3, 2]
        input_columns = [2, 1, 3]
        output = torch.ops.fbgemm.batch_index_select_dim0(
            inputs,
            indices,
            input_num_indices,
            input_rows,
            input_columns,
            True,
        )
        grad_output = torch.randn(
            output.shape, generator=torch.Generator().manual_seed(0)
        )
        output.backward(grad_output)
        num_indices = torch.tensor(input_num_indices)
        actual = torch.ops.fbgemm.batch_index_select_dim0_backward_cpu_impl(
            grad_output,
            indices,
            num_indices,
            num_indices,
            torch.tensor(input_rows),
            torch.tensor(input_columns),
            True,
            torch.tensor([inputs.numel()]),
        )

        torch.testing.assert_close(actual, inputs.grad)

    @classmethod
    def tearDownClass(cls) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()

    def _setup_tpu(self) -> None:
        if not tpu_is_available():
            self.skipTest("requires an attached TPU")
        from torchrec.experimental.torch_tpu.pallas import dispatcher  # noqa: F401

        _ = dispatcher
        if not dist.is_initialized():
            dist.init_process_group(backend="tpu_dist")

    def test_sparse_core_matches_fbgemm_cpu(self) -> None:
        batch_size = 16
        input_rows = [3, 4, 2]
        input_columns = [16, 32, 16]
        input_num_indices = [batch_size] * len(input_columns)
        inputs = torch.arange(
            sum(rows * columns for rows, columns in zip(input_rows, input_columns)),
            dtype=torch.float32,
        )
        inputs.requires_grad_()
        indices = torch.cat(
            [torch.arange(batch_size) % rows for rows in input_rows]
        ).to(torch.int64)
        expected = torch.ops.fbgemm.batch_index_select_dim0(
            inputs,
            indices,
            input_num_indices,
            input_rows,
            input_columns,
            True,
        )

        self._setup_tpu()
        tpu_inputs = inputs.detach().to("tpu").requires_grad_()
        actual = torch.ops.fbgemm.batch_index_select_dim0(
            tpu_inputs,
            indices.to("tpu"),
            input_num_indices,
            input_rows,
            input_columns,
            True,
        )

        torch.testing.assert_close(actual.cpu(), expected)
        generator = torch.Generator().manual_seed(0)
        grad_output = torch.randn(expected.shape, generator=generator)
        expected.backward(grad_output)
        actual.backward(grad_output.to("tpu"))
        assert tpu_inputs.grad is not None
        assert inputs.grad is not None
        torch.testing.assert_close(tpu_inputs.grad.cpu(), inputs.grad)


if __name__ == "__main__":
    unittest.main()
