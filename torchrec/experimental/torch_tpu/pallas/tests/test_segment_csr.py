#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest
from itertools import accumulate
from unittest.mock import patch

import fbgemm_gpu.sparse_ops  # noqa: F401
import torch
import torch.distributed as dist
from parameterized import parameterized


def tpu_is_available() -> bool:
    try:
        import torch_tpu  # pyrefly: ignore[missing-import]  # noqa: F401
    except ImportError:
        return False
    tpu = getattr(torch, "tpu", None)
    if tpu is None:
        return False
    is_available = getattr(tpu, "is_available", None)
    if callable(is_available):
        return bool(is_available())
    return bool(getattr(tpu, "available", False))


class SegmentSumCsrTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not tpu_is_available():
            raise unittest.SkipTest("requires an attached TPU")
        from torchrec.experimental.torch_tpu.pallas import dispatcher, impl

        _ = dispatcher
        cls._impl = impl
        if not dist.is_initialized():
            dist.init_process_group(backend="tpu_dist")
            cls.addClassCleanup(dist.destroy_process_group)

    @parameterized.expand(
        [
            (f"{name}_{dtype_name}", batch_size, lengths, dtype, csr_dtype)
            for name, batch_size, lengths in (
                ("short_segments", 1, [2, 3, 1]),
                ("lane_boundaries", 1, [1, 15, 16, 17]),
                ("chunk_boundaries", 1, [159, 160, 161]),
                ("multiple_chunks", 1, [7, 321, 3]),
                ("empty_segments", 1, [0, 3, 0, 1, 0]),
                ("all_empty", 1, [0, 0, 0]),
                ("no_segments", 1, []),
                ("batch_stride_two", 2, [2, 1, 2]),
                ("batch_stride_three", 3, [0, 1, 54, 0, 2]),
                ("many_segments", 1, [i % 19 for i in range(131)]),
            )
            for dtype_name, dtype, csr_dtype in (
                ("int32", torch.int32, torch.int32),
                ("float32", torch.float32, torch.int32),
                ("int64", torch.int64, torch.int64),
                ("float32_int64_offsets", torch.float32, torch.int64),
            )
        ]
    )
    def test_matches_fbgemm_cpu(
        self,
        _name: str,
        batch_size: int,
        lengths: list[int],
        dtype: torch.dtype,
        csr_dtype: torch.dtype,
    ) -> None:
        csr_seg = torch.tensor([0, *accumulate(lengths)], dtype=csr_dtype, device="cpu")
        values = (
            torch.arange(batch_size * sum(lengths), dtype=torch.int32, device="cpu")
            % 23
            - 11
        ).to(dtype=dtype)
        if dtype.is_floating_point:
            values = values / 8
        expected = torch.ops.fbgemm.segment_sum_csr(batch_size, csr_seg, values)

        with patch.object(
            self._impl,
            "_segment_sum_csr_sc",
            wraps=self._impl._segment_sum_csr_sc,
        ) as kernel:
            actual = torch.ops.fbgemm.segment_sum_csr(
                batch_size,
                csr_seg.to("tpu"),
                values.to("tpu"),
            )
            kernel.assert_called_once()

        self.assertEqual(actual.device.type, "tpu")
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
