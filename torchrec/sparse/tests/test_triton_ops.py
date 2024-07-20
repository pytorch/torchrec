#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import unittest

import torch

from torchrec.sparse.jagged_tensor import _desugar_keyed_tensors, _regroup_keyed_tensors
from torchrec.sparse.tests.utils import build_groups, build_kts
from torchrec.sparse.triton_ops import (
    triton_permute_multi_embs,
    triton_permute_pooled_embs,
)


class TestPermutePooledEmbs(unittest.TestCase):
    # pyre-ignore[56]
    @unittest.skipIf(not torch.cuda.is_available(), "Skip when CUDA is not available")
    def test_triton_permute_pooled_embs_forward(self) -> None:
        kts = build_kts(
            dense_features=2,
            sparse_features=2,
            dim_dense=16,
            dim_sparse=16,
            batch_size=8,
            device=torch.device("cuda"),
            run_backward=False,
        )
        groups = build_groups(
            kts,
            4,
        )
        keys, lengths, values = _desugar_keyed_tensors(kts)
        output, splits = triton_permute_pooled_embs(values, keys, lengths, groups)
        refs = _regroup_keyed_tensors(kts, groups)
        outputs = torch.split(output, splits, dim=1)
        for ref, output in zip(refs, outputs):
            torch.testing.assert_close(ref, output)


class TestPermuteMultiEmbs(unittest.TestCase):
    # pyre-ignore[56]
    @unittest.skipIf(not torch.cuda.is_available(), "Skip when CUDA is not available")
    def test_triton_permute_multi_embs_forward(self) -> None:
        kts = build_kts(
            dense_features=2,
            sparse_features=2,
            dim_dense=16,
            dim_sparse=16,
            batch_size=8,
            device=torch.device("cuda"),
            run_backward=False,
        )
        groups = build_groups(
            kts,
            4,
        )
        keys, lengths, values = _desugar_keyed_tensors(kts)
        outputs = triton_permute_multi_embs(values, keys, lengths, groups)
        refs = _regroup_keyed_tensors(kts, groups)
        for ref, output in zip(refs, outputs):
            torch.testing.assert_close(ref, output)
