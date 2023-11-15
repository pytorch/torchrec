#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import copy
import re
import unittest
from contextlib import contextmanager
from typing import List

import torch
import torch._dynamo.skipfiles
from fbgemm_gpu import sparse_ops  # noqa: F401, E402
from torch._dynamo import skipfiles
from torch._export import dynamic_dim
from torchrec.distributed.test_utils.infer_utils import (
    assert_close,
    KJTInputExportWrapper,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


def make_kjt(values: List[int], lengths: List[int]) -> KeyedJaggedTensor:
    values_tensor = torch.tensor(values, dtype=torch.int32)
    lengths_tensor = torch.tensor(lengths, dtype=torch.int32)
    torch._check(torch.sum(lengths_tensor).item() == values_tensor.size(0))
    kjt = KeyedJaggedTensor(
        keys=[f"key{i}" for i in range(len(lengths))],
        values=values_tensor,
        lengths=lengths_tensor,
    )
    dynamic_dim(kjt._values, 0)
    return kjt


# TODO(ivankobzarev): Remove once torchrec is not in dynamo skipfiles
@contextmanager
# pyre-ignore
def dynamo_skipfiles_allow(exclude_from_skipfiles_pattern: str):
    original_FBCODE_SKIP_DIRS_RE = copy.deepcopy(skipfiles.FBCODE_SKIP_DIRS_RE)
    new_FBCODE_SKIP_DIRS = {
        s for s in skipfiles.FBCODE_SKIP_DIRS if exclude_from_skipfiles_pattern not in s
    }
    skipfiles.FBCODE_SKIP_DIRS_RE = re.compile(
        # pyre-ignore
        f".*({'|'.join(map(re.escape, new_FBCODE_SKIP_DIRS))})"
    )
    yield
    skipfiles.FBCODE_SKIP_DIRS_RE = original_FBCODE_SKIP_DIRS_RE


class TestPt2(unittest.TestCase):
    def _test_kjt_input_module(
        self,
        kjt_input_module: torch.nn.Module,
        kjt_keys: List[str],
        # pyre-ignore
        inputs,
    ) -> None:
        with dynamo_skipfiles_allow("torchrec"):
            EM: torch.nn.Module = KJTInputExportWrapper(kjt_input_module, kjt_keys)
            eager_output = EM(*inputs)
            x = torch._dynamo.export(EM, same_signature=True)(*inputs)

            export_gm = x.graph_module
            export_gm_output = export_gm(*inputs)

            assert_close(eager_output, export_gm_output)

    def test_kjt_split(self) -> None:
        class M(torch.nn.Module):
            def forward(self, kjt: KeyedJaggedTensor, segments: List[int]):
                return kjt.split(segments)

        kjt: KeyedJaggedTensor = make_kjt([2, 3, 4, 5, 6], [1, 2, 1, 1])
        segments: List[int] = [1, 2, 1]
        self._test_kjt_input_module(
            M(), kjt.keys(), (kjt._values, kjt._lengths, segments)
        )

    def test_kjt_permute(self) -> None:
        class M(torch.nn.Module):
            def forward(self, kjt: KeyedJaggedTensor, indices: List[int]):
                return kjt.permute(indices)

        kjt: KeyedJaggedTensor = make_kjt([2, 3, 4, 5, 6], [1, 2, 1, 1])
        indices: List[int] = [1, 0, 3, 2]
        self._test_kjt_input_module(
            M(), kjt.keys(), (kjt._values, kjt._lengths, indices)
        )
