#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchrec.distributed.stream_sync_tensor import WrapInStreamSyncTensorFunc

from torchrec.distributed.test_utils.multi_process import MultiProcessTestBase
from torchrec.test_utils import skip_if_asan_class


@skip_if_asan_class
class TestStreamSyncCollectiveTensor(MultiProcessTestBase):
    # pyre-ignore
    @unittest.skipIf(
        torch.cuda.device_count() < 1,
        "Not enough GPUs, this test requires at least one GPUs",
    )
    def test_stream_sync_collective_tensor(self) -> None:
        s = torch.cuda.Stream()
        a = torch.randn(10, device=torch.device("cuda"), requires_grad=True)
        b = torch.randn(10, device=torch.device("cuda"), requires_grad=True)

        y = torch.randn(10, device=torch.device("cuda"), requires_grad=True)
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            x = a + b
        x_stream = WrapInStreamSyncTensorFunc.apply(x, s)
        z = x_stream + y
        z.sum().backward()


import unittest

if __name__ == "__main__":
    unittest.main()