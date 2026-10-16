#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torchrec.distributed.model_parallel import DistributedModelParallel


def _dmp_with_device(device: torch.device) -> DistributedModelParallel:
    # _init_parameters only uses self.device; avoid full DMP construction
    # (which requires a process group and sharding plan).
    dmp = DistributedModelParallel.__new__(DistributedModelParallel)
    dmp.device = device
    return dmp


class InitParametersFakeModeTest(unittest.TestCase):
    def test_materialize_meta_under_leaked_fake_mode(self) -> None:
        # A leaked FakeTensorMode must not turn materialization into silent
        # FakeTensor factories: meta params must still materialize to real.
        m = torch.nn.Linear(8, 8, device="meta")
        dmp = _dmp_with_device(torch.device("cpu"))
        with FakeTensorMode(
            shape_env=ShapeEnv(), allow_non_fake_inputs=True
        ):
            dmp._init_parameters(m)
        self.assertTrue(type(m.weight) is torch.Tensor)
        self.assertEqual(m.weight.device.type, "cpu")

    def test_refuse_already_fake_params(self) -> None:
        m = torch.nn.Linear(8, 8)
        with FakeTensorMode(
            shape_env=ShapeEnv(), allow_non_fake_inputs=True
        ):
            fake_w = torch.empty_like(m.weight)
        self.assertTrue(isinstance(fake_w, FakeTensor))
        m.weight = torch.nn.Parameter(fake_w)
        dmp = _dmp_with_device(torch.device("cpu"))
        with self.assertRaisesRegex(RuntimeError, "already a FakeTensor"):
            dmp._init_parameters(m)


if __name__ == "__main__":
    unittest.main()
