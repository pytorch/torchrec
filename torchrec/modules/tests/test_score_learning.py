#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torchrec.fx import Tracer
from torchrec.modules.score_learning import PositionWeightsAttacher
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class PositionWeightsAttacherTest(unittest.TestCase):
    def test_populate_weights(self) -> None:
        features_max_length = {"f1": 10, "f2": 3}
        pw = PositionWeightsAttacher(features_max_length)

        #     0       1        2  <-- batch
        # 0   [0,1] None    [2]
        # 1   [3]    [4]    [5,6,7]
        # ^
        # feature
        features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f2"],
            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
            offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8]),
        )

        weighted_features = pw(features)
        self.assertEqual(weighted_features.weights().size(), (8,))
        self.assertEqual(weighted_features["f1"].weights().size(), (3,))
        self.assertEqual(weighted_features["f2"].weights().size(), (5,))
        pw_f1_ref = torch.gather(
            pw.state_dict()["position_weights.f1"], 0, torch.tensor([0, 1, 0])
        )
        pw_f1 = weighted_features["f1"].weights().detach()
        self.assertTrue(torch.allclose(pw_f1_ref, pw_f1))
        pw_f2_ref = torch.gather(
            pw.state_dict()["position_weights.f2"], 0, torch.tensor([0, 0, 0, 1, 2])
        )
        pw_f2 = weighted_features["f2"].weights().detach()
        self.assertTrue(torch.allclose(pw_f2_ref, pw_f2))

    def test_fx_script_PositionWeightsAttacher(self) -> None:
        features_max_length = {"f1": 10, "f2": 3}
        pw = PositionWeightsAttacher(features_max_length)

        gm = torch.fx.GraphModule(pw, Tracer().trace(pw))
        torch.jit.script(gm)


if __name__ == "__main__":
    unittest.main()
