#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torchrec.modules.feature_processor_ import PositionWeightedModule
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class PositionWeightedModuleTest(unittest.TestCase):
    def test_populate_weights(self) -> None:
        pw = PositionWeightedModule(max_feature_length=10)

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
        features = features.to_dict()

        jt = features["f1"]
        weighted_features = pw(jt)
        print("weighted features", weighted_features)

        self.assertEqual(weighted_features.weights().size(), (3,))

        pw_f1_ref = torch.gather(
            pw.state_dict()["position_weight"], 0, torch.tensor([0, 1, 0])
        )

        pw_f1 = weighted_features.weights().detach()
        self.assertTrue(torch.allclose(pw_f1_ref, pw_f1))
