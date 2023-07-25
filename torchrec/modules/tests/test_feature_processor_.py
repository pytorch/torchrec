#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchrec.fx.tracer import symbolic_trace
from torchrec.modules.feature_processor_ import (
    PositionWeightedModule,
    PositionWeightedModuleCollection,
)
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

        self.assertEqual(weighted_features.weights().size(), (3,))

        pw_f1_ref = torch.gather(
            pw.state_dict()["position_weight"], 0, torch.tensor([0, 1, 0])
        )

        pw_f1 = weighted_features.weights().detach()
        self.assertTrue(torch.allclose(pw_f1_ref, pw_f1))

        position_weighted_module_gm = symbolic_trace(pw)
        position_weighted_module_gm_script = torch.jit.script(
            position_weighted_module_gm
        )

        weighted_features_gm_script = position_weighted_module_gm_script(jt)
        torch.testing.assert_close(
            weighted_features.values(), weighted_features_gm_script.values()
        )
        torch.testing.assert_close(
            weighted_features.lengths(), weighted_features_gm_script.lengths()
        )


class PositionWeightedCollectionModuleTest(unittest.TestCase):
    def test_populate_weights(self) -> None:
        position_weighted_module_collection = PositionWeightedModuleCollection(
            {"f1": 10, "f2": 10}
        )

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

        fp_kjt = position_weighted_module_collection(features)

        position_weighted_module_collection_gm = symbolic_trace(
            position_weighted_module_collection
        )
        position_weighted_module_collection_gm_script = torch.jit.script(
            position_weighted_module_collection_gm
        )
        fp_kjt_gm_script = position_weighted_module_collection_gm_script(features)

        torch.testing.assert_close(fp_kjt.values(), fp_kjt_gm_script.values())
        torch.testing.assert_close(fp_kjt.lengths(), fp_kjt_gm_script.lengths())
        torch.testing.assert_close(
            fp_kjt.length_per_key(), fp_kjt_gm_script.length_per_key()
        )
