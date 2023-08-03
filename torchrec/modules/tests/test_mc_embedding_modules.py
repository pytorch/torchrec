#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import cast

import torch
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.modules.managed_collision_modules import (
    ManagedCollisionModule,
    TrivialManagedCollisionModule,
)
from torchrec.modules.mc_embedding_modules import ManagedCollisionEmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class TriviallyManagedCollisionEmbeddingBagCollectionTest(unittest.TestCase):
    def test_trivial_managed_ebc(self) -> None:
        device = torch.device("cpu")
        #     0       1        2  <-- batch
        # 0   [0,1] None    [2]
        # 1   [3]    [4]    [5,6,7]
        # ^
        # feature

        features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f2"],
            values=torch.tensor([100, 101, 102, 103, 104, 105, 106, 107]),
            offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8]),
        ).to(device)

        ebc = EmbeddingBagCollection(
            tables=[
                EmbeddingBagConfig(
                    name="t1", embedding_dim=8, num_embeddings=16, feature_names=["f1"]
                ),
                EmbeddingBagConfig(
                    name="t2", embedding_dim=8, num_embeddings=16, feature_names=["f2"]
                ),
            ],
            device=device,
        )
        mc_modules = {
            "t1": cast(
                ManagedCollisionModule,
                TrivialManagedCollisionModule(
                    max_output_id=16, max_input_id=32, device=device
                ),
            ),
            "t2": cast(
                ManagedCollisionModule,
                TrivialManagedCollisionModule(
                    max_output_id=16, max_input_id=32, device=device
                ),
            ),
        }

        mc_ebc = ManagedCollisionEmbeddingBagCollection(ebc, mc_modules)

        self.assertEqual(mc_ebc.is_weighted(), ebc.is_weighted())
        self.assertEqual(mc_ebc.embedding_bag_configs(), ebc.embedding_bag_configs())
        self.assertEqual(mc_ebc.device, ebc.device)

        pooled_embeddings = mc_ebc(features)

        self.assertEqual(pooled_embeddings.keys(), ["f1", "f2"])
        self.assertEqual(pooled_embeddings.values().size(), (3, 16))
        self.assertEqual(pooled_embeddings.offset_per_key(), [0, 8, 16])

        print("state dict", mc_ebc.state_dict())
