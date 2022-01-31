#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Union, Dict

import torch
import torch.fx
from torchrec.fx import symbolic_trace
from torchrec.modules.embedding_configs import (
    EmbeddingBagConfig,
    EmbeddingConfig,
)
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, JaggedTensor


class EmbeddingBagCollectionTest(unittest.TestCase):
    def test_unweighted(self) -> None:
        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2", embedding_dim=4, num_embeddings=10, feature_names=["f2"]
        )
        ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])

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

        pooled_embeddings = ebc(features)
        self.assertEqual(pooled_embeddings.values().size(), (3, 7))
        self.assertEqual(pooled_embeddings.keys(), ["f1", "f2"])
        self.assertEqual(pooled_embeddings.offset_per_key(), [0, 3, 7])

    def test_shared_tables(self) -> None:
        eb_config = EmbeddingBagConfig(
            name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1", "f2"]
        )
        ebc = EmbeddingBagCollection(tables=[eb_config])

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

        pooled_embeddings = ebc(features)
        self.assertEqual(pooled_embeddings.values().size(), (3, 6))
        self.assertEqual(pooled_embeddings.keys(), ["f1", "f2"])
        self.assertEqual(pooled_embeddings.offset_per_key(), [0, 3, 6])

    def test_shared_features(self) -> None:
        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2", embedding_dim=4, num_embeddings=10, feature_names=["f1"]
        )
        ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])

        #     0       1        2  <-- batch
        # 0   [0,1] None    [2]
        # 1   [3]    [4]    [5,6,7]
        # ^
        # feature
        features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1"],
            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
            offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8]),
        )

        pooled_embeddings = ebc(features)
        self.assertEqual(pooled_embeddings.values().size(), (6, 7))
        self.assertEqual(pooled_embeddings.keys(), ["f1@t1", "f1@t2"])
        self.assertEqual(pooled_embeddings.offset_per_key(), [0, 3, 7])

    def test_weighted(self) -> None:
        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1", "f3"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=4,
            num_embeddings=10,
            feature_names=["f2"],
        )
        ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config], is_weighted=True)

        features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f3", "f2"],
            values=torch.tensor([1, 2, 4, 5, 4, 3, 2, 9, 1, 3, 4, 7]),
            offsets=torch.tensor([0, 2, 4, 6, 8, 10, 12]),
            weights=torch.tensor(
                [0.1, 0.2, 0.4, 0.5, 0.4, 0.3, 0.2, 0.9, 0.1, 0.3, 0.4, 0.7]
            ),
        )

        pooled_embeddings = ebc(features)
        self.assertEqual(pooled_embeddings.values().size(), (2, 10))
        self.assertEqual(pooled_embeddings.keys(), ["f1", "f3", "f2"])
        self.assertEqual(pooled_embeddings.offset_per_key(), [0, 3, 6, 10])

    def test_fx(self) -> None:
        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1", "f3"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=4,
            num_embeddings=10,
            feature_names=["f2"],
        )
        ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config], is_weighted=True)

        gm = symbolic_trace(ebc)

        features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f3", "f2"],
            values=torch.tensor([1, 2, 4, 5, 4, 3, 2, 9, 1, 3, 4, 7]),
            offsets=torch.tensor([0, 2, 4, 6, 8, 10, 12]),
            weights=torch.tensor(
                [0.1, 0.2, 0.4, 0.5, 0.4, 0.3, 0.2, 0.9, 0.1, 0.3, 0.4, 0.7]
            ),
        )

        pooled_embeddings = gm(features)
        self.assertEqual(pooled_embeddings.values().size(), (2, 10))
        self.assertEqual(pooled_embeddings.keys(), ["f1", "f3", "f2"])
        self.assertEqual(pooled_embeddings.offset_per_key(), [0, 3, 6, 10])

    # TODO(T89043538): Auto-generate this test.
    def test_fx_script(self) -> None:
        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2", embedding_dim=4, num_embeddings=10, feature_names=["f2"]
        )

        ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])

        gm = symbolic_trace(ebc)
        torch.jit.script(gm)

    def test_duplicate_config_name_fails(self) -> None:
        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t1", embedding_dim=4, num_embeddings=10, feature_names=["f2"]
        )
        with self.assertRaises(ValueError):
            EmbeddingBagCollection(tables=[eb1_config, eb2_config])

    def test_device(self) -> None:
        config = EmbeddingBagConfig(
            name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1"]
        )
        ebc = EmbeddingBagCollection(tables=[config], device=torch.device("meta"))
        self.assertEqual(torch.device("meta"), ebc.embedding_bags["t1"].weight.device)


class EmbeddingCollectionTest(unittest.TestCase):
    def test_simple(self) -> None:
        e1_config = EmbeddingConfig(
            name="t1", embedding_dim=2, num_embeddings=10, feature_names=["f1"]
        )
        e2_config = EmbeddingConfig(
            name="t2", embedding_dim=3, num_embeddings=10, feature_names=["f2"]
        )
        ec = EmbeddingCollection(tables=[e1_config, e2_config])

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
        feature_embeddings = ec(features)

        self.assertEqual(feature_embeddings["f1"].values().size(), (3, 2))
        self.assertTrue(
            torch.allclose(feature_embeddings["f1"].lengths(), torch.tensor([2, 0, 1]))
        )
        self.assertEqual(feature_embeddings["f2"].values().size(), (5, 3))
        self.assertTrue(
            torch.allclose(feature_embeddings["f2"].lengths(), torch.tensor([1, 1, 3]))
        )

    def test_fx(self) -> None:
        e1_config = EmbeddingConfig(
            name="t1", embedding_dim=2, num_embeddings=10, feature_names=["f1"]
        )
        e2_config = EmbeddingConfig(
            name="t2", embedding_dim=3, num_embeddings=10, feature_names=["f2"]
        )
        ec = EmbeddingCollection(tables=[e1_config, e2_config])

        gm = symbolic_trace(ec)

        features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f2"],
            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
            offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8]),
        )
        feature_embeddings = gm(features)
        self.assertEqual(feature_embeddings["f1"].values().size(), (3, 2))
        self.assertTrue(
            torch.allclose(feature_embeddings["f1"].lengths(), torch.tensor([2, 0, 1]))
        )
        self.assertEqual(feature_embeddings["f2"].values().size(), (5, 3))
        self.assertTrue(
            torch.allclose(feature_embeddings["f2"].lengths(), torch.tensor([1, 1, 3]))
        )

    # TODO(T89043538): Auto-generate this test.
    def test_fx_script(self) -> None:
        e1_config = EmbeddingConfig(
            name="t1", embedding_dim=2, num_embeddings=10, feature_names=["f1"]
        )
        e2_config = EmbeddingConfig(
            name="t2", embedding_dim=3, num_embeddings=10, feature_names=["f2"]
        )
        ec = EmbeddingCollection(tables=[e1_config, e2_config])

        gm = symbolic_trace(ec)
        torch.jit.script(gm)

    def test_duplicate_config_name_fails(self) -> None:
        e1_config = EmbeddingConfig(
            name="t1", embedding_dim=2, num_embeddings=10, feature_names=["f1"]
        )
        e2_config = EmbeddingConfig(
            name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f2"]
        )
        with self.assertRaises(ValueError):
            EmbeddingCollection(tables=[e1_config, e2_config])

    def test_device(self) -> None:
        config = EmbeddingConfig(
            name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1"]
        )
        eb = EmbeddingCollection(tables=[config], device=torch.device("meta"))
        self.assertEqual(torch.device("meta"), eb.embeddings["t1"].weight.device)
