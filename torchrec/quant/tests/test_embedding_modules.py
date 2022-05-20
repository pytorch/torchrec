#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import List

import hypothesis.strategies as st
import torch
from hypothesis import given, settings, Verbosity
from torchrec.modules.embedding_configs import (
    DataType,
    EmbeddingBagConfig,
    EmbeddingConfig,
)
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)
from torchrec.quant.embedding_modules import (
    EmbeddingBagCollection as QuantEmbeddingBagCollection,
    EmbeddingCollection as QuantEmbeddingCollection,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class EmbeddingBagCollectionTest(unittest.TestCase):
    def _test_ebc(
        self, tables: List[EmbeddingBagConfig], features: KeyedJaggedTensor
    ) -> None:
        ebc = EmbeddingBagCollection(tables=tables)

        embeddings = ebc(features)

        # test forward
        # pyre-ignore [16]
        ebc.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.PlaceholderObserver.with_args(
                dtype=torch.qint8
            ),
            weight=torch.quantization.PlaceholderObserver.with_args(dtype=torch.qint8),
        )

        qebc = QuantEmbeddingBagCollection.from_float(ebc)
        quantized_embeddings = qebc(features)

        self.assertEqual(embeddings.keys(), quantized_embeddings.keys())
        for key in embeddings.keys():
            self.assertEqual(embeddings[key].shape, quantized_embeddings[key].shape)
            self.assertTrue(
                torch.allclose(
                    embeddings[key].cpu().float(),
                    quantized_embeddings[key].cpu().float(),
                    atol=1,
                )
            )

        # test state dict
        state_dict = ebc.state_dict()
        quantized_state_dict = qebc.state_dict()
        self.assertEqual(state_dict.keys(), quantized_state_dict.keys())

    # pyre-fixme[56]
    @given(
        data_type=st.sampled_from(
            [
                DataType.FP32,
                DataType.FP16,
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=2, deadline=None)
    def test_ebc(self, data_type: DataType) -> None:
        eb1_config = EmbeddingBagConfig(
            name="t1",
            embedding_dim=16,
            num_embeddings=10,
            feature_names=["f1"],
            data_type=data_type,
        )
        eb2_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=16,
            num_embeddings=10,
            feature_names=["f2"],
            data_type=data_type,
        )
        features = KeyedJaggedTensor(
            keys=["f1", "f2"],
            values=torch.as_tensor([0, 1]),
            lengths=torch.as_tensor([1, 1]),
        )
        self._test_ebc([eb1_config, eb2_config], features)

    def test_shared_tables(self) -> None:
        eb_config = EmbeddingBagConfig(
            name="t1", embedding_dim=16, num_embeddings=10, feature_names=["f1", "f2"]
        )
        features = KeyedJaggedTensor(
            keys=["f1", "f2"],
            values=torch.as_tensor([0, 1]),
            lengths=torch.as_tensor([1, 1]),
        )
        self._test_ebc([eb_config], features)

    def test_shared_features(self) -> None:
        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=16, num_embeddings=10, feature_names=["f1"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2", embedding_dim=16, num_embeddings=10, feature_names=["f1"]
        )
        features = KeyedJaggedTensor(
            keys=["f1"],
            values=torch.as_tensor([0, 1]),
            lengths=torch.as_tensor([1, 1]),
        )
        self._test_ebc([eb1_config, eb2_config], features)


class EmbeddingCollectionTest(unittest.TestCase):
    def _test_ec(
        self, tables: List[EmbeddingConfig], features: KeyedJaggedTensor
    ) -> None:
        eb = EmbeddingCollection(tables=tables)

        embeddings = eb(features)

        # test forward
        # pyre-ignore [16]
        eb.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.PlaceholderObserver.with_args(
                dtype=torch.qint8
            ),
            weight=torch.quantization.PlaceholderObserver.with_args(dtype=torch.qint8),
        )

        qeb = QuantEmbeddingCollection.from_float(eb)
        quantized_embeddings = qeb(features)

        self.assertEqual(embeddings.keys(), quantized_embeddings.keys())
        for key in embeddings.keys():
            self.assertEqual(
                embeddings[key].values().size(),
                quantized_embeddings[key].values().size(),
            )
            self.assertTrue(
                torch.allclose(
                    embeddings[key].values().cpu().float(),
                    quantized_embeddings[key].values().cpu().float(),
                    atol=1,
                )
            )

        # test state dict
        state_dict = eb.state_dict()
        quantized_state_dict = qeb.state_dict()
        self.assertEqual(state_dict.keys(), quantized_state_dict.keys())

    # pyre-fixme[56]
    @given(
        data_type=st.sampled_from(
            [
                DataType.FP32,
                DataType.INT8,
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=2, deadline=None)
    def test_ec(self, data_type: DataType) -> None:
        eb1_config = EmbeddingConfig(
            name="t1",
            embedding_dim=16,
            num_embeddings=10,
            feature_names=["f1"],
            data_type=data_type,
        )
        eb2_config = EmbeddingConfig(
            name="t2",
            embedding_dim=16,
            num_embeddings=10,
            feature_names=["f2"],
            data_type=data_type,
        )
        features = KeyedJaggedTensor(
            keys=["f1", "f2"],
            values=torch.as_tensor([0, 1]),
            lengths=torch.as_tensor([1, 1]),
        )
        self._test_ec([eb1_config, eb2_config], features)

    def test_shared_tables(self) -> None:
        eb_config = EmbeddingConfig(
            name="t1", embedding_dim=16, num_embeddings=10, feature_names=["f1", "f2"]
        )
        features = KeyedJaggedTensor(
            keys=["f1", "f2"],
            values=torch.as_tensor([0, 1]),
            lengths=torch.as_tensor([1, 1]),
        )
        self._test_ec([eb_config], features)

    def test_shared_features(self) -> None:
        eb1_config = EmbeddingConfig(
            name="t1", embedding_dim=16, num_embeddings=10, feature_names=["f1"]
        )
        eb2_config = EmbeddingConfig(
            name="t2", embedding_dim=16, num_embeddings=10, feature_names=["f1"]
        )
        features = KeyedJaggedTensor(
            keys=["f1"],
            values=torch.as_tensor([0, 1]),
            lengths=torch.as_tensor([1, 1]),
        )
        self._test_ec([eb1_config, eb2_config], features)
