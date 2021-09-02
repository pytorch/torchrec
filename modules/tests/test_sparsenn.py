#!/usr/bin/env python3

import math
import unittest

import torch
from torch.testing import FileCheck  # @manual
from torchrec.distributed.types import NoWait
from torchrec.fx import symbolic_trace
from torchrec.modules.embedding_configs import (
    EmbeddingBagConfig,
)
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
)
from torchrec.modules.sparsenn import (
    SparseArch,
    DenseArch,
    InteractionArch,
    SimpleSparseNN,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor


class SparseArchTest(unittest.TestCase):
    def test_basic(self) -> None:
        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1", "f3"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=4,
            num_embeddings=10,
            feature_names=["f2"],
        )

        ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])
        sparse_arch = SparseArch(ebc)

        features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f2", "f3", "f4", "f5"],
            values=torch.tensor(
                [1, 2, 4, 5, 4, 3, 2, 9, 1, 2, 4, 5, 4, 3, 2, 9, 1, 2, 3]
            ),
            offsets=torch.tensor([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 19]),
        )

        kt = sparse_arch(features)
        self.assertEqual(kt.values().size(), (2, 10))
        self.assertEqual(kt.keys(), ["f1", "f3", "f2"])
        self.assertEqual(kt.offset_per_key(), [0, 3, 6, 10])

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

        ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])
        sparse_arch = SparseArch(ebc)
        gm = symbolic_trace(sparse_arch)

        FileCheck().check("KeyedJaggedTensor").check("cat").run(gm.code)

        features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f2", "f3", "f4", "f5"],
            values=torch.tensor(
                [1, 2, 4, 5, 4, 3, 2, 9, 1, 2, 4, 5, 4, 3, 2, 9, 1, 2, 3]
            ),
            offsets=torch.tensor([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 19]),
        )

        kt = gm(features)
        self.assertEqual(kt.values().size(), (2, 10))
        self.assertEqual(kt.keys(), ["f1", "f3", "f2"])
        self.assertEqual(kt.offset_per_key(), [0, 3, 6, 10])

    # TODO(T89043538): Auto-generate this test.
    def test_fx_script(self) -> None:
        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2", embedding_dim=4, num_embeddings=10, feature_names=["f2"]
        )

        ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])
        sparse_arch = SparseArch(ebc)

        gm = symbolic_trace(sparse_arch)
        torch.jit.script(gm)


class DenseArchTest(unittest.TestCase):
    def test_basic(self) -> None:
        B = 20
        D = 3
        dense_arch = DenseArch(hidden_layer_size=10, embedding_dim=D)
        dense_embedded = dense_arch(torch.rand((B, 10)))

        self.assertEqual(dense_embedded.size(), (B, D))


class InteractionArchTest(unittest.TestCase):
    def test_basic(self) -> None:
        D = 3
        B = 10
        keys = ["f1", "f2"]
        F = len(keys)
        inter_arch = InteractionArch(sparse_feature_names=keys)

        dense_features = torch.rand((B, D))

        embeddings = KeyedTensor(
            keys=keys,
            length_per_key=[D] * F,
            values=torch.rand((B, D * F)),
        )
        concat_dense = inter_arch(dense_features, embeddings)
        #  B X (D + F + F choose 2)
        self.assertEqual(concat_dense.size(), (B, D + F + math.comb(F, 2)))

    def test_larger(self) -> None:
        D = 8
        B = 20
        keys = ["f1", "f2", "f3", "f4"]
        F = len(keys)
        inter_arch = InteractionArch(sparse_feature_names=keys)

        dense_features = torch.rand((B, D))

        embeddings = KeyedTensor(
            keys=keys,
            length_per_key=[D] * F,
            values=torch.rand((B, D * F)),
        )

        concat_dense = inter_arch(dense_features, embeddings)
        #  B X (D + F + F choose 2)
        self.assertEqual(concat_dense.size(), (B, D + F + math.comb(F, 2)))


class SimpleSparseNNTest(unittest.TestCase):
    def test_basic(self) -> None:
        B = 2
        D = 8

        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=D, num_embeddings=100, feature_names=["f1", "f3"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=D,
            num_embeddings=100,
            feature_names=["f2"],
        )

        ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])
        sparse_nn = SimpleSparseNN(
            embedding_bag_collection=ebc, hidden_layer_size=20, over_embedding_dim=5
        )

        features = torch.rand((B, 100))
        sparse_features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f3", "f2"],
            values=torch.tensor([1, 2, 4, 5, 4, 3, 2, 9, 1, 2, 3]),
            offsets=torch.tensor([0, 2, 4, 6, 8, 10, 11]),
        )

        logits = sparse_nn(
            dense_features=features,
            sparse_features=sparse_features,
        )
        self.assertEqual(logits.size(), (B, 1))

    def test_one_sparse(self) -> None:
        B = 2
        D = 8

        eb1_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=D,
            num_embeddings=100,
            feature_names=["f2"],
        )

        ebc = EmbeddingBagCollection(tables=[eb1_config])
        sparse_nn = SimpleSparseNN(
            embedding_bag_collection=ebc, hidden_layer_size=20, over_embedding_dim=5
        )

        features = torch.rand((B, 100))
        sparse_features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f2"],
            values=torch.tensor(range(3)),
            offsets=torch.tensor([0, 2, 3]),
        )

        logits = sparse_nn(
            dense_features=features,
            sparse_features=sparse_features,
        )
        self.assertEqual(logits.size(), (B, 1))

    def test_no_sparse(self) -> None:
        ebc = EmbeddingBagCollection(tables=[])
        with self.assertRaises(AssertionError):
            SimpleSparseNN(
                embedding_bag_collection=ebc, hidden_layer_size=20, over_embedding_dim=5
            )

    def test_fx(self) -> None:
        B = 2
        D = 8

        eb1_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=D,
            num_embeddings=100,
            feature_names=["f2"],
        )

        ebc = EmbeddingBagCollection(tables=[eb1_config])
        sparse_nn = SimpleSparseNN(
            embedding_bag_collection=ebc, hidden_layer_size=20, over_embedding_dim=5
        )
        gm = symbolic_trace(sparse_nn)
        FileCheck().check("KeyedJaggedTensor").check("cat").check("f2").run(gm.code)

        features = torch.rand((B, 100))
        sparse_features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f2"],
            values=torch.tensor(range(3)),
            offsets=torch.tensor([0, 2, 3]),
        )

        logits = gm(
            dense_features=features,
            sparse_features=sparse_features,
        )
        self.assertEqual(logits.size(), (B, 1))

    # TODO(T89043538): Auto-generate this test.
    def test_fx_script(self) -> None:
        B = 2
        D = 8

        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=D, num_embeddings=100, feature_names=["f1", "f3"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=D,
            num_embeddings=100,
            feature_names=["f2"],
        )

        ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])
        sparse_nn = SimpleSparseNN(
            embedding_bag_collection=ebc, hidden_layer_size=20, over_embedding_dim=5
        )

        features = torch.rand((B, 100))
        sparse_features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f3", "f2"],
            values=torch.tensor([1, 2, 4, 5, 4, 3, 2, 9, 1, 2, 3]),
            offsets=torch.tensor([0, 2, 4, 6, 8, 10, 11]),
        )

        sparse_nn(
            dense_features=features,
            sparse_features=sparse_features,
        )

        gm = symbolic_trace(sparse_nn)

        scripted_gm = torch.jit.script(gm)

        logits = scripted_gm(features, sparse_features)
        self.assertEqual(logits.size(), (B, 1))
