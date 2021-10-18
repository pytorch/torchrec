#!/usr/bin/env python3

import math
import unittest
from itertools import combinations
from typing import List

import torch
from torch.testing import FileCheck  # @manual
from torchrec.fx import symbolic_trace
from torchrec.models.dlrm import (
    SparseArch,
    DenseArch,
    InteractionArch,
    DLRM,
)
from torchrec.modules.embedding_configs import (
    EmbeddingBagConfig,
)
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor


class SparseArchTest(unittest.TestCase):
    def test_basic(self) -> None:
        torch.manual_seed(0)
        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1", "f3"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=4,
            num_embeddings=10,
            feature_names=["f2"],
        )
        D = sum(
            eb_config.embedding_dim * len(eb_config.feature_names)
            for eb_config in [eb1_config, eb2_config]
        )

        ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])
        sparse_arch = SparseArch(ebc)

        keys = ["f1", "f2", "f3", "f4", "f5"]
        offsets = torch.tensor([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 19])
        features = KeyedJaggedTensor.from_offsets_sync(
            keys=keys,
            values=torch.tensor(
                [1, 2, 4, 5, 4, 3, 2, 9, 1, 2, 4, 5, 4, 3, 2, 9, 1, 2, 3]
            ),
            offsets=offsets,
        )
        B = (len(offsets) - 1) // len(keys)

        kt = sparse_arch(features)
        self.assertEqual(kt.values().size(), (B, D))
        self.assertEqual(kt.keys(), ["f1", "f3", "f2"])
        self.assertEqual(kt.offset_per_key(), [0, 3, 6, 10])

        expected_values = torch.tensor(
            [
                [
                    -0.7499,
                    -1.2665,
                    1.0143,
                    -0.7499,
                    -1.2665,
                    1.0143,
                    2.0283,
                    2.8195,
                    -2.1004,
                    -0.3142,
                ],
                [
                    0.0082,
                    0.6241,
                    -0.1119,
                    0.0082,
                    0.6241,
                    -0.1119,
                    -0.6147,
                    3.3314,
                    -0.8118,
                    -0.5584,
                ],
            ]
        )
        self.assertTrue(
            torch.allclose(
                kt.values(),
                expected_values,
                rtol=1e-4,
                atol=1e-4,
            ),
        )

    def test_fx_and_shape(self) -> None:
        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1", "f3"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=4,
            num_embeddings=10,
            feature_names=["f2"],
        )

        D = sum(
            eb_config.embedding_dim * len(eb_config.feature_names)
            for eb_config in [eb1_config, eb2_config]
        )

        ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])
        sparse_arch = SparseArch(ebc)
        gm = symbolic_trace(sparse_arch)

        FileCheck().check("KeyedJaggedTensor").check("cat").run(gm.code)

        keys = ["f1", "f2", "f3", "f4", "f5"]
        offsets = torch.tensor([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 19])
        features = KeyedJaggedTensor.from_offsets_sync(
            keys=keys,
            values=torch.tensor(
                [1, 2, 4, 5, 4, 3, 2, 9, 1, 2, 4, 5, 4, 3, 2, 9, 1, 2, 3]
            ),
            offsets=offsets,
        )
        B = (len(offsets) - 1) // len(keys)

        kt = gm(features)
        self.assertEqual(kt.values().size(), (B, D))
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
        torch.manual_seed(0)
        B = 4
        D = 3
        in_features = 10
        dense_arch = DenseArch(in_features=in_features, layer_sizes=[10, D])
        dense_embedded = dense_arch(torch.rand((B, in_features)))
        self.assertEqual(dense_embedded.size(), (B, D))

        expected = torch.tensor(
            [
                [0.2351, 0.1578, 0.2784],
                [0.1579, 0.1012, 0.2660],
                [0.2459, 0.2379, 0.2749],
                [0.2582, 0.2178, 0.2860],
            ]
        )
        self.assertTrue(
            torch.allclose(
                dense_embedded,
                expected,
                rtol=1e-4,
                atol=1e-4,
            )
        )

    def test_fx_and_shape(self) -> None:
        B = 20
        D = 3
        in_features = 10
        dense_arch = DenseArch(in_features=in_features, layer_sizes=[10, D])
        gm = symbolic_trace(dense_arch)
        dense_embedded = gm(torch.rand((B, in_features)))
        self.assertEqual(dense_embedded.size(), (B, D))

    # TODO(T89043538): Auto-generate this test.
    def test_fx_script(self) -> None:
        B = 20
        D = 3
        in_features = 10
        dense_arch = DenseArch(in_features=in_features, layer_sizes=[10, D])
        gm = symbolic_trace(dense_arch)
        scripted_gm = torch.jit.script(gm)
        dense_embedded = scripted_gm(torch.rand((B, in_features)))
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

    def test_fx_and_shape(self) -> None:
        D = 3
        B = 10
        keys = ["f1", "f2"]
        F = len(keys)
        inter_arch = InteractionArch(sparse_feature_names=keys)
        gm = symbolic_trace(inter_arch)

        dense_features = torch.rand((B, D))

        embeddings = KeyedTensor(
            keys=keys,
            length_per_key=[D] * F,
            values=torch.rand((B, D * F)),
        )

        concat_dense = gm(dense_features, embeddings)
        #  B X (D + F + F choose 2)
        self.assertEqual(concat_dense.size(), (B, D + F + math.comb(F, 2)))

    # TODO(T89043538): Auto-generate this test.
    def test_fx_script(self) -> None:
        D = 3
        B = 10
        keys = ["f1", "f2"]
        F = len(keys)
        inter_arch = InteractionArch(sparse_feature_names=keys)
        gm = symbolic_trace(inter_arch)
        scripted_gm = torch.jit.script(gm)

        dense_features = torch.rand((B, D))

        embeddings = KeyedTensor(
            keys=keys,
            length_per_key=[D] * F,
            values=torch.rand((B, D * F)),
        )

        concat_dense = scripted_gm(dense_features, embeddings)
        #  B X (D + F + F choose 2)
        self.assertEqual(concat_dense.size(), (B, D + F + math.comb(F, 2)))

    def test_correctness(self) -> None:
        D = 11
        B = 25
        keys = ["f1", "f2", "f3", "f4", "f5", "f6"]
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

        expected = self._test_correctness_helper(
            dense_features=dense_features,
            sparse_features=embeddings,
            sparse_feature_names=keys,
        )
        self.assertTrue(
            torch.allclose(
                concat_dense,
                expected,
                rtol=1e-4,
                atol=1e-4,
            )
        )

    def _test_correctness_helper(
        self,
        dense_features: torch.Tensor,
        sparse_features: KeyedTensor,
        sparse_feature_names: List[str],
    ) -> torch.Tensor:
        interactions: List[torch.Tensor] = []
        # dense/sparse interaction
        # size B X F
        for feature_name in sparse_feature_names:
            sparse_values = sparse_features[feature_name]
            dots = torch.sum(sparse_values * dense_features, dim=1)
            # dots is size B
            interactions.append(dots)

        # sparse/sparse interaction
        # size B X (F choose 2)
        for (f1, f2) in list(combinations(sparse_feature_names, 2)):
            f1_values = sparse_features[f1]
            f2_values = sparse_features[f2]
            dots = torch.sum(f1_values * f2_values, dim=1)
            interactions.append(dots)

        interactions_tensor = torch.stack(interactions).transpose(1, 0)
        return torch.cat((dense_features, interactions_tensor), dim=1)

    def test_numerical_stability(self) -> None:
        D = 3
        B = 6
        keys = ["f1", "f2"]
        F = len(keys)
        inter_arch = InteractionArch(sparse_feature_names=keys)
        torch.manual_seed(0)
        dense_features = torch.randint(0, 10, (B, D))

        embeddings = KeyedTensor(
            keys=keys,
            length_per_key=[D] * F,
            values=torch.randint(0, 10, (B, D * F)),
        )

        concat_dense = inter_arch(dense_features, embeddings)
        expected = torch.LongTensor(
            [
                [4, 9, 3, 61, 57, 63],
                [0, 3, 9, 84, 27, 45],
                [7, 3, 7, 34, 50, 25],
                [3, 1, 6, 21, 50, 91],
                [6, 9, 8, 125, 109, 74],
                [6, 6, 8, 18, 80, 21],
            ]
        )

        self.assertTrue(torch.equal(concat_dense, expected))


class DLRMTest(unittest.TestCase):
    def test_basic(self) -> None:
        torch.manual_seed(0)
        B = 2
        D = 8
        dense_in_features = 100

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
        sparse_nn = DLRM(
            embedding_bag_collection=ebc,
            dense_in_features=dense_in_features,
            dense_arch_layer_sizes=[20, D],
            over_arch_layer_sizes=[5, 1],
        )

        features = torch.rand((B, dense_in_features))
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

        expected_logits = torch.tensor([[0.5805], [0.5909]])
        self.assertTrue(
            torch.allclose(
                logits,
                expected_logits,
                rtol=1e-4,
                atol=1e-4,
            )
        )

    def test_one_sparse(self) -> None:
        B = 2
        D = 8
        dense_in_features = 100

        eb1_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=D,
            num_embeddings=100,
            feature_names=["f2"],
        )

        ebc = EmbeddingBagCollection(tables=[eb1_config])
        sparse_nn = DLRM(
            embedding_bag_collection=ebc,
            dense_in_features=dense_in_features,
            dense_arch_layer_sizes=[20, D],
            over_arch_layer_sizes=[5, 1],
        )

        features = torch.rand((B, dense_in_features))
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
        D_unused = 1
        with self.assertRaises(AssertionError):
            DLRM(
                embedding_bag_collection=ebc,
                dense_in_features=100,
                dense_arch_layer_sizes=[20, D_unused],
                over_arch_layer_sizes=[5, 1],
            )

    def test_fx(self) -> None:
        B = 2
        D = 8
        dense_in_features = 100

        eb1_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=D,
            num_embeddings=100,
            feature_names=["f2"],
        )

        ebc = EmbeddingBagCollection(tables=[eb1_config])
        sparse_nn = DLRM(
            embedding_bag_collection=ebc,
            dense_in_features=dense_in_features,
            dense_arch_layer_sizes=[20, D],
            over_arch_layer_sizes=[5, 1],
        )
        gm = symbolic_trace(sparse_nn)
        FileCheck().check("KeyedJaggedTensor").check("cat").check("f2").run(gm.code)

        features = torch.rand((B, dense_in_features))
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
        dense_in_features = 100

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
        sparse_nn = DLRM(
            embedding_bag_collection=ebc,
            dense_in_features=dense_in_features,
            dense_arch_layer_sizes=[20, D],
            over_arch_layer_sizes=[5, 1],
        )

        features = torch.rand((B, dense_in_features))
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
