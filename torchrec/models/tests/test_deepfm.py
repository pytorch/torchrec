#!/usr/bin/env python3

import unittest

import torch
from torch.testing import FileCheck  # @manual
from torchrec.fx import Tracer
from torchrec.fx import symbolic_trace
from torchrec.models.deepfm import (
    FMInteractionArch,
    SimpleDeepFMNN,
)
from torchrec.modules.embedding_configs import (
    EmbeddingBagConfig,
)
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor


class FMInteractionArchTest(unittest.TestCase):
    def test_basic(self) -> None:
        torch.manual_seed(0)

        D = 3
        B = 3
        DI = 2
        keys = ["f1", "f2"]
        F = len(keys)
        dense_features = torch.rand((B, D))

        embeddings = KeyedTensor(
            keys=keys,
            length_per_key=[D] * F,
            values=torch.rand((B, D * F)),
        )
        inter_arch = FMInteractionArch(
            fm_in_features=D + D * F,
            sparse_feature_names=keys,
            deep_fm_dimension=DI,
        )
        inter_output = inter_arch(dense_features, embeddings)
        self.assertEqual(inter_output.size(), (B, D + DI + 1))

        # check output forward numerical accuracy
        expected_output = torch.Tensor(
            [
                [0.4963, 0.7682, 0.0885, 0.0000, 0.2646, 4.3660],
                [0.1320, 0.3074, 0.6341, 0.0000, 0.0834, 7.6417],
                [0.4901, 0.8964, 0.4556, 0.0000, 0.0671, 15.5230],
            ],
        )
        self.assertTrue(
            torch.allclose(
                inter_output,
                expected_output,
                rtol=1e-4,
                atol=1e-4,
            )
        )

        # check tracer compatibility
        gm = torch.fx.GraphModule(inter_arch, Tracer().trace(inter_arch))
        torch.jit.script(gm)


class SimpleDeepFMNNTest(unittest.TestCase):
    def test_basic(self) -> None:
        B = 2
        D = 8
        num_dense_features = 100
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

        features = torch.rand((B, num_dense_features))
        sparse_features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f3", "f2"],
            values=torch.tensor([1, 2, 4, 5, 4, 3, 2, 9, 1, 2, 3]),
            offsets=torch.tensor([0, 2, 4, 6, 8, 10, 11]),
        )

        deepfm_nn = SimpleDeepFMNN(
            num_dense_features=num_dense_features,
            embedding_bag_collection=ebc,
            hidden_layer_size=20,
            deep_fm_dimension=5,
        )

        logits = deepfm_nn(
            dense_features=features,
            sparse_features=sparse_features,
        )
        self.assertEqual(logits.size(), (B, 1))

    def test_no_sparse(self) -> None:
        ebc = EmbeddingBagCollection(tables=[])
        with self.assertRaises(AssertionError):
            SimpleDeepFMNN(
                num_dense_features=10,
                embedding_bag_collection=ebc,
                hidden_layer_size=20,
                deep_fm_dimension=5,
            )

    def test_fx(self) -> None:
        B = 2
        D = 8
        num_dense_features = 100

        eb1_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=D,
            num_embeddings=100,
            feature_names=["f2"],
        )

        ebc = EmbeddingBagCollection(tables=[eb1_config])
        deepfm_nn = SimpleDeepFMNN(
            num_dense_features=num_dense_features,
            embedding_bag_collection=ebc,
            hidden_layer_size=20,
            deep_fm_dimension=5,
        )
        gm = symbolic_trace(deepfm_nn)
        FileCheck().check("KeyedJaggedTensor").check("cat").check("f2").run(gm.code)

        features = torch.rand((B, num_dense_features))
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

    def test_fx_script(self) -> None:
        B = 2
        D = 8
        num_dense_features = 100

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
        deepfm_nn = SimpleDeepFMNN(
            num_dense_features=num_dense_features,
            embedding_bag_collection=ebc,
            hidden_layer_size=20,
            deep_fm_dimension=5,
        )

        features = torch.rand((B, num_dense_features))
        sparse_features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f3", "f2"],
            values=torch.tensor([1, 2, 4, 5, 4, 3, 2, 9, 1, 2, 3]),
            offsets=torch.tensor([0, 2, 4, 6, 8, 10, 11]),
        )

        deepfm_nn(
            dense_features=features,
            sparse_features=sparse_features,
        )

        gm = symbolic_trace(deepfm_nn)

        scripted_gm = torch.jit.script(gm)

        logits = scripted_gm(features, sparse_features)
        self.assertEqual(logits.size(), (B, 1))


if __name__ == "__main__":
    unittest.main()
