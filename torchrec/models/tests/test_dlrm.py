#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch import nn
from torch.testing import FileCheck  # @manual
from torchrec.datasets.utils import Batch
from torchrec.fx import symbolic_trace
from torchrec.models.dlrm import (
    choose,
    DenseArch,
    DLRM,
    DLRMTrain,
    DLRMV2,
    InteractionArch,
    InteractionV2Arch,
    SparseArch,
)
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class SparseArchTest(unittest.TestCase):
    def test_basic(self) -> None:
        torch.manual_seed(0)

        D = 3
        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=D, num_embeddings=10, feature_names=["f1", "f3"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=D,
            num_embeddings=10,
            feature_names=["f2"],
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

        sparse_features = sparse_arch(features)
        F = len(sparse_arch.sparse_feature_names)
        self.assertEqual(sparse_features.shape, (B, F, D))

        expected_values = torch.tensor(
            [
                [
                    [-0.7499, -1.2665, 1.0143],
                    [-0.7499, -1.2665, 1.0143],
                    [3.2276, 2.9643, -0.3816],
                ],
                [
                    [0.0082, 0.6241, -0.1119],
                    [0.0082, 0.6241, -0.1119],
                    [2.0722, -2.2734, -1.6307],
                ],
            ]
        )

        self.assertTrue(
            torch.allclose(
                sparse_features,
                expected_values,
                rtol=1e-4,
                atol=1e-4,
            ),
        )

    def test_fx_and_shape(self) -> None:
        D = 3
        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=D, num_embeddings=10, feature_names=["f1", "f3"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=D,
            num_embeddings=10,
            feature_names=["f2"],
        )

        ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])
        sparse_arch = SparseArch(ebc)
        F = len(sparse_arch.sparse_feature_names)
        gm = symbolic_trace(sparse_arch)

        # pyre-fixme[16]: `FileCheck` has no attribute `check`.
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

        sparse_features = gm(features)
        self.assertEqual(sparse_features.shape, (B, F, D))

    # TODO(T89043538): Auto-generate this test.
    def test_fx_script(self) -> None:
        D = 3
        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=D, num_embeddings=10, feature_names=["f1"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2", embedding_dim=D, num_embeddings=10, feature_names=["f2"]
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
        inter_arch = InteractionArch(num_sparse_features=F)

        dense_features = torch.rand((B, D))

        sparse_features = torch.rand((B, F, D))
        concat_dense = inter_arch(dense_features, sparse_features)
        #  B X (D + F + F choose 2)
        self.assertEqual(concat_dense.size(), (B, D + F + choose(F, 2)))

    def test_larger(self) -> None:
        D = 8
        B = 20
        keys = ["f1", "f2", "f3", "f4"]
        F = len(keys)
        inter_arch = InteractionArch(num_sparse_features=F)

        dense_features = torch.rand((B, D))
        sparse_features = torch.rand((B, F, D))

        concat_dense = inter_arch(dense_features, sparse_features)
        #  B X (D + F + F choose 2)
        self.assertEqual(concat_dense.size(), (B, D + F + choose(F, 2)))

    def test_fx_and_shape(self) -> None:
        D = 3
        B = 10
        keys = ["f1", "f2"]
        F = len(keys)
        inter_arch = InteractionArch(num_sparse_features=F)

        gm = symbolic_trace(inter_arch)

        dense_features = torch.rand((B, D))
        sparse_features = torch.rand((B, F, D))

        concat_dense = gm(dense_features, sparse_features)
        #  B X (D + F + F choose 2)
        self.assertEqual(concat_dense.size(), (B, D + F + choose(F, 2)))

    # TODO(T89043538): Auto-generate this test.
    def test_fx_script(self) -> None:
        D = 3
        B = 10
        keys = ["f1", "f2"]
        F = len(keys)
        inter_arch = InteractionArch(num_sparse_features=F)

        gm = symbolic_trace(inter_arch)
        scripted_gm = torch.jit.script(gm)

        dense_features = torch.rand((B, D))
        sparse_features = torch.rand((B, F, D))

        concat_dense = scripted_gm(dense_features, sparse_features)
        #  B X (D + F + F choose 2)
        self.assertEqual(concat_dense.size(), (B, D + F + choose(F, 2)))

    def test_correctness(self) -> None:
        D = 4
        B = 3
        keys = [
            "f1",
            "f2",
            "f3",
            "f4",
        ]
        F = len(keys)
        inter_arch = InteractionArch(num_sparse_features=F)
        torch.manual_seed(0)

        dense_features = torch.rand((B, D))
        sparse_features = torch.rand((B, F, D))

        concat_dense = inter_arch(dense_features, sparse_features)
        #  B X (D + F + F choose 2)
        self.assertEqual(concat_dense.size(), (B, D + F + choose(F, 2)))

        expected = torch.tensor(
            [
                [
                    0.4963,
                    0.7682,
                    0.0885,
                    0.1320,
                    0.2353,
                    1.0123,
                    1.1919,
                    0.7220,
                    0.3444,
                    0.7397,
                    0.4015,
                    1.5184,
                    0.8986,
                    1.2018,
                ],
                [
                    0.3074,
                    0.6341,
                    0.4901,
                    0.8964,
                    1.2787,
                    0.3275,
                    1.6734,
                    0.6325,
                    0.2089,
                    1.2982,
                    0.3977,
                    0.4200,
                    0.2475,
                    0.7834,
                ],
                [
                    0.4556,
                    0.6323,
                    0.3489,
                    0.4017,
                    0.8195,
                    1.1181,
                    1.0511,
                    0.4919,
                    1.6147,
                    1.0786,
                    0.4264,
                    1.3576,
                    0.5860,
                    0.6559,
                ],
            ]
        )

        self.assertTrue(
            torch.allclose(
                concat_dense,
                expected,
                rtol=1e-4,
                atol=1e-4,
            )
        )

    def test_numerical_stability(self) -> None:
        D = 3
        B = 6
        keys = ["f1", "f2"]
        F = len(keys)

        inter_arch = InteractionArch(num_sparse_features=F)
        torch.manual_seed(0)
        dense_features = torch.randint(0, 10, (B, D))

        sparse_features = torch.randint(0, 10, (B, F, D))

        concat_dense = inter_arch(dense_features, sparse_features)

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
        # pyre-fixme[16]: `FileCheck` has no attribute `check`.
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


class DLRMTrainTest(unittest.TestCase):
    def test_basic(self) -> None:
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
        dlrm_module = DLRM(
            embedding_bag_collection=ebc,
            dense_in_features=dense_in_features,
            dense_arch_layer_sizes=[20, D],
            over_arch_layer_sizes=[5, 1],
        )
        dlrm = DLRMTrain(dlrm_module)

        features = torch.rand((B, dense_in_features))
        sparse_features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f2"],
            values=torch.tensor(range(3)),
            offsets=torch.tensor([0, 2, 3]),
        )
        batch = Batch(
            dense_features=features,
            sparse_features=sparse_features,
            labels=torch.randint(2, (B,)),
        )

        _, (_, logits, _) = dlrm(batch)
        self.assertEqual(logits.size(), (B,))


class InteractionV2ArchTest(unittest.TestCase):
    def test_basic(self) -> None:
        D = 3
        B = 10
        keys = ["f1", "f2"]
        F = len(keys)
        F1 = 2
        F2 = 2
        I1 = DenseArch(
            in_features=2 * D + D,
            layer_sizes=[2 * D, F1 * D],
        )
        I2 = DenseArch(
            in_features=2 * D + D,
            layer_sizes=[2 * D, F2 * D],
        )
        inter_arch = InteractionV2Arch(
            num_sparse_features=F,
            interaction_branch1=I1,
            interaction_branch2=I2,
        )

        dense_features = torch.rand((B, D))

        sparse_features = torch.rand((B, F, D))
        concat_dense = inter_arch(dense_features, sparse_features)
        #  B X (D + F1 * F2)
        self.assertEqual(concat_dense.size(), (B, D + F1 * F2))

    def test_larger(self) -> None:
        D = 8
        B = 20
        keys = ["f1", "f2", "f3", "f4"]
        F = len(keys)
        F1 = 4
        F2 = 4
        I1 = DenseArch(
            in_features=4 * D + D,
            layer_sizes=[4 * D, F1 * D],  # F1 = 4
        )
        I2 = DenseArch(
            in_features=4 * D + D,
            layer_sizes=[4 * D, F2 * D],  # F2 = 4
        )
        inter_arch = InteractionV2Arch(
            num_sparse_features=F,
            interaction_branch1=I1,
            interaction_branch2=I2,
        )

        dense_features = torch.rand((B, D))
        sparse_features = torch.rand((B, F, D))

        concat_dense = inter_arch(dense_features, sparse_features)
        #  B X (D + F1 * F2)
        self.assertEqual(concat_dense.size(), (B, D + F1 * F2))

    def test_correctness(self) -> None:
        D = 4
        B = 3
        keys = [
            "f1",
            "f2",
            "f3",
            "f4",
        ]
        F = len(keys)
        F1 = 5
        F2 = 5
        I1 = nn.Identity()
        I2 = nn.Identity()
        inter_arch = InteractionV2Arch(
            num_sparse_features=F,
            interaction_branch1=I1,
            interaction_branch2=I2,
        )
        torch.manual_seed(0)

        dense_features = torch.rand((B, D))
        sparse_features = torch.rand((B, F, D))

        concat_dense = inter_arch(dense_features, sparse_features)
        #  B X (D + F1 * F2)
        self.assertEqual(concat_dense.size(), (B, D + F1 * F2))

        expected = torch.tensor(
            [
                [
                    0.4963,
                    0.7682,
                    0.0885,
                    0.1320,
                    0.5057,
                    0.6874,
                    0.5756,
                    0.8082,
                    0.6656,
                    0.5402,
                    0.3672,
                    0.5765,
                    0.8837,
                    0.2710,
                    0.7540,
                    0.9349,
                    0.7424,
                    1.0666,
                    0.7297,
                    1.3209,
                    1.2713,
                    1.2888,
                    1.9248,
                    0.9367,
                    0.4865,
                    0.7688,
                    0.9932,
                    1.3475,
                    0.8313,
                ],
                [
                    0.3074,
                    0.6341,
                    0.4901,
                    0.8964,
                    1.0706,
                    0.8757,
                    1.0621,
                    1.3669,
                    0.6122,
                    0.9342,
                    0.7316,
                    0.7294,
                    1.0603,
                    0.3866,
                    0.2011,
                    0.2153,
                    0.3768,
                    0.3638,
                    0.2154,
                    1.0712,
                    0.8293,
                    1.3000,
                    1.4564,
                    0.8369,
                    0.3655,
                    0.4440,
                    0.6148,
                    1.0776,
                    0.5871,
                ],
                [
                    0.4556,
                    0.6323,
                    0.3489,
                    0.4017,
                    0.7294,
                    1.3899,
                    0.9493,
                    0.6186,
                    0.7565,
                    0.9535,
                    1.5688,
                    0.8992,
                    0.7077,
                    1.0088,
                    1.1206,
                    1.9778,
                    1.1639,
                    0.8642,
                    1.1966,
                    1.1827,
                    1.8592,
                    1.3003,
                    0.9441,
                    1.1177,
                    0.4730,
                    0.7631,
                    0.4304,
                    0.3937,
                    0.3230,
                ],
            ]
        )

        self.assertTrue(
            torch.allclose(
                concat_dense,
                expected,
                rtol=1e-4,
                atol=1e-4,
            )
        )


class DLRMV2Test(unittest.TestCase):
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
        sparse_nn = DLRMV2(
            embedding_bag_collection=ebc,
            dense_in_features=dense_in_features,
            dense_arch_layer_sizes=[20, D],
            interaction_branch1_layer_sizes=[3 * D + D, 4 * D],
            interaction_branch2_layer_sizes=[3 * D + D, 4 * D],
            over_arch_layer_sizes=[5, 1],
        )
        self.assertTrue(isinstance(sparse_nn.inter_arch, InteractionV2Arch))

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

        expected_logits = torch.tensor([[-0.0036], [-0.0260]])
        self.assertTrue(
            torch.allclose(
                logits,
                expected_logits,
                rtol=1e-4,
                atol=1e-4,
            )
        )

    def test_dense_size(self) -> None:
        torch.manual_seed(0)
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
        with self.assertRaises(ValueError):
            sparse_nn = DLRMV2(  # noqa
                embedding_bag_collection=ebc,
                dense_in_features=dense_in_features,
                dense_arch_layer_sizes=[20, D + 1],
                interaction_branch1_layer_sizes=[3 * D + D, 4 * D],
                interaction_branch2_layer_sizes=[3 * D + D, 4 * D],
                over_arch_layer_sizes=[5, 1],
            )

    def test_interaction_size(self) -> None:
        torch.manual_seed(0)
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
        with self.assertRaises(ValueError):
            sparse_nn = DLRMV2(
                embedding_bag_collection=ebc,
                dense_in_features=dense_in_features,
                dense_arch_layer_sizes=[20, D],
                interaction_branch1_layer_sizes=[3 * D + D, 4 * D + 1],
                interaction_branch2_layer_sizes=[3 * D + D, 4 * D],
                over_arch_layer_sizes=[5, 1],
            )

        with self.assertRaises(ValueError):
            sparse_nn = DLRMV2(  # noqa
                embedding_bag_collection=ebc,
                dense_in_features=dense_in_features,
                dense_arch_layer_sizes=[20, D],
                interaction_branch1_layer_sizes=[3 * D + D, 4 * D],
                interaction_branch2_layer_sizes=[3 * D + D, 4 * D + 1],
                over_arch_layer_sizes=[5, 1],
            )


class DLRMV2TrainTest(unittest.TestCase):
    def test_basic(self) -> None:
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
        dlrmv2_module = DLRMV2(
            embedding_bag_collection=ebc,
            dense_in_features=dense_in_features,
            dense_arch_layer_sizes=[20, D],
            over_arch_layer_sizes=[5, 1],
            interaction_branch1_layer_sizes=[80, 40],
            interaction_branch2_layer_sizes=[80, 48],
        )
        dlrm = DLRMTrain(dlrmv2_module)

        features = torch.rand((B, dense_in_features))
        sparse_features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f2"],
            values=torch.tensor(range(3)),
            offsets=torch.tensor([0, 2, 3]),
        )
        batch = Batch(
            dense_features=features,
            sparse_features=sparse_features,
            labels=torch.randint(2, (B,)),
        )

        _, (_, logits, _) = dlrm(batch)
        self.assertEqual(logits.size(), (B,))
