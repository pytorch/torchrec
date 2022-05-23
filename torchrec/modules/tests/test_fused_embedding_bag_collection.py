#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest
from collections import OrderedDict
from typing import Optional

import torch
import torch.fx
from fbgemm_gpu.split_table_batched_embeddings_ops import EmbeddingLocation
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.modules.fused_embedding_bag_collection import (
    fuse_optimizer,
    FusedEmbeddingBagCollection,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class FusedEmbeddingBagCollectionTest(unittest.TestCase):
    def test_unweighted(self) -> None:
        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=4, num_embeddings=10, feature_names=["f1"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2", embedding_dim=4, num_embeddings=10, feature_names=["f2"]
        )
        eb3_config = EmbeddingBagConfig(
            name="t3", embedding_dim=4, num_embeddings=10, feature_names=["f3"]
        )

        ebc = FusedEmbeddingBagCollection(
            tables=[eb1_config, eb2_config, eb3_config],
            optimizer_type=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.02},
        )

        #     0       1        2  <-- batch
        # f1   [0,1] None    [2]
        # f2   [3]    [4]    [5,6,7]
        # f3   []    [8]    []
        # ^
        # feature
        features = KeyedJaggedTensor.from_lengths_sync(
            keys=["f1", "f2", "f3"],
            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8]),
            lengths=torch.tensor([2, 0, 1, 1, 1, 3, 0, 1, 0]),
        )

        pooled_embeddings = ebc(features)

        self.assertEqual(pooled_embeddings.keys(), ["f1", "f2", "f3"])
        self.assertEqual(pooled_embeddings["f1"].shape, (features.stride(), 4))
        self.assertEqual(pooled_embeddings["f2"].shape, (features.stride(), 4))
        self.assertEqual(pooled_embeddings["f3"].shape, (features.stride(), 4))

        torch.testing.assert_close(pooled_embeddings["f1"][1], torch.zeros(4))
        torch.testing.assert_close(pooled_embeddings["f3"][0], torch.zeros(4))
        torch.testing.assert_close(pooled_embeddings["f3"][2], torch.zeros(4))

    def test_shared_tables(self) -> None:
        ebc_config = EmbeddingBagConfig(
            name="t1", embedding_dim=4, num_embeddings=10, feature_names=["f1", "f2"]
        )
        ebc = FusedEmbeddingBagCollection(
            tables=[ebc_config],
            optimizer_type=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.02},
        )

        #     0       1        2  <-- batch
        # f1   [0,1] None    [2]
        # f2   [3]    [4]    [5,6,7]
        # ^
        # feature
        features = KeyedJaggedTensor.from_lengths_sync(
            keys=["f1", "f2"],
            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
            lengths=torch.tensor([2, 0, 1, 1, 1, 3]),
        )

        pooled_embeddings = ebc(features)

        self.assertEqual(pooled_embeddings.keys(), ["f1", "f2"])
        self.assertEqual(pooled_embeddings["f1"].shape, (features.stride(), 4))
        self.assertEqual(pooled_embeddings["f2"].shape, (features.stride(), 4))

        torch.testing.assert_close(pooled_embeddings["f1"][1], torch.zeros(4))

    def test_state_dict(self) -> None:
        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=4, num_embeddings=2, feature_names=["f1"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2", embedding_dim=4, num_embeddings=2, feature_names=["f2"]
        )
        eb3_config = EmbeddingBagConfig(
            name="t3", embedding_dim=4, num_embeddings=2, feature_names=["f3"]
        )

        ebc = FusedEmbeddingBagCollection(
            tables=[eb1_config, eb2_config, eb3_config],
            optimizer_type=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.02},
        )

        # pyre-ignore
        ebc.load_state_dict(ebc.state_dict())

    def test_state_dict_manual(self) -> None:
        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=4, num_embeddings=2, feature_names=["f1"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2", embedding_dim=4, num_embeddings=2, feature_names=["f2"]
        )
        eb3_config = EmbeddingBagConfig(
            name="t3", embedding_dim=4, num_embeddings=2, feature_names=["f3"]
        )

        ebc = FusedEmbeddingBagCollection(
            tables=[eb1_config, eb2_config, eb3_config],
            optimizer_type=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.02},
        )

        ebc.load_state_dict(
            OrderedDict(
                [
                    (
                        "embedding_bags.t1.weight",
                        torch.Tensor([[1, 1, 1, 1], [2, 2, 2, 2]]),
                    ),
                    (
                        "embedding_bags.t2.weight",
                        torch.Tensor([[4, 4, 4, 4], [8, 8, 8, 8]]),
                    ),
                    (
                        "embedding_bags.t3.weight",
                        torch.Tensor([[16, 16, 16, 16], [32, 32, 32, 32]]),
                    ),
                ]
            )
        )

        state_dict = ebc.state_dict()
        torch.testing.assert_close(
            state_dict["embedding_bags.t1.weight"],
            torch.Tensor([[1, 1, 1, 1], [2, 2, 2, 2]]),
        ),
        torch.testing.assert_close(
            state_dict["embedding_bags.t2.weight"],
            torch.Tensor([[4, 4, 4, 4], [8, 8, 8, 8]]),
        ),
        torch.testing.assert_close(
            state_dict["embedding_bags.t3.weight"],
            torch.Tensor([[16, 16, 16, 16], [32, 32, 32, 32]]),
        )

        #     0       1        2  <-- batch
        # f1   [0,1] []    [0]
        # f2   [0]    [1]    [0,1]
        # f3   []    []    [0]
        # ^
        # feature
        features = KeyedJaggedTensor.from_lengths_sync(
            keys=["f1", "f2", "f3"],
            values=torch.tensor([0, 1, 0, 0, 1, 0, 1, 0]),
            lengths=torch.tensor([2, 0, 1, 1, 1, 2, 0, 0, 1]),
        )

        pooled_embeddings = ebc(features)

        torch.testing.assert_close(
            pooled_embeddings["f1"],
            torch.tensor(
                [[3, 3, 3, 3], [0, 0, 0, 0], [1, 1, 1, 1]], dtype=torch.float32
            ),
        )
        torch.testing.assert_close(
            pooled_embeddings["f2"],
            torch.tensor(
                [[4, 4, 4, 4], [8, 8, 8, 8], [12, 12, 12, 12]], dtype=torch.float32
            ),
        )
        torch.testing.assert_close(
            pooled_embeddings["f3"],
            torch.tensor(
                [[0, 0, 0, 0], [0, 0, 0, 0], [16, 16, 16, 16]], dtype=torch.float32
            ),
        )

    def test_shared_tables_shared_features(self) -> None:
        eb1_config = EmbeddingBagConfig(
            name="t1",
            embedding_dim=4,
            num_embeddings=2,
            feature_names=["f1", "shared_f1"],
        )
        eb2_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=4,
            num_embeddings=2,
            feature_names=["f2", "shared_f1"],
        )

        ebc = FusedEmbeddingBagCollection(
            tables=[eb1_config, eb2_config],
            optimizer_type=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.02},
        )

        ebc.load_state_dict(
            OrderedDict(
                [
                    (
                        "embedding_bags.t1.weight",
                        torch.Tensor([[1, 1, 1, 1], [2, 2, 2, 2]]),
                    ),
                    (
                        "embedding_bags.t2.weight",
                        torch.Tensor([[4, 4, 4, 4], [8, 8, 8, 8]]),
                    ),
                ]
            )
        )

        #     0       1        2  <-- batch
        # f1   [0,1] []    [0]
        # f2   [0]    [1]    [0,1]
        # shared_f1   []    [0]    [0,1]

        features = KeyedJaggedTensor.from_lengths_sync(
            keys=["f1", "f2", "shared_f1"],
            values=torch.tensor([0, 1, 0, 0, 1, 0, 1, 0, 0, 1]),
            lengths=torch.tensor([2, 0, 1, 1, 1, 2, 0, 1, 2]),
        )

        pooled_embeddings = ebc(features)

        torch.testing.assert_close(
            pooled_embeddings["f1"],
            torch.tensor(
                [[3, 3, 3, 3], [0, 0, 0, 0], [1, 1, 1, 1]], dtype=torch.float32
            ),
        )

        torch.testing.assert_close(
            pooled_embeddings["shared_f1@t1"],
            torch.tensor(
                [
                    [0, 0, 0, 0],
                    [1, 1, 1, 1],
                    [3, 3, 3, 3],
                ],
                dtype=torch.float32,
            ),
        )

        torch.testing.assert_close(
            pooled_embeddings["f2"],
            torch.tensor(
                [[4, 4, 4, 4], [8, 8, 8, 8], [12, 12, 12, 12]], dtype=torch.float32
            ),
        )

        torch.testing.assert_close(
            pooled_embeddings["shared_f1@t2"],
            torch.tensor(
                [[0, 0, 0, 0], [4, 4, 4, 4], [12, 12, 12, 12]], dtype=torch.float32
            ),
        )

    def test_weighted(self) -> None:
        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=4, num_embeddings=2, feature_names=["f1"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2", embedding_dim=4, num_embeddings=2, feature_names=["f2"]
        )

        ebc = FusedEmbeddingBagCollection(
            tables=[eb1_config, eb2_config],
            optimizer_type=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.02},
            is_weighted=True,
        )

        ebc.load_state_dict(
            OrderedDict(
                [
                    (
                        "embedding_bags.t1.weight",
                        torch.Tensor([[1, 1, 1, 1], [2, 2, 2, 2]]),
                    ),
                    (
                        "embedding_bags.t2.weight",
                        torch.Tensor([[4, 4, 4, 4], [8, 8, 8, 8]]),
                    ),
                ]
            )
        )

        #     0       1        2  <-- batch
        # f1   [0,1] []    [0]
        # f2   [0]    [1]    [0,1]
        # feature

        #     0       1        2  <-- batch
        # f1  [1.0,2.0] [] [3.0]
        # f2   [5.0]    [7.0]    [11.0, 13.0]
        # weight
        features = KeyedJaggedTensor.from_lengths_sync(
            keys=["f1", "f2"],
            values=torch.tensor([0, 1, 0, 0, 1, 0, 1]),
            weights=torch.tensor([1.0, 2.0, 3.0, 5.0, 7.0, 11.0, 13.0]),
            lengths=torch.tensor([2, 0, 1, 1, 1, 2]),
        )

        pooled_embeddings = ebc(features)

        torch.testing.assert_close(
            pooled_embeddings["f1"],
            torch.tensor([[5.0] * 4, [0.0] * 4, [3.0] * 4], dtype=torch.float32),
        )

        torch.testing.assert_close(
            pooled_embeddings["f2"],
            torch.tensor(
                [[20.0] * 4, [56.0] * 4, [4 * 11 + 8 * 13] * 4], dtype=torch.float32
            ),
        )

    def test_optimizer_fusion(self) -> None:
        tables = [
            EmbeddingBagConfig(
                num_embeddings=2,
                embedding_dim=4,
                name="table_0",
                feature_names=["feature_0"],
            ),
            EmbeddingBagConfig(
                num_embeddings=2,
                embedding_dim=4,
                name="table_1",
                feature_names=["feature_1"],
            ),
        ]

        fused_ebc = FusedEmbeddingBagCollection(
            tables=tables,
            optimizer_type=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.1},
        )

        ebc = EmbeddingBagCollection(tables=tables)

        state_dict = OrderedDict(
            [
                (
                    "embedding_bags.table_0.weight",
                    torch.Tensor([[1, 1, 1, 1], [2, 2, 2, 2]]),
                ),
                (
                    "embedding_bags.table_1.weight",
                    torch.Tensor([[4, 4, 4, 4], [8, 8, 8, 8]]),
                ),
            ]
        )
        fused_ebc.load_state_dict(state_dict)
        ebc.load_state_dict(state_dict)

        #        0       1        2  <-- batch
        # "f1"   [] [0]    [0,1]
        # "f2"   [1]    [0,1]    []
        #  ^
        # feature
        features = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0", "feature_1"],
            values=torch.tensor([0, 0, 1, 1, 0, 1]),
            lengths=torch.tensor([0, 1, 2, 1, 2, 0]),
        )

        opt = torch.optim.SGD(ebc.parameters(), lr=0.1)
        # pyre-ignore
        def run_one_training_step() -> None:
            fused_pooled_embeddings = fused_ebc(features)
            fused_vals = []
            for _name, param in fused_pooled_embeddings.to_dict().items():
                fused_vals.append(param)
            torch.cat(fused_vals, dim=1).sum().backward()

            opt.zero_grad()
            pooled_embeddings = ebc(features)

            vals = []
            for _name, param in pooled_embeddings.to_dict().items():
                vals.append(param)
            torch.cat(vals, dim=1).sum().backward()
            opt.step()

        run_one_training_step()
        torch.testing.assert_close(
            ebc.state_dict()["embedding_bags.table_0.weight"],
            fused_ebc.state_dict()["embedding_bags.table_0.weight"],
        )

        torch.testing.assert_close(
            fused_ebc.state_dict()["embedding_bags.table_0.weight"],
            torch.Tensor([[1.0 - 2 * 0.1] * 4, [2.0 - 1 * 0.1] * 4]),
        )

        run_one_training_step()
        torch.testing.assert_close(
            ebc.state_dict()["embedding_bags.table_0.weight"],
            fused_ebc.state_dict()["embedding_bags.table_0.weight"],
        )

        torch.testing.assert_close(
            fused_ebc.state_dict()["embedding_bags.table_0.weight"],
            torch.Tensor([[1.0 - 2 * 2 * 0.1] * 4, [2.0 - 2 * 1 * 0.1] * 4]),
        )

        # TODO, ensure this state dict is loaded correctly
        # SGD does not have any state (momentum etc, so need to expand this test)
        fused_optimizer = fused_ebc.fused_optimizer
        fused_optimizer.load_state_dict(fused_optimizer.state_dict())

    def unweighted_replacement(
        self, device: torch.device, location: Optional[EmbeddingLocation] = None
    ) -> None:
        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=4, num_embeddings=10, feature_names=["f1"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2", embedding_dim=4, num_embeddings=10, feature_names=["f2"]
        )
        eb3_config = EmbeddingBagConfig(
            name="t3", embedding_dim=4, num_embeddings=10, feature_names=["f3"]
        )

        ebc = EmbeddingBagCollection(
            tables=[eb1_config, eb2_config, eb3_config],
            device=device,
        )

        fused_ebc = fuse_optimizer(
            ebc,
            optimizer_type=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.02},
            device=device,
        )

        #     0       1        2  <-- batch
        # f1   [0,1] None    [2]
        # f2   [3]    [4]    [5,6,7]
        # f3   []    [8]    []
        # ^
        # feature
        features = KeyedJaggedTensor.from_lengths_sync(
            keys=["f1", "f2", "f3"],
            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8]),
            lengths=torch.tensor([2, 0, 1, 1, 1, 3, 0, 1, 0]),
        ).to(device)

        pooled_embeddings = fused_ebc(features)

        self.assertEqual(pooled_embeddings.keys(), ["f1", "f2", "f3"])
        self.assertEqual(pooled_embeddings["f1"].shape, (features.stride(), 4))
        self.assertEqual(pooled_embeddings["f2"].shape, (features.stride(), 4))
        self.assertEqual(pooled_embeddings["f3"].shape, (features.stride(), 4))

        torch.testing.assert_close(
            pooled_embeddings["f1"][1], torch.zeros(4).to(device)
        )
        torch.testing.assert_close(
            pooled_embeddings["f3"][0], torch.zeros(4).to(device)
        )
        torch.testing.assert_close(
            pooled_embeddings["f3"][2], torch.zeros(4).to(device)
        )

    def test_fuse_embedding_optimizer_replacement_cpu(self) -> None:
        self.unweighted_replacement(torch.device("cpu"))

    # pyre-ignore
    @unittest.skipIf(
        torch.cuda.device_count() < 1,
        "This test requires a gpu",
    )
    def test_fuse_embedding_optimizer_replacement_cuda(self) -> None:
        self.unweighted_replacement(torch.device("cuda"))

    # pyre-ignore
    @unittest.skipIf(
        torch.cuda.device_count() < 1,
        "This test requires a gpu",
    )
    def test_fuse_embedding_optimizer_replacement_cuda_uvm_caching(self) -> None:
        self.unweighted_replacement(
            torch.device("cuda"), location=EmbeddingLocation.MANAGED_CACHING
        )
