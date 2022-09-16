#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest
from unittest.mock import ANY, MagicMock, patch

import torch
from torchrec import (
    EmbeddingBagCollection,
    EmbeddingBagConfig,
    EmbeddingCollection,
    EmbeddingConfig,
    KeyedJaggedTensor,
)
from torchrec.optim.overlapped_optimizer_utils import (
    apply_overlapped_optimizer,
    apply_overlapped_optimizer_to_module,
)
from torchrec.test_utils.test_models import TestModel, TestSequentialModel


class ApplyOverlappedOptimizerTest(unittest.TestCase):
    def test_apply_overlapped_optimizer(self) -> None:
        ebc = EmbeddingBagCollection(
            tables=[
                EmbeddingBagConfig(
                    name="t1", embedding_dim=4, num_embeddings=2, feature_names=["f1"]
                ),
                EmbeddingBagConfig(
                    name="t2", embedding_dim=4, num_embeddings=2, feature_names=["f2"]
                ),
            ]
        )

        apply_overlapped_optimizer(
            torch.optim.SGD,
            ebc.embedding_bags["t1"].parameters(),
            optimizer_kwargs={"lr": 1.0},
        )

        apply_overlapped_optimizer(
            torch.optim.SGD,
            ebc.embedding_bags["t2"].parameters(),
            optimizer_kwargs={"lr": 2.0},
        )

        ebc.load_state_dict(
            {
                "embedding_bags.t1.weight": torch.FloatTensor(
                    [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]]
                ),
                "embedding_bags.t2.weight": torch.FloatTensor(
                    [[10.0, 10.0, 10.0, 10.0], [12.0, 12.0, 12.0, 12.0]]
                ),
            }
        )

        #     0       1        2  <-- batch
        # f1   [0,1] None    [0]
        # f2   [0,1]    [1]    [0]
        # ^
        # feature
        kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=["f1", "f2"],
            values=torch.tensor([0, 1, 0, 0, 1, 1, 0]),
            lengths=torch.tensor([2, 0, 1, 2, 1, 1]),
        )

        kt_out = ebc(kjt).to_dict()
        stack = []
        for _key, val in kt_out.items():
            stack.append(val)
        torch.stack(stack).sum().backward()

        t1_weight = next(ebc.embedding_bags["t1"].parameters())
        t2_weight = next(ebc.embedding_bags["t2"].parameters())

        self.assertIsNone(t1_weight.grad)
        self.assertIsNone(t2_weight.grad)

        self.assertTrue(hasattr(t1_weight, "_optimizer_class"))
        self.assertEqual(t1_weight._optimizer_class, torch.optim.SGD)
        self.assertTrue(hasattr(t1_weight, "_optimizer_kwargs"))
        self.assertEqual(t1_weight._optimizer_kwargs, {"lr": 1.0})

        self.assertTrue(hasattr(t2_weight, "_optimizer_class"))
        self.assertEqual(t2_weight._optimizer_class, torch.optim.SGD)
        self.assertTrue(hasattr(t2_weight, "_optimizer_kwargs"))
        self.assertEqual(t2_weight._optimizer_kwargs, {"lr": 2.0})

        expected_state_dict = {
            "embedding_bags.t1.weight": torch.FloatTensor(
                [[-1.0, -1.0, -1.0, -1.0], [1.0, 1.0, 1.0, 1.0]]
            ),
            "embedding_bags.t2.weight": torch.FloatTensor(
                [[6.0, 6.0, 6.0, 6.0], [8.0, 8.0, 8.0, 8.0]]
            ),
        }

        for key, state in ebc.state_dict().items():
            self.assertIn(key, expected_state_dict)
            torch.testing.assert_close(state, expected_state_dict[key])

    @patch("torchrec.optim.overlapped_optimizer_utils.apply_overlapped_optimizer")
    def test_apply_overlapped_optimizer_to_ebc_module(
        self, apply_overlapped_optimizer_mock: MagicMock
    ) -> None:
        ebc = EmbeddingBagCollection(
            tables=[
                EmbeddingBagConfig(
                    name="t1", embedding_dim=4, num_embeddings=2, feature_names=["f1"]
                ),
                EmbeddingBagConfig(
                    name="t2", embedding_dim=4, num_embeddings=2, feature_names=["f2"]
                ),
            ]
        )

        model = TestModel(ebc)
        apply_overlapped_optimizer_to_module(
            torch.optim.SGD,
            model,
            optimizer_kwargs={"lr": 1.0},
        )

        assert apply_overlapped_optimizer_mock.call_count == 2
        apply_overlapped_optimizer_mock.assert_called_with(
            torch.optim.SGD, ANY, {"lr": 1.0}
        )

    @patch("torchrec.optim.overlapped_optimizer_utils.apply_overlapped_optimizer")
    def test_apply_overlapped_optimizer_ec_module(
        self, apply_overlapped_optimizer_mock: MagicMock
    ) -> None:

        ec = EmbeddingCollection(
            tables=[
                EmbeddingConfig(
                    name="t1", embedding_dim=4, num_embeddings=2, feature_names=["f1"]
                ),
                EmbeddingConfig(
                    name="t2", embedding_dim=4, num_embeddings=2, feature_names=["f2"]
                ),
            ]
        )

        model = TestSequentialModel(ec)

        apply_overlapped_optimizer_to_module(
            torch.optim.SGD,
            model,
            optimizer_kwargs={"lr": 1.0},
        )

        assert apply_overlapped_optimizer_mock.call_count == 2
        apply_overlapped_optimizer_mock.assert_called_with(
            torch.optim.SGD, ANY, {"lr": 1.0}
        )
