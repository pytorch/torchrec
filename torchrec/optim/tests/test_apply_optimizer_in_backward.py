#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import torch
from torchrec import EmbeddingBagCollection, EmbeddingBagConfig, KeyedJaggedTensor
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward


class ApplyOverlappedOptimizerTest(unittest.TestCase):
    def test_apply_optimizer_in_backward(self) -> None:
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

        apply_optimizer_in_backward(
            torch.optim.SGD,
            ebc.embedding_bags["t1"].parameters(),
            optimizer_kwargs={"lr": 1.0},
        )

        apply_optimizer_in_backward(
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

        self.assertTrue(hasattr(t1_weight, "_optimizer_classes"))
        self.assertEqual(t1_weight._optimizer_classes, [torch.optim.SGD])
        self.assertTrue(hasattr(t1_weight, "_optimizer_kwargs"))
        self.assertEqual(t1_weight._optimizer_kwargs, [{"lr": 1.0}])

        self.assertTrue(hasattr(t2_weight, "_optimizer_classes"))
        self.assertEqual(t2_weight._optimizer_classes, [torch.optim.SGD])
        self.assertTrue(hasattr(t2_weight, "_optimizer_kwargs"))
        self.assertEqual(t2_weight._optimizer_kwargs, [{"lr": 2.0}])

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
