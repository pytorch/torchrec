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
import torchrec
from hypothesis import given, settings

from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.modules.fused_embedding_modules import fuse_embedding_optimizer
from torchrec.optim.fused import get_fused_optimizers
from torchrec.optim.keyed import CombinedOptimizer

from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

devices: List[torch.device] = [torch.device("cpu")]
if torch.cuda.device_count() > 1:
    devices.append(torch.device("cuda"))


class TestModel(torch.nn.Module):
    def __init__(self, ebc: EmbeddingBagCollection) -> None:
        super().__init__()
        self.ebc = ebc
        self.over_arch = torch.nn.Linear(
            4,
            1,
        )

    def forward(self, kjt: KeyedJaggedTensor) -> torch.Tensor:
        ebc_output = self.ebc.forward(kjt).to_dict()
        sparse_features = []
        for key in kjt.keys():
            sparse_features.append(ebc_output[key])
        sparse_features = torch.cat(sparse_features, dim=0)
        return self.over_arch(sparse_features)


class TestFused(unittest.TestCase):
    @settings(deadline=None)
    # pyre-ignore
    @given(device=st.sampled_from(devices))
    def test_get_fused_optimizers(self, device: torch.device) -> None:
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
        )

        test_model = TestModel(ebc).to(device)
        test_model = fuse_embedding_optimizer(
            test_model,
            optimizer_type=torchrec.optim.RowWiseAdagrad,
            optimizer_kwargs={"lr": 0.02},
            device=device,
        )

        #     0       1        2  <-- batch
        # f1   [0,1] None    [2]
        # f2   [3]    [4]    [5,6,7]
        # f3   []    [8]    []
        # ^
        # feature
        features: KeyedJaggedTensor = KeyedJaggedTensor.from_lengths_sync(
            keys=["f1", "f2", "f3"],
            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8]),
            lengths=torch.tensor([2, 0, 1, 1, 1, 3, 0, 1, 0]),
        ).to(device)

        fused_optims: CombinedOptimizer = get_fused_optimizers(test_model)

        test_model(features).sum().backward()
        fused_optims.step()

    @settings(deadline=None)
    # pyre-ignore
    @given(device=st.sampled_from(devices))
    def test_get_fused_optimizers_empty(self, device: torch.device) -> None:
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
        )

        test_model = TestModel(ebc).to(device)

        #     0       1        2  <-- batch
        # f1   [0,1] None    [2]
        # f2   [3]    [4]    [5,6,7]
        # f3   []    [8]    []
        # ^
        # feature
        features: KeyedJaggedTensor = KeyedJaggedTensor.from_lengths_sync(
            keys=["f1", "f2", "f3"],
            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8]),
            lengths=torch.tensor([2, 0, 1, 1, 1, 3, 0, 1, 0]),
        ).to(device)

        fused_optims: CombinedOptimizer = get_fused_optimizers(test_model)

        test_model(features).sum().backward()
        fused_optims.step()
