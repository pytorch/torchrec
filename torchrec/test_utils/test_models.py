#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class TestModel(torch.nn.Module):
    """
    Simple TestModel class, contains a EmbeddingBagCollection and Linear layer.
    """

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


class TestSequentialModel(torch.nn.Module):
    """
    Simple TestModel class, contains a EmbeddingCollection and Linear layer.
    """

    def __init__(self, ec: EmbeddingCollection) -> None:
        super().__init__()
        self.ec = ec
        self.over_arch = torch.nn.Linear(
            ec.embedding_dim(),
            1,
        )

    def forward(self, kjt: KeyedJaggedTensor) -> torch.Tensor:
        ec_output = self.ec.forward(kjt)
        sparse_features = []
        for key in kjt.keys():
            sparse_features.extend(ec_output[key].to_dense())
        sparse_features = torch.Tensor(sparse_features)
        sparse_features = torch.sum(sparse_features)
        return self.over_arch(sparse_features)
