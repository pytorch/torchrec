#!/usr/bin/env python3

# pyre-strict

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3
# @nolint

import unittest

import torch
from torchrec.distributed.test_utils.infer_utils import TorchTypesModelInputWrapper
from torchrec.distributed.test_utils.test_model import TestSparseNN
from torchrec.inference.modules import quantize_inference_model, shard_quant_model
from torchrec.modules.embedding_configs import EmbeddingBagConfig


class Simple(torch.nn.Module):
    def __init__(self, N: int, M: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(N, M))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.weight + input
        return output

    def set_weight(self, weight: torch.Tensor) -> None:
        self.weight[:] = torch.nn.Parameter(weight)


class Nested(torch.nn.Module):
    def __init__(self, N: int, M: int) -> None:
        super().__init__()
        self.simple = Simple(N, M)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.simple(input)

class EagerModelProcessingTests(unittest.TestCase):
    def test_quantize_shard_cuda(self) -> None:
        tables = [
            EmbeddingBagConfig(
                num_embeddings=10,
                embedding_dim=4,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(10)
        ]

        model = TorchTypesModelInputWrapper(
            TestSparseNN(
                tables=tables,
            )
        )

        table_fqns = ["table_" + str(i) for i in range(10)]

        quantized_model = quantize_inference_model(model)
        sharded_model, _ = shard_quant_model(quantized_model, table_fqns)

        sharded_qebc = sharded_model._module.sparse.ebc
        self.assertEqual(len(sharded_qebc.tbes), 1)
