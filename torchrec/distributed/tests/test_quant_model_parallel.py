#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest
from typing import List

import torch
from torch import nn
from torch import quantization as quant
from torchrec.distributed.embedding_lookup import (
    GroupedEmbeddingBag,
    BatchedFusedEmbeddingBag,
    BatchedDenseEmbeddingBag,
    QuantBatchedEmbeddingBag,
)
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.quant_embeddingbag import (
    QuantEmbeddingBagCollectionSharder,
)
from torchrec.distributed.test_utils.test_model import (
    TestSparseNN,
    TestEBCSharder,
)
from torchrec.distributed.types import ShardingType, ShardingEnv
from torchrec.distributed.utils import sharded_model_copy
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.quant.embedding_modules import (
    EmbeddingBagCollection as QuantEmbeddingBagCollection,
)


class TestQuantEBCSharder(QuantEmbeddingBagCollectionSharder):
    def __init__(self, sharding_type: str, kernel_type: str) -> None:
        self._sharding_type = sharding_type
        self._kernel_type = kernel_type

    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [self._sharding_type]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [self._kernel_type]


def _quantize_sharded(module: nn.Module, inplace: bool) -> nn.Module:
    qconfig = quant.QConfig(
        activation=quant.PlaceholderObserver,
        weight=quant.PlaceholderObserver.with_args(dtype=torch.qint8),
    )
    return quant.quantize_dynamic(
        module,
        qconfig_spec={
            GroupedEmbeddingBag: qconfig,
            BatchedFusedEmbeddingBag: qconfig,
            BatchedDenseEmbeddingBag: qconfig,
        },
        mapping={
            GroupedEmbeddingBag: QuantBatchedEmbeddingBag,
            BatchedFusedEmbeddingBag: QuantBatchedEmbeddingBag,
            BatchedDenseEmbeddingBag: QuantBatchedEmbeddingBag,
        },
        inplace=inplace,
    )


def _quantize(module: nn.Module, inplace: bool) -> nn.Module:
    qconfig = quant.QConfig(
        activation=quant.PlaceholderObserver,
        weight=quant.PlaceholderObserver.with_args(dtype=torch.qint8),
    )
    return quant.quantize_dynamic(
        module,
        qconfig_spec={
            EmbeddingBagCollection: qconfig,
        },
        mapping={
            EmbeddingBagCollection: QuantEmbeddingBagCollection,
        },
        inplace=inplace,
    )


class QuantModelParallelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)

        num_features = 4
        num_weighted_features = 2

        self.tables = [
            EmbeddingBagConfig(
                num_embeddings=(i + 1) * 10,
                embedding_dim=(i + 1) * 4,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(num_features)
        ]
        self.weighted_tables = [
            EmbeddingBagConfig(
                num_embeddings=(i + 1) * 10,
                embedding_dim=(i + 1) * 4,
                name="weighted_table_" + str(i),
                feature_names=["weighted_feature_" + str(i)],
            )
            for i in range(num_weighted_features)
        ]
