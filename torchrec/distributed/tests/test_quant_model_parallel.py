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
from torchrec.distributed.tests.test_model import (
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

    # pyre-fixme[56]
    @unittest.skipIf(
        torch.cuda.device_count() <= 0,
        "No GPUs available",
    )
    def test_quant_pred(self) -> None:
        device = torch.device("cuda:0")
        model = TestSparseNN(
            tables=self.tables,
            weighted_tables=self.weighted_tables,
            num_float_features=10,
            dense_device=device,
            sparse_device=torch.device("meta"),
        )
        quant_model = _quantize(model, inplace=True)
        _ = DistributedModelParallel(
            quant_model,
            # pyre-ignore [6]
            sharders=[
                TestQuantEBCSharder(
                    sharding_type=ShardingType.DATA_PARALLEL.value,
                    kernel_type=EmbeddingComputeKernel.BATCHED_QUANT.value,
                )
            ],
            device=device,
            env=ShardingEnv.from_local(world_size=1, rank=0),
            init_data_parallel=False,
        )

    # pyre-fixme[56]
    @unittest.skipIf(
        torch.cuda.device_count() <= 0,
        "No GPUs available",
    )
    def test_quant_train(self) -> None:
        device = torch.device("cuda:0")
        model = TestSparseNN(
            tables=self.tables,
            weighted_tables=self.weighted_tables,
            num_float_features=10,
            dense_device=device,
            sparse_device=torch.device("meta"),
        )
        sharded_model = DistributedModelParallel(
            model,
            # pyre-ignore [6]
            sharders=[
                TestEBCSharder(
                    sharding_type=ShardingType.DATA_PARALLEL.value,
                    kernel_type=EmbeddingComputeKernel.DENSE.value,
                )
            ],
            device=device,
            env=ShardingEnv.from_local(world_size=1, rank=0),
            init_data_parallel=False,
        )
        with sharded_model_copy(device="cpu"):
            sharded_model_cpu = copy.deepcopy(sharded_model)
        _ = _quantize_sharded(sharded_model_cpu, inplace=True)
