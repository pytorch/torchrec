#!/usr/bin/env python3

import copy
import os
import unittest
from typing import List

import torch
import torch.distributed as dist
from torch import nn
from torch import quantization as quant
from torchrec.distributed.embedding import QuantEmbeddingBagCollectionSharder
from torchrec.distributed.embedding_lookup import (
    GroupedEmbeddingBag,
    BatchedFusedEmbeddingBag,
    BatchedDenseEmbeddingBag,
    QuantBatchedEmbeddingBag,
)
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.tests.test_model import (
    TestSparseNN,
    TestEBCSharder,
)
from torchrec.distributed.types import ShardingType
from torchrec.distributed.utils import sharded_model_copy
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.quant.embedding_modules import (
    EmbeddingBagCollection as QuantEmbeddingBagCollection,
)
from torchrec.tests.utils import get_free_port


class TestQuantEBCSharder(QuantEmbeddingBagCollectionSharder):
    def __init__(self, sharding_type: str, kernel_type: str) -> None:
        self._sharding_type = sharding_type
        self._kernel_type = kernel_type

    @property
    def sharding_types(self) -> List[str]:
        return [self._sharding_type]

    def compute_kernels(self, sharding_type: str, device: torch.device) -> List[str]:
        return [self._kernel_type]


def _quantize_sharded(module: nn.Module, inplace: bool) -> nn.Module:
    qconfig = quant.QConfigDynamic(
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
    qconfig = quant.QConfigDynamic(
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
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = str("localhost")
        os.environ["MASTER_PORT"] = str(get_free_port())
        os.environ["GLOO_DEVICE_TRANSPORT"] = "TCP"
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            backend = "nccl"
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
            backend = "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)

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
                    sharding_type=ShardingType.ROW_WISE.value,
                    kernel_type=EmbeddingComputeKernel.BATCHED_QUANT.value,
                )
            ],
            device=device,
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
                    sharding_type=ShardingType.ROW_WISE.value,
                    kernel_type=EmbeddingComputeKernel.BATCHED_FUSED.value,
                )
            ],
            device=device,
        )
        with sharded_model_copy(device="cpu"):
            sharded_model_cpu = copy.deepcopy(sharded_model)
        _ = _quantize_sharded(sharded_model_cpu, inplace=True)
