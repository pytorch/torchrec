#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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
from torchrec.distributed.embedding_types import (
    EmbeddingComputeKernel,
)
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.quant_embeddingbag import (
    QuantEmbeddingBagCollectionSharder,
)
from torchrec.distributed.test_utils.test_model import TestSparseNN
from torchrec.distributed.types import ShardedModule, ShardingType, ShardingEnv
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


class QuantModelParallelModelCopyTest(unittest.TestCase):
    def setUp(self) -> None:
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

    def _buffer_param_check(
        self,
        module: nn.Module,
        module_copy: nn.Module,
        device: torch.device,
        device_copy: torch.device,
    ) -> None:
        # check all buffer/param under the module is value-identical
        # but device-different with the copied module.
        for (name, buffer), (name_copy, buffer_copy) in zip(
            # pyre-ignore [58]
            list(module.named_buffers()) + list(module.named_parameters()),
            # pyre-ignore [58]
            list(module_copy.named_buffers()) + list(module_copy.named_parameters()),
        ):
            self.assertEquals(name, name_copy)
            torch.testing.assert_allclose(
                buffer.detach().cpu(), buffer_copy.detach().cpu()
            )
            self.assertEquals(buffer.detach().device, device)
            self.assertEquals(buffer_copy.detach().device, device_copy)

    def _recursive_device_check(
        self,
        module: nn.Module,
        module_copy: nn.Module,
        device: torch.device,
        device_copy: torch.device,
    ) -> None:
        if isinstance(module, ShardedModule):
            # sparse part parameter needs to be the same reference
            # TBE ops's parameter is accessed via buffer.
            for name_buffer, name_buffer_copy in zip(
                module.named_buffers(),
                module_copy.named_buffers(),
            ):
                name, buffer = name_buffer
                name_copy, buffer_copy = name_buffer_copy
                self.assertEqual(name, name_copy)
                # compare tensor storage reference
                self.assertTrue(buffer.detach().is_set_to(buffer_copy.detach()))
        for name_child, name_child_copy in zip(
            module.named_children(), module_copy.named_children()
        ):
            name, child = name_child
            name_copy, child_copy = name_child_copy
            if not any(
                [isinstance(submodule, ShardedModule) for submodule in child.modules()]
            ):
                # other part parameter/buffer needs to be
                # identical in value and different in device
                self._buffer_param_check(child, child_copy, device, device_copy)
            else:
                self._recursive_device_check(child, child_copy, device, device_copy)

    # pyre-fixme[56]
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
    )
    def test_quant_pred(self) -> None:
        device = torch.device("cuda:0")
        device_1 = torch.device("cuda:1")
        model = TestSparseNN(
            tables=self.tables,
            weighted_tables=self.weighted_tables,
            num_float_features=10,
            dense_device=device,
            sparse_device=torch.device("meta"),
        )
        quant_model = _quantize(model, inplace=True)
        dmp = DistributedModelParallel(
            quant_model,
            # pyre-ignore [6]
            sharders=[
                TestQuantEBCSharder(
                    sharding_type=ShardingType.TABLE_WISE.value,
                    kernel_type=EmbeddingComputeKernel.BATCHED_QUANT.value,
                )
            ],
            device=device,
            env=ShardingEnv.from_local(world_size=2, rank=0),
            init_data_parallel=False,
        )
        dmp_1 = dmp.copy(device_1)
        self._recursive_device_check(dmp.dmp_module, dmp_1.dmp_module, device, device_1)
