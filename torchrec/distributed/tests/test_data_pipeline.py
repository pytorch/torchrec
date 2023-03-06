#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
import unittest
from typing import cast

import torch
from torch import distributed as dist, nn, optim
from torchrec.distributed.data_pipeline import CudaCopyingPipeline, SparseDistPipeline

from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.test_utils.test_model import (
    ModelInput,
    ModelInputSimple,
    TestCustomEBCSharder,
    TestModule,
    TestSparseNN,
)
from torchrec.distributed.test_utils.test_sharding import copy_state_dict
from torchrec.distributed.types import ModuleSharder, ShardingEnv
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.optim.keyed import KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter
from torchrec.test_utils import get_free_port, init_distributed_single_host


class CudaCopyingPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cuda:0")
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPUs",
    )
    def test_equal_to_non_pipelined(self) -> None:
        model_cpu = TestModule()
        model_gpu = TestModule().to(self.device)
        model_gpu.load_state_dict(model_cpu.state_dict())
        optimizer_cpu = optim.SGD(model_cpu.model.parameters(), lr=0.01)
        optimizer_gpu = optim.SGD(model_gpu.model.parameters(), lr=0.01)
        data = [
            ModelInputSimple(
                float_features=torch.rand((10,)),
                label=torch.randint(2, (1,), dtype=torch.float32),
            )
            for b in range(5)
        ]
        dataloader = iter(data)
        pipeline = CudaCopyingPipeline(dataloader, self.device)

        for i, batch in enumerate(pipeline):
            # cpu model train
            optimizer_cpu.zero_grad()
            loss, pred = model_cpu(data[i])
            loss.backward()
            optimizer_cpu.step()

            # pipelined train
            optimizer_gpu.zero_grad()
            loss_gpu, pred_gpu = model_gpu(batch)
            loss_gpu.backward()
            optimizer_gpu.step()

            self.assertEqual(pred_gpu.device, self.device)
            torch.testing.assert_close(pred_gpu.cpu(), pred)


class TestDataPipeline(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["MASTER_ADDR"] = str("localhost")
        os.environ["MASTER_PORT"] = str(get_free_port())
        self.pg = init_distributed_single_host(backend="nccl", rank=0, world_size=1)

        num_features = 4
        num_weighted_features = 2

        self.tables = [
            EmbeddingBagConfig(
                num_embeddings=(i + 1) * 100,
                embedding_dim=(i + 1) * 4,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(num_features)
        ]
        self.weighted_tables = [
            EmbeddingBagConfig(
                num_embeddings=(i + 1) * 100,
                embedding_dim=(i + 1) * 4,
                name="weighted_table_" + str(i),
                feature_names=["weighted_feature_" + str(i)],
            )
            for i in range(num_weighted_features)
        ]

        self.device = torch.device("cuda:0")

    def tearDown(self) -> None:
        super().tearDown()
        dist.destroy_process_group(self.pg)

    # pyre-ignore
    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPUs",
    )
    def test_sparse_dist_data_pipelining(
        self,
    ) -> None:
        unsharded_model = TestSparseNN(
            tables=self.tables,
            weighted_tables=self.weighted_tables,
            dense_device=self.device,
            sparse_device=torch.device("meta"),
        )
        distributed_model = DistributedModelParallel(
            unsharded_model,
            env=ShardingEnv.from_process_group(self.pg),
            init_data_parallel=False,
            device=self.device,
            sharders=[cast(ModuleSharder[nn.Module], TestCustomEBCSharder())],
        )

        model_cpu = TestSparseNN(
            tables=self.tables,
            weighted_tables=self.weighted_tables,
        )
        optimizer = optim.SGD(model_cpu.parameters(), lr=0.1)
        optimizer_distributed = KeyedOptimizerWrapper(
            dict(in_backward_optimizer_filter(distributed_model.named_parameters())),
            lambda params: optim.SGD(params, lr=0.1),
        )

        data = [
            ModelInput.generate(
                tables=self.tables,
                weighted_tables=self.weighted_tables,
                batch_size=1,
                world_size=1,
                num_float_features=10,
            )[0]
            for i in range(4)
        ]

        dataloader = iter(data)

        copy_state_dict(
            distributed_model.state_dict(), copy.deepcopy(model_cpu.state_dict())
        )

        pipeline = SparseDistPipeline(
            dataloader_iter=dataloader, device=self.device, model=distributed_model
        )

        for i, batch in enumerate(pipeline):
            # cpu model train
            optimizer.zero_grad()
            loss, pred = model_cpu(data[i])
            loss.backward()
            optimizer.step()

            # pipelined train
            optimizer_distributed.zero_grad()
            loss_pipelined, pred_pipelined = distributed_model(batch)
            loss_pipelined.backward()
            optimizer_distributed.step()

            self.assertEqual(len(pipeline._pipelined_modules), 2)

            self.assertEqual(pred_pipelined.device, self.device)
            self.assertEqual(pred_pipelined.cpu().size(), pred.size())
            torch.testing.assert_close(pred_pipelined.cpu(), pred)
