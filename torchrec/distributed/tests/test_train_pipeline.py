#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
from typing import cast, List

import torch
import torch.distributed as dist
from torch import nn, optim
from torchrec.distributed import DistributedModelParallel
from torchrec.distributed.embedding_types import EmbeddingComputeKernel

from torchrec.distributed.test_utils.test_model import (
    ModelInput,
    ModelInputSimple,
    TestCustomEBCSharder,
    TestEBCSharder,
    TestModule,
    TestSparseNN,
)
from torchrec.distributed.test_utils.test_sharding import copy_state_dict
from torchrec.distributed.train_pipeline import (
    TrainPipelineBase,
    TrainPipelineSparseDist,
)
from torchrec.distributed.types import ModuleSharder, ShardingEnv, ShardingType
from torchrec.modules.embedding_configs import EmbeddingBagConfig

from torchrec.optim.keyed import KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter
from torchrec.test_utils import get_free_port, init_distributed_single_host


class TrainPipelineBaseTest(unittest.TestCase):
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
        pipeline = TrainPipelineBase(model_gpu, optimizer_gpu, self.device)

        for example in data[:-1]:
            optimizer_cpu.zero_grad()
            loss, pred = model_cpu(example)
            loss.backward()
            optimizer_cpu.step()

            pred_gpu = pipeline.progress(dataloader)

            self.assertEqual(pred_gpu.device, self.device)
            self.assertTrue(torch.isclose(pred_gpu.cpu(), pred))


class TrainPipelineSparseDistTest(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["MASTER_ADDR"] = str("localhost")
        os.environ["MASTER_PORT"] = str(get_free_port())
        self.pg = init_distributed_single_host(backend="gloo", rank=0, world_size=1)

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

    def _test_feature_processor_helper(
        self,
        unsharded_model: TestSparseNN,
        distributed_model: DistributedModelParallel,
        fp_tables: List[EmbeddingBagConfig],
    ) -> None:
        copy_state_dict(unsharded_model.state_dict(), distributed_model.state_dict())
        optimizer_cpu = optim.SGD(unsharded_model.parameters(), lr=0.1)
        optimizer_distributed = KeyedOptimizerWrapper(
            dict(in_backward_optimizer_filter(distributed_model.named_parameters())),
            lambda params: optim.SGD(params, lr=0.1),
        )
        pipeline = TrainPipelineSparseDist(
            distributed_model, optimizer_distributed, self.device
        )

        data = [
            ModelInput.generate(
                tables=self.tables + fp_tables,
                weighted_tables=self.weighted_tables,
                batch_size=1,
                world_size=1,
                num_float_features=10,
            )[0]
            for i in range(5)
        ]
        dataloader = iter(data)

        for example in data[:-2]:
            optimizer_cpu.zero_grad()
            loss, pred = unsharded_model(example)
            example.idlist_features._jt_dict = None
            if example.idscore_features is not None:
                example.idscore_features._jt_dict = None
            loss.backward()
            optimizer_cpu.step()
            pred_gpu = pipeline.progress(dataloader)

            self.assertEqual(pred_gpu.device, self.device)
            self.assertEqual(pred_gpu.cpu().size(), pred.size())
            torch.testing.assert_close(pred_gpu.cpu(), pred)
            self.assertEqual(len(pipeline._pipelined_modules), 3)

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPUs",
    )
    def test_position_weighted_feature_processor(self) -> None:
        max_feature_length = 100
        table_num = 2
        fp_tables = [
            EmbeddingBagConfig(
                num_embeddings=(i + 1) * 100,
                embedding_dim=(i + 1) * 4,
                name="fp_table_" + str(i),
                feature_names=["fp_feature_" + str(i)],
                need_pos=True,
            )
            for i in range(table_num)
        ]
        # chained feature_processors, the output is only 1 feature
        max_feature_lengths_list = [
            {
                name: max_feature_length
                for table in reversed(fp_tables)
                for name in table.feature_names
            }
            for i in range(table_num)
        ]

        unsharded_model = TestSparseNN(
            tables=self.tables + fp_tables,
            weighted_tables=self.weighted_tables,
            dense_device=self.device,
            sparse_device=torch.device("meta"),
            max_feature_lengths_list=max_feature_lengths_list,
        )
        distributed_model = DistributedModelParallel(
            unsharded_model,
            env=ShardingEnv.from_process_group(self.pg),
            init_data_parallel=True,
            device=self.device,
            sharders=[cast(ModuleSharder[nn.Module], TestCustomEBCSharder())],
        )
        test_unsharded_model = TestSparseNN(
            tables=self.tables + fp_tables,
            weighted_tables=self.weighted_tables,
            max_feature_lengths_list=max_feature_lengths_list,
        )
        self._test_feature_processor_helper(
            test_unsharded_model, distributed_model, fp_tables
        )

    def _test_move_cpu_gpu_helper(
        self, distributed_model: DistributedModelParallel
    ) -> None:
        model_cpu = TestSparseNN(
            tables=self.tables, weighted_tables=self.weighted_tables
        )
        optimizer_cpu = optim.SGD(model_cpu.parameters(), lr=0.1)
        optimizer_distributed = KeyedOptimizerWrapper(
            dict(in_backward_optimizer_filter(distributed_model.named_parameters())),
            lambda params: optim.SGD(params, lr=0.1),
        )
        pipeline = TrainPipelineSparseDist(
            distributed_model, optimizer_distributed, self.device
        )

        data = [
            ModelInput.generate(
                tables=self.tables,
                weighted_tables=self.weighted_tables,
                batch_size=1,
                world_size=1,
                num_float_features=10,
            )[0]
            for i in range(5)
        ]
        dataloader = iter(data)

        for example in data[:-2]:
            optimizer_cpu.zero_grad()
            loss, pred = model_cpu(example)
            loss.backward()
            optimizer_cpu.step()

            pred_gpu = pipeline.progress(dataloader)

            self.assertEqual(pred_gpu.device, self.device)
            self.assertEqual(pred_gpu.cpu().size(), pred.size())
            self.assertEqual(len(pipeline._pipelined_modules), 2)

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPUs",
    )
    def test_move_cpu_gpu(self) -> None:
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
            sharders=[
                cast(
                    ModuleSharder[nn.Module],
                    TestEBCSharder(
                        sharding_type=ShardingType.TABLE_WISE.value,
                        kernel_type=EmbeddingComputeKernel.DENSE.value,
                    ),
                )
            ],
        )
        self._test_move_cpu_gpu_helper(distributed_model)

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPUs",
    )
    def test_pipelining(self) -> None:
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
        self._test_move_cpu_gpu_helper(distributed_model)
