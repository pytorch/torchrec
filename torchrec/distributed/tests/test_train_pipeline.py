#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
from dataclasses import dataclass
from typing import cast, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from hypothesis import given, settings, strategies as st, Verbosity
from torch import nn, optim
from torchrec.distributed import DistributedModelParallel
from torchrec.distributed.embedding_types import EmbeddingComputeKernel, KJTList
from torchrec.distributed.embeddingbag import (
    EmbeddingBagCollectionContext,
    EmbeddingBagCollectionSharder,
    ShardedEmbeddingBagCollection,
)
from torchrec.distributed.test_utils.test_model import (
    ModelInput,
    TestEBCSharder,
    TestSparseNN,
)
from torchrec.distributed.test_utils.test_sharding import copy_state_dict
from torchrec.distributed.train_pipeline import (
    TrainPipelineBase,
    TrainPipelineSparseDist,
)
from torchrec.distributed.types import (
    Awaitable,
    ModuleSharder,
    ParameterSharding,
    ShardingEnv,
    ShardingType,
)
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection

from torchrec.optim.keyed import KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.streamable import Pipelineable
from torchrec.test_utils import get_free_port, init_distributed_single_host


class TestShardedEmbeddingBagCollection(ShardedEmbeddingBagCollection):
    def input_dist(
        self,
        ctx: EmbeddingBagCollectionContext,
        features: KeyedJaggedTensor,
    ) -> Awaitable[Awaitable[KJTList]]:
        return super().input_dist(ctx, features)


class TestCustomEBCSharder(EmbeddingBagCollectionSharder):
    def shard(
        self,
        module: EmbeddingBagCollection,
        params: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
    ) -> TestShardedEmbeddingBagCollection:
        return TestShardedEmbeddingBagCollection(
            module, params, env, self.fused_params, device
        )

    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [
            ShardingType.ROW_WISE.value,
        ]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [EmbeddingComputeKernel.DENSE.value]


@dataclass
class ModelInputSimple(Pipelineable):
    float_features: torch.Tensor
    label: torch.Tensor

    def to(self, device: torch.device, non_blocking: bool) -> "ModelInputSimple":
        return ModelInputSimple(
            float_features=self.float_features.to(
                device=device, non_blocking=non_blocking
            ),
            label=self.label.to(device=device, non_blocking=non_blocking),
        )

    def record_stream(self, stream: torch.cuda.streams.Stream) -> None:
        # pyre-fixme[6]: For 1st param expected `Stream` but got `Stream`.
        self.float_features.record_stream(stream)
        # pyre-fixme[6]: For 1st param expected `Stream` but got `Stream`.
        self.label.record_stream(stream)


class TestModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Linear(10, 1)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self, model_input: ModelInputSimple
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pred = self.model(model_input.float_features)
        loss = self.loss_fn(pred, model_input.label)
        return (loss, pred)


class TrainPipelineBaseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cuda:0")
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPUs",
    )
    # pyre-fixme[56]
    @given(iter_api=st.sampled_from([True, False]))
    @settings(verbosity=Verbosity.verbose, deadline=None)
    def test_equal_to_non_pipelined(self, iter_api: bool) -> None:
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
        if iter_api:
            pipeline = iter(
                TrainPipelineBase(model_gpu, optimizer_gpu, self.device, dataloader)
            )
        else:
            pipeline = TrainPipelineBase(model_gpu, optimizer_gpu, self.device)

        for example in data[:-1]:
            optimizer_cpu.zero_grad()
            loss, pred = model_cpu(example)
            loss.backward()
            optimizer_cpu.step()

            if iter_api:
                pred_gpu = next(pipeline)
            else:
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
        iter_api: bool,
    ) -> None:
        copy_state_dict(unsharded_model.state_dict(), distributed_model.state_dict())
        optimizer_cpu = optim.SGD(unsharded_model.parameters(), lr=0.1)
        optimizer_distributed = KeyedOptimizerWrapper(
            dict(in_backward_optimizer_filter(distributed_model.named_parameters())),
            lambda params: optim.SGD(params, lr=0.1),
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

        if iter_api:
            pipeline = iter(
                TrainPipelineSparseDist(
                    distributed_model, optimizer_distributed, self.device, dataloader
                )
            )
        else:
            pipeline = TrainPipelineSparseDist(
                distributed_model, optimizer_distributed, self.device
            )

        for example in data[:-2]:
            optimizer_cpu.zero_grad()
            loss, pred = unsharded_model(example)
            example.idlist_features._jt_dict = None
            example.idscore_features._jt_dict = None
            loss.backward()
            optimizer_cpu.step()

            if iter_api:
                pred_gpu = next(pipeline)
            else:
                pred_gpu = pipeline.progress(dataloader)

            self.assertEqual(pred_gpu.device, self.device)
            self.assertEqual(pred_gpu.cpu().size(), pred.size())
            torch.testing.assert_close(pred_gpu.cpu(), pred)
            self.assertEqual(len(pipeline._pipelined_modules), 3)

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPUs",
    )
    # pyre-fixme[56]
    @given(iter_api=st.sampled_from([True, False]))
    @settings(verbosity=Verbosity.verbose, deadline=None)
    def test_position_weighted_feature_processor(self, iter_api: bool) -> None:
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
            test_unsharded_model, distributed_model, fp_tables, iter_api
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

        pipeline = iter(
            TrainPipelineSparseDist(
                distributed_model, optimizer_distributed, self.device, dataloader
            )
        )

        for example in data[:-2]:
            optimizer_cpu.zero_grad()
            loss, pred = model_cpu(example)
            loss.backward()
            optimizer_cpu.step()

            pred_gpu = next(pipeline)

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
