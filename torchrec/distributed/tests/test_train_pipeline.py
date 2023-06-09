#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
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
from torchrec.distributed.fp_embeddingbag import (
    FeatureProcessedEmbeddingBagCollectionSharder,
    ShardedFeatureProcessedEmbeddingBagCollection,
)
from torchrec.distributed.sharding_plan import (
    construct_module_sharding_plan,
    table_wise,
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
    ShardingPlan,
    ShardingType,
)
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection

from torchrec.optim.keyed import KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.streamable import Pipelineable
from torchrec.test_utils import get_free_port, init_distributed_single_host

from .test_fp_embeddingbag_utils import create_module_and_freeze


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

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
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

        for batch in data[:-1]:
            optimizer_cpu.zero_grad()
            loss, pred = model_cpu(batch)
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

        for batch in data[:-2]:
            optimizer_cpu.zero_grad()
            loss, pred = unsharded_model(batch)
            batch.idlist_features._jt_dict = None
            if batch.idscore_features is not None:
                batch.idscore_features._jt_dict = None
            loss.backward()
            optimizer_cpu.step()
            pred_gpu = pipeline.progress(dataloader)

            self.assertEqual(pred_gpu.device, self.device)
            torch.testing.assert_close(pred_gpu.cpu(), pred)
            self.assertEqual(len(pipeline._pipelined_modules), 3)

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
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

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_feature_processed_ebc(self) -> None:
        embedding_bag_configs = [
            EmbeddingBagConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=3 * 16,
                num_embeddings=16,
            ),
            EmbeddingBagConfig(
                name="table_1",
                feature_names=["feature_1"],
                embedding_dim=8,
                num_embeddings=16,
            ),
            EmbeddingBagConfig(
                name="table_2",
                feature_names=["feature_2"],
                embedding_dim=8,
                num_embeddings=16,
            ),
            EmbeddingBagConfig(
                name="table_3",
                feature_names=["feature_3"],
                embedding_dim=3 * 16,
                num_embeddings=16,
            ),
        ]

        sharder = cast(
            ModuleSharder[nn.Module], FeatureProcessedEmbeddingBagCollectionSharder()
        )

        class DummyWrapper(nn.Module):
            def __init__(self, sparse_arch):
                super().__init__()
                self.m = sparse_arch

            def forward(self, model_input) -> Tuple[torch.Tensor, torch.Tensor]:
                return self.m(model_input.idlist_features)

        sparse_arch = DummyWrapper(
            create_module_and_freeze(
                tables=embedding_bag_configs,
                device=self.device,
                use_fp_collection=False,
            )
        )
        module_sharding_plan = construct_module_sharding_plan(
            sparse_arch.m._fp_ebc,
            per_param_sharding={
                "table_0": table_wise(rank=0),
                "table_1": table_wise(rank=0),
                "table_2": table_wise(rank=0),
                "table_3": table_wise(rank=0),
            },
            local_size=1,
            world_size=1,
            device_type=self.device.type,
            sharder=sharder,
        )
        sharded_sparse_arch_no_pipeline = DistributedModelParallel(
            module=copy.deepcopy(sparse_arch),
            plan=ShardingPlan({"m._fp_ebc": module_sharding_plan}),
            env=ShardingEnv.from_process_group(self.pg),
            sharders=[sharder],
            device=self.device,
        )

        sharded_sparse_arch_pipeline = DistributedModelParallel(
            module=copy.deepcopy(sparse_arch),
            plan=ShardingPlan({"m._fp_ebc": module_sharding_plan}),
            env=ShardingEnv.from_process_group(self.pg),
            sharders=[sharder],
            device=self.device,
        )

        copy_state_dict(
            sharded_sparse_arch_no_pipeline.state_dict(),
            sharded_sparse_arch_pipeline.state_dict(),
        )

        data = [
            ModelInput.generate(
                tables=embedding_bag_configs,
                weighted_tables=[],
                batch_size=1,
                world_size=1,
                num_float_features=0,
                pooling_avg=5,
            )[0].to(self.device)
            for i in range(10)
        ]
        dataloader = iter(data)

        optimizer_no_pipeline = optim.SGD(
            sharded_sparse_arch_no_pipeline.parameters(), lr=0.1
        )
        optimizer_pipeline = optim.SGD(
            sharded_sparse_arch_pipeline.parameters(), lr=0.1
        )

        pipeline = TrainPipelineSparseDist(
            sharded_sparse_arch_pipeline,
            optimizer_pipeline,
            self.device,
        )

        for batch in data[:-2]:
            optimizer_no_pipeline.zero_grad()
            loss, pred = sharded_sparse_arch_no_pipeline(batch)
            loss.backward()
            optimizer_no_pipeline.step()

            pred_pipeline = pipeline.progress(dataloader)
            torch.testing.assert_close(pred_pipeline.cpu(), pred.cpu())

        self.assertEqual(len(pipeline._pipelined_modules), 1)
        self.assertIsInstance(
            pipeline._pipelined_modules[0],
            ShardedFeatureProcessedEmbeddingBagCollection,
        )

    def _setup_pipeline(
        self, sharder: EmbeddingBagCollectionSharder, execute_all_batches: bool
    ) -> TrainPipelineSparseDist[ModelInput, torch.Tensor]:
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
                    sharder,
                )
            ],
        )
        optimizer_distributed = KeyedOptimizerWrapper(
            dict(in_backward_optimizer_filter(distributed_model.named_parameters())),
            lambda params: optim.SGD(params, lr=0.1),
        )
        return TrainPipelineSparseDist(
            model=distributed_model,
            optimizer=optimizer_distributed,
            device=self.device,
            execute_all_batches=execute_all_batches,
        )

    def _setup_cpu_model_and_opt(self) -> Tuple[TestSparseNN, optim.SGD]:
        cpu_model = TestSparseNN(
            tables=self.tables, weighted_tables=self.weighted_tables
        )
        cpu_optimizer = optim.SGD(cpu_model.parameters(), lr=0.1)
        return cpu_model, cpu_optimizer

    def _generate_data(self, num_batches: int = 5) -> List[ModelInput]:
        return [
            ModelInput.generate(
                tables=self.tables,
                weighted_tables=self.weighted_tables,
                batch_size=1,
                world_size=1,
                num_float_features=10,
            )[0]
            for i in range(num_batches)
        ]

    def _test_pipelining(
        self,
        sharder: EmbeddingBagCollectionSharder,
        execute_all_batches: bool,
    ) -> None:
        pipeline = self._setup_pipeline(sharder, execute_all_batches)
        cpu_model, cpu_optimizer = self._setup_cpu_model_and_opt()
        data = self._generate_data()

        dataloader = iter(data)
        if not execute_all_batches:
            data = data[:-2]

        for batch in data:
            cpu_optimizer.zero_grad()
            loss, pred = cpu_model(batch)
            loss.backward()
            cpu_optimizer.step()

            pred_gpu = pipeline.progress(dataloader)

            self.assertEqual(len(pipeline._pipelined_modules), 2)
            self.assertEqual(pred_gpu.device, self.device)
            self.assertEqual(pred_gpu.cpu().size(), pred.size())

    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    # pyre-ignore[56]
    @given(execute_all_batches=st.booleans())
    @settings(verbosity=Verbosity.verbose, max_examples=2, deadline=None)
    def test_pipelining(self, execute_all_batches: bool) -> None:
        pipeline = self._setup_pipeline(
            TestEBCSharder(
                sharding_type=ShardingType.TABLE_WISE.value,
                kernel_type=EmbeddingComputeKernel.FUSED.value,
            ),
            execute_all_batches,
        )
        cpu_model, cpu_optimizer = self._setup_cpu_model_and_opt()
        data = self._generate_data()

        dataloader = iter(data)
        if not execute_all_batches:
            data = data[:-2]

        for batch in data:
            cpu_optimizer.zero_grad()
            loss, pred = cpu_model(batch)
            loss.backward()
            cpu_optimizer.step()

            pred_gpu = pipeline.progress(dataloader)

            self.assertEqual(len(pipeline._pipelined_modules), 2)
            self.assertEqual(pred_gpu.device, self.device)
            self.assertEqual(pred_gpu.cpu().size(), pred.size())

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @unittest.skipIf(
        not torch.cuda.is_available(),
        "Not enough GPUs, this test requires at least one GPU",
    )
    def test_multi_dataloader_pipelining(self) -> None:
        pipeline = self._setup_pipeline(
            sharder=TestEBCSharder(
                sharding_type=ShardingType.TABLE_WISE.value,
                kernel_type=EmbeddingComputeKernel.FUSED.value,
            ),
            execute_all_batches=True,
        )
        cpu_model, cpu_optimizer = self._setup_cpu_model_and_opt()
        data = self._generate_data(num_batches=7)

        cpu_preds = []
        for batch in data:
            cpu_optimizer.zero_grad()
            loss, pred = cpu_model(batch)
            loss.backward()
            cpu_optimizer.step()
            cpu_preds.append(pred)

        dataloaders = [iter(data[:-3]), iter(data[-3:-2]), iter(data[-2:])]
        gpu_preds = []
        for dataloader in dataloaders:
            while True:
                try:
                    pred = pipeline.progress(dataloader)
                    self.assertEqual(pred.device, self.device)
                    self.assertEqual(len(pipeline._pipelined_modules), 2)
                    gpu_preds.append(pred.cpu())
                except StopIteration:
                    break

        self.assertEqual(len(pipeline._pipelined_modules), 2)
        self.assertEqual(len(cpu_preds), len(gpu_preds))
        self.assertTrue(
            all(
                [
                    cpu_pred.size() == gpu_pred.size()
                    for cpu_pred, gpu_pred in zip(cpu_preds, gpu_preds)
                ]
            )
        )
