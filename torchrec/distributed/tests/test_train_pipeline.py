#!/usr/bin/env python3

import os
import unittest
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict

import torch
import torch.distributed as dist
from torch import nn, optim
from torchrec.distributed import DistributedModelParallel
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.embedding_types import (
    SparseFeaturesList,
)
from torchrec.distributed.embeddingbag import (
    ShardedEmbeddingBagCollection,
    EmbeddingBagCollectionSharder,
)
from torchrec.distributed.tests.test_model import (
    TestSparseNN,
    ModelInput,
    TestEBCSharder,
)
from torchrec.distributed.train_pipeline import (
    TrainPipelineBase,
    TrainPipelineSparseDist,
)
from torchrec.distributed.types import (
    Awaitable,
    ParameterSharding,
    ShardedModuleContext,
    ShardingEnv,
)
from torchrec.distributed.types import (
    ShardingType,
)
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.optim.keyed import KeyedOptimizerWrapper
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.tests.utils import get_free_port, init_distributed_single_host


class TestShardedEmbeddingBagCollection(ShardedEmbeddingBagCollection):
    def input_dist(
        self,
        ctx: ShardedModuleContext,
        features: KeyedJaggedTensor,
    ) -> Awaitable[SparseFeaturesList]:
        return super().input_dist(ctx, features)


class TestCustomEBCSharder(EmbeddingBagCollectionSharder[EmbeddingBagCollection]):
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
            ShardingType.TABLE_WISE.value,
        ]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [EmbeddingComputeKernel.DENSE.value]


@dataclass
class ModelInputSimple:
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
        self.float_features.record_stream(stream)
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

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
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

            self.assertEquals(pred_gpu.device, self.device)
            self.assertTrue(torch.isclose(pred_gpu.cpu(), pred))


class TrainPipelineSparseDistTest(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["MASTER_ADDR"] = str("localhost")
        os.environ["MASTER_PORT"] = str(get_free_port())
        if not dist.is_initialized():
            self.pg = init_distributed_single_host(backend="gloo", rank=0, world_size=1)
        else:
            self.pg = dist.group.WORLD

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

    def _test_move_cpu_gpu_helper(
        self, distributed_model: DistributedModelParallel
    ) -> None:
        model_cpu = TestSparseNN(
            tables=self.tables, weighted_tables=self.weighted_tables
        )
        optimizer_cpu = optim.SGD(model_cpu.parameters(), lr=0.1)
        optimizer_distributed = KeyedOptimizerWrapper(
            dict(distributed_model.named_parameters()),
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

            self.assertEquals(pred_gpu.device, self.device)
            self.assertEquals(pred_gpu.cpu().size(), pred.size())
            self.assertEquals(len(pipeline._pipelined_modules), 2)

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
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
            # pyre-ignore [6]
            sharders=[
                TestEBCSharder(
                    sharding_type=ShardingType.TABLE_WISE.value,
                    kernel_type=EmbeddingComputeKernel.DENSE.value,
                )
            ],
        )
        self._test_move_cpu_gpu_helper(distributed_model)

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
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
            # pyre-fixme [6]
            sharders=[TestCustomEBCSharder()],
        )
        self._test_move_cpu_gpu_helper(distributed_model)
