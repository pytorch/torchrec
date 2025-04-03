#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

#!/usr/bin/env python3

import copy

from dataclasses import dataclass
from typing import Any, cast, Dict, List, Optional, Tuple, Type, Union

import click

import torch
import torch.distributed as dist
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from torch import nn, optim
from torch.optim import Optimizer
from torchrec.distributed import DistributedModelParallel
from torchrec.distributed.benchmark.benchmark_utils import benchmark_func, cmd_conf
from torchrec.distributed.embedding_types import EmbeddingComputeKernel

from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    run_multi_process_func,
)
from torchrec.distributed.test_utils.test_input import (
    ModelInput,
    TdModelInput,
    TestSparseNNInputConfig,
)
from torchrec.distributed.test_utils.test_model import (
    TestEBCSharder,
    TestOverArchLarge,
    TestSparseNN,
)
from torchrec.distributed.train_pipeline import (
    TrainPipeline,
    TrainPipelineBase,
    TrainPipelineSparseDist,
)
from torchrec.distributed.train_pipeline.train_pipelines import (
    PrefetchTrainPipelineSparseDist,
    TrainPipelineSemiSync,
)
from torchrec.distributed.types import ModuleSharder, ShardingEnv, ShardingType
from torchrec.modules.embedding_configs import EmbeddingBagConfig


@dataclass
class RunOptions:
    world_size: int = 4
    num_batches: int = 20
    sharding_type: ShardingType = ShardingType.TABLE_WISE
    input_type: str = "kjt"
    profile: str = ""


@dataclass
class EmbeddingTablesConfig:
    num_unweighted_features: int = 4
    num_weighted_features: int = 4
    embedding_feature_dim: int = 512

    def generate_tables(
        self,
    ) -> Tuple[
        List[EmbeddingBagConfig],
        List[EmbeddingBagConfig],
    ]:
        tables = [
            EmbeddingBagConfig(
                num_embeddings=max(i + 1, 100) * 1000,
                embedding_dim=self.embedding_feature_dim,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(self.num_unweighted_features)
        ]
        weighted_tables = [
            EmbeddingBagConfig(
                num_embeddings=max(i + 1, 100) * 1000,
                embedding_dim=self.embedding_feature_dim,
                name="weighted_table_" + str(i),
                feature_names=["weighted_feature_" + str(i)],
            )
            for i in range(self.num_weighted_features)
        ]
        return tables, weighted_tables


@dataclass
class PipelineConfig:
    pipeline: str = "base"

    def generate_pipeline(
        self, model: nn.Module, opt: torch.optim.Optimizer, device: torch.device
    ) -> Union[TrainPipelineBase, TrainPipelineSparseDist]:
        _pipeline_cls: Dict[
            str, Type[Union[TrainPipelineBase, TrainPipelineSparseDist]]
        ] = {
            "base": TrainPipelineBase,
            "sparse": TrainPipelineSparseDist,
            "semi": TrainPipelineSemiSync,
            "prefetch": PrefetchTrainPipelineSparseDist,
        }

        if self.pipeline == "semi":
            return TrainPipelineSemiSync(
                model=model, optimizer=opt, device=device, start_batch=0
            )
        elif self.pipeline in _pipeline_cls:
            Pipeline = _pipeline_cls[self.pipeline]
            return Pipeline(model=model, optimizer=opt, device=device)
        else:
            raise RuntimeError(f"unknown pipeline option {self.pipeline}")


@click.command()
@cmd_conf(RunOptions, EmbeddingTablesConfig, TestSparseNNInputConfig, PipelineConfig)
def main(
    run_option: RunOptions,
    table_config: EmbeddingTablesConfig,
    input_config: TestSparseNNInputConfig,
    pipeline_config: PipelineConfig,
) -> None:
    # sparse table config is available on each trainer
    tables, weighted_tables = table_config.generate_tables()

    # launch trainers
    run_multi_process_func(
        func=runner,
        world_size=run_option.world_size,
        tables=tables,
        weighted_tables=weighted_tables,
        run_option=run_option,
        input_config=input_config,
        pipeline_config=pipeline_config,
    )


def _generate_data(
    tables: List[EmbeddingBagConfig],
    weighted_tables: List[EmbeddingBagConfig],
    input_config: TestSparseNNInputConfig,
    num_batches: int,
) -> List[ModelInput]:
    return [
        input_config.generate_model_input(
            tables=tables,
            weighted_tables=weighted_tables,
        )
        for _ in range(num_batches)
    ]


def _generate_model(
    tables: List[EmbeddingBagConfig],
    weighted_tables: List[EmbeddingBagConfig],
    dense_device: torch.device,
) -> nn.Module:
    return TestSparseNN(
        tables=tables,
        weighted_tables=weighted_tables,
        dense_device=dense_device,
        sparse_device=torch.device("meta"),
        over_arch_clazz=TestOverArchLarge,
    )


def _generate_sharded_model_and_optimizer(
    model: nn.Module,
    sharding_type: str,
    kernel_type: str,
    pg: dist.ProcessGroup,
    device: torch.device,
    fused_params: Optional[Dict[str, Any]] = None,
) -> Tuple[nn.Module, Optimizer]:
    sharder = TestEBCSharder(
        sharding_type=sharding_type,
        kernel_type=kernel_type,
        fused_params=fused_params,
    )
    sharded_model = DistributedModelParallel(
        module=copy.deepcopy(model),
        env=ShardingEnv.from_process_group(pg),
        init_data_parallel=True,
        device=device,
        sharders=[
            cast(
                ModuleSharder[nn.Module],
                sharder,
            )
        ],
    ).to(device)
    optimizer = optim.SGD(
        [
            param
            for name, param in sharded_model.named_parameters()
            if "sparse" not in name
        ],
        lr=0.1,
    )
    return sharded_model, optimizer


def runner(
    rank: int,
    world_size: int,
    tables: List[EmbeddingBagConfig],
    weighted_tables: List[EmbeddingBagConfig],
    run_option: RunOptions,
    input_config: TestSparseNNInputConfig,
    pipeline_config: PipelineConfig,
) -> None:

    torch.autograd.set_detect_anomaly(True)
    with MultiProcessContext(
        rank=rank,
        world_size=world_size,
        backend="nccl",
        use_deterministic_algorithms=False,
    ) as ctx:
        unsharded_model = _generate_model(
            tables=tables,
            weighted_tables=weighted_tables,
            dense_device=ctx.device,
        )

        sharded_model, optimizer = _generate_sharded_model_and_optimizer(
            model=unsharded_model,
            sharding_type=run_option.sharding_type.value,
            kernel_type=EmbeddingComputeKernel.FUSED.value,
            # pyre-ignore
            pg=ctx.pg,
            device=ctx.device,
            fused_params={
                "optimizer": EmbOptimType.EXACT_ADAGRAD,
                "learning_rate": 0.1,
            },
        )
        bench_inputs = _generate_data(
            tables=tables,
            weighted_tables=weighted_tables,
            input_config=input_config,
            num_batches=run_option.num_batches,
        )
        pipeline = pipeline_config.generate_pipeline(
            sharded_model, optimizer, ctx.device
        )
        pipeline.progress(iter(bench_inputs))

        def _func_to_benchmark(
            bench_inputs: List[ModelInput],
            model: nn.Module,
            pipeline: TrainPipeline,
        ) -> None:
            dataloader = iter(bench_inputs)
            while True:
                try:
                    pipeline.progress(dataloader)
                except StopIteration:
                    break

        result = benchmark_func(
            name=type(pipeline).__name__,
            bench_inputs=bench_inputs,  # pyre-ignore
            prof_inputs=bench_inputs,  # pyre-ignore
            num_benchmarks=5,
            num_profiles=2,
            profile_dir=run_option.profile,
            world_size=run_option.world_size,
            func_to_benchmark=_func_to_benchmark,
            benchmark_func_kwargs={"model": sharded_model, "pipeline": pipeline},
            rank=rank,
        )
        if rank == 0:
            print(result)


if __name__ == "__main__":
    main()
