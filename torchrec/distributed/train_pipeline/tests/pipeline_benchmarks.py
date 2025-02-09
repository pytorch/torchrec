#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

#!/usr/bin/env python3

import copy
import multiprocessing
import os
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Type, Union

import click

import torch
import torch.distributed as dist
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from torch import nn, optim
from torch.optim import Optimizer
from torchrec.distributed import DistributedModelParallel
from torchrec.distributed.benchmark.benchmark_utils import benchmark_func
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.test_utils.multi_process import MultiProcessContext
from torchrec.distributed.test_utils.test_model import (
    ModelInput,
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

from torchrec.test_utils import get_free_port


_pipeline_cls: Dict[str, Type[Union[TrainPipelineBase, TrainPipelineSparseDist]]] = {
    "base": TrainPipelineBase,
    "sparse": TrainPipelineSparseDist,
    "semi": TrainPipelineSemiSync,
    "prefetch": PrefetchTrainPipelineSparseDist,
}


def _gen_pipelines(
    pipelines: str,
) -> List[Type[Union[TrainPipelineBase, TrainPipelineSparseDist]]]:
    if pipelines == "all":
        return list(_pipeline_cls.values())
    else:
        return [_pipeline_cls[pipelines]]


@click.command()
@click.option(
    "--world_size",
    type=int,
    default=4,
    help="Num of GPUs to run",
)
@click.option(
    "--n_features",
    default=100,
    help="Total number of sparse embeddings to be used.",
)
@click.option(
    "--ratio_features_weighted",
    default=0.4,
    help="percentage of features weighted vs unweighted",
)
@click.option(
    "--dim_emb",
    type=int,
    default=512,
    help="Dim embeddings embedding.",
)
@click.option(
    "--n_batches",
    type=int,
    default=20,
    help="Num of batchs to run.",
)
@click.option(
    "--batch_size",
    type=int,
    default=8192,
    help="Batch size.",
)
@click.option(
    "--sharding_type",
    type=ShardingType,
    default=ShardingType.TABLE_WISE,
    help="ShardingType.",
)
@click.option(
    "--pooling_factor",
    type=int,
    default=100,
    help="Pooling Factor.",
)
@click.option(
    "--input_type",
    type=str,
    default="kjt",
    help="Input type: kjt, td",
)
@click.option(
    "--pipeline",
    type=str,
    default="all",
    help="Pipeline to run: all, base, sparse, semi, prefetch",
)
@click.option(
    "--multi_process",
    type=bool,
    default=True,
    help="Run in multi process mode.",
)
@click.option(
    "--profile",
    type=str,
    default="",
    help="profile output directory",
)
def main(
    world_size: int,
    n_features: int,
    ratio_features_weighted: float,
    dim_emb: int,
    n_batches: int,
    batch_size: int,
    sharding_type: ShardingType,
    pooling_factor: int,
    input_type: str,
    pipeline: str,
    multi_process: bool,
    profile: str,
) -> None:
    """
    Checks that pipelined training is equivalent to non-pipelined training.
    """

    os.environ["MASTER_ADDR"] = str("localhost")
    os.environ["MASTER_PORT"] = str(get_free_port())

    num_weighted_features = int(n_features * ratio_features_weighted)
    num_features = n_features - num_weighted_features

    tables = [
        EmbeddingBagConfig(
            num_embeddings=max(i + 1, 100) * 1000,
            embedding_dim=dim_emb,
            name="table_" + str(i),
            feature_names=["feature_" + str(i)],
        )
        for i in range(num_features)
    ]
    weighted_tables = [
        EmbeddingBagConfig(
            num_embeddings=max(i + 1, 100) * 1000,
            embedding_dim=dim_emb,
            name="weighted_table_" + str(i),
            feature_names=["weighted_feature_" + str(i)],
        )
        for i in range(num_weighted_features)
    ]
    batches = _generate_data(
        tables=tables,
        weighted_tables=weighted_tables,
        num_float_features=10,
        num_batches=n_batches,
        batch_size=batch_size,
        world_size=world_size,
        pooling_factor=pooling_factor,
        input_type=input_type,
    )

    if multi_process:
        _run_multi_process_test(
            callable=runner,
            tables=tables,
            weighted_tables=weighted_tables,
            sharding_type=sharding_type.value,
            kernel_type=EmbeddingComputeKernel.FUSED.value,
            batches=batches,
            fused_params={},
            world_size=world_size,
            pipelines=pipeline,
            profile=profile,
        )
    else:
        single_runner(
            tables=tables,
            weighted_tables=weighted_tables,
            sharding_type=sharding_type.value,
            kernel_type=EmbeddingComputeKernel.FUSED.value,
            batches=batches,
            fused_params={},
            world_size=1,
            pipelines=pipeline,
            profile=profile,
        )


def _run_multi_process_test(
    *,
    callable: Callable[
        ...,
        None,
    ],
    world_size: int,
    # pyre-ignore
    **kwargs,
) -> None:
    ctx = multiprocessing.get_context("spawn")
    processes = []
    if world_size == 1:
        kwargs["world_size"] = 1
        kwargs["rank"] = 0
        callable(**kwargs)
        return

    for rank in range(world_size):
        kwargs["rank"] = rank
        kwargs["world_size"] = world_size
        p = ctx.Process(
            target=callable,
            kwargs=kwargs,
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def _generate_data(
    tables: List[EmbeddingBagConfig],
    weighted_tables: List[EmbeddingBagConfig],
    num_float_features: int = 10,
    num_batches: int = 100,
    batch_size: int = 4096,
    world_size: int = 1,
    pooling_factor: int = 10,
    input_type: str = "kjt",
) -> List[List[ModelInput]]:
    return [
        ModelInput.generate(
            tables=tables,
            weighted_tables=weighted_tables,
            batch_size=batch_size,
            world_size=world_size,
            num_float_features=num_float_features,
            pooling_avg=pooling_factor,
            input_type=input_type,
        )[1]
        for i in range(num_batches)
    ]


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
    tables: List[EmbeddingBagConfig],
    weighted_tables: List[EmbeddingBagConfig],
    rank: int,
    sharding_type: str,
    kernel_type: str,
    fused_params: Dict[str, Any],
    world_size: int,
    batches: List[List[ModelInput]],
    pipelines: str,
    profile: str,
) -> None:

    torch.autograd.set_detect_anomaly(True)
    with MultiProcessContext(
        rank=rank,
        world_size=world_size,
        backend="nccl",
        use_deterministic_algorithms=False,
    ) as ctx:

        unsharded_model = TestSparseNN(
            tables=tables,
            weighted_tables=weighted_tables,
            dense_device=ctx.device,
            sparse_device=torch.device("meta"),
            over_arch_clazz=TestOverArchLarge,
        )

        sharded_model, optimizer = _generate_sharded_model_and_optimizer(
            model=unsharded_model,
            sharding_type=sharding_type,
            kernel_type=kernel_type,
            # pyre-ignore
            pg=ctx.pg,
            device=ctx.device,
            fused_params={
                "optimizer": EmbOptimType.EXACT_ADAGRAD,
                "learning_rate": 0.1,
            },
        )
        bench_inputs = [batch[rank] for batch in batches]
        for pipeline_clazz in _gen_pipelines(pipelines=pipelines):
            if pipeline_clazz == TrainPipelineSemiSync:
                # pyre-ignore [28]
                pipeline = pipeline_clazz(
                    model=sharded_model,
                    optimizer=optimizer,
                    device=ctx.device,
                    start_batch=0,
                )
            else:
                pipeline = pipeline_clazz(
                    model=sharded_model,
                    optimizer=optimizer,
                    device=ctx.device,
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
                name=pipeline_clazz.__name__,
                bench_inputs=bench_inputs,  # pyre-ignore
                prof_inputs=bench_inputs,  # pyre-ignore
                num_benchmarks=5,
                num_profiles=2,
                profile_dir=profile,
                world_size=world_size,
                func_to_benchmark=_func_to_benchmark,
                benchmark_func_kwargs={"model": sharded_model, "pipeline": pipeline},
                rank=rank,
            )
            if rank == 0:
                print(result)


def single_runner(
    tables: List[EmbeddingBagConfig],
    weighted_tables: List[EmbeddingBagConfig],
    sharding_type: str,
    kernel_type: str,
    fused_params: Dict[str, Any],
    world_size: int,
    batches: List[List[ModelInput]],
    pipelines: str,
    profile: str,
) -> None:
    device = torch.device("cuda")
    torch.autograd.set_detect_anomaly(True)
    model = TestSparseNN(
        tables=tables,
        weighted_tables=weighted_tables,
        dense_device=device,
        sparse_device=device,
        over_arch_clazz=TestOverArchLarge,
    ).to(device)

    optimizer = optim.SGD(
        [param for name, param in model.named_parameters() if "sparse" not in name],
        lr=0.1,
    )

    bench_inputs = [batch[0] for batch in batches]
    for pipeline_clazz in _gen_pipelines(pipelines=pipelines):
        if pipeline_clazz == TrainPipelineSemiSync:
            # pyre-ignore [28]
            pipeline = pipeline_clazz(
                model=model,
                optimizer=optimizer,
                device=device,
                start_batch=0,
            )
        else:
            pipeline = pipeline_clazz(
                model=model,
                optimizer=optimizer,
                device=device,
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
            name=pipeline_clazz.__name__,
            bench_inputs=bench_inputs,  # pyre-ignore
            prof_inputs=bench_inputs,  # pyre-ignore
            num_benchmarks=5,
            num_profiles=2,
            profile_dir=profile,
            world_size=world_size,
            func_to_benchmark=_func_to_benchmark,
            benchmark_func_kwargs={"model": model, "pipeline": pipeline},
            rank=0,
        )

        print(result)


if __name__ == "__main__":
    main()
