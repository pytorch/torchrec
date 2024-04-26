#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

import copy
import multiprocessing
import os
import time
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

import click

import torch
import torch.distributed as dist
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from torch import nn, optim
from torch.optim import Optimizer
from torchrec.distributed import DistributedModelParallel
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.test_utils.multi_process import MultiProcessContext
from torchrec.distributed.test_utils.test_model import (
    ModelInput,
    TestEBCSharder,
    TestOverArchLarge,
    TestSparseNN,
)
from torchrec.distributed.train_pipeline import TrainPipelineSparseDist
from torchrec.distributed.train_pipeline.train_pipelines import (
    EmbeddingTrainPipeline,
    TrainPipelineBase,
    # TrainPipelineSparseDist,
)
from torchrec.distributed.types import ModuleSharder, ShardingEnv, ShardingType
from torchrec.modules.embedding_configs import EmbeddingBagConfig

from torchrec.test_utils import get_free_port


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
    "--dim_emb",
    type=int,
    default=512,
    help="Dim embeddings embedding.",
)
@click.option(
    "--n_batches",
    type=int,
    default=100,
    help="Num of batchs to run.",
)
@click.option(
    "--batch_size",
    type=int,
    default=8192,
    help="Batch size.",
)
def main(
    world_size: int,
    n_features: int,
    dim_emb: int,
    n_batches: int,
    batch_size: int,
) -> None:
    """
    Checks that pipelined training is equivalent to non-pipelined training.
    """

    fused_params = {
        "stochastic_rounding": False,  # disable non-deterministic behavior when converting fp32<->fp16
    }

    os.environ["MASTER_ADDR"] = str("localhost")
    os.environ["MASTER_PORT"] = str(get_free_port())

    num_features = n_features // 2
    num_weighted_features = n_features // 2
    tables = [
        EmbeddingBagConfig(
            num_embeddings=(i + 1) * 1000,
            embedding_dim=dim_emb,
            name="table_" + str(i),
            feature_names=["feature_" + str(i)],
        )
        for i in range(num_features)
    ]
    weighted_tables = [
        EmbeddingBagConfig(
            num_embeddings=(i + 1) * 1000,
            embedding_dim=dim_emb,
            name="weighted_table_" + str(i),
            feature_names=["weighted_feature_" + str(i)],
        )
        for i in range(num_weighted_features)
    ]

    _run_multi_process_test(
        callable=runner,
        tables=tables,
        weighted_tables=weighted_tables,
        sharding_type=ShardingType.TABLE_WISE.value,
        kernel_type=EmbeddingComputeKernel.FUSED.value,
        fused_params=fused_params,
        world_size=world_size,
        n_batches=n_batches,
        batch_size=batch_size,
    )


def _run_multi_process_test(
    *,
    callable: Callable[
        ...,
        None,
    ],
    world_size: int = 2,
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
    num_batches: int = 100,
    batch_size: int = 4096 * 4,
    rank: int = 0,
) -> List[ModelInput]:
    return [
        ModelInput.generate(
            tables=tables,
            weighted_tables=weighted_tables,
            batch_size=batch_size,
            world_size=1,
            num_float_features=10,
        )[0]
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
    n_batches: int,
    batch_size: int,
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

        data = _generate_data(
            tables=tables,
            weighted_tables=weighted_tables,
            rank=rank,
            num_batches=n_batches,
            batch_size=batch_size,
        )
        for pipeline_clazz in [
            TrainPipelineBase,
            TrainPipelineSparseDist,
            EmbeddingTrainPipeline,
        ]:
            dataloader = iter(data)

            pipeline = pipeline_clazz(
                model=sharded_model,
                optimizer=optimizer,
                device=ctx.device,
            )
            start = None
            print(f"{pipeline_clazz.__name__}: START")
            while True:
                try:
                    pipeline.progress(dataloader)
                except StopIteration:
                    break
                if start is None:
                    start = time.time()
            if rank == 0 and start is not None:
                print(f"{pipeline_clazz.__name__} took {time.time() - start}")
            dist.barrier()


if __name__ == "__main__":
    main()
