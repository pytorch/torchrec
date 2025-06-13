#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

#!/usr/bin/env python3

import copy

from dataclasses import dataclass, field
from typing import Any, cast, Dict, List, Optional, Tuple, Type, Union

import click

import torch
import torch.distributed as dist
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from torch import nn, optim
from torch.optim import Optimizer
from torchrec.distributed import DistributedModelParallel
from torchrec.distributed.benchmark.benchmark_utils import benchmark_func, cmd_conf
from torchrec.distributed.comm import get_local_size
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.constants import NUM_POOLINGS, POOLING_FACTOR
from torchrec.distributed.planner.planners import HeteroEmbeddingShardingPlanner
from torchrec.distributed.planner.types import ParameterConstraints

from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    run_multi_process_func,
)
from torchrec.distributed.test_utils.test_input import (
    ModelInput,
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
    TrainPipelineFusedSparseDist,
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
    """
    Configuration options for running sparse neural network benchmarks.

    This class defines the parameters that control how the benchmark is executed,
    including distributed training settings, batch configuration, and profiling options.

    Args:
        world_size (int): Number of processes/GPUs to use for distributed training.
            Default is 2.
        num_batches (int): Number of batches to process during the benchmark.
            Default is 10.
        sharding_type (ShardingType): Strategy for sharding embedding tables across devices.
            Default is ShardingType.TABLE_WISE (entire tables are placed on single devices).
        compute_kernel (EmbeddingComputeKernel): Compute kernel to use for embedding tables.
            Default is EmbeddingComputeKernel.FUSED.
        input_type (str): Type of input format to use for the model.
            Default is "kjt" (KeyedJaggedTensor).
        profile (str): Directory to save profiling results. If empty, profiling is disabled.
            Default is "" (disabled).
        planner_type (str): Type of sharding planner to use. Options are:
            - "embedding": EmbeddingShardingPlanner (default)
            - "hetero": HeteroEmbeddingShardingPlanner
        pooling_factors (Optional[List[float]]): Pooling factors for each feature of the table.
            This is the average number of values each sample has for the feature.
        num_poolings (Optional[List[float]]): Number of poolings for each feature of the table.
    """

    world_size: int = 2
    num_batches: int = 10
    sharding_type: ShardingType = ShardingType.TABLE_WISE
    compute_kernel: EmbeddingComputeKernel = EmbeddingComputeKernel.FUSED
    input_type: str = "kjt"
    profile: str = ""
    planner_type: str = "embedding"
    pooling_factors: Optional[List[float]] = None
    num_poolings: Optional[List[float]] = None


@dataclass
class EmbeddingTablesConfig:
    """
    Configuration for embedding tables used in sparse neural network benchmarks.

    This class defines the parameters for generating embedding tables with both weighted
    and unweighted features. It provides a method to generate the actual embedding bag
    configurations that can be used to create embedding tables.

    Args:
        num_unweighted_features (int): Number of unweighted features to generate.
            Default is 100.
        num_weighted_features (int): Number of weighted features to generate.
            Default is 100.
        embedding_feature_dim (int): Dimension of the embedding vectors.
            Default is 512.
    """

    num_unweighted_features: int = 100
    num_weighted_features: int = 100
    embedding_feature_dim: int = 128

    def generate_tables(
        self,
    ) -> Tuple[
        List[EmbeddingBagConfig],
        List[EmbeddingBagConfig],
    ]:
        """
        Generate embedding bag configurations for both unweighted and weighted features.

        This method creates two lists of EmbeddingBagConfig objects:
        1. Unweighted tables: Named as "table_{i}" with feature names "feature_{i}"
        2. Weighted tables: Named as "weighted_table_{i}" with feature names "weighted_feature_{i}"

        For both types, the number of embeddings scales with the feature index,
        calculated as max(i + 1, 100) * 1000.

        Returns:
            Tuple[List[EmbeddingBagConfig], List[EmbeddingBagConfig]]: A tuple containing
            two lists - the first for unweighted embedding tables and the second for
            weighted embedding tables.
        """
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
    """
    Configuration for training pipelines used in sparse neural network benchmarks.

    This class defines the parameters for configuring the training pipeline and provides
    a method to generate the appropriate pipeline instance based on the configuration.

    Args:
        pipeline (str): The type of training pipeline to use. Options include:
            - "base": Basic training pipeline
            - "sparse": Pipeline optimized for sparse operations
            - "fused": Pipeline with fused sparse distribution
            - "semi": Semi-synchronous training pipeline
            - "prefetch": Pipeline with prefetching for sparse distribution
            Default is "base".
        emb_lookup_stream (str): The stream to use for embedding lookups.
            Only used by certain pipeline types (e.g., "fused").
            Default is "data_dist".
    """

    pipeline: str = "base"
    emb_lookup_stream: str = "data_dist"

    def generate_pipeline(
        self, model: nn.Module, opt: torch.optim.Optimizer, device: torch.device
    ) -> Union[TrainPipelineBase, TrainPipelineSparseDist]:
        """
        Generate a training pipeline instance based on the configuration.

        This method creates and returns the appropriate training pipeline object
        based on the pipeline type specified in the configuration. Different
        pipeline types are optimized for different training scenarios.

        Args:
            model (nn.Module): The model to be trained.
            opt (torch.optim.Optimizer): The optimizer to use for training.
            device (torch.device): The device to run the training on.

        Returns:
            Union[TrainPipelineBase, TrainPipelineSparseDist]: An instance of the
            appropriate training pipeline class based on the configuration.

        Raises:
            RuntimeError: If an unknown pipeline type is specified.
        """
        _pipeline_cls: Dict[
            str, Type[Union[TrainPipelineBase, TrainPipelineSparseDist]]
        ] = {
            "base": TrainPipelineBase,
            "sparse": TrainPipelineSparseDist,
            "fused": TrainPipelineFusedSparseDist,
            "semi": TrainPipelineSemiSync,
            "prefetch": PrefetchTrainPipelineSparseDist,
        }

        if self.pipeline == "semi":
            return TrainPipelineSemiSync(
                model=model, optimizer=opt, device=device, start_batch=0
            )
        elif self.pipeline == "fused":
            return TrainPipelineFusedSparseDist(
                model=model,
                optimizer=opt,
                device=device,
                emb_lookup_stream=self.emb_lookup_stream,
            )
        elif self.pipeline in _pipeline_cls:
            Pipeline = _pipeline_cls[self.pipeline]
            return Pipeline(model=model, optimizer=opt, device=device)
        else:
            raise RuntimeError(f"unknown pipeline option {self.pipeline}")


@cmd_conf
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


def _generate_planner(
    planner_type: str,
    topology: Topology,
    tables: Optional[List[EmbeddingBagConfig]],
    weighted_tables: Optional[List[EmbeddingBagConfig]],
    sharding_type: ShardingType,
    compute_kernel: EmbeddingComputeKernel,
    num_batches: int,
    batch_size: int,
    pooling_factors: Optional[List[float]],
    num_poolings: Optional[List[float]],
) -> Union[EmbeddingShardingPlanner, HeteroEmbeddingShardingPlanner]:
    # Create parameter constraints for tables
    constraints = {}

    if pooling_factors is None:
        pooling_factors = [POOLING_FACTOR] * num_batches

    if num_poolings is None:
        num_poolings = [NUM_POOLINGS] * num_batches

    batch_sizes = [batch_size] * num_batches

    assert (
        len(pooling_factors) == num_batches and len(num_poolings) == num_batches
    ), "The length of pooling_factors and num_poolings must match the number of batches."

    if tables is not None:
        for table in tables:
            constraints[table.name] = ParameterConstraints(
                sharding_types=[sharding_type.value],
                compute_kernels=[compute_kernel.value],
                device_group="cuda",
                pooling_factors=pooling_factors,
                num_poolings=num_poolings,
                batch_sizes=batch_sizes,
            )

    if weighted_tables is not None:
        for table in weighted_tables:
            constraints[table.name] = ParameterConstraints(
                sharding_types=[sharding_type.value],
                compute_kernels=[compute_kernel.value],
                device_group="cuda",
                pooling_factors=pooling_factors,
                num_poolings=num_poolings,
                batch_sizes=batch_sizes,
                is_weighted=True,
            )

    if planner_type == "embedding":
        return EmbeddingShardingPlanner(
            topology=topology,
            constraints=constraints if constraints else None,
        )
    elif planner_type == "hetero":
        topology_groups = {"cuda": topology}
        return HeteroEmbeddingShardingPlanner(
            topology_groups=topology_groups,
            constraints=constraints if constraints else None,
        )
    else:
        raise RuntimeError(f"Unknown planner type: {planner_type}")


def _generate_sharded_model_and_optimizer(
    model: nn.Module,
    sharding_type: str,
    kernel_type: str,
    pg: dist.ProcessGroup,
    device: torch.device,
    fused_params: Optional[Dict[str, Any]] = None,
    planner: Optional[
        Union[
            EmbeddingShardingPlanner,
            HeteroEmbeddingShardingPlanner,
        ]
    ] = None,
) -> Tuple[nn.Module, Optimizer]:
    sharder = TestEBCSharder(
        sharding_type=sharding_type,
        kernel_type=kernel_type,
        fused_params=fused_params,
    )

    sharders = [cast(ModuleSharder[nn.Module], sharder)]

    # Use planner if provided
    plan = None
    if planner is not None:
        if pg is not None:
            plan = planner.collective_plan(model, sharders, pg)
        else:
            plan = planner.plan(model, sharders)

    sharded_model = DistributedModelParallel(
        module=copy.deepcopy(model),
        env=ShardingEnv.from_process_group(pg),
        init_data_parallel=True,
        device=device,
        sharders=sharders,
        plan=plan,
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
    # Ensure GPUs are available and we have enough of them
    assert (
        torch.cuda.is_available() and torch.cuda.device_count() >= world_size
    ), "CUDA not available or insufficient GPUs for the requested world_size"

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

        # Create a topology for sharding
        topology = Topology(
            local_world_size=get_local_size(world_size),
            world_size=world_size,
            compute_device=ctx.device.type,
        )

        # Create a planner for sharding based on the specified type
        planner = _generate_planner(
            planner_type=run_option.planner_type,
            topology=topology,
            tables=tables,
            weighted_tables=weighted_tables,
            sharding_type=run_option.sharding_type,
            compute_kernel=run_option.compute_kernel,
            num_batches=run_option.num_batches,
            batch_size=input_config.batch_size,
            pooling_factors=run_option.pooling_factors,
            num_poolings=run_option.num_poolings,
        )

        sharded_model, optimizer = _generate_sharded_model_and_optimizer(
            model=unsharded_model,
            sharding_type=run_option.sharding_type.value,
            kernel_type=run_option.compute_kernel.value,
            # pyre-ignore
            pg=ctx.pg,
            device=ctx.device,
            fused_params={
                "optimizer": EmbOptimType.EXACT_ADAGRAD,
                "learning_rate": 0.1,
            },
            planner=planner,
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
