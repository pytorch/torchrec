#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Example usage:

Buck2 (internal):
    buck2 run @fbcode//mode/opt fbcode//torchrec/distributed/benchmark:benchmark_train_pipeline -- --world_size=2 --pipeline=sparse --batch_size=10

OSS (external):
    python -m torchrec.distributed.benchmark.benchmark_train_pipeline --world_size=4 --pipeline=sparse --batch_size=10

Adding New Model Support:
    See benchmark_pipeline_utils.py for step-by-step instructions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type, Union

import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from torch import nn
from torchrec.distributed.benchmark.benchmark_pipeline_utils import (
    BaseModelConfig,
    create_model_config,
    DeepFMConfig,
    DLRMConfig,
    generate_data,
    generate_pipeline,
    generate_planner,
    generate_sharded_model_and_optimizer,
    generate_tables,
    TestSparseNNConfig,
    TestTowerCollectionSparseNNConfig,
    TestTowerSparseNNConfig,
)
from torchrec.distributed.benchmark.benchmark_utils import benchmark_func, cmd_conf
from torchrec.distributed.comm import get_local_size
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.planner import Topology

from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    run_multi_process_func,
)
from torchrec.distributed.test_utils.test_input import ModelInput
from torchrec.distributed.test_utils.test_model import TestOverArchLarge
from torchrec.distributed.train_pipeline import TrainPipeline
from torchrec.distributed.types import ShardingType
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
        dense_optimizer (str): Optimizer to use for dense parameters.
            Default is "SGD".
        dense_lr (float): Learning rate for dense parameters.
            Default is 0.1.
        sparse_optimizer (str): Optimizer to use for sparse parameters.
            Default is "EXACT_ADAGRAD".
        sparse_lr (float): Learning rate for sparse parameters.
            Default is 0.1.
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
    dense_optimizer: str = "SGD"
    dense_lr: float = 0.1
    dense_momentum: Optional[float] = None
    dense_weight_decay: Optional[float] = None
    sparse_optimizer: str = "EXACT_ADAGRAD"
    sparse_lr: float = 0.1
    sparse_momentum: Optional[float] = None
    sparse_weight_decay: Optional[float] = None


@dataclass
class EmbeddingTablesConfig:
    """
    Configuration for embedding tables.

    This class defines the parameters for generating embedding tables with both weighted
    and unweighted features.

    Args:
        num_unweighted_features (int): Number of unweighted features to generate.
            Default is 100.
        num_weighted_features (int): Number of weighted features to generate.
            Default is 100.
        embedding_feature_dim (int): Dimension of the embedding vectors.
            Default is 128.
    """

    num_unweighted_features: int = 100
    num_weighted_features: int = 100
    embedding_feature_dim: int = 128


@dataclass
class PipelineConfig:
    """
    Configuration for training pipelines.

    This class defines the parameters for configuring the training pipeline.

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


@dataclass
class ModelSelectionConfig:
    model_name: str = "test_sparse_nn"

    # Common config for all model types
    batch_size: int = 8192
    num_float_features: int = 10
    feature_pooling_avg: int = 10
    use_offsets: bool = False
    dev_str: str = ""
    long_kjt_indices: bool = True
    long_kjt_offsets: bool = True
    long_kjt_lengths: bool = True
    pin_memory: bool = True

    # TestSparseNN specific config
    embedding_groups: Optional[Dict[str, List[str]]] = None
    feature_processor_modules: Optional[Dict[str, torch.nn.Module]] = None
    max_feature_lengths: Optional[Dict[str, int]] = None
    over_arch_clazz: Type[nn.Module] = TestOverArchLarge
    postproc_module: Optional[nn.Module] = None
    zch: bool = False

    # DeepFM specific config
    hidden_layer_size: int = 20
    deep_fm_dimension: int = 5

    # DLRM specific config
    dense_arch_layer_sizes: List[int] = field(default_factory=lambda: [20, 10])
    over_arch_layer_sizes: List[int] = field(default_factory=lambda: [5, 3])


@cmd_conf
def main(
    run_option: RunOptions,
    table_config: EmbeddingTablesConfig,
    model_selection: ModelSelectionConfig,
    pipeline_config: PipelineConfig,
    model_config: Optional[
        Union[
            TestSparseNNConfig,
            TestTowerCollectionSparseNNConfig,
            TestTowerSparseNNConfig,
            DeepFMConfig,
            DLRMConfig,
        ]
    ] = None,
) -> None:
    tables, weighted_tables = generate_tables(
        num_unweighted_features=table_config.num_unweighted_features,
        num_weighted_features=table_config.num_weighted_features,
        embedding_feature_dim=table_config.embedding_feature_dim,
    )

    if model_config is None:
        model_config = create_model_config(
            model_name=model_selection.model_name,
            batch_size=model_selection.batch_size,
            num_float_features=model_selection.num_float_features,
            feature_pooling_avg=model_selection.feature_pooling_avg,
            use_offsets=model_selection.use_offsets,
            dev_str=model_selection.dev_str,
            long_kjt_indices=model_selection.long_kjt_indices,
            long_kjt_offsets=model_selection.long_kjt_offsets,
            long_kjt_lengths=model_selection.long_kjt_lengths,
            pin_memory=model_selection.pin_memory,
            embedding_groups=model_selection.embedding_groups,
            feature_processor_modules=model_selection.feature_processor_modules,
            max_feature_lengths=model_selection.max_feature_lengths,
            over_arch_clazz=model_selection.over_arch_clazz,
            postproc_module=model_selection.postproc_module,
            zch=model_selection.zch,
            hidden_layer_size=model_selection.hidden_layer_size,
            deep_fm_dimension=model_selection.deep_fm_dimension,
            dense_arch_layer_sizes=model_selection.dense_arch_layer_sizes,
            over_arch_layer_sizes=model_selection.over_arch_layer_sizes,
        )

    # launch trainers
    run_multi_process_func(
        func=runner,
        world_size=run_option.world_size,
        tables=tables,
        weighted_tables=weighted_tables,
        run_option=run_option,
        model_config=model_config,
        pipeline_config=pipeline_config,
    )


def runner(
    rank: int,
    world_size: int,
    tables: List[EmbeddingBagConfig],
    weighted_tables: List[EmbeddingBagConfig],
    run_option: RunOptions,
    model_config: BaseModelConfig,
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
        unsharded_model = model_config.generate_model(
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
        planner = generate_planner(
            planner_type=run_option.planner_type,
            topology=topology,
            tables=tables,
            weighted_tables=weighted_tables,
            sharding_type=run_option.sharding_type,
            compute_kernel=run_option.compute_kernel,
            num_batches=run_option.num_batches,
            batch_size=model_config.batch_size,
            pooling_factors=run_option.pooling_factors,
            num_poolings=run_option.num_poolings,
        )
        bench_inputs = generate_data(
            tables=tables,
            weighted_tables=weighted_tables,
            model_config=model_config,
            num_batches=run_option.num_batches,
        )

        # Prepare fused_params for sparse optimizer
        fused_params = {
            "optimizer": getattr(EmbOptimType, run_option.sparse_optimizer.upper()),
            "learning_rate": run_option.sparse_lr,
        }

        # Add momentum and weight_decay to fused_params if provided
        if run_option.sparse_momentum is not None:
            fused_params["momentum"] = run_option.sparse_momentum

        if run_option.sparse_weight_decay is not None:
            fused_params["weight_decay"] = run_option.sparse_weight_decay

        sharded_model, optimizer = generate_sharded_model_and_optimizer(
            model=unsharded_model,
            sharding_type=run_option.sharding_type.value,
            kernel_type=run_option.compute_kernel.value,
            # pyre-ignore
            pg=ctx.pg,
            device=ctx.device,
            fused_params=fused_params,
            dense_optimizer=run_option.dense_optimizer,
            dense_lr=run_option.dense_lr,
            dense_momentum=run_option.dense_momentum,
            dense_weight_decay=run_option.dense_weight_decay,
            planner=planner,
        )
        pipeline = generate_pipeline(
            pipeline_type=pipeline_config.pipeline,
            emb_lookup_stream=pipeline_config.emb_lookup_stream,
            model=sharded_model,
            opt=optimizer,
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
