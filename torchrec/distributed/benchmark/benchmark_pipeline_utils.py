#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
from dataclasses import dataclass
from typing import Any, cast, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.distributed as dist
from torch import nn, optim
from torch.optim import Optimizer
from torchrec.distributed import DistributedModelParallel
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.constants import NUM_POOLINGS, POOLING_FACTOR
from torchrec.distributed.planner.planners import HeteroEmbeddingShardingPlanner
from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.distributed.test_utils.test_input import ModelInput
from torchrec.distributed.test_utils.test_model import (
    TestEBCSharder,
    TestOverArchLarge,
    TestSparseNN,
    TestTowerCollectionSparseNN,
    TestTowerSparseNN,
)
from torchrec.distributed.train_pipeline import (
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
class ModelConfig:
    model_name: str = "test_sparsenn"

    batch_size: int = 8192
    num_float_features: int = 10
    feature_pooling_avg: int = 10
    use_offsets: bool = False
    dev_str: str = ""
    long_kjt_indices: bool = True
    long_kjt_offsets: bool = True
    long_kjt_lengths: bool = True
    pin_memory: bool = True

    def generate_model(
        self,
        tables: List[EmbeddingBagConfig],
        weighted_tables: List[EmbeddingBagConfig],
        dense_device: torch.device,
    ) -> nn.Module:
        if self.model_name == "test_sparsenn":
            return TestSparseNN(
                tables=tables,
                weighted_tables=weighted_tables,
                dense_device=dense_device,
                sparse_device=torch.device("meta"),
                over_arch_clazz=TestOverArchLarge,
            )
        elif self.model_name == "test_tower_sparsenn":
            return TestTowerSparseNN(
                tables=tables,
                weighted_tables=weighted_tables,
                dense_device=dense_device,
                sparse_device=torch.device("meta"),
                num_float_features=self.num_float_features,
            )
        elif self.model_name == "test_tower_collection_sparsenn":
            return TestTowerCollectionSparseNN(
                tables=tables,
                weighted_tables=weighted_tables,
                dense_device=dense_device,
                sparse_device=torch.device("meta"),
                num_float_features=self.num_float_features,
            )
        else:
            raise RuntimeError(f"Unknown model name: {self.model_name}")


def generate_tables(
    num_unweighted_features: int,
    num_weighted_features: int,
    embedding_feature_dim: int,
) -> Tuple[
    List[EmbeddingBagConfig],
    List[EmbeddingBagConfig],
]:
    """
    Generate embedding bag configurations for both unweighted and weighted features.

    This function creates two lists of EmbeddingBagConfig objects:
    1. Unweighted tables: Named as "table_{i}" with feature names "feature_{i}"
    2. Weighted tables: Named as "weighted_table_{i}" with feature names "weighted_feature_{i}"

    For both types, the number of embeddings scales with the feature index,
    calculated as max(i + 1, 100) * 1000.

    Args:
        num_unweighted_features (int): Number of unweighted features to generate.
        num_weighted_features (int): Number of weighted features to generate.
        embedding_feature_dim (int): Dimension of the embedding vectors.

    Returns:
        Tuple[List[EmbeddingBagConfig], List[EmbeddingBagConfig]]: A tuple containing
        two lists - the first for unweighted embedding tables and the second for
        weighted embedding tables.
    """
    tables = [
        EmbeddingBagConfig(
            num_embeddings=max(i + 1, 100) * 1000,
            embedding_dim=embedding_feature_dim,
            name="table_" + str(i),
            feature_names=["feature_" + str(i)],
        )
        for i in range(num_unweighted_features)
    ]
    weighted_tables = [
        EmbeddingBagConfig(
            num_embeddings=max(i + 1, 100) * 1000,
            embedding_dim=embedding_feature_dim,
            name="weighted_table_" + str(i),
            feature_names=["weighted_feature_" + str(i)],
        )
        for i in range(num_weighted_features)
    ]
    return tables, weighted_tables


def generate_pipeline(
    pipeline_type: str,
    emb_lookup_stream: str,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    device: torch.device,
) -> Union[TrainPipelineBase, TrainPipelineSparseDist]:
    """
    Generate a training pipeline instance based on the configuration.

    This function creates and returns the appropriate training pipeline object
    based on the pipeline type specified. Different pipeline types are optimized
    for different training scenarios.

    Args:
        pipeline_type (str): The type of training pipeline to use. Options include:
            - "base": Basic training pipeline
            - "sparse": Pipeline optimized for sparse operations
            - "fused": Pipeline with fused sparse distribution
            - "semi": Semi-synchronous training pipeline
            - "prefetch": Pipeline with prefetching for sparse distribution
        emb_lookup_stream (str): The stream to use for embedding lookups.
            Only used by certain pipeline types (e.g., "fused").
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

    if pipeline_type == "semi":
        return TrainPipelineSemiSync(
            model=model, optimizer=opt, device=device, start_batch=0
        )
    elif pipeline_type == "fused":
        return TrainPipelineFusedSparseDist(
            model=model,
            optimizer=opt,
            device=device,
            emb_lookup_stream=emb_lookup_stream,
        )
    elif pipeline_type in _pipeline_cls:
        Pipeline = _pipeline_cls[pipeline_type]
        return Pipeline(model=model, optimizer=opt, device=device)
    else:
        raise RuntimeError(f"unknown pipeline option {pipeline_type}")


def generate_planner(
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
    """
    Generate an embedding sharding planner based on the specified configuration.

    Args:
        planner_type: Type of planner to use ("embedding" or "hetero")
        topology: Network topology for distributed training
        tables: List of unweighted embedding tables
        weighted_tables: List of weighted embedding tables
        sharding_type: Strategy for sharding embedding tables
        compute_kernel: Compute kernel to use for embedding tables
        num_batches: Number of batches to process
        batch_size: Size of each batch
        pooling_factors: Pooling factors for each feature of the table
        num_poolings: Number of poolings for each feature of the table

    Returns:
        An instance of EmbeddingShardingPlanner or HeteroEmbeddingShardingPlanner

    Raises:
        RuntimeError: If an unknown planner type is specified
    """
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


def generate_sharded_model_and_optimizer(
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
    # Ensure fused_params is always a dictionary
    fused_params_dict = {} if fused_params is None else fused_params

    sharder = TestEBCSharder(
        sharding_type=sharding_type,
        kernel_type=kernel_type,
        fused_params=fused_params_dict,
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


def generate_data(
    model_class_name: str,
    tables: List[EmbeddingBagConfig],
    weighted_tables: List[EmbeddingBagConfig],
    model_config: ModelConfig,
    num_batches: int,
) -> List[ModelInput]:
    """
    Generate model input data for benchmarking.

    Args:
        tables: List of unweighted embedding tables
        weighted_tables: List of weighted embedding tables
        model_config: Configuration for model generation
        num_batches: Number of batches to generate

    Returns:
        A list of ModelInput objects representing the generated batches
    """
    device = torch.device(model_config.dev_str) if model_config.dev_str else None

    if (
        model_class_name == "TestSparseNN"
        or model_class_name == "TestTowerSparseNN"
        or model_class_name == "TestTowerCollectionSparseNN"
    ):
        return [
            ModelInput.generate(
                batch_size=model_config.batch_size,
                tables=tables,
                weighted_tables=weighted_tables,
                num_float_features=model_config.num_float_features,
                pooling_avg=model_config.feature_pooling_avg,
                use_offsets=model_config.use_offsets,
                device=device,
                indices_dtype=(
                    torch.int64 if model_config.long_kjt_indices else torch.int32
                ),
                offsets_dtype=(
                    torch.int64 if model_config.long_kjt_offsets else torch.int32
                ),
                lengths_dtype=(
                    torch.int64 if model_config.long_kjt_lengths else torch.int32
                ),
                pin_memory=model_config.pin_memory,
            )
            for _ in range(num_batches)
        ]
    else:
        raise RuntimeError(f"Unknown model name: {model_config.model_name}")
