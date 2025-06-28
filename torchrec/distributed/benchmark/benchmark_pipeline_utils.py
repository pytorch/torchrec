#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Utilities for benchmarking training pipelines with different model configurations.

Adding New Model Support:
    1. Create config class inheriting from BaseModelConfig with generate_model() method
    2. Add the model to model_configs dict in create_model_config()
    3. Add model-specific params to ModelSelectionConfig and create_model_config's arguments in benchmark_train_pipeline.py
"""

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
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
class BaseModelConfig(ABC):
    """
    Abstract base class for model configurations.

    This class defines the common parameters shared across all model types
    and requires each concrete implementation to provide its own generate_model method.
    """

    # Common parameters for all model types
    batch_size: int
    num_float_features: int
    feature_pooling_avg: int
    use_offsets: bool
    dev_str: str
    long_kjt_indices: bool
    long_kjt_offsets: bool
    long_kjt_lengths: bool
    pin_memory: bool

    @abstractmethod
    def generate_model(
        self,
        tables: List[EmbeddingBagConfig],
        weighted_tables: List[EmbeddingBagConfig],
        dense_device: torch.device,
    ) -> nn.Module:
        """
        Generate a model instance based on the configuration.

        Args:
            tables: List of unweighted embedding tables
            weighted_tables: List of weighted embedding tables
            dense_device: Device to place dense layers on

        Returns:
            A neural network module instance
        """
        pass


@dataclass
class TestSparseNNConfig(BaseModelConfig):
    """Configuration for TestSparseNN model."""

    embedding_groups: Optional[Dict[str, List[str]]]
    feature_processor_modules: Optional[Dict[str, torch.nn.Module]]
    max_feature_lengths: Optional[Dict[str, int]]
    over_arch_clazz: Type[nn.Module]
    postproc_module: Optional[nn.Module]
    zch: bool

    def generate_model(
        self,
        tables: List[EmbeddingBagConfig],
        weighted_tables: List[EmbeddingBagConfig],
        dense_device: torch.device,
    ) -> nn.Module:
        return TestSparseNN(
            tables=tables,
            num_float_features=self.num_float_features,
            weighted_tables=weighted_tables,
            dense_device=dense_device,
            sparse_device=torch.device("meta"),
            max_feature_lengths=self.max_feature_lengths,
            feature_processor_modules=self.feature_processor_modules,
            over_arch_clazz=self.over_arch_clazz,
            postproc_module=self.postproc_module,
            embedding_groups=self.embedding_groups,
            zch=self.zch,
        )


@dataclass
class TestTowerSparseNNConfig(BaseModelConfig):
    """Configuration for TestTowerSparseNN model."""

    embedding_groups: Optional[Dict[str, List[str]]]
    feature_processor_modules: Optional[Dict[str, torch.nn.Module]]

    def generate_model(
        self,
        tables: List[EmbeddingBagConfig],
        weighted_tables: List[EmbeddingBagConfig],
        dense_device: torch.device,
    ) -> nn.Module:
        return TestTowerSparseNN(
            num_float_features=self.num_float_features,
            tables=tables,
            weighted_tables=weighted_tables,
            dense_device=dense_device,
            sparse_device=torch.device("meta"),
            embedding_groups=self.embedding_groups,
            feature_processor_modules=self.feature_processor_modules,
        )


@dataclass
class TestTowerCollectionSparseNNConfig(BaseModelConfig):
    """Configuration for TestTowerCollectionSparseNN model."""

    embedding_groups: Optional[Dict[str, List[str]]]
    feature_processor_modules: Optional[Dict[str, torch.nn.Module]]

    def generate_model(
        self,
        tables: List[EmbeddingBagConfig],
        weighted_tables: List[EmbeddingBagConfig],
        dense_device: torch.device,
    ) -> nn.Module:
        return TestTowerCollectionSparseNN(
            tables=tables,
            weighted_tables=weighted_tables,
            dense_device=dense_device,
            sparse_device=torch.device("meta"),
            num_float_features=self.num_float_features,
            embedding_groups=self.embedding_groups,
            feature_processor_modules=self.feature_processor_modules,
        )


@dataclass
class DeepFMConfig(BaseModelConfig):
    """Configuration for DeepFM model."""

    hidden_layer_size: int
    deep_fm_dimension: int

    def generate_model(
        self,
        tables: List[EmbeddingBagConfig],
        weighted_tables: List[EmbeddingBagConfig],
        dense_device: torch.device,
    ) -> nn.Module:
        # TODO: Implement DeepFM model generation
        raise NotImplementedError("DeepFM model generation not yet implemented")


@dataclass
class DLRMConfig(BaseModelConfig):
    """Configuration for DLRM model."""

    dense_arch_layer_sizes: List[int]
    over_arch_layer_sizes: List[int]

    def generate_model(
        self,
        tables: List[EmbeddingBagConfig],
        weighted_tables: List[EmbeddingBagConfig],
        dense_device: torch.device,
    ) -> nn.Module:
        # TODO: Implement DLRM model generation
        raise NotImplementedError("DLRM model generation not yet implemented")


# pyre-ignore[2]: Missing parameter annotation
def create_model_config(model_name: str, **kwargs) -> BaseModelConfig:

    model_configs = {
        "test_sparse_nn": TestSparseNNConfig,
        "test_tower_sparse_nn": TestTowerSparseNNConfig,
        "test_tower_collection_sparse_nn": TestTowerCollectionSparseNNConfig,
        "deepfm": DeepFMConfig,
        "dlrm": DLRMConfig,
    }

    if model_name not in model_configs:
        raise ValueError(f"Unknown model name: {model_name}")

    # Filter kwargs to only include valid parameters for the specific model config class
    model_class = model_configs[model_name]
    valid_field_names = {field.name for field in fields(model_class)}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_field_names}

    return model_class(**filtered_kwargs)


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
    fused_params: Dict[str, Any],
    dense_optimizer: str,
    dense_lr: float,
    dense_momentum: Optional[float],
    dense_weight_decay: Optional[float],
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

    # Get dense parameters
    dense_params = [
        param
        for name, param in sharded_model.named_parameters()
        if "sparse" not in name
    ]

    # Create optimizer based on the specified type
    optimizer_class = getattr(optim, dense_optimizer)

    # Create optimizer with momentum and/or weight_decay if provided
    optimizer_kwargs = {"lr": dense_lr}

    if dense_momentum is not None:
        optimizer_kwargs["momentum"] = dense_momentum

    if dense_weight_decay is not None:
        optimizer_kwargs["weight_decay"] = dense_weight_decay

    optimizer = optimizer_class(dense_params, **optimizer_kwargs)

    return sharded_model, optimizer


def generate_data(
    tables: List[EmbeddingBagConfig],
    weighted_tables: List[EmbeddingBagConfig],
    model_config: BaseModelConfig,
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
