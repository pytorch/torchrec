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

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Dict, List, Optional, Type, Union

import torch
from torch import nn
from torchrec.distributed.test_utils.test_input import ModelInput
from torchrec.distributed.test_utils.test_model import (
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
from torchrec.models.deepfm import SimpleDeepFMNNWrapper
from torchrec.models.dlrm import DLRMWrapper
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection


@dataclass
class BaseModelConfig(ABC):
    """
    Abstract base class for model configurations.

    This class defines the common parameters shared across all model types
    and requires each concrete implementation to provide its own generate_model method.
    """

    # Common parameters for all model types
    batch_size: int
    batch_sizes: Optional[List[int]]
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
        # DeepFM only uses unweighted tables
        ebc = EmbeddingBagCollection(tables=tables, device=torch.device("meta"))

        # Create and return SimpleDeepFMNN model
        return SimpleDeepFMNNWrapper(
            num_dense_features=self.num_float_features,
            embedding_bag_collection=ebc,
            hidden_layer_size=self.hidden_layer_size,
            deep_fm_dimension=self.deep_fm_dimension,
        )


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
        # DLRM only uses unweighted tables
        ebc = EmbeddingBagCollection(tables=tables, device=torch.device("meta"))

        return DLRMWrapper(
            embedding_bag_collection=ebc,
            dense_in_features=self.num_float_features,
            dense_arch_layer_sizes=self.dense_arch_layer_sizes,
            over_arch_layer_sizes=self.over_arch_layer_sizes,
            dense_device=dense_device,
        )


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


def generate_pipeline(
    pipeline_type: str,
    emb_lookup_stream: str,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    device: torch.device,
    apply_jit: bool = False,
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
        apply_jit (bool): Whether to apply JIT (Just-In-Time) compilation to the model.
            Default is False.

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
            model=model,
            optimizer=opt,
            device=device,
            start_batch=0,
            apply_jit=apply_jit,
        )
    elif pipeline_type == "fused":
        return TrainPipelineFusedSparseDist(
            model=model,
            optimizer=opt,
            device=device,
            emb_lookup_stream=emb_lookup_stream,
            apply_jit=apply_jit,
        )
    elif pipeline_type == "base":
        assert apply_jit is False, "JIT is not supported for base pipeline"

        return TrainPipelineBase(model=model, optimizer=opt, device=device)
    else:
        Pipeline = _pipeline_cls[pipeline_type]
        # pyre-ignore[28]
        return Pipeline(model=model, optimizer=opt, device=device, apply_jit=apply_jit)


def generate_data(
    tables: List[EmbeddingBagConfig],
    weighted_tables: List[EmbeddingBagConfig],
    model_config: BaseModelConfig,
    batch_sizes: List[int],
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
            batch_size=batch_size,
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
        for batch_size in batch_sizes
    ]
