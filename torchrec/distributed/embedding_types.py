#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
from dataclasses import dataclass
from enum import Enum, unique
from typing import Generic, List, Optional, Dict, Any, TypeVar, Iterator

import torch
from torch import nn
from torchrec.distributed.types import (
    ModuleSharder,
    ShardingType,
    ParameterStorage,
    ShardMetadata,
    ShardedTensorMetadata,
)
from torchrec.modules.embedding_configs import (
    PoolingType,
    DataType,
    EmbeddingTableConfig,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.streamable import Multistreamable


@unique
class OptimType(Enum):
    SGD = "SGD"
    LARS_SGD = "LARS_SGD"
    LAMB = "LAMB"
    PARTIAL_ROWWISE_LAMB = "PARTIAL_ROWWISE_LAMB"
    ADAM = "ADAM"
    PARTIAL_ROWWISE_ADAM = "PARTIAL_ROWWISE_ADAM"
    ADAGRAD = "ADAGRAD"
    ROWWISE_ADAGRAD = "ROWWISE_ADAGRAD"


@unique
class EmbeddingComputeKernel(Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    BATCHED_DENSE = "batched_dense"
    BATCHED_FUSED = "batched_fused"
    BATCHED_FUSED_UVM = "batched_fused_uvm"
    BATCHED_FUSED_UVM_CACHING = "batched_fused_uvm_caching"
    BATCHED_QUANT = "batched_quant"


@dataclass
class SparseFeatures(Multistreamable):
    id_list_features: Optional[KeyedJaggedTensor] = None
    id_score_list_features: Optional[KeyedJaggedTensor] = None

    def record_stream(self, stream: torch.cuda.streams.Stream) -> None:
        if self.id_list_features is not None:
            self.id_list_features.record_stream(stream)
        if self.id_score_list_features is not None:
            self.id_score_list_features.record_stream(stream)


class SparseFeaturesList(Multistreamable):
    def __init__(self, features: List[SparseFeatures]) -> None:
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __setitem__(self, key: int, item: SparseFeatures) -> None:
        self.features[key] = item

    def __getitem__(self, key: int) -> SparseFeatures:
        return self.features[key]

    def __iter__(self) -> Iterator[SparseFeatures]:
        return iter(self.features)

    def record_stream(self, stream: torch.cuda.streams.Stream) -> None:
        for feature in self.features:
            feature.record_stream(stream)


class ListOfSparseFeaturesList(Multistreamable):
    def __init__(self, features: List[SparseFeaturesList]) -> None:
        self.features_list = features

    def __len__(self) -> int:
        return len(self.features_list)

    def __setitem__(self, key: int, item: SparseFeaturesList) -> None:
        self.features_list[key] = item

    def __getitem__(self, key: int) -> SparseFeaturesList:
        return self.features_list[key]

    def __iter__(self) -> Iterator[SparseFeaturesList]:
        return iter(self.features_list)

    def record_stream(self, stream: torch.cuda.streams.Stream) -> None:
        for feature in self.features_list:
            feature.record_stream(stream)


@dataclass
class ShardedConfig:
    local_rows: int = 0
    local_cols: int = 0


@dataclass
class ShardedMetaConfig(ShardedConfig):
    local_metadata: Optional[ShardMetadata] = None
    global_metadata: Optional[ShardedTensorMetadata] = None


@dataclass
class EmbeddingAttributes:
    compute_kernel: EmbeddingComputeKernel = EmbeddingComputeKernel.DENSE


@dataclass
class ShardedEmbeddingTable(
    ShardedMetaConfig,
    EmbeddingAttributes,
    EmbeddingTableConfig,
):
    pass


@dataclass
class GroupedEmbeddingConfig:
    data_type: DataType
    pooling: PoolingType
    is_weighted: bool
    has_feature_processor: bool
    compute_kernel: EmbeddingComputeKernel
    embedding_tables: List[ShardedEmbeddingTable]

    def feature_hash_sizes(self) -> List[int]:
        feature_hash_sizes = []
        for table in self.embedding_tables:
            feature_hash_sizes.extend(table.num_features() * [table.num_embeddings])
        return feature_hash_sizes

    def num_features(self) -> int:
        num_features = 0
        for table in self.embedding_tables:
            num_features += table.num_features()
        return num_features

    def dim_sum(self) -> int:
        dim_sum = 0
        for table in self.embedding_tables:
            dim_sum += table.num_features() * table.local_cols
        return dim_sum

    def feature_names(self) -> List[str]:
        feature_names = []
        for table in self.embedding_tables:
            feature_names.extend(table.feature_names)
        return feature_names

    def embedding_dims(self) -> List[int]:
        embedding_dims = []
        for table in self.embedding_tables:
            embedding_dims.extend([table.local_cols] * table.num_features())
        return embedding_dims

    def embedding_names(self) -> List[str]:
        embedding_names = []
        for table in self.embedding_tables:
            embedding_names.extend(table.embedding_names)
        return embedding_names

    def embedding_shard_metadata(self) -> List[Optional[ShardMetadata]]:
        embedding_shard_metadata: List[Optional[ShardMetadata]] = []
        for table in self.embedding_tables:
            for _ in table.feature_names:
                embedding_shard_metadata.append(table.local_metadata)
        return embedding_shard_metadata


F = TypeVar("F", bound=Multistreamable)
T = TypeVar("T")


class BaseEmbeddingLookup(abc.ABC, nn.Module, Generic[F, T]):
    """
    Interface implemented by different embedding implementations:
    e.g. one, which relies on `nn.EmbeddingBag` or table-batched one, etc.
    """

    @abc.abstractmethod
    def forward(
        self,
        sparse_features: F,
    ) -> T:
        pass

    def sparse_grad_parameter_names(
        self, destination: Optional[List[str]] = None, prefix: str = ""
    ) -> List[str]:
        destination = [] if destination is None else destination
        return destination


M = TypeVar("M", bound=nn.Module)


class BaseEmbeddingSharder(ModuleSharder[M]):
    def __init__(self, fused_params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()

        self._fused_params = fused_params

    def sharding_types(self, compute_device_type: str) -> List[str]:
        types = [
            ShardingType.DATA_PARALLEL.value,
            ShardingType.TABLE_WISE.value,
            ShardingType.ROW_WISE.value,
            ShardingType.COLUMN_WISE.value,
            ShardingType.TABLE_COLUMN_WISE.value,
        ]
        if compute_device_type in {"cuda"}:
            types.append(ShardingType.TABLE_ROW_WISE.value)

        return types

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        ret = [
            EmbeddingComputeKernel.DENSE.value,
            EmbeddingComputeKernel.BATCHED_DENSE.value,
        ]
        if sharding_type != ShardingType.DATA_PARALLEL.value:
            ret += [
                EmbeddingComputeKernel.BATCHED_FUSED.value,
                EmbeddingComputeKernel.SPARSE.value,
            ]
            if compute_device_type in {"cuda"}:
                ret += [
                    EmbeddingComputeKernel.BATCHED_FUSED_UVM.value,
                    EmbeddingComputeKernel.BATCHED_FUSED_UVM_CACHING.value,
                ]
        return ret

    @property
    def fused_params(self) -> Optional[Dict[str, Any]]:
        return self._fused_params

    def storage_usage(
        self, tensor: torch.Tensor, compute_device_type: str, compute_kernel: str
    ) -> Dict[str, int]:
        """
        List of system resources and corresponding usage given a compute device and
        compute kernel
        """
        tensor_bytes = tensor.element_size() * tensor.nelement()
        if compute_kernel in {
            EmbeddingComputeKernel.BATCHED_FUSED_UVM.value,
            EmbeddingComputeKernel.BATCHED_FUSED_UVM_CACHING.value,
        }:
            assert compute_device_type in {"cuda"}
            return {ParameterStorage.DDR.value: tensor_bytes}
        else:
            assert compute_device_type in {"cuda", "cpu"}
            storage_map = {"cuda": ParameterStorage.HBM, "cpu": ParameterStorage.DDR}
            return {
                storage_map[compute_device_type].value: tensor.element_size()
                * tensor.nelement()
            }


class BaseGroupedFeatureProcessor(nn.Module):
    """
    Abstract base class for grouped feature processor
    """

    @abc.abstractmethod
    def forward(
        self,
        features: KeyedJaggedTensor,
    ) -> KeyedJaggedTensor:
        pass

    def sparse_grad_parameter_names(
        self, destination: Optional[List[str]] = None, prefix: str = ""
    ) -> List[str]:
        destination = [] if destination is None else destination
        return destination
