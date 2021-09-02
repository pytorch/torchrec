#!/usr/bin/env python3

import abc
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import List, Optional, Dict, Any, TypeVar

import torch
from torch import nn
from torchrec.distributed.types import (
    ModuleSharder,
    ShardingType,
    ParameterStorage,
)
from torchrec.distributed.types import ShardedTensorMetadata
from torchrec.modules.embedding_configs import (
    PoolingType,
    DataType,
    BaseEmbeddingConfig,
    EmbeddingTableConfig,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


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
    SSD = "ssd"


@dataclass
class SparseFeatures:
    id_list_features: Optional[KeyedJaggedTensor] = None
    id_score_list_features: Optional[KeyedJaggedTensor] = None

    def record_stream(self, stream: torch.cuda.streams.Stream) -> None:
        if self.id_list_features is not None:
            self.id_list_features.record_stream(stream)
        if self.id_score_list_features is not None:
            self.id_score_list_features.record_stream(stream)


@dataclass
class ShardedConfig:
    compute_kernel: EmbeddingComputeKernel = EmbeddingComputeKernel.DENSE
    embedding_names: List[str] = field(default_factory=list)
    rank: int = 0
    local_rows: int = 0
    local_cols: int = 0
    sharded_tensor: bool = False
    metadata: ShardedTensorMetadata = field(default_factory=ShardedTensorMetadata)


@dataclass
class ShardedEmbeddingTable(ShardedConfig, EmbeddingTableConfig):
    embedding_names: List[str] = field(default_factory=list)


@dataclass
class GroupedEmbeddingConfig:
    data_type: DataType
    pooling: PoolingType
    is_weighted: bool
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
            dim_sum += table.num_features() * table.embedding_dim
        return dim_sum

    def feature_names(self) -> List[str]:
        feature_names = []
        for table in self.embedding_tables:
            feature_names.extend(table.feature_names)
        return feature_names

    def embedding_dims(self) -> List[int]:
        embedding_dims = []
        for table in self.embedding_tables:
            embedding_dims.extend([table.embedding_dim] * table.num_features())
        return embedding_dims

    def embedding_names(self) -> List[str]:
        embedding_names = []
        for table in self.embedding_tables:
            embedding_names.extend(table.embedding_names)
        return embedding_names


class BaseEmbeddingLookup(abc.ABC, nn.Module):
    """
    Interface implemented by different embedding implementations:
    e.g. one, which relies on nn.EmbeddingBag or table-batched one, etc.
    """

    @abc.abstractmethod
    def forward(
        self,
        sparse_features: SparseFeatures,
    ) -> torch.Tensor:
        pass

    def sparse_grad_parameter_names(
        self, destination: Optional[List[str]] = None, prefix: str = ""
    ) -> List[str]:
        destination = [] if destination is None else destination
        return destination


M = TypeVar("M", bound=nn.Module)


class BaseEmbeddingSharder(ModuleSharder[M]):
    def __init__(self, fused_params: Optional[Dict[str, Any]] = None) -> None:
        self._fused_params = fused_params

    @property
    def sharding_types(self) -> List[str]:
        return [
            ShardingType.DATA_PARALLEL.value,
            ShardingType.TABLE_WISE.value,
            ShardingType.TABLE_ROW_WISE.value,
            ShardingType.ROW_WISE.value,
        ]

    def compute_kernels(self, sharding_type: str, device: torch.device) -> List[str]:
        ret = [
            EmbeddingComputeKernel.DENSE.value,
            EmbeddingComputeKernel.BATCHED_DENSE.value,
        ]
        if sharding_type != ShardingType.DATA_PARALLEL:
            ret += [
                EmbeddingComputeKernel.BATCHED_FUSED.value,
                EmbeddingComputeKernel.SPARSE.value,
            ]
            if device.type in {"cuda"}:
                ret += [
                    EmbeddingComputeKernel.BATCHED_FUSED_UVM.value,
                    EmbeddingComputeKernel.BATCHED_FUSED_UVM_CACHING.value,
                    EmbeddingComputeKernel.SSD.value,
                ]
        return ret

    @property
    def fused_params(self) -> Optional[Dict[str, Any]]:
        return self._fused_params

    def storage_usage(
        self, tensor: torch.Tensor, device: torch.device, compute_kernel: str
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
            assert device.type in {"cuda"}
            return {ParameterStorage.DDR.value: tensor_bytes}
        elif compute_kernel in {EmbeddingComputeKernel.SSD.value}:
            assert device.type in {"cuda"}
            return {ParameterStorage.SSD.value: tensor_bytes}
        else:
            assert device.type in {"cuda", "cpu"}
            storage_map = {"cuda": ParameterStorage.HBM, "cpu": ParameterStorage.DDR}
            return {
                storage_map[device.type].value: tensor.element_size()
                * tensor.nelement()
            }
