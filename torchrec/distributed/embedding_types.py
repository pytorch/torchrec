#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
from dataclasses import dataclass
from enum import Enum, unique
from typing import Any, Dict, Generic, Iterator, List, Optional, TypeVar

import torch
from fbgemm_gpu.split_table_batched_embeddings_ops import EmbeddingLocation
from torch import fx, nn
from torchrec.distributed.types import (
    ModuleSharder,
    ParameterStorage,
    QuantizedCommCodecs,
    ShardedTensorMetadata,
    ShardingType,
    ShardMetadata,
)
from torchrec.modules.embedding_configs import (
    DataType,
    EmbeddingTableConfig,
    PoolingType,
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
    SHAMPOO = "SHAMPOO"


@unique
class EmbeddingComputeKernel(Enum):
    DENSE = "dense"
    FUSED = "fused"
    FUSED_UVM = "fused_uvm"
    FUSED_UVM_CACHING = "fused_uvm_caching"
    QUANT = "quant"
    QUANT_UVM = "quant_uvm"
    QUANT_UVM_CACHING = "quant_uvm_caching"


def compute_kernel_to_embedding_location(
    compute_kernel: EmbeddingComputeKernel,
) -> EmbeddingLocation:
    if compute_kernel in [
        EmbeddingComputeKernel.DENSE,
        EmbeddingComputeKernel.FUSED,
        EmbeddingComputeKernel.QUANT,
    ]:
        return EmbeddingLocation.DEVICE
    elif compute_kernel in [
        EmbeddingComputeKernel.FUSED_UVM,
        EmbeddingComputeKernel.QUANT_UVM,
    ]:
        return EmbeddingLocation.MANAGED
    elif compute_kernel in [
        EmbeddingComputeKernel.FUSED_UVM_CACHING,
        EmbeddingComputeKernel.QUANT_UVM_CACHING,
    ]:
        return EmbeddingLocation.MANAGED_CACHING
    else:
        raise ValueError(f"Invalid EmbeddingComputeKernel {compute_kernel}")


@dataclass
class SparseFeatures(Multistreamable):
    id_list_features: Optional[KeyedJaggedTensor] = None
    id_score_list_features: Optional[KeyedJaggedTensor] = None

    def record_stream(self, stream: torch.cuda.streams.Stream) -> None:
        if self.id_list_features is not None:
            self.id_list_features.record_stream(stream)
        if self.id_score_list_features is not None:
            self.id_score_list_features.record_stream(stream)

    def __fx_create_arg__(self, tracer: torch.fx.Tracer) -> fx.node.Argument:
        return tracer.create_node(
            "call_function",
            SparseFeatures,
            args=(
                tracer.create_arg(self.id_list_features),
                tracer.create_arg(self.id_score_list_features),
            ),
            kwargs={},
        )


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

    def __fx_create_arg__(self, tracer: torch.fx.Tracer) -> fx.node.Argument:
        return tracer.create_node(
            "call_function",
            SparseFeaturesList,
            args=(tracer.create_arg(self.features),),
            kwargs={},
        )


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

    def __fx_create_arg__(self, tracer: torch.fx.Tracer) -> fx.node.Argument:
        return tracer.create_node(
            "call_function",
            ListOfSparseFeaturesList,
            args=(tracer.create_arg(self.features_list),),
            kwargs={},
        )


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
    fused_params: Optional[Dict[str, Any]] = None


@dataclass
class GroupedEmbeddingConfig:
    data_type: DataType
    pooling: PoolingType
    is_weighted: bool
    has_feature_processor: bool
    compute_kernel: EmbeddingComputeKernel
    embedding_tables: List[ShardedEmbeddingTable]
    fused_params: Optional[Dict[str, Any]] = None

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
    def __init__(
        self,
        fused_params: Optional[Dict[str, Any]] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
        variable_batch_size: bool = False,
    ) -> None:
        super().__init__(qcomm_codecs_registry=qcomm_codecs_registry)
        self._variable_batch_size = variable_batch_size

        # TODO remove after decoupling
        self._fused_params = fused_params

    def sharding_types(self, compute_device_type: str) -> List[str]:
        types = [
            ShardingType.DATA_PARALLEL.value,
            ShardingType.TABLE_WISE.value,
            ShardingType.COLUMN_WISE.value,
            ShardingType.TABLE_COLUMN_WISE.value,
        ]
        if compute_device_type in {"cuda"}:
            types += [
                ShardingType.ROW_WISE.value,
                ShardingType.TABLE_ROW_WISE.value,
            ]

        return types

    def compute_kernels(
        # TODO remove after decoupling
        self,
        sharding_type: str,
        compute_device_type: str,
    ) -> List[str]:
        ret: List[str] = []
        if sharding_type != ShardingType.DATA_PARALLEL.value:
            ret += [
                EmbeddingComputeKernel.FUSED.value,
            ]
            if compute_device_type in {"cuda"}:
                ret += [
                    EmbeddingComputeKernel.FUSED_UVM.value,
                    EmbeddingComputeKernel.FUSED_UVM_CACHING.value,
                ]
        else:
            ret += [
                EmbeddingComputeKernel.DENSE.value,
            ]
        return ret

    @property
    def fused_params(self) -> Optional[Dict[str, Any]]:
        return self._fused_params

    @property
    def variable_batch_size(self) -> bool:
        return self._variable_batch_size

    def storage_usage(
        self, tensor: torch.Tensor, compute_device_type: str, compute_kernel: str
    ) -> Dict[str, int]:
        """
        List of system resources and corresponding usage given a compute device and
        compute kernel
        """
        tensor_bytes = tensor.element_size() * tensor.nelement()
        if compute_kernel in {
            EmbeddingComputeKernel.FUSED_UVM.value,
            EmbeddingComputeKernel.FUSED_UVM_CACHING.value,
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


class BaseQuantEmbeddingSharder(ModuleSharder[M]):
    def __init__(
        self,
        fused_params: Optional[Dict[str, Any]] = None,
        shardable_params: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self._fused_params = fused_params
        if not shardable_params:
            shardable_params = []
        self._shardable_params: List[str] = shardable_params

    def sharding_types(self, compute_device_type: str) -> List[str]:
        types = [
            ShardingType.TABLE_WISE.value,
        ]

        return types

    def shardable_parameters(self, module: M) -> Dict[str, nn.Parameter]:

        shardable_params: Dict[str, nn.Parameter] = {}
        for name, param in module.state_dict().items():
            if name.endswith(".weight"):
                table_name = name.split(".")[-2]
                shardable_params[table_name] = param

        if self._shardable_params:
            assert all(
                [
                    table_name in self._shardable_params
                    for table_name in shardable_params.keys()
                ]
            ) or all(
                [
                    table_name not in self._shardable_params
                    for table_name in shardable_params.keys()
                ]
            ), f"Cannot partially shard {type(module)}, please check sharder kwargs"
            shardable_params = {
                table_name: param
                for table_name, param in shardable_params.items()
                if table_name in self._shardable_params
            }

        return shardable_params

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        ret = [
            EmbeddingComputeKernel.QUANT.value,
        ]
        if compute_device_type in {"cuda"}:
            ret += [
                EmbeddingComputeKernel.QUANT_UVM.value,
                EmbeddingComputeKernel.QUANT_UVM_CACHING.value,
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
        tensor_bytes = tensor.element_size() * tensor.nelement() + tensor.shape[0] * 4
        if compute_kernel in {
            EmbeddingComputeKernel.QUANT_UVM.value,
            EmbeddingComputeKernel.QUANT_UVM_CACHING.value,
        }:
            assert compute_device_type in {"cuda"}
            return {ParameterStorage.DDR.value: tensor_bytes}
        else:
            assert compute_device_type in {"cuda", "cpu"}
            storage_map = {"cuda": ParameterStorage.HBM, "cpu": ParameterStorage.DDR}
            return {storage_map[compute_device_type].value: tensor_bytes}
