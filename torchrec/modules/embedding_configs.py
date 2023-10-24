#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from enum import Enum, unique
from functools import partial
from math import sqrt
from typing import Callable, Dict, List, NamedTuple, Optional

import torch
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    BoundsCheckMode as FbgemmBoundsCheckMode,
    CacheAlgorithm as FbgemmCacheAlgorithm,
    PoolingMode,
)
from torchrec.distributed.types import BoundsCheckMode, CacheAlgorithm, DataType


@unique
class PoolingType(Enum):
    SUM = "SUM"
    MEAN = "MEAN"
    NONE = "NONE"


DATA_TYPE_NUM_BITS: Dict[DataType, int] = {
    DataType.FP32: 32,
    DataType.FP16: 16,
    DataType.BF16: 16,
    DataType.INT8: 8,
    DataType.UINT8: 8,
    DataType.INT4: 4,
    DataType.INT2: 2,
}


def to_fbgemm_bounds_check_mode(
    bounds_check_mode: BoundsCheckMode,
) -> FbgemmBoundsCheckMode:
    if bounds_check_mode == BoundsCheckMode.FATAL:
        return FbgemmBoundsCheckMode.FATAL
    elif bounds_check_mode == BoundsCheckMode.WARNING:
        return FbgemmBoundsCheckMode.WARNING
    elif bounds_check_mode == BoundsCheckMode.IGNORE:
        return FbgemmBoundsCheckMode.IGNORE
    elif bounds_check_mode == BoundsCheckMode.NONE:
        return FbgemmBoundsCheckMode.NONE
    else:
        raise Exception(f"Invalid bounds check mode {bounds_check_mode}")


def to_fbgemm_cache_algorithm(cache_algorithm: CacheAlgorithm) -> FbgemmCacheAlgorithm:
    if cache_algorithm == CacheAlgorithm.LRU:
        return FbgemmCacheAlgorithm.LRU
    elif cache_algorithm == CacheAlgorithm.LFU:
        return FbgemmCacheAlgorithm.LFU
    else:
        raise Exception(f"Invalid cache algorithm {cache_algorithm}")


def dtype_to_data_type(dtype: torch.dtype) -> DataType:
    if dtype == torch.float:
        return DataType.FP32
    elif dtype == torch.float16 or dtype == torch.half:
        return DataType.FP16
    elif dtype == torch.bfloat16:
        return DataType.BF16
    elif dtype in [torch.int, torch.int32]:
        return DataType.INT32
    elif dtype in [torch.long, torch.int64]:
        return DataType.INT64
    elif dtype in [torch.quint8, torch.qint8, torch.int8]:
        return DataType.INT8
    elif dtype == torch.uint8:
        return DataType.UINT8
    elif dtype == torch.quint4x2:
        return DataType.INT4
    elif dtype == torch.quint2x4:
        return DataType.INT2
    else:
        raise Exception(f"Invalid data type {dtype}")


def pooling_type_to_pooling_mode(pooling_type: PoolingType) -> PoolingMode:
    if pooling_type.value == PoolingType.SUM.value:
        return PoolingMode.SUM
    elif pooling_type.value == PoolingType.MEAN.value:
        return PoolingMode.MEAN
    elif pooling_type.value == PoolingType.NONE.value:
        return PoolingMode.NONE
    else:
        raise Exception(f"Invalid pooling type {pooling_type}")


def pooling_type_to_str(pooling_type: PoolingType) -> str:
    if pooling_type.value == PoolingType.SUM.value:
        return "sum"
    elif pooling_type.value == PoolingType.MEAN.value:
        return "mean"
    else:
        raise ValueError(f"Unsupported pooling type {pooling_type}")


def data_type_to_sparse_type(data_type: DataType) -> SparseType:
    if data_type == DataType.FP32:
        return SparseType.FP32
    elif data_type == DataType.FP16:
        return SparseType.FP16
    elif data_type == DataType.BF16:
        return SparseType.BF16
    elif data_type == DataType.INT8 or data_type == DataType.UINT8:
        return SparseType.INT8
    elif data_type == DataType.INT4:
        return SparseType.INT4
    elif data_type == DataType.INT2:
        return SparseType.INT2
    else:
        raise ValueError(f"Invalid DataType {data_type}")


def data_type_to_dtype(data_type: DataType) -> torch.dtype:
    if data_type.value == DataType.FP32.value:
        return torch.float32
    elif data_type.value == DataType.FP16.value:
        return torch.float16
    elif data_type.value == DataType.BF16.value:
        return torch.bfloat16
    elif data_type.value == DataType.INT64.value:
        return torch.int64
    elif data_type.value == DataType.INT32.value:
        return torch.int32
    elif data_type.value == DataType.INT8.value:
        return torch.int8
    elif data_type.value == DataType.UINT8.value:
        return torch.uint8
    elif data_type.value == DataType.INT4.value:
        return torch.quint4x2
    elif data_type.value == DataType.INT2.value:
        return torch.quint2x4
    else:
        raise ValueError(f"DataType {data_type} cannot be converted to dtype")


@dataclass
class BaseEmbeddingConfig:
    num_embeddings: int
    embedding_dim: int
    name: str = ""
    data_type: DataType = DataType.FP32
    feature_names: List[str] = field(default_factory=list)
    weight_init_max: Optional[float] = None
    weight_init_min: Optional[float] = None

    init_fn: Optional[Callable[[torch.Tensor], Optional[torch.Tensor]]] = None
    # when the position_weighted feature is in this table config,
    # enable this flag to support rw_sharding
    need_pos: bool = False

    def get_weight_init_max(self) -> float:
        if self.weight_init_max is None:
            return sqrt(1 / self.num_embeddings)
        else:
            return self.weight_init_max

    def get_weight_init_min(self) -> float:
        if self.weight_init_min is None:
            return -sqrt(1 / self.num_embeddings)
        else:
            return self.weight_init_min

    def num_features(self) -> int:
        return len(self.feature_names)

    def __post_init__(self) -> None:
        if self.init_fn is None:
            self.init_fn = partial(
                torch.nn.init.uniform_,
                a=self.get_weight_init_min(),
                b=self.get_weight_init_max(),
            )


# this class will be deprecated after migration
# and all the following code in sharding itself
# which contains has_feature_processor
@dataclass
class EmbeddingTableConfig(BaseEmbeddingConfig):
    pooling: PoolingType = PoolingType.SUM
    is_weighted: bool = False
    has_feature_processor: bool = False
    embedding_names: List[str] = field(default_factory=list)


@dataclass
class EmbeddingBagConfig(BaseEmbeddingConfig):
    pooling: PoolingType = PoolingType.SUM


@dataclass
class EmbeddingConfig(BaseEmbeddingConfig):
    pass


class QuantConfig(NamedTuple):
    activation: torch.quantization.PlaceholderObserver
    weight: torch.quantization.PlaceholderObserver
    per_table_weight_dtype: Optional[Dict[str, torch.dtype]] = None
