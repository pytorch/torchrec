#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from enum import Enum, unique
from math import sqrt
from typing import Dict, List, Optional

import torch
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops import PoolingMode


@unique
class PoolingType(Enum):
    SUM = "SUM"
    MEAN = "MEAN"
    NONE = "NONE"


@unique
class DataType(Enum):
    """
    Our fusion implementation supports only certain types of data
    so it makes sense to retrict in a non-fused version as well.
    """

    FP32 = "FP32"
    FP16 = "FP16"
    INT64 = "INT64"
    INT32 = "INT32"
    INT8 = "INT8"
    INT4 = "INT4"
    INT2 = "INT2"

    def __str__(self) -> str:
        return self.value


DATA_TYPE_NUM_BITS: Dict[DataType, int] = {
    DataType.FP32: 32,
    DataType.FP16: 16,
    DataType.INT8: 8,
    DataType.INT4: 4,
    DataType.INT2: 2,
}


def dtype_to_data_type(dtype: torch.dtype) -> DataType:
    if dtype == torch.float:
        return DataType.FP32
    elif dtype == torch.float16 or dtype == torch.half:
        return DataType.FP16
    elif dtype in {torch.int, torch.int32}:
        return DataType.INT32
    elif dtype in {torch.long, torch.int64}:
        return DataType.INT64
    elif dtype in {torch.quint8, torch.qint8, torch.int8, torch.uint8}:
        return DataType.INT8
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
    if pooling_type == PoolingType.SUM:
        return "sum"
    elif pooling_type == PoolingType.MEAN:
        return "mean"
    else:
        raise ValueError(f"Unsupported pooling type {pooling_type}")


def data_type_to_sparse_type(data_type: DataType) -> SparseType:
    if data_type == DataType.FP32:
        return SparseType.FP32
    elif data_type == DataType.FP16:
        return SparseType.FP16
    elif data_type == DataType.INT8:
        return SparseType.INT8
    elif data_type == DataType.INT4:
        return SparseType.INT4
    elif data_type == DataType.INT2:
        return SparseType.INT2
    else:
        raise ValueError(f"Invalid DataType {data_type}")


def data_type_to_dtype(data_type: DataType) -> torch.dtype:
    if data_type == DataType.FP32:
        return torch.float32
    elif data_type == DataType.FP16:
        return torch.float16
    elif data_type == DataType.INT64:
        return torch.int64
    elif data_type == DataType.INT32:
        return torch.int32
    elif data_type == DataType.INT8:
        return torch.int8
    elif data_type == DataType.INT4:
        return torch.quint4x2
    elif data_type == DataType.INT2:
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
