#!/usr/bin/env python3

from dataclasses import dataclass, field
from enum import Enum, unique
from typing import List, Dict


@unique
class PoolingType(Enum):
    SUM = "SUM"
    MEAN = "MEAN"
    NONE = "NONE"


@unique
class DataType(Enum):
    """
    Our fusion impl supports only certain types of data
    so it makes sense to retrict in a non-fused version as well.
    """

    FP32 = "FP32"
    FP16 = "FP16"
    INT8 = "INT8"


ELEMENT_SIZE: Dict[DataType, int] = {
    DataType.FP32: 4,
    DataType.FP16: 2,
    DataType.INT8: 1,
}


@dataclass
class BaseEmbeddingConfig:
    # FKA EmbeddingTableConfig
    num_embeddings: int
    embedding_dim: int
    name: str = ""
    data_type: DataType = DataType.FP32
    feature_names: List[str] = field(default_factory=list)

    def num_features(self) -> int:
        return len(self.feature_names)


@dataclass
class EmbeddingTableConfig(BaseEmbeddingConfig):
    pooling: PoolingType = PoolingType.SUM
    is_weighted: bool = False


@dataclass
class EmbeddingBagConfig(BaseEmbeddingConfig):
    # FKA PooledEmbeddingTableConfig
    pooling: PoolingType = PoolingType.SUM


@dataclass
class EmbeddingConfig(BaseEmbeddingConfig):
    # FKA SequenceEmbeddingTableConfig
    pass
