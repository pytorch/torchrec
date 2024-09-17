#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import inspect
import unittest
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch
from torchrec.modules.embedding_configs import (
    DataType,
    EmbeddingBagConfig,
    EmbeddingConfig,
    PoolingType,
)

from torchrec.schema.utils import is_signature_compatible


@dataclass
class StableEmbeddingBagConfig:
    num_embeddings: int
    embedding_dim: int
    name: str = ""
    data_type: DataType = DataType.FP32
    feature_names: List[str] = field(default_factory=list)
    weight_init_max: Optional[float] = None
    weight_init_min: Optional[float] = None
    num_embeddings_post_pruning: Optional[int] = None

    init_fn: Optional[Callable[[torch.Tensor], Optional[torch.Tensor]]] = None
    # when the position_weighted feature is in this table config,
    # enable this flag to support rw_sharding
    need_pos: bool = False
    pooling: PoolingType = PoolingType.SUM


@dataclass
class StableEmbeddingConfig:
    num_embeddings: int
    embedding_dim: int
    name: str = ""
    data_type: DataType = DataType.FP32
    feature_names: List[str] = field(default_factory=list)
    weight_init_max: Optional[float] = None
    weight_init_min: Optional[float] = None
    num_embeddings_post_pruning: Optional[int] = None

    init_fn: Optional[Callable[[torch.Tensor], Optional[torch.Tensor]]] = None
    # when the position_weighted feature is in this table config,
    # enable this flag to support rw_sharding
    need_pos: bool = False


class TestEmbeddingConfigSchema(unittest.TestCase):
    def test_embedding_bag_config(self) -> None:
        self.assertTrue(
            is_signature_compatible(
                inspect.signature(StableEmbeddingBagConfig.__init__),
                inspect.signature(EmbeddingBagConfig.__init__),
            )
        )

    def test_embedding_config(self) -> None:
        self.assertTrue(
            is_signature_compatible(
                inspect.signature(StableEmbeddingConfig.__init__),
                inspect.signature(EmbeddingConfig.__init__),
            )
        )
