#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Any, Dict, Union, cast

import torch
import torch.nn as nn
from torchrec.distributed.embedding import EmbeddingCollectionSharder
from torchrec.distributed.test_utils.test_model import (
    ModelInput,
    TestDenseArch,
)
from torchrec.distributed.test_utils.test_model import TestSparseNNBase
from torchrec.modules.embedding_configs import BaseEmbeddingConfig, EmbeddingConfig
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

try:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
except OSError:
    pass


class TestSequenceSparseArch(nn.Module):
    def __init__(
        self,
        tables: List[EmbeddingConfig],
        embedding_names: List[str],
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if device is None:
            device = torch.device("cpu")
        self.ec: EmbeddingCollection = EmbeddingCollection(
            tables=tables,
            device=device,
        )
        self.embedding_names = embedding_names
        self.embedding_dim: int = self.ec.embedding_dim

    def forward(
        self,
        id_list_features: KeyedJaggedTensor,
    ) -> torch.Tensor:
        jt_dict = self.ec(id_list_features)
        padded_embeddings = [
            torch.ops.fbgemm.jagged_2d_to_dense(
                values=jt_dict[e].values(),
                offsets=jt_dict[e].offsets(),
                max_sequence_length=20,
            ).view(-1, 20 * self.embedding_dim)
            for e in self.embedding_names
        ]
        return torch.cat(
            padded_embeddings,
            dim=1,
        )


class TestSequenceOverArch(nn.Module):
    def __init__(
        self,
        tables: List[EmbeddingConfig],
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if device is None:
            device = torch.device("cpu")
        in_features = 8 + sum(
            [table.embedding_dim * len(table.feature_names) * 20 for table in tables]
        )
        self.linear: nn.modules.Linear = nn.Linear(
            in_features=in_features, out_features=16, device=device
        )

    def forward(
        self,
        dense: torch.Tensor,
        sparse: torch.Tensor,
    ) -> torch.Tensor:
        return self.linear(torch.cat([dense, sparse], dim=1))


class TestSequenceSparseNN(TestSparseNNBase):
    def __init__(
        self,
        tables: List[EmbeddingConfig],
        weighted_tables: Optional[List[EmbeddingConfig]] = None,
        num_float_features: int = 10,
        embedding_groups: Optional[Dict[str, List[str]]] = None,
        dense_device: Optional[torch.device] = None,
        sparse_device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(
            tables=cast(List[BaseEmbeddingConfig], tables),
            weighted_tables=cast(Optional[List[BaseEmbeddingConfig]], weighted_tables),
            embedding_groups=embedding_groups,
            dense_device=dense_device,
            sparse_device=sparse_device,
        )
        if embedding_groups is None:
            embedding_groups = {}

        self.dense = TestDenseArch(
            device=dense_device, num_float_features=num_float_features
        )
        self.sparse = TestSequenceSparseArch(
            tables,
            list(embedding_groups.values())[0] if embedding_groups.values() else [],
            device=sparse_device,
        )
        self.over = TestSequenceOverArch(tables=tables, device=dense_device)

    def forward(
        self,
        input: ModelInput,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        dense_r = self.dense(input.float_features)
        sparse_r = self.sparse(input.idlist_features)
        over_r = self.over(dense_r, sparse_r)
        pred = torch.sigmoid(torch.mean(over_r, dim=1))
        if self.training:
            return (
                torch.nn.functional.binary_cross_entropy_with_logits(pred, input.label),
                pred,
            )
        else:
            return pred


class TestEmbeddingCollectionSharder(EmbeddingCollectionSharder[EmbeddingCollection]):
    def __init__(self, sharding_type: str, kernel_type: str) -> None:
        self._sharding_type = sharding_type
        self._kernel_type = kernel_type

    """
    Restricts sharding to single type only.
    """

    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [self._sharding_type]

    """
    Restricts to single impl.
    """

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [self._kernel_type]

    @property
    def fused_params(self) -> Optional[Dict[str, Any]]:
        return {"learning_rate": 0.1}
