#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
from dataclasses import dataclass, field
from typing import List, TypeVar, Optional

import torch
from torchrec.distributed.embedding_sharding import BaseEmbeddingDist
from torchrec.distributed.types import Awaitable
from torchrec.streamable import Multistreamable


@dataclass
class VariableBatchShardingContext(Multistreamable):
    """
    For variable batch size case, we need pass batch_size_per_rank to
    PooledEmbeddingsAllToAll and it can be get from SparseFeaturesAllToAll.

    batch_size_per_rank: stores batch size in each rank.
    batch_size_per_rank_tensor: batch_size_per_rank stored in tensor.
    """

    batch_size_per_rank: List[int] = field(default_factory=list)
    batch_size_per_rank_tensor: Optional[torch.Tensor] = None

    def record_stream(self, stream: torch.cuda.streams.Stream) -> None:
        if self.batch_size_per_rank_tensor is not None:
            self.batch_size_per_rank_tensor.record_stream(stream)


T = TypeVar("T")


class BaseVariableBatchEmbeddingDist(BaseEmbeddingDist[T]):
    """
    Base class for converting output of EmbeddingLookup
    from model-parallel to data-parallel for variable batch size.
    """

    @abc.abstractmethod
    def forward(
        self,
        local_embs: T,
        sharding_ctx: VariableBatchShardingContext,
    ) -> Awaitable[torch.Tensor]:
        pass
