#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
from dataclasses import dataclass, field
from typing import List, Optional, TypeVar

import torch
import torch.distributed as dist  # noqa
from torchrec.distributed.embedding_sharding import BaseEmbeddingDist
from torchrec.distributed.types import Awaitable
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.streamable import Multistreamable


@dataclass
class SequenceShardingContext(Multistreamable):
    """
    SequenceEmbeddingAllToAll has the same comm pattern as KJTAllToAll.
    Stores KJTAllToAll context and reuses it in SequenceEmbeddingAllToAll.

    features_before_input_dist: stores the original KJT before input dist.
    input_splits: stores the input splits of KJT AlltoAll.
    output_splits: stores the output splits of KJT AlltoAll.
    unbucketize_permute_tensor: stores the permute order of
        KJT bucketize (for row-wise sharding only).
    lengths_after_input_dist: stores the KJT length after input dist.
    """

    features_before_input_dist: Optional[KeyedJaggedTensor] = None
    input_splits: List[int] = field(default_factory=list)
    output_splits: List[int] = field(default_factory=list)
    unbucketize_permute_tensor: Optional[torch.Tensor] = None
    lengths_after_input_dist: Optional[torch.Tensor] = None

    def record_stream(self, stream: torch.cuda.streams.Stream) -> None:
        if self.features_before_input_dist is not None:
            self.features_before_input_dist.record_stream(stream)
        if self.unbucketize_permute_tensor is not None:
            self.unbucketize_permute_tensor.record_stream(stream)
        if self.lengths_after_input_dist is not None:
            self.lengths_after_input_dist.record_stream(stream)


T = TypeVar("T")


class BaseSequenceEmbeddingDist(BaseEmbeddingDist[T]):
    """
    Converts output of Sequence EmbeddingLookup from model-parallel to data-parallel.
    """

    @abc.abstractmethod
    def forward(
        self,
        local_embs: T,
        sharding_ctx: SequenceShardingContext,
    ) -> Awaitable[torch.Tensor]:
        pass
