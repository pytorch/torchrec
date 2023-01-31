#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.distributed as dist  # noqa
from torchrec.distributed.embedding_sharding import EmbeddingShardingContext
from torchrec.distributed.embedding_types import KJTList
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.streamable import Multistreamable


@dataclass
class SequenceShardingContext(EmbeddingShardingContext):
    """
    Stores KJTAllToAll context and reuses it in SequenceEmbeddingsAllToAll.
    SequenceEmbeddingsAllToAll has the same comm pattern as KJTAllToAll.

    Attributes:
        features_before_input_dist (Optional[KeyedJaggedTensor]): stores the original
            KJT before input dist.
        input_splits(List[int]): stores the input splits of KJT AlltoAll.
        output_splits (List[int]): stores the output splits of KJT AlltoAll.
        unbucketize_permute_tensor (Optional[torch.Tensor]): stores the permute order of
            KJT bucketize (for row-wise sharding only).
        lengths_after_input_dist (Optional[torch.Tensor]): stores the KJT length after
            input dist.
    """

    features_before_input_dist: Optional[KeyedJaggedTensor] = None
    input_splits: List[int] = field(default_factory=list)
    output_splits: List[int] = field(default_factory=list)
    sparse_features_recat: Optional[torch.Tensor] = None
    unbucketize_permute_tensor: Optional[torch.Tensor] = None
    lengths_after_input_dist: Optional[torch.Tensor] = None

    def record_stream(self, stream: torch.cuda.streams.Stream) -> None:
        if self.features_before_input_dist is not None:
            self.features_before_input_dist.record_stream(stream)
        if self.sparse_features_recat is not None:
            # pyre-fixme[6]: For 1st param expected `Stream` but got `Stream`.
            self.sparse_features_recat.record_stream(stream)
        if self.unbucketize_permute_tensor is not None:
            # pyre-fixme[6]: For 1st param expected `Stream` but got `Stream`.
            self.unbucketize_permute_tensor.record_stream(stream)
        if self.lengths_after_input_dist is not None:
            # pyre-fixme[6]: For 1st param expected `Stream` but got `Stream`.
            self.lengths_after_input_dist.record_stream(stream)


@dataclass
class InferSequenceShardingContext(Multistreamable):
    """
    Stores inference context and reuses it in sequence embedding output_dist or result return.

    Attributes:
        features (Optional[List[KeyedJaggedTensor]]): stores the original
            shards of KJT after input dist.
    """

    features: Optional[KJTList] = None

    def record_stream(self, stream: torch.cuda.streams.Stream) -> None:
        if self.features is not None:
            # pyre-ignore [16]
            for feature in self.features:
                feature.record_stream(stream)
