#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import List, Optional

import torch
from torchrec.streamable import Multistreamable


@dataclass
class VariableBatchShardingContext(Multistreamable):
    """
    For variable batch size case, we need pass `batch_size_per_rank` to
    PooledEmbeddingsAllToAll and it can be retrieved from SparseFeaturesAllToAll.

    Attributes:
        batch_size_per_rank (List[int]): stores batch size in each rank.
        batch_size_per_rank_tensor (Optional[torch.Tensor]): batch_size_per_rank stored in
            tensor.
    """

    batch_size_per_rank: List[int] = field(default_factory=list)
    batch_size_per_rank_tensor: Optional[torch.Tensor] = None

    def record_stream(self, stream: torch.cuda.streams.Stream) -> None:
        if self.batch_size_per_rank_tensor is not None:
            # pyre-fixme[6]: For 1st param expected `Stream` but got `Stream`.
            self.batch_size_per_rank_tensor.record_stream(stream)
