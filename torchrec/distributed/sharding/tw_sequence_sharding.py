#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Any, Dict

import torch
import torch.distributed as dist
from torchrec.distributed.dist_data import SequenceEmbeddingAllToAll
from torchrec.distributed.embedding_lookup import GroupedEmbeddingsLookup
from torchrec.distributed.embedding_sharding import (
    BaseSparseFeaturesDist,
    BaseEmbeddingDist,
    BaseEmbeddingLookup,
)
from torchrec.distributed.embedding_types import BaseGroupedFeatureProcessor
from torchrec.distributed.embedding_types import SparseFeatures
from torchrec.distributed.sharding.sequence_sharding import (
    SequenceShardingContext,
    BaseSequenceEmbeddingDist,
)
from torchrec.distributed.sharding.tw_sharding import (
    TwSparseFeaturesDist,
    BaseTwEmbeddingSharding,
)
from torchrec.distributed.types import Awaitable


class TwSequenceEmbeddingDist(BaseSequenceEmbeddingDist[torch.Tensor]):
    """
    Redistributes sequence embedding tensor in TW fashion with an AlltoAll
    operation.

    Constructor Args:
        pg (dist.ProcessGroup): ProcessGroup for AlltoAll communication.
        features_per_rank (List[int]): number of features (sum of dimensions) of the
            embedding for each host.
        device (Optional[torch.device]): device on which buffers will be allocated.
    """

    def __init__(
        self,
        # pyre-fixme[11]
        pg: dist.ProcessGroup,
        features_per_rank: List[int],
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self._dist = SequenceEmbeddingAllToAll(pg, features_per_rank, device)

    def forward(
        self,
        local_embs: torch.Tensor,
        sharding_ctx: SequenceShardingContext,
    ) -> Awaitable[torch.Tensor]:
        """
        Performs AlltoAll operation on sequence embeddings tensor.

        Call Args:
            sharding_ctx (SequenceShardingContext): shared context from KJTAllToAll
                operation.
            local_embs (torch.Tensor): tensor of values to distribute.

        Returns:
            Awaitable[torch.Tensor]: awaitable of sequence embeddings.
        """

        return self._dist(
            local_embs,
            lengths=sharding_ctx.lengths_after_input_dist,
            input_splits=sharding_ctx.input_splits,
            output_splits=sharding_ctx.output_splits,
            unbucketize_permute_tensor=None,
        )


class TwSequenceEmbeddingSharding(
    BaseTwEmbeddingSharding[SparseFeatures, torch.Tensor]
):
    """
    Shards sequence (unpooled) table-wise, i.e.. a given embedding table is entirely placed
    on a selected rank.
    """

    def create_input_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseSparseFeaturesDist[SparseFeatures]:
        return TwSparseFeaturesDist(
            self._pg,
            self._id_list_features_per_rank(),
            self._id_score_list_features_per_rank(),
            device if device is not None else self._device,
        )

    def create_lookup(
        self,
        device: Optional[torch.device] = None,
        fused_params: Optional[Dict[str, Any]] = None,
        feature_processor: Optional[BaseGroupedFeatureProcessor] = None,
    ) -> BaseEmbeddingLookup:
        assert feature_processor is None
        return GroupedEmbeddingsLookup(
            grouped_configs=self._grouped_embedding_configs,
            fused_params=fused_params,
            pg=self._pg,
            device=device if device is not None else self._device,
        )

    def create_output_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseSequenceEmbeddingDist[torch.Tensor]:
        return TwSequenceEmbeddingDist(
            self._pg,
            self._id_list_features_per_rank(),
            device if device is not None else self._device,
        )
