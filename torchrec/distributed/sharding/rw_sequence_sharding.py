#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Dict, Any

import torch
import torch.distributed as dist
from torchrec.distributed.dist_data import SequenceEmbeddingAllToAll
from torchrec.distributed.embedding_lookup import GroupedEmbeddingsLookup
from torchrec.distributed.embedding_sharding import (
    BaseEmbeddingDist,
    BaseSparseFeaturesDist,
    BaseEmbeddingLookup,
)
from torchrec.distributed.embedding_types import (
    SparseFeatures,
    BaseGroupedFeatureProcessor,
)
from torchrec.distributed.sharding.rw_sharding import (
    BaseRwEmbeddingSharding,
    RwSparseFeaturesDist,
)
from torchrec.distributed.sharding.sequence_sharding import (
    SequenceShardingContext,
    BaseSequenceEmbeddingDist,
)
from torchrec.distributed.types import Awaitable


class RwSequenceEmbeddingDist(BaseSequenceEmbeddingDist[torch.Tensor]):
    """
    Redistributes sequence embedding tensor in RW fashion with an AlltoAll operation.

    Constructor Args:
        pg (dist.ProcessGroup): ProcessGroup for AlltoAll communication.
        num_features (int): total number of features.
        device (Optional[torch.device]): device on which buffers will be allocated.
    """

    def __init__(
        self,
        # pyre-fixme[11]
        pg: dist.ProcessGroup,
        num_features: int,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self._dist = SequenceEmbeddingAllToAll(pg, [num_features] * pg.size(), device)

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
            unbucketize_permute_tensor=sharding_ctx.unbucketize_permute_tensor,
        )


class RwSequenceEmbeddingSharding(
    BaseRwEmbeddingSharding[SparseFeatures, torch.Tensor]
):
    """
    Shards sequence (unpooled) row-wise, i.e.. a given embedding table is evenly distributed
    by rows and table slices are placed on all ranks.
    """

    def create_input_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseSparseFeaturesDist[SparseFeatures]:
        num_id_list_features = self._get_id_list_features_num()
        num_id_score_list_features = self._get_id_score_list_features_num()
        id_list_feature_hash_sizes = self._get_id_list_features_hash_sizes()
        id_score_list_feature_hash_sizes = self._get_id_score_list_features_hash_sizes()
        return RwSparseFeaturesDist(
            pg=self._pg,
            num_id_list_features=num_id_list_features,
            num_id_score_list_features=num_id_score_list_features,
            id_list_feature_hash_sizes=id_list_feature_hash_sizes,
            id_score_list_feature_hash_sizes=id_score_list_feature_hash_sizes,
            device=device if device is not None else self._device,
            is_sequence=True,
            has_feature_processor=self._has_feature_processor,
        )

    def create_lookup(
        self,
        device: Optional[torch.device] = None,
        fused_params: Optional[Dict[str, Any]] = None,
        feature_processor: Optional[BaseGroupedFeatureProcessor] = None,
    ) -> BaseEmbeddingLookup:
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
        return RwSequenceEmbeddingDist(
            self._pg,
            self._get_id_list_features_num(),
            device if device is not None else self._device,
        )
