#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, List, Optional

import torch
import torch.distributed as dist
from torchrec.distributed.dist_data import PooledEmbeddingsAllToAll
from torchrec.distributed.embedding_lookup import GroupedPooledEmbeddingsLookup
from torchrec.distributed.embedding_sharding import (
    BaseEmbeddingDist,
    BaseEmbeddingLookup,
    BaseSparseFeaturesDist,
    SparseFeaturesAllToAll,
)
from torchrec.distributed.embedding_types import (
    BaseGroupedFeatureProcessor,
    SparseFeatures,
)
from torchrec.distributed.sharding.tw_sharding import BaseTwEmbeddingSharding
from torchrec.distributed.sharding.vb_sharding import VariableBatchShardingContext
from torchrec.distributed.types import Awaitable, CommOp, QuantizedCommCodecs


class VariableBatchTwSparseFeaturesDist(BaseSparseFeaturesDist[SparseFeatures]):
    """
    Redistributes sparse features in TW fashion with an AlltoAll collective
    operation.

    Supports variable batch size.

    Args:
        pg (dist.ProcessGroup): ProcessGroup for AlltoAll communication.
        id_list_features_per_rank (List[int]): number of id list features to send to
            each rank.
        id_score_list_features_per_rank (List[int]): number of id score list features to
            send to each rank
        device (Optional[torch.device]): device on which buffers will be allocated.
    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
        id_list_features_per_rank: List[int],
        id_score_list_features_per_rank: List[int],
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self._dist = SparseFeaturesAllToAll(
            pg=pg,
            id_list_features_per_rank=id_list_features_per_rank,
            id_score_list_features_per_rank=id_score_list_features_per_rank,
            device=device,
            variable_batch_size=True,
        )

    def forward(
        self,
        sparse_features: SparseFeatures,
    ) -> Awaitable[Awaitable[SparseFeatures]]:
        """
        Performs AlltoAll operation on sparse features.

        Args:
            sparse_features (SparseFeatures): sparse features to redistribute.

        Returns:
            Awaitable[Awaitable[SparseFeatures]]: awaitable of awaitable of SparseFeatures.
        """

        return self._dist(sparse_features)


class VariableBatchTwPooledEmbeddingDist(
    BaseEmbeddingDist[VariableBatchShardingContext, torch.Tensor, torch.Tensor]
):
    def __init__(
        self,
        pg: dist.ProcessGroup,
        dim_sum_per_rank: List[int],
        device: Optional[torch.device] = None,
        callbacks: Optional[List[Callable[[torch.Tensor], torch.Tensor]]] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__()
        self._dist = PooledEmbeddingsAllToAll(
            pg,
            dim_sum_per_rank,
            device,
            callbacks,
            codecs=qcomm_codecs_registry.get(
                CommOp.POOLED_EMBEDDINGS_ALL_TO_ALL.name, None
            )
            if qcomm_codecs_registry
            else None,
        )

    def forward(
        self,
        local_embs: torch.Tensor,
        sharding_ctx: Optional[VariableBatchShardingContext] = None,
    ) -> Awaitable[torch.Tensor]:
        assert sharding_ctx is not None
        # do not remove the keyword for quantized communication hook injection.
        return self._dist(
            local_embs, batch_size_per_rank=sharding_ctx.batch_size_per_rank
        )


class VariableBatchTwPooledEmbeddingSharding(
    BaseTwEmbeddingSharding[
        VariableBatchShardingContext, SparseFeatures, torch.Tensor, torch.Tensor
    ]
):
    """
    Shards pooled embeddings table-wise, i.e.. a given embedding table is entirely placed
    on a selected rank.

    Supports variable batch size.
    """

    def create_input_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseSparseFeaturesDist[SparseFeatures]:
        return VariableBatchTwSparseFeaturesDist(
            # pyre-fixme[6]: For 1st param expected `ProcessGroup` but got
            #  `Optional[ProcessGroup]`.
            self._pg,
            self.id_list_features_per_rank(),
            self.id_score_list_features_per_rank(),
            device if device is not None else self._device,
        )

    def create_lookup(
        self,
        device: Optional[torch.device] = None,
        fused_params: Optional[Dict[str, Any]] = None,
        feature_processor: Optional[BaseGroupedFeatureProcessor] = None,
    ) -> BaseEmbeddingLookup:
        return GroupedPooledEmbeddingsLookup(
            grouped_configs=self._grouped_embedding_configs,
            grouped_score_configs=self._score_grouped_embedding_configs,
            pg=self._pg,
            device=device if device is not None else self._device,
            feature_processor=feature_processor,
        )

    def create_output_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseEmbeddingDist[VariableBatchShardingContext, torch.Tensor, torch.Tensor]:
        return VariableBatchTwPooledEmbeddingDist(
            # pyre-fixme[6]: For 1st param expected `ProcessGroup` but got
            #  `Optional[ProcessGroup]`.
            self._pg,
            self._dim_sum_per_rank(),
            device if device is not None else self._device,
            qcomm_codecs_registry=self.qcomm_codecs_registry,
        )
