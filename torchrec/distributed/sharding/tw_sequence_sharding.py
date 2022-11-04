#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from torchrec.distributed.dist_data import (
    SeqEmbeddingsAllToOne,
    SequenceEmbeddingsAllToAll,
)
from torchrec.distributed.embedding_lookup import (
    GroupedEmbeddingsLookup,
    InferGroupedEmbeddingsLookup,
)
from torchrec.distributed.embedding_sharding import (
    BaseEmbeddingDist,
    BaseEmbeddingLookup,
    BaseSparseFeaturesDist,
)
from torchrec.distributed.embedding_types import (
    BaseGroupedFeatureProcessor,
    SparseFeatures,
    SparseFeaturesList,
)
from torchrec.distributed.sharding.sequence_sharding import (
    InferSequenceShardingContext,
    SequenceShardingContext,
)
from torchrec.distributed.sharding.tw_sharding import (
    BaseTwEmbeddingSharding,
    InferTwSparseFeaturesDist,
    TwSparseFeaturesDist,
)
from torchrec.distributed.types import Awaitable, CommOp, QuantizedCommCodecs


class TwSequenceEmbeddingDist(
    BaseEmbeddingDist[SequenceShardingContext, torch.Tensor, torch.Tensor]
):
    """
    Redistributes sequence embedding tensor in TW fashion with an AlltoAll operation.

    Args:
        pg (dist.ProcessGroup): ProcessGroup for AlltoAll communication.
        features_per_rank (List[int]): number of features (sum of dimensions) of the
            embedding for each rank.
        device (Optional[torch.device]): device on which buffers will be allocated.
    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
        features_per_rank: List[int],
        device: Optional[torch.device] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__()
        self._dist = SequenceEmbeddingsAllToAll(
            pg,
            features_per_rank,
            device,
            codecs=qcomm_codecs_registry.get(
                CommOp.SEQUENCE_EMBEDDINGS_ALL_TO_ALL.name, None
            )
            if qcomm_codecs_registry
            else None,
        )

    def forward(
        self,
        local_embs: torch.Tensor,
        sharding_ctx: Optional[SequenceShardingContext] = None,
    ) -> Awaitable[torch.Tensor]:
        """
        Performs AlltoAll operation on sequence embeddings tensor.

        Args:
            local_embs (torch.Tensor): tensor of values to distribute.
            sharding_ctx (SequenceShardingContext): shared context from KJTAllToAll
                operation.

        Returns:
            Awaitable[torch.Tensor]: awaitable of sequence embeddings.
        """

        assert sharding_ctx is not None
        return self._dist(
            local_embs,
            lengths=sharding_ctx.lengths_after_input_dist,
            input_splits=sharding_ctx.input_splits,
            output_splits=sharding_ctx.output_splits,
            batch_size_per_rank=sharding_ctx.batch_size_per_rank,
            sparse_features_recat=sharding_ctx.sparse_features_recat,
            unbucketize_permute_tensor=None,
        )


class TwSequenceEmbeddingSharding(
    BaseTwEmbeddingSharding[
        SequenceShardingContext, SparseFeatures, torch.Tensor, torch.Tensor
    ]
):
    """
    Shards sequence (unpooled) embedding table-wise, i.e.. a given embedding table is
    placed entirely on a selected rank.
    """

    def create_input_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseSparseFeaturesDist[SparseFeatures]:
        return TwSparseFeaturesDist(
            # pyre-fixme[6]: For 1st param expected `ProcessGroup` but got
            #  `Optional[ProcessGroup]`.
            self._pg,
            self.id_list_features_per_rank(),
            self.id_score_list_features_per_rank(),
            device if device is not None else self._device,
            variable_batch_size=self._variable_batch_size,
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
            pg=self._pg,
            device=device if device is not None else self._device,
        )

    def create_output_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseEmbeddingDist[SequenceShardingContext, torch.Tensor, torch.Tensor]:
        assert self._pg is not None
        return TwSequenceEmbeddingDist(
            self._pg,
            self.id_list_features_per_rank(),
            device if device is not None else self._device,
            qcomm_codecs_registry=self.qcomm_codecs_registry,
        )


class InferTwSequenceEmbeddingDist(
    BaseEmbeddingDist[
        InferSequenceShardingContext, List[torch.Tensor], List[torch.Tensor]
    ]
):
    """
    Redistributes sequence embedding tensor in hierarchical fashion with an AlltoOne
    operation.

    Args:
        device (torch.device): device on which the tensors will be communicated to.
        world_size (int): number of devices in the topology.
    """

    def __init__(
        self,
        device: torch.device,
        world_size: int,
    ) -> None:
        super().__init__()
        self._dist: SeqEmbeddingsAllToOne = SeqEmbeddingsAllToOne(device, world_size)

    def forward(
        self,
        local_embs: List[torch.Tensor],
        sharding_ctx: Optional[InferSequenceShardingContext] = None,
    ) -> Awaitable[List[torch.Tensor]]:
        """
        Performs AlltoOne operation on sequence embeddings tensor.

        Args:
            local_embs (List[orch.Tensor]): tensor of values to distribute.
            sharding_ctx (InferSequenceShardingContext): shared context from KJTAllToOne
                operation.


        Returns:
            Awaitable[torch.Tensor]: awaitable of sequence embeddings.
        """
        return self._dist.forward(local_embs)


class InferTwSequenceEmbeddingSharding(
    BaseTwEmbeddingSharding[
        InferSequenceShardingContext,
        SparseFeaturesList,
        List[torch.Tensor],
        List[torch.Tensor],
    ]
):
    """
    Shards sequence (unpooled) embedding table-wise, i.e.. a given embedding table is
    placed entirely on a selected rank, for inference.
    """

    def create_input_dist(
        self, device: Optional[torch.device] = None
    ) -> BaseSparseFeaturesDist[SparseFeaturesList]:
        return InferTwSparseFeaturesDist(
            self.id_list_features_per_rank(),
            self.id_score_list_features_per_rank(),
            self._world_size,
        )

    def create_lookup(
        self,
        device: Optional[torch.device] = None,
        fused_params: Optional[Dict[str, Any]] = None,
        feature_processor: Optional[BaseGroupedFeatureProcessor] = None,
    ) -> BaseEmbeddingLookup[SparseFeaturesList, List[torch.Tensor]]:
        return InferGroupedEmbeddingsLookup(
            grouped_configs_per_rank=self._grouped_embedding_configs_per_rank,
            world_size=self._world_size,
            fused_params=fused_params,
        )

    def create_output_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseEmbeddingDist[
        InferSequenceShardingContext, List[torch.Tensor], List[torch.Tensor]
    ]:
        device = device if device is not None else self._device
        return InferTwSequenceEmbeddingDist(
            # pyre-fixme [6]
            device,
            self._world_size,
        )
