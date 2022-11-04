#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional

import torch
from torchrec.distributed.embedding_lookup import GroupedEmbeddingsLookup
from torchrec.distributed.embedding_sharding import (
    BaseEmbeddingDist,
    BaseEmbeddingLookup,
    BaseSparseFeaturesDist,
)
from torchrec.distributed.embedding_types import (
    BaseGroupedFeatureProcessor,
    SparseFeatures,
)
from torchrec.distributed.sharding.cw_sharding import BaseCwEmbeddingSharding
from torchrec.distributed.sharding.sequence_sharding import SequenceShardingContext
from torchrec.distributed.sharding.tw_sequence_sharding import TwSequenceEmbeddingDist
from torchrec.distributed.sharding.tw_sharding import TwSparseFeaturesDist


class CwSequenceEmbeddingSharding(
    BaseCwEmbeddingSharding[
        SequenceShardingContext, SparseFeatures, torch.Tensor, torch.Tensor
    ]
):
    """
    Shards sequence (unpooled) embeddings column-wise, i.e.. a given embedding is
    partitioned along its columns and placed on specified ranks.
    """

    def create_input_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseSparseFeaturesDist[SparseFeatures]:
        assert self._pg is not None
        return TwSparseFeaturesDist(
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
