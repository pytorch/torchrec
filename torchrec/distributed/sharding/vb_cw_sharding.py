#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional

import torch
from fbgemm_gpu.permute_pooled_embedding_modules_split import (
    PermutePooledEmbeddingsSplit,
)
from torchrec.distributed.embedding_lookup import GroupedPooledEmbeddingsLookup
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
from torchrec.distributed.sharding.vb_sharding import VariableBatchShardingContext
from torchrec.distributed.sharding.vb_tw_sharding import (
    VariableBatchTwPooledEmbeddingDist,
    VariableBatchTwSparseFeaturesDist,
)


class VariableBatchCwPooledEmbeddingSharding(
    BaseCwEmbeddingSharding[
        VariableBatchShardingContext, SparseFeatures, torch.Tensor, torch.Tensor
    ]
):
    """
    Shards embedding bags column-wise, i.e.. a given embedding table is partitioned
    along its columns and placed on specified ranks.

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
        callbacks = None
        if self._permute_embeddings and self._embedding_order != list(
            range(len(self._embedding_order))
        ):
            assert len(self._embedding_order) == len(self._embedding_dims)
            embedding_permute_op = PermutePooledEmbeddingsSplit(
                self._embedding_dims,
                self._embedding_order,
            ).to(device=device)
            callbacks = [embedding_permute_op]
        return VariableBatchTwPooledEmbeddingDist(
            # pyre-fixme[6]: For 1st param expected `ProcessGroup` but got
            #  `Optional[ProcessGroup]`.
            self._pg,
            self._dim_sum_per_rank(),
            device if device is not None else self._device,
            # pyre-ignore [6]
            callbacks,
            qcomm_codecs_registry=self.qcomm_codecs_registry,
        )
