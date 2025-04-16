#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch

from torchrec.distributed.dist_data import SeqEmbeddingsAllToOne
from torchrec.distributed.embedding_sharding import (
    BaseEmbeddingDist,
    EmbeddingShardingContext,
)
from torchrec.distributed.embedding_types import KJTList

from torchrec.modules.utils import (
    _fx_trec_get_feature_length,
    _get_batching_hinted_output,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.streamable import Multistreamable

torch.fx.wrap("_get_batching_hinted_output")
torch.fx.wrap("_fx_trec_get_feature_length")


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

    # Torch Dynamo does not support default_factory=list:
    # https://github.com/pytorch/pytorch/issues/120108
    # TODO(ivankobzarev): Make this a dataclass once supported

    def __init__(
        self,
        # Fields of EmbeddingShardingContext
        batch_size_per_rank: Optional[List[int]] = None,
        batch_size_per_rank_per_feature: Optional[List[List[int]]] = None,
        batch_size_per_feature_pre_a2a: Optional[List[int]] = None,
        variable_batch_per_feature: bool = False,
        # Fields of SequenceShardingContext
        features_before_input_dist: Optional[KeyedJaggedTensor] = None,
        input_splits: Optional[List[int]] = None,
        output_splits: Optional[List[int]] = None,
        sparse_features_recat: Optional[torch.Tensor] = None,
        unbucketize_permute_tensor: Optional[torch.Tensor] = None,
        lengths_after_input_dist: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__(
            batch_size_per_rank,
            batch_size_per_rank_per_feature,
            batch_size_per_feature_pre_a2a,
            variable_batch_per_feature,
        )
        self.features_before_input_dist: Optional[KeyedJaggedTensor] = (
            features_before_input_dist
        )
        self.input_splits: List[int] = input_splits if input_splits is not None else []
        self.output_splits: List[int] = (
            output_splits if output_splits is not None else []
        )
        self.sparse_features_recat: Optional[torch.Tensor] = sparse_features_recat
        self.unbucketize_permute_tensor: Optional[torch.Tensor] = (
            unbucketize_permute_tensor
        )
        self.lengths_after_input_dist: Optional[torch.Tensor] = lengths_after_input_dist

    def record_stream(self, stream: torch.Stream) -> None:
        if self.features_before_input_dist is not None:
            self.features_before_input_dist.record_stream(stream)
        if self.sparse_features_recat is not None:
            self.sparse_features_recat.record_stream(stream)
        if self.unbucketize_permute_tensor is not None:
            self.unbucketize_permute_tensor.record_stream(stream)
        if self.lengths_after_input_dist is not None:
            self.lengths_after_input_dist.record_stream(stream)


@dataclass
class InferSequenceShardingContext(Multistreamable):
    """
    Stores inference context and reuses it in sequence embedding output_dist or result return.

    Attributes:
        features KJTList: stores the shards of KJT after input dist.
        features_before_input_dist KJT: stores the original input KJT (before input dist).
        unbucketize_permute_tensor Optional[torch.Tensor]: stores unbucketize tensor, only for RowWise sharding.
    """

    features: KJTList
    features_before_input_dist: Optional[KeyedJaggedTensor] = None
    unbucketize_permute_tensor: Optional[torch.Tensor] = None
    bucket_mapping_tensor: Optional[torch.Tensor] = None
    bucketized_length: Optional[torch.Tensor] = None
    embedding_names_per_rank: Optional[List[List[str]]] = None

    def record_stream(self, stream: torch.Stream) -> None:
        for feature in self.features:
            feature.record_stream(stream)
        if self.features_before_input_dist is not None:
            self.features_before_input_dist.record_stream(stream)
        if self.unbucketize_permute_tensor is not None:
            self.unbucketize_permute_tensor.record_stream(stream)
        if self.bucket_mapping_tensor is not None:
            self.bucket_mapping_tensor.record_stream(stream)
        if self.bucketized_length is not None:
            self.bucketized_length.record_stream(stream)


class InferSequenceEmbeddingDist(
    BaseEmbeddingDist[
        InferSequenceShardingContext, List[torch.Tensor], List[torch.Tensor]
    ]
):
    def __init__(
        self,
        device: torch.device,
        world_size: int,
        device_type_from_sharding_infos: Optional[Union[str, Tuple[str, ...]]] = None,
    ) -> None:
        super().__init__()
        self._device_type_from_sharding_infos: Optional[Union[str, Tuple[str, ...]]] = (
            device_type_from_sharding_infos
        )
        non_cpu_ranks = 0
        if self._device_type_from_sharding_infos and isinstance(
            self._device_type_from_sharding_infos, tuple
        ):
            for device_type in self._device_type_from_sharding_infos:
                if device_type != "cpu":
                    non_cpu_ranks += 1
        elif self._device_type_from_sharding_infos == "cpu":
            non_cpu_ranks = 0
        else:
            non_cpu_ranks = world_size

        self._device_dist: SeqEmbeddingsAllToOne = SeqEmbeddingsAllToOne(
            device, non_cpu_ranks
        )

    def forward(
        self,
        local_embs: List[torch.Tensor],
        sharding_ctx: Optional[InferSequenceShardingContext] = None,
    ) -> List[torch.Tensor]:
        assert (
            self._device_type_from_sharding_infos is not None
        ), "_device_type_from_sharding_infos should always be set for InferRwSequenceEmbeddingDist"
        if isinstance(self._device_type_from_sharding_infos, tuple):
            assert sharding_ctx is not None
            assert sharding_ctx.embedding_names_per_rank is not None
            assert len(self._device_type_from_sharding_infos) == len(
                local_embs
            ), "For heterogeneous sharding, the number of local_embs should be equal to the number of device types"
            non_cpu_local_embs = []
            # Here looping through local_embs is also compatible with tracing
            # given the number of looks up / shards withing ShardedQuantEmbeddingCollection
            # are fixed and local_embs is the output of those looks ups. However, still
            # using _device_type_from_sharding_infos to iterate on local_embs list as
            # that's a better practice.
            for i, device_type in enumerate(self._device_type_from_sharding_infos):
                if device_type != "cpu":
                    non_cpu_local_embs.append(
                        _get_batching_hinted_output(
                            _fx_trec_get_feature_length(
                                sharding_ctx.features[i],
                                # pyre-fixme [16]
                                sharding_ctx.embedding_names_per_rank[i],
                            ),
                            local_embs[i],
                        )
                    )
            non_cpu_local_embs_dist = self._device_dist(non_cpu_local_embs)
            index = 0
            result = []
            for i, device_type in enumerate(self._device_type_from_sharding_infos):
                if device_type == "cpu":
                    result.append(local_embs[i])
                else:
                    result.append(non_cpu_local_embs_dist[index])
                    index += 1
            return result
        elif self._device_type_from_sharding_infos == "cpu":
            # for cpu sharder, output dist should be a no-op
            return local_embs
        else:
            return self._device_dist(local_embs)
