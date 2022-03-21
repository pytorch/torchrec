#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Dict, Any, cast, List

import torch
import torch.distributed as dist
from torchrec.distributed.embedding_lookup import GroupedEmbeddingsLookup
from torchrec.distributed.embedding_sharding import (
    BaseEmbeddingDist,
    BaseSparseFeaturesDist,
    BaseEmbeddingLookup,
    bucketize_kjt_before_all2all,
)
from torchrec.distributed.embedding_types import (
    SparseFeatures,
    BaseGroupedFeatureProcessor,
)
from torchrec.distributed.sharding.rw_sequence_sharding import RwSequenceEmbeddingDist
from torchrec.distributed.sharding.twrw_sharding import (
    TwRwSparseFeaturesDist,
    BaseTwRwEmbeddingSharding,
)
from torchrec.distributed.types import Awaitable


class TwRwSequenceSparseFeaturesDist(TwRwSparseFeaturesDist):
    """
    Bucketizes sparse features in TWRW fashion and then redistributes with an AlltoAll
    collective operation.

    Args:
        pg (dist.ProcessGroup): ProcessGroup for AlltoAll communication.
        intra_pg (dist.ProcessGroup): ProcessGroup within single host group for AlltoAll
        communication.
        id_list_features_per_rank (List[int]): number of id list features to send to
        each rank.
        id_score_list_features_per_rank (List[int]): number of id score list features to
        send to each rank
        id_list_feature_hash_sizes (List[int]): hash sizes of id list features.
        id_score_list_feature_hash_sizes (List[int]): hash sizes of id score list features.
        device (Optional[torch.device]): device on which buffers will be allocated.
        has_feature_processor (bool): existence of feature processor (ie. position
        weighted features).

    Example::

        3 features
        2 hosts with 2 devices each

        Bucketize each feature into 2 buckets
        Staggered shuffle with feature splits [2, 1]
        AlltoAll operation

        NOTE: result of staggered shuffle and AlltoAll operation look the same after
        reordering in AlltoAll

        Result:
            host 0 device 0:
                feature 0 bucket 0
                feature 1 bucket 0

            host 0 device 1:
                feature 0 bucket 1
                feature 1 bucket 1

            host 1 device 0:
                feature 2 bucket 0

            host 1 device 1:
                feature 2 bucket 1
    """

    def __init__(
        self,
        # pyre-ignore [11]
        pg: dist.ProcessGroup,
        intra_pg: dist.ProcessGroup,
        id_list_features_per_rank: List[int],
        id_list_feature_hash_sizes: List[int],
        device: Optional[torch.device] = None,
        has_feature_processor: bool = False,
        is_sequence: bool = False,
    ) -> None:
        super().__init__(
            pg=pg,
            intra_pg=intra_pg,
            id_list_features_per_rank=id_list_features_per_rank,
            id_score_list_features_per_rank=[0] * pg.size(),
            id_list_feature_hash_sizes=id_list_feature_hash_sizes,
            id_score_list_feature_hash_sizes=[],
            device=device,
            has_feature_processor=has_feature_processor,
            cross_node_stagger=False,
        )
        self.unbucketize_permute_tensor: Optional[torch.Tensor] = None
        self._device = device

    def forward(
        self,
        sparse_features: SparseFeatures,
    ) -> Awaitable[Awaitable[SparseFeatures]]:
        """
        Bucketizes sparse feature values into local world size number of buckets,
        performs staggered shuffle on the sparse features, and then performs AlltoAll
        operation.

        Call Args:
            sparse_features (SparseFeatures): sparse features to bucketize and
                redistribute.

        Returns:
            Awaitable[SparseFeatures]: awaitable of SparseFeatures.
        """
        if sparse_features.id_list_features is not None:
            (
                id_list_features,
                unbucketize_permute_tensor,
            ) = bucketize_kjt_before_all2all(
                sparse_features.id_list_features,
                num_buckets=self._local_size,
                block_sizes=self._id_list_feature_block_sizes_tensor,
                output_permute=True,
                bucketize_pos=self._has_feature_processor,
            )

            indices_before_2nd_permute = []
            batch_size = id_list_features.stride()
            lengths = id_list_features.lengths()
            start = 0
            for i in lengths:
                indices_before_2nd_permute.append(list(range(start, start + i)))
                start += i
            indices_before_2nd_permute = [
                indices_before_2nd_permute[j : j + batch_size]
                for j in range(0, len(indices_before_2nd_permute), batch_size)
            ]
            second_permute = []
            for i in self._id_list_sf_staggered_shuffle:
                second_permute.append(indices_before_2nd_permute[i])
            second_permute = [k for i in second_permute for j in i for k in j]
            self.unbucketize_permute_tensor = torch.tensor(
                [
                    second_permute.index(i)
                    # pyre-ignore [16]
                    for i in unbucketize_permute_tensor.tolist()
                ],
                device=self._device,
                dtype=torch.int32,
            )

            id_list_features = id_list_features.permute(
                self._id_list_sf_staggered_shuffle,
                self._id_list_sf_staggered_shuffle_tensor,
            )
        else:
            id_list_features = None

        bucketized_sparse_features = SparseFeatures(
            id_list_features=id_list_features,
            id_score_list_features=None,
        )
        return self._dist(bucketized_sparse_features)


class TwRwSequenceEmbeddingSharding(
    BaseTwRwEmbeddingSharding[SparseFeatures, torch.Tensor]
):
    """
    Shards embedding bags table-wise then row-wise.
    """

    def create_input_dist(
        self, device: Optional[torch.device] = None
    ) -> BaseSparseFeaturesDist[SparseFeatures]:
        id_list_features_per_rank = self._features_per_rank(
            self._grouped_embedding_configs_per_rank
        )
        id_list_feature_hash_sizes = self._get_id_list_features_hash_sizes()
        return TwRwSequenceSparseFeaturesDist(
            pg=self._pg,
            intra_pg=cast(dist.ProcessGroup, self._intra_pg),
            id_list_features_per_rank=id_list_features_per_rank,
            id_list_feature_hash_sizes=id_list_feature_hash_sizes,
            device=device if device is not None else self._device,
            has_feature_processor=self._has_feature_processor,
            is_sequence=True,
        )

    def create_lookup(
        self,
        device: Optional[torch.device] = None,
        fused_params: Optional[Dict[str, Any]] = None,
        feature_processor: Optional[BaseGroupedFeatureProcessor] = None,
    ) -> BaseEmbeddingLookup:
        return GroupedEmbeddingsLookup(
            grouped_configs=self._grouped_embedding_configs_per_rank[self._rank],
            fused_params=fused_params,
            pg=self._pg,
            device=device if device is not None else self._device,
        )

    def create_output_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseEmbeddingDist[torch.Tensor]:
        id_list_features_per_rank = self._features_per_rank(
            self._grouped_embedding_configs_per_rank
        )
        return RwSequenceEmbeddingDist(
            self._pg,
            id_list_features_per_rank,
            device if device is not None else self._device,
        )
