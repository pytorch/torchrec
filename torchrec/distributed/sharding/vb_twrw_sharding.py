#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

import itertools
import math
from typing import Any, cast, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torchrec.distributed.dist_data import (
    PooledEmbeddingsAllToAll,
    PooledEmbeddingsReduceScatter,
)
from torchrec.distributed.embedding_lookup import GroupedPooledEmbeddingsLookup
from torchrec.distributed.embedding_sharding import (
    BaseEmbeddingDist,
    BaseEmbeddingLookup,
    BaseSparseFeaturesDist,
    bucketize_kjt_before_all2all,
    SparseFeaturesAllToAll,
)
from torchrec.distributed.embedding_types import (
    BaseGroupedFeatureProcessor,
    SparseFeatures,
)
from torchrec.distributed.sharding.twrw_sharding import BaseTwRwEmbeddingSharding
from torchrec.distributed.sharding.vb_sharding import VariableBatchShardingContext
from torchrec.distributed.types import Awaitable, CommOp, QuantizedCommCodecs

torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")


class VariableBatchTwRwSparseFeaturesDist(BaseSparseFeaturesDist[SparseFeatures]):
    """
    Bucketizes sparse features in TWRW fashion and then redistributes with an AlltoAll
    collective operation.

    Supports variable batch size.

    Args:
        pg (dist.ProcessGroup): ProcessGroup for AlltoAll communication.
        intra_pg (dist.ProcessGroup): ProcessGroup within single host group for AlltoAll
            communication.
        id_list_features_per_rank (List[int]): number of id list features to send to
            each rank.
        id_score_list_features_per_rank (List[int]): number of id score list features to
            send to each rank
        id_list_feature_hash_sizes (List[int]): hash sizes of id list features.
        id_score_list_feature_hash_sizes (List[int]): hash sizes of id score list
            features.
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
        pg: dist.ProcessGroup,
        intra_pg: dist.ProcessGroup,
        id_list_features_per_rank: List[int],
        id_score_list_features_per_rank: List[int],
        id_list_feature_hash_sizes: List[int],
        id_score_list_feature_hash_sizes: List[int],
        device: Optional[torch.device] = None,
        has_feature_processor: bool = False,
    ) -> None:
        super().__init__()
        assert (
            pg.size() % intra_pg.size() == 0
        ), "currently group granularity must be node"

        self._world_size: int = pg.size()
        self._local_size: int = intra_pg.size()
        self._num_cross_nodes: int = self._world_size // self._local_size
        id_list_feature_block_sizes = [
            math.ceil(hash_size / self._local_size)
            for hash_size in id_list_feature_hash_sizes
        ]
        id_score_list_feature_block_sizes = [
            math.ceil(hash_size / self._local_size)
            for hash_size in id_score_list_feature_hash_sizes
        ]

        self._id_list_sf_staggered_shuffle: List[int] = self._staggered_shuffle(
            id_list_features_per_rank
        )
        self._id_score_list_sf_staggered_shuffle: List[int] = self._staggered_shuffle(
            id_score_list_features_per_rank
        )
        self.register_buffer(
            "_id_list_feature_block_sizes_tensor",
            torch.tensor(
                id_list_feature_block_sizes,
                device=device,
                dtype=torch.int32,
            ),
        )
        self.register_buffer(
            "_id_score_list_feature_block_sizes_tensor",
            torch.tensor(
                id_score_list_feature_block_sizes,
                device=device,
                dtype=torch.int32,
            ),
        )
        self.register_buffer(
            "_id_list_sf_staggered_shuffle_tensor",
            torch.tensor(
                self._id_list_sf_staggered_shuffle,
                device=device,
                dtype=torch.int32,
            ),
        )
        self.register_buffer(
            "_id_score_list_sf_staggered_shuffle_tensor",
            torch.tensor(
                self._id_score_list_sf_staggered_shuffle,
                device=device,
                dtype=torch.int32,
            ),
        )
        self._dist = SparseFeaturesAllToAll(
            pg=pg,
            id_list_features_per_rank=id_list_features_per_rank,
            id_score_list_features_per_rank=id_score_list_features_per_rank,
            device=device,
            stagger=self._num_cross_nodes,
            variable_batch_size=True,
        )
        self._has_feature_processor = has_feature_processor

    def forward(
        self,
        sparse_features: SparseFeatures,
    ) -> Awaitable[Awaitable[SparseFeatures]]:
        """
        Bucketizes sparse feature values into local world size number of buckets,
        performs staggered shuffle on the sparse features, and then performs AlltoAll
        operation.

        Args:
            sparse_features (SparseFeatures): sparse features to bucketize and
                redistribute.

        Returns:
            Awaitable[SparseFeatures]: awaitable of SparseFeatures.
        """

        bucketized_sparse_features = SparseFeatures(
            id_list_features=bucketize_kjt_before_all2all(
                sparse_features.id_list_features,
                num_buckets=self._local_size,
                block_sizes=self._id_list_feature_block_sizes_tensor,
                output_permute=False,
                bucketize_pos=self._has_feature_processor,
            )[0].permute(
                self._id_list_sf_staggered_shuffle,
                self._id_list_sf_staggered_shuffle_tensor,
            )
            if sparse_features.id_list_features is not None
            else None,
            id_score_list_features=bucketize_kjt_before_all2all(
                sparse_features.id_score_list_features,
                num_buckets=self._local_size,
                block_sizes=self._id_score_list_feature_block_sizes_tensor,
                output_permute=False,
                bucketize_pos=False,
            )[0].permute(
                self._id_score_list_sf_staggered_shuffle,
                self._id_score_list_sf_staggered_shuffle_tensor,
            )
            if sparse_features.id_score_list_features is not None
            else None,
        )
        return self._dist(bucketized_sparse_features)

    def _staggered_shuffle(self, features_per_rank: List[int]) -> List[int]:
        """
        Reorders sparse data such that data is in contiguous blocks and correctly
        ordered for global TWRW layout.
        """

        nodes = self._world_size // self._local_size
        features_per_node = [
            features_per_rank[node * self._local_size] for node in range(nodes)
        ]
        node_offsets = [0] + list(itertools.accumulate(features_per_node))
        num_features = node_offsets[-1]

        return [
            bucket * num_features + feature
            for node in range(nodes)
            for bucket in range(self._local_size)
            for feature in range(node_offsets[node], node_offsets[node + 1])
        ]


class VariableBatchTwRwPooledEmbeddingDist(
    BaseEmbeddingDist[VariableBatchShardingContext, torch.Tensor, torch.Tensor]
):
    def __init__(
        self,
        rank: int,
        cross_pg: dist.ProcessGroup,
        intra_pg: dist.ProcessGroup,
        dim_sum_per_node: List[int],
        device: Optional[torch.device] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__()
        self._rank = rank
        self._intra_pg: dist.ProcessGroup = intra_pg
        self._cross_pg: dist.ProcessGroup = cross_pg
        self._device: Optional[torch.device] = device
        self._intra_dist = PooledEmbeddingsReduceScatter(
            intra_pg,
            codecs=qcomm_codecs_registry.get(
                CommOp.POOLED_EMBEDDINGS_REDUCE_SCATTER.name, None
            )
            if qcomm_codecs_registry
            else None,
        )
        self._cross_dist = PooledEmbeddingsAllToAll(
            cross_pg,
            dim_sum_per_node,
            device,
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
        # preprocess batch_size_per_rank
        (
            batch_size_per_rank_by_cross_group,
            batch_size_sum_by_cross_group,
        ) = self._preprocess_batch_size_per_rank(
            self._intra_pg.size(),
            self._cross_pg.size(),
            sharding_ctx.batch_size_per_rank,
        )

        # Perform ReduceScatterV within one host
        lengths = batch_size_sum_by_cross_group
        rs_result = self._intra_dist(
            local_embs.view(sum(lengths), -1), input_splits=lengths
        ).wait()

        local_rank = self._rank % self._intra_pg.size()

        return self._cross_dist(
            rs_result,
            batch_size_per_rank=batch_size_per_rank_by_cross_group[local_rank],
        )

    def _preprocess_batch_size_per_rank(
        self, local_size: int, nodes: int, batch_size_per_rank: List[int]
    ) -> Tuple[List[List[int]], List[int]]:
        """
        Reorders `batch_size_per_rank` so it's aligned with reordered features after
        AlltoAll.
        """
        batch_size_per_rank_by_cross_group: List[List[int]] = []
        batch_size_sum_by_cross_group: List[int] = []
        for local_rank in range(local_size):
            batch_size_per_rank_: List[int] = []
            batch_size_sum = 0
            for node in range(nodes):
                batch_size_per_rank_.append(
                    batch_size_per_rank[local_rank + node * local_size]
                )
                batch_size_sum += batch_size_per_rank[local_rank + node * local_size]
            batch_size_per_rank_by_cross_group.append(batch_size_per_rank_)
            batch_size_sum_by_cross_group.append(batch_size_sum)

        return batch_size_per_rank_by_cross_group, batch_size_sum_by_cross_group


class VariableBatchTwRwPooledEmbeddingSharding(
    BaseTwRwEmbeddingSharding[
        VariableBatchShardingContext, SparseFeatures, torch.Tensor, torch.Tensor
    ]
):
    """
    Shards embedding bags table-wise then row-wise.

    Supports variable batch size.
    """

    def create_input_dist(
        self, device: Optional[torch.device] = None
    ) -> BaseSparseFeaturesDist[SparseFeatures]:
        id_list_features_per_rank = self._features_per_rank(
            self._grouped_embedding_configs_per_rank
        )
        id_score_list_features_per_rank = self._features_per_rank(
            self._score_grouped_embedding_configs_per_rank
        )
        id_list_feature_hash_sizes = self._get_id_list_features_hash_sizes()
        id_score_list_feature_hash_sizes = self._get_id_score_list_features_hash_sizes()
        return VariableBatchTwRwSparseFeaturesDist(
            # pyre-fixme[6]: For 1st param expected `ProcessGroup` but got
            #  `Optional[ProcessGroup]`.
            pg=self._pg,
            intra_pg=cast(dist.ProcessGroup, self._intra_pg),
            id_list_features_per_rank=id_list_features_per_rank,
            id_score_list_features_per_rank=id_score_list_features_per_rank,
            id_list_feature_hash_sizes=id_list_feature_hash_sizes,
            id_score_list_feature_hash_sizes=id_score_list_feature_hash_sizes,
            device=device if device is not None else self._device,
            has_feature_processor=self._has_feature_processor,
        )

    def create_lookup(
        self,
        device: Optional[torch.device] = None,
        fused_params: Optional[Dict[str, Any]] = None,
        feature_processor: Optional[BaseGroupedFeatureProcessor] = None,
    ) -> BaseEmbeddingLookup:
        return GroupedPooledEmbeddingsLookup(
            grouped_configs=self._grouped_embedding_configs_per_rank[self._rank],
            grouped_score_configs=self._score_grouped_embedding_configs_per_rank[
                self._rank
            ],
            pg=self._pg,
            device=device if device is not None else self._device,
            feature_processor=feature_processor,
        )

    def create_output_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseEmbeddingDist[VariableBatchShardingContext, torch.Tensor, torch.Tensor]:
        return VariableBatchTwRwPooledEmbeddingDist(
            rank=self._rank,
            cross_pg=cast(dist.ProcessGroup, self._cross_pg),
            intra_pg=cast(dist.ProcessGroup, self._intra_pg),
            dim_sum_per_node=self._dim_sum_per_node(),
            device=device if device is not None else self._device,
            qcomm_codecs_registry=self.qcomm_codecs_registry,
        )
