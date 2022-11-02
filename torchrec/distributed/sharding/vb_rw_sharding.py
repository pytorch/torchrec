#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from torchrec.distributed.dist_data import PooledEmbeddingsReduceScatter
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
from torchrec.distributed.sharding.rw_sharding import BaseRwEmbeddingSharding
from torchrec.distributed.sharding.vb_sharding import VariableBatchShardingContext
from torchrec.distributed.types import Awaitable, CommOp, QuantizedCommCodecs


torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")


class VariableBatchRwSparseFeaturesDist(BaseSparseFeaturesDist[SparseFeatures]):
    """
    Bucketizes sparse features in RW fashion and then redistributes with an AlltoAll
    collective operation.

    Supports variable batch size.

    Args:
        pg (dist.ProcessGroup): ProcessGroup for AlltoAll communication.
        intra_pg (dist.ProcessGroup): ProcessGroup within single host group for AlltoAll
            communication.
        num_id_list_features (int): total number of id list features.
        num_id_score_list_features (int): total number of id score list features
        id_list_feature_hash_sizes (List[int]): hash sizes of id list features.
        id_score_list_feature_hash_sizes (List[int]): hash sizes of id score list features.
        device (Optional[torch.device]): device on which buffers will be allocated.
        has_feature_processor (bool): existence of feature processor (ie. position
            weighted features).
    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
        num_id_list_features: int,
        num_id_score_list_features: int,
        id_list_feature_hash_sizes: List[int],
        id_score_list_feature_hash_sizes: List[int],
        device: Optional[torch.device] = None,
        has_feature_processor: bool = False,
    ) -> None:
        super().__init__()
        self._world_size: int = pg.size()
        self._num_id_list_features = num_id_list_features
        self._num_id_score_list_features = num_id_score_list_features
        id_list_feature_block_sizes = [
            (hash_size + self._world_size - 1) // self._world_size
            for hash_size in id_list_feature_hash_sizes
        ]
        id_score_list_feature_block_sizes = [
            (hash_size + self._world_size - 1) // self._world_size
            for hash_size in id_score_list_feature_hash_sizes
        ]
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
        self._dist = SparseFeaturesAllToAll(
            pg=pg,
            id_list_features_per_rank=self._world_size * [self._num_id_list_features],
            id_score_list_features_per_rank=self._world_size
            * [self._num_id_score_list_features],
            device=device,
            variable_batch_size=True,
        )
        self._has_feature_processor = has_feature_processor
        self.unbucketize_permute_tensor: Optional[torch.Tensor] = None

    def forward(
        self,
        sparse_features: SparseFeatures,
    ) -> Awaitable[Awaitable[SparseFeatures]]:
        """
        Bucketizes sparse feature values into world size number of buckets, and then
        performs AlltoAll operation.

        Args:
            sparse_features (SparseFeatures): sparse features to bucketize and
                redistribute.

        Returns:
            Awaitable[SparseFeatures]: awaitable of SparseFeatures.
        """

        if self._num_id_list_features > 0:
            assert sparse_features.id_list_features is not None
            (
                id_list_features,
                self.unbucketize_permute_tensor,
            ) = bucketize_kjt_before_all2all(
                sparse_features.id_list_features,
                num_buckets=self._world_size,
                block_sizes=self._id_list_feature_block_sizes_tensor,
                output_permute=False,
                bucketize_pos=self._has_feature_processor,
            )
        else:
            id_list_features = None

        if self._num_id_score_list_features > 0:
            assert sparse_features.id_score_list_features is not None
            id_score_list_features, _ = bucketize_kjt_before_all2all(
                sparse_features.id_score_list_features,
                num_buckets=self._world_size,
                block_sizes=self._id_score_list_feature_block_sizes_tensor,
                output_permute=False,
                bucketize_pos=False,
            )
        else:
            id_score_list_features = None

        bucketized_sparse_features = SparseFeatures(
            id_list_features=id_list_features,
            id_score_list_features=id_score_list_features,
        )
        return self._dist(bucketized_sparse_features)


class VariableBatchRwEmbeddingDistAwaitable(Awaitable[torch.Tensor]):
    def __init__(self, awaitable: Awaitable[torch.Tensor], batch_size: int) -> None:
        super().__init__()
        self._awaitable = awaitable
        self._batch_size = batch_size

    def _wait_impl(self) -> torch.Tensor:
        embedding = self._awaitable.wait()

        return embedding


class VariableBatchRwPooledEmbeddingDist(
    BaseEmbeddingDist[VariableBatchShardingContext, torch.Tensor, torch.Tensor]
):
    def __init__(
        self,
        pg: dist.ProcessGroup,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__()
        self._workers: int = pg.size()
        self._rank: int = pg.rank()
        self._dist = PooledEmbeddingsReduceScatter(
            pg,
            codecs=qcomm_codecs_registry.get(
                CommOp.POOLED_EMBEDDINGS_REDUCE_SCATTER.name, None
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
        batch_size_per_rank = sharding_ctx.batch_size_per_rank
        batch_size = batch_size_per_rank[self._rank]

        awaitable_tensor = self._dist(
            local_embs.view(sum(batch_size_per_rank), -1),
            input_splits=batch_size_per_rank,
        )
        return VariableBatchRwEmbeddingDistAwaitable(awaitable_tensor, batch_size)


class VariableBatchRwPooledEmbeddingSharding(
    BaseRwEmbeddingSharding[
        VariableBatchShardingContext, SparseFeatures, torch.Tensor, torch.Tensor
    ]
):
    """
    Shards pooled embeddings row-wise, i.e.. a given embedding table is evenly
    distributed by rows and table slices are placed on all ranks.

    Supports variable batch size.
    """

    def create_input_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseSparseFeaturesDist[SparseFeatures]:
        num_id_list_features = self._get_id_list_features_num()
        num_id_score_list_features = self._get_id_score_list_features_num()
        id_list_feature_hash_sizes = self._get_id_list_features_hash_sizes()
        id_score_list_feature_hash_sizes = self._get_id_score_list_features_hash_sizes()
        return VariableBatchRwSparseFeaturesDist(
            # pyre-fixme[6]: For 1st param expected `ProcessGroup` but got
            #  `Optional[ProcessGroup]`.
            pg=self._pg,
            num_id_list_features=num_id_list_features,
            num_id_score_list_features=num_id_score_list_features,
            id_list_feature_hash_sizes=id_list_feature_hash_sizes,
            id_score_list_feature_hash_sizes=id_score_list_feature_hash_sizes,
            device=self._device,
            has_feature_processor=self._has_feature_processor,
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
        return VariableBatchRwPooledEmbeddingDist(
            # pyre-fixme[6]: For 1st param expected `ProcessGroup` but got
            #  `Optional[ProcessGroup]`.
            self._pg,
            qcomm_codecs_registry=self.qcomm_codecs_registry,
        )
