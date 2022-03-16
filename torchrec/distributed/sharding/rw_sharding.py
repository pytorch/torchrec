#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Dict, Any, Tuple, TypeVar

import torch
import torch.distributed as dist
from torchrec.distributed.dist_data import PooledEmbeddingsReduceScatter
from torchrec.distributed.embedding_lookup import GroupedPooledEmbeddingsLookup
from torchrec.distributed.embedding_sharding import (
    group_tables,
    SparseFeaturesAllToAll,
    BaseEmbeddingDist,
    BaseSparseFeaturesDist,
    EmbeddingSharding,
    BaseEmbeddingLookup,
    bucketize_kjt_before_all2all,
)
from torchrec.distributed.embedding_types import (
    ShardedEmbeddingTable,
    GroupedEmbeddingConfig,
    SparseFeatures,
    EmbeddingComputeKernel,
    BaseGroupedFeatureProcessor,
)
from torchrec.distributed.types import (
    ShardingEnv,
    ShardedTensorMetadata,
    ShardMetadata,
    Awaitable,
    ParameterSharding,
)
from torchrec.modules.embedding_configs import EmbeddingTableConfig
from torchrec.streamable import Multistreamable


F = TypeVar("F", bound=Multistreamable)
T = TypeVar("T")


class BaseRwEmbeddingSharding(EmbeddingSharding[F, T]):
    """
    base class for row-wise sharding
    """

    def __init__(
        self,
        embedding_configs: List[
            Tuple[EmbeddingTableConfig, ParameterSharding, torch.Tensor]
        ],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self._env = env
        # pyre-ignore[11]
        self._pg: Optional[dist.ProcessGroup] = self._env.process_group
        self._world_size: int = self._env.world_size
        self._rank: int = self._env.rank
        if device is None:
            device = torch.device("cpu")
        self._device = device
        sharded_tables_per_rank = self._shard(embedding_configs)
        self._grouped_embedding_configs_per_rank: List[
            List[GroupedEmbeddingConfig]
        ] = []
        self._score_grouped_embedding_configs_per_rank: List[
            List[GroupedEmbeddingConfig]
        ] = []
        (
            self._grouped_embedding_configs_per_rank,
            self._score_grouped_embedding_configs_per_rank,
        ) = group_tables(sharded_tables_per_rank)
        self._grouped_embedding_configs: List[
            GroupedEmbeddingConfig
        ] = self._grouped_embedding_configs_per_rank[self._rank]
        self._score_grouped_embedding_configs: List[
            GroupedEmbeddingConfig
        ] = self._score_grouped_embedding_configs_per_rank[self._rank]

        self._has_feature_processor: bool = False
        for group_config in self._grouped_embedding_configs:
            if group_config.has_feature_processor:
                self._has_feature_processor = True

    def _shard(
        self,
        embedding_configs: List[
            Tuple[EmbeddingTableConfig, ParameterSharding, torch.Tensor]
        ],
    ) -> List[List[ShardedEmbeddingTable]]:
        tables_per_rank: List[List[ShardedEmbeddingTable]] = [
            [] for i in range(self._world_size)
        ]
        for config in embedding_configs:
            # pyre-fixme [16]
            shards = config[1].sharding_spec.shards

            # construct the global sharded_tensor_metadata
            global_metadata = ShardedTensorMetadata(
                shards_metadata=shards,
                size=torch.Size([config[0].num_embeddings, config[0].embedding_dim]),
            )

            for rank in range(self._world_size):
                tables_per_rank[rank].append(
                    ShardedEmbeddingTable(
                        num_embeddings=config[0].num_embeddings,
                        embedding_dim=config[0].embedding_dim,
                        name=config[0].name,
                        embedding_names=config[0].embedding_names,
                        data_type=config[0].data_type,
                        feature_names=config[0].feature_names,
                        pooling=config[0].pooling,
                        is_weighted=config[0].is_weighted,
                        has_feature_processor=config[0].has_feature_processor,
                        local_rows=shards[rank].shard_sizes[0],
                        local_cols=config[0].embedding_dim,
                        compute_kernel=EmbeddingComputeKernel(config[1].compute_kernel),
                        local_metadata=shards[rank],
                        global_metadata=global_metadata,
                        weight_init_max=config[0].weight_init_max,
                        weight_init_min=config[0].weight_init_min,
                    )
                )
        return tables_per_rank

    def embedding_dims(self) -> List[int]:
        embedding_dims = []
        for grouped_config in self._grouped_embedding_configs:
            embedding_dims.extend(grouped_config.embedding_dims())
        for grouped_config in self._score_grouped_embedding_configs:
            embedding_dims.extend(grouped_config.embedding_dims())
        return embedding_dims

    def embedding_names(self) -> List[str]:
        embedding_names = []
        for grouped_config in self._grouped_embedding_configs:
            embedding_names.extend(grouped_config.embedding_names())
        for grouped_config in self._score_grouped_embedding_configs:
            embedding_names.extend(grouped_config.embedding_names())
        return embedding_names

    def embedding_shard_metadata(self) -> List[Optional[ShardMetadata]]:
        embedding_shard_metadata = []
        for grouped_config in self._grouped_embedding_configs:
            embedding_shard_metadata.extend(grouped_config.embedding_shard_metadata())
        for grouped_config in self._score_grouped_embedding_configs:
            embedding_shard_metadata.extend(grouped_config.embedding_shard_metadata())
        return embedding_shard_metadata

    def id_list_feature_names(self) -> List[str]:
        id_list_feature_names = []
        for grouped_config in self._grouped_embedding_configs:
            id_list_feature_names.extend(grouped_config.feature_names())
        return id_list_feature_names

    def id_score_list_feature_names(self) -> List[str]:
        id_score_list_feature_names = []
        for grouped_config in self._score_grouped_embedding_configs:
            id_score_list_feature_names.extend(grouped_config.feature_names())
        return id_score_list_feature_names

    def _get_id_list_features_num(self) -> int:
        return sum(
            group_config.num_features()
            for group_config in self._grouped_embedding_configs
        )

    def _get_id_score_list_features_num(self) -> int:
        return sum(
            group_config.num_features()
            for group_config in self._score_grouped_embedding_configs
        )

    def _get_id_list_features_hash_sizes(self) -> List[int]:
        id_list_feature_hash_sizes: List[int] = []
        for group_config in self._grouped_embedding_configs:
            id_list_feature_hash_sizes.extend(group_config.feature_hash_sizes())
        return id_list_feature_hash_sizes

    def _get_id_score_list_features_hash_sizes(self) -> List[int]:
        id_score_list_feature_hash_sizes: List[int] = []
        for group_config in self._score_grouped_embedding_configs:
            id_score_list_feature_hash_sizes.extend(group_config.feature_hash_sizes())
        return id_score_list_feature_hash_sizes


class RwSparseFeaturesDist(BaseSparseFeaturesDist[SparseFeatures]):
    """
    Bucketizes sparse features in RW fashion and then redistributes with an AlltoAll
    collective operation.

    Args:

        pg (dist.ProcessGroup): ProcessGroup for AlltoAll communication.
        intra_pg (dist.ProcessGroup): ProcessGroup within single host group for AlltoAll
        communication.
        num_id_list_features (int): total number of id list features.
        num_id_score_list_features (int): total number of id score list features
        id_list_feature_hash_sizes (List[int]): hash sizes of id list features.
        id_score_list_feature_hash_sizes (List[int]): hash sizes of id score list features.
        device (Optional[torch.device]): device on which buffers will be allocated.
        is_sequence (bool): if this is for a sequence embedding.
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
        is_sequence: bool = False,
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
        )
        self._is_sequence = is_sequence
        self._has_feature_processor = has_feature_processor
        self.unbucketize_permute_tensor: Optional[torch.Tensor] = None

    def forward(
        self,
        sparse_features: SparseFeatures,
    ) -> Awaitable[Awaitable[SparseFeatures]]:
        """
        Bucketizes sparse feature values into  world size number of buckets, and then
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
                output_permute=self._is_sequence,
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


class RwPooledEmbeddingDist(BaseEmbeddingDist[torch.Tensor]):
    """
    Redistributes pooled embedding tensor in RW fashion by performing a reduce-scatter
    operation.

    Args:
        pg (dist.ProcessGroup): ProcessGroup for reduce-scatter communication.
    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
    ) -> None:
        super().__init__()
        self._dist = PooledEmbeddingsReduceScatter(pg)

    def forward(self, local_embs: torch.Tensor) -> Awaitable[torch.Tensor]:
        """
        Performs reduce-scatter pooled operation on pooled embeddings tensor.

        Args:
            local_embs (torch.Tensor): pooled embeddings tensor to distribute.

        Returns:
            Awaitable[torch.Tensor]: awaitable of pooled embeddings tensor.
        """

        return self._dist(local_embs)


class RwPooledEmbeddingSharding(BaseRwEmbeddingSharding[SparseFeatures, torch.Tensor]):
    """
    Shards embedding bags row-wise, i.e.. a given embedding table is evenly distributed
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
            is_sequence=False,
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
            fused_params=fused_params,
            pg=self._pg,
            device=device if device is not None else self._device,
            feature_processor=feature_processor,
        )

    def create_output_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseEmbeddingDist[torch.Tensor]:
        return RwPooledEmbeddingDist(self._pg)
