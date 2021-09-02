#!/usr/bin/env python3

from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.distributed as dist
from torchrec.distributed.dist_data import (
    PooledEmbeddingsReduceScatter,
    SequenceEmbeddingAllToAll,
)
from torchrec.distributed.embedding_lookup import (
    GroupedPooledEmbeddingsLookup,
    GroupedEmbeddingsLookup,
)
from torchrec.distributed.embedding_sharding import (
    group_tables,
    SparseFeaturesAllToAll,
    ShardedEmbeddingTable,
    BasePooledEmbeddingDist,
    BaseSparseFeaturesDist,
    EmbeddingSharding,
    BaseSequenceEmbeddingDist,
    SequenceShardingContext,
    BaseEmbeddingLookup,
)
from torchrec.distributed.embedding_types import GroupedEmbeddingConfig, SparseFeatures
from torchrec.distributed.types import (
    ShardedTensorMetadata,
    ShardMetadata,
    Awaitable,
)


class RwSparseFeaturesDist(BaseSparseFeaturesDist):
    def __init__(
        self,
        pg: dist.ProcessGroup,
        num_id_list_features: int,
        num_id_score_list_features: int,
        id_list_feature_hash_sizes: List[int],
        id_score_list_feature_hash_sizes: List[int],
        device: Optional[torch.device] = None,
        is_sequence: bool = False,
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
            pg,
            self._world_size * [self._num_id_list_features],
            self._world_size * [self._num_id_score_list_features],
            device,
        )
        self._is_sequence = is_sequence
        self.unbucketize_permute_tensor: Optional[torch.Tensor] = None

    def forward(
        self,
        sparse_features: SparseFeatures,
    ) -> Awaitable[SparseFeatures]:
        if self._num_id_list_features > 0:
            (
                id_list_features,
                self.unbucketize_permute_tensor,
                # pyre-ignore [16]
            ) = sparse_features.id_list_features.bucketize(
                num_buckets=self._world_size,
                block_sizes=self._id_list_feature_block_sizes_tensor,
                output_permute=self._is_sequence,
                bucketize_pos=False,
            )
        else:
            id_list_features = None

        if self._num_id_score_list_features > 0:
            (
                id_score_list_features,
                _,
            ) = sparse_features.id_score_list_features.bucketize(
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


class RwPooledEmbeddingDist(BasePooledEmbeddingDist):
    def __init__(
        self,
        pg: dist.ProcessGroup,
    ) -> None:
        super().__init__()
        self._dist = PooledEmbeddingsReduceScatter(pg)

    def forward(self, local_embs: torch.Tensor) -> Awaitable[torch.Tensor]:
        return self._dist(local_embs)


class RwSequenceEmbeddingDist(BaseSequenceEmbeddingDist):
    def __init__(
        self,
        pg: dist.ProcessGroup,
        num_features: int,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self._dist = SequenceEmbeddingAllToAll(pg, [num_features] * pg.size(), device)

    def forward(
        self, sharding_ctx: SequenceShardingContext, local_embs: torch.Tensor
    ) -> Awaitable[torch.Tensor]:
        return self._dist(
            local_embs=local_embs,
            lengths=sharding_ctx.lengths_after_input_dist,
            input_splits=sharding_ctx.input_splits,
            output_splits=sharding_ctx.output_splits,
            unbucketize_permute_tensor=sharding_ctx.unbucketize_permute_tensor,
        )


class RwEmbeddingSharding(EmbeddingSharding):
    """
    Shards embedding bags row-wise, i.e.. a given embedding table is evenly distribued by rows and table slices are placed on all ranks.
    """

    def __init__(
        self,
        sharded_tables: List[ShardedEmbeddingTable],
        pg: dist.ProcessGroup,
        device: Optional[torch.device] = None,
        is_sequence: bool = False,
    ) -> None:
        super().__init__()
        self._pg = pg
        self._device = device
        self._is_sequence = is_sequence
        sharded_tables_per_rank = self._shard(sharded_tables)
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
        ] = self._grouped_embedding_configs_per_rank[dist.get_rank(pg)]
        self._score_grouped_embedding_configs: List[
            GroupedEmbeddingConfig
        ] = self._score_grouped_embedding_configs_per_rank[dist.get_rank(pg)]

    def _shard(
        self, tables: List[ShardedEmbeddingTable]
    ) -> List[List[ShardedEmbeddingTable]]:
        world_size = self._pg.size()
        tables_per_rank: List[List[ShardedEmbeddingTable]] = [
            [] for i in range(world_size)
        ]

        def _shard_table_rows(
            hash_size: int, world_size: int
        ) -> Tuple[List[int], int, int]:
            block_size = (hash_size + world_size - 1) // world_size
            last_rank = hash_size // block_size
            last_block_size = hash_size - block_size * last_rank
            local_rows: List[int] = []
            for rank in range(world_size):
                if rank < last_rank:
                    local_row = block_size
                elif rank == last_rank:
                    local_row = last_block_size
                else:
                    local_row = 0
                local_rows.append(local_row)
            return (local_rows, block_size, last_rank)

        for table in tables:
            local_rows, block_size, last_rank = _shard_table_rows(
                table.num_embeddings, world_size
            )
            shards = [
                ShardMetadata(
                    dims=[
                        local_rows[rank],
                        table.embedding_dim,
                    ],
                    offsets=[block_size * min(rank, last_rank), 0],
                )
                for rank in range(world_size)
            ]
            for rank in range(world_size):
                tables_per_rank[rank].append(
                    ShardedEmbeddingTable(
                        num_embeddings=table.num_embeddings,
                        embedding_dim=table.embedding_dim,
                        name=table.name,
                        embedding_names=table.embedding_names,
                        data_type=table.data_type,
                        feature_names=table.feature_names,
                        pooling=table.pooling,
                        compute_kernel=table.compute_kernel,
                        is_weighted=table.is_weighted,
                        rank=table.rank,
                        local_rows=local_rows[rank],
                        local_cols=table.embedding_dim,
                        sharded_tensor=True,
                        metadata=ShardedTensorMetadata(shards=shards),
                    )
                )
        return tables_per_rank

    def create_input_dist(self) -> BaseSparseFeaturesDist:
        num_id_list_features = self._get_id_list_features_num()
        num_id_score_list_features = self._get_id_score_list_features_num()
        id_list_feature_hash_sizes = self._get_id_list_features_hash_sizes()
        id_score_list_feature_hash_sizes = self._get_id_score_list_features_hash_sizes()
        return RwSparseFeaturesDist(
            self._pg,
            num_id_list_features,
            num_id_score_list_features,
            id_list_feature_hash_sizes,
            id_score_list_feature_hash_sizes,
            self._device,
            self._is_sequence,
        )

    def create_lookup(
        self,
        fused_params: Optional[Dict[str, Any]],
    ) -> BaseEmbeddingLookup:
        if self._is_sequence:
            return GroupedEmbeddingsLookup(
                grouped_configs=self._grouped_embedding_configs,
                fused_params=fused_params,
                device=self._device,
            )
        else:
            return GroupedPooledEmbeddingsLookup(
                grouped_configs=self._grouped_embedding_configs,
                grouped_score_configs=self._score_grouped_embedding_configs,
                fused_params=fused_params,
                device=self._device,
            )

    def create_pooled_output_dist(self) -> RwPooledEmbeddingDist:
        return RwPooledEmbeddingDist(self._pg)

    def create_sequence_output_dist(self) -> RwSequenceEmbeddingDist:
        return RwSequenceEmbeddingDist(
            self._pg,
            self._get_id_list_features_num(),
            self._device,
        )

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
