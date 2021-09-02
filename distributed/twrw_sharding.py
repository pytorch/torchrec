#!/usr/bin/env python3

import itertools
import math
from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.distributed as dist
from torchrec.distributed.comm import intra_and_cross_node_pg
from torchrec.distributed.dist_data import (
    PooledEmbeddingsReduceScatter,
    PooledEmbeddingsAllToAll,
)
from torchrec.distributed.embedding_lookup import GroupedPooledEmbeddingsLookup
from torchrec.distributed.embedding_sharding import (
    group_tables,
    SparseFeaturesAllToAll,
    ShardedEmbeddingTable,
    BasePooledEmbeddingDist,
    BaseSequenceEmbeddingDist,
    BaseSparseFeaturesDist,
    EmbeddingSharding,
    BaseEmbeddingLookup,
)
from torchrec.distributed.embedding_types import GroupedEmbeddingConfig, SparseFeatures
from torchrec.distributed.types import (
    ShardedTensorMetadata,
    ShardMetadata,
    Awaitable,
)


class TwRwSparseFeaturesDist(BaseSparseFeaturesDist):
    def __init__(
        self,
        pg: dist.ProcessGroup,
        intra_pg: dist.ProcessGroup,
        num_id_list_features: int,
        num_id_score_list_features: int,
        id_list_features_per_rank: List[int],
        id_score_list_features_per_rank: List[int],
        id_list_feature_hash_sizes: List[int],
        id_score_list_feature_hash_sizes: List[int],
        device: Optional[torch.device] = None,
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
            "_id_list_sf_staggerd_shuffle_tensor",
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
            pg,
            id_list_features_per_rank,
            id_score_list_features_per_rank,
            device,
            self._num_cross_nodes,
        )

    def forward(
        self,
        sparse_features: SparseFeatures,
    ) -> Awaitable[SparseFeatures]:
        bucketized_sparse_features = SparseFeatures(
            id_list_features=sparse_features.id_list_features.bucketize(
                num_buckets=self._local_size,
                block_sizes=self._id_list_feature_block_sizes_tensor,
                output_permute=False,
                bucketize_pos=False,
            )[0].permute(
                self._id_list_sf_staggered_shuffle,
                self._id_list_sf_staggerd_shuffle_tensor,
            )
            if sparse_features.id_list_features is not None
            else None,
            id_score_list_features=sparse_features.id_score_list_features.bucketize(
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


class TwRwEmbeddingDist(BasePooledEmbeddingDist):
    def __init__(
        self,
        cross_pg: dist.ProcessGroup,
        intra_pg: dist.ProcessGroup,
        dim_sum_per_node: List[int],
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self._intra_dist = PooledEmbeddingsReduceScatter(intra_pg)
        self._cross_dist = PooledEmbeddingsAllToAll(
            cross_pg,
            dim_sum_per_node,
            device,
        )

    def forward(self, local_embs: torch.Tensor) -> Awaitable[torch.Tensor]:
        return self._cross_dist(self._intra_dist(local_embs).wait())


class TwRwEmbeddingSharding(EmbeddingSharding):
    """
    Shards embedding bags table-wise then row-wise
    """

    def __init__(
        self,
        sharded_tables: List[ShardedEmbeddingTable],
        pg: dist.ProcessGroup,
        device: Optional[torch.device] = None,
        is_sequence: bool = False,
    ) -> None:
        super().__init__()
        if is_sequence:
            raise RuntimeError(
                "TABLE_ROW_WISE sharding does not support sequence embeddings."
            )
        self._pg = pg
        self._device = device
        self._is_sequence = is_sequence
        intra_pg, cross_pg = intra_and_cross_node_pg()
        self._intra_pg: dist.ProcessGroup = intra_pg
        self._cross_pg: dist.ProcessGroup = cross_pg
        self._world_size: int = self._pg.size()
        self._local_size: int = self._intra_pg.size()
        self._my_rank: int = self._pg.rank()

        sharded_tables_per_rank = self._shard(sharded_tables)
        self._grouped_embedding_configs_per_rank: List[
            List[GroupedEmbeddingConfig]
        ] = []
        self._score_grouped_embedding_configs_per_rank: List[
            List[GroupedEmbeddingConfig]
        ] = []
        self._grouped_embedding_configs_per_node: List[
            List[GroupedEmbeddingConfig]
        ] = []
        self._score_grouped_embedding_configs_per_node: List[
            List[GroupedEmbeddingConfig]
        ] = []
        (
            self._grouped_embedding_configs_per_rank,
            self._score_grouped_embedding_configs_per_rank,
        ) = group_tables(sharded_tables_per_rank)
        self._grouped_embedding_configs_per_node = [
            self._grouped_embedding_configs_per_rank[rank]
            for rank in range(self._world_size)
            if rank % self._local_size == 0
        ]
        self._score_grouped_embedding_configs_per_node = [
            self._score_grouped_embedding_configs_per_rank[rank]
            for rank in range(self._world_size)
            if rank % self._local_size == 0
        ]

    def _shard(
        self, tables: List[ShardedEmbeddingTable]
    ) -> List[List[ShardedEmbeddingTable]]:
        world_size = self._world_size
        local_size = self._local_size
        tables_per_rank: List[List[ShardedEmbeddingTable]] = [
            [] for i in range(world_size)
        ]

        def _shard_table_rows(
            table_node: int,
            hash_size: int,
            embedding_dim: int,
            world_size: int,
            local_size: int,
        ) -> Tuple[List[int], List[int], List[int]]:
            block_size = math.ceil(hash_size / local_size)
            last_block_size = hash_size - block_size * (local_size - 1)
            first_local_rank = (table_node) * local_size
            last_local_rank = first_local_rank + local_size - 1
            local_rows: List[int] = []
            local_cols: List[int] = []
            local_row_offsets: List[int] = []
            cumul_row_offset = 0
            for rank in range(world_size):
                if rank < first_local_rank:
                    local_row = 0
                    local_col = 0
                elif rank < last_local_rank:
                    local_row = block_size
                    local_col = embedding_dim
                elif rank == last_local_rank:
                    local_row = last_block_size
                    local_col = embedding_dim
                else:
                    cumul_row_offset = 0
                    local_row = 0
                    local_col = 0
                local_rows.append(local_row)
                local_cols.append(local_col)
                local_row_offsets.append(cumul_row_offset)
                cumul_row_offset += local_row

            return (local_rows, local_cols, local_row_offsets)

        for table in tables:
            table_node = table.rank // local_size
            local_rows, local_cols, local_row_offsets = _shard_table_rows(
                table_node=table_node,
                embedding_dim=table.embedding_dim,
                hash_size=table.num_embeddings,
                world_size=world_size,
                local_size=local_size,
            )
            shards = [
                ShardMetadata(
                    dims=[
                        local_rows[rank],
                        local_cols[rank],
                    ],
                    offsets=[local_row_offsets[rank], 0],
                )
                for rank in range(world_size)
            ]

            for rank in range(
                table_node * local_size,
                (table_node + 1) * local_size,
            ):
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
        id_list_features_per_rank = self._features_per_rank(
            self._grouped_embedding_configs_per_rank
        )
        id_score_list_features_per_rank = self._features_per_rank(
            self._score_grouped_embedding_configs_per_rank
        )
        id_list_feature_hash_sizes = self._get_id_list_features_hash_sizes()
        id_score_list_feature_hash_sizes = self._get_id_score_list_features_hash_sizes()
        return TwRwSparseFeaturesDist(
            pg=self._pg,
            intra_pg=self._intra_pg,
            num_id_list_features=num_id_list_features,
            num_id_score_list_features=num_id_score_list_features,
            id_list_features_per_rank=id_list_features_per_rank,
            id_score_list_features_per_rank=id_score_list_features_per_rank,
            id_list_feature_hash_sizes=id_list_feature_hash_sizes,
            id_score_list_feature_hash_sizes=id_score_list_feature_hash_sizes,
            device=self._device,
        )

    def create_lookup(
        self,
        fused_params: Optional[Dict[str, Any]],
    ) -> BaseEmbeddingLookup:
        return GroupedPooledEmbeddingsLookup(
            grouped_configs=self._grouped_embedding_configs_per_rank[self._my_rank],
            grouped_score_configs=self._score_grouped_embedding_configs_per_rank[
                self._my_rank
            ],
            fused_params=fused_params,
            device=self._device,
        )

    def create_pooled_output_dist(self) -> BasePooledEmbeddingDist:
        return TwRwEmbeddingDist(
            cross_pg=self._cross_pg,
            intra_pg=self._intra_pg,
            dim_sum_per_node=self._dim_sum_per_node(),
            device=self._device,
        )

    def create_sequence_output_dist(self) -> BaseSequenceEmbeddingDist:
        raise NotImplementedError

    def embedding_dims(self) -> List[int]:
        embedding_dims = []
        for grouped_embedding_configs, score_grouped_embedding_configs in zip(
            self._grouped_embedding_configs_per_node,
            self._score_grouped_embedding_configs_per_node,
        ):
            for grouped_config in grouped_embedding_configs:
                embedding_dims.extend(grouped_config.embedding_dims())
            for grouped_config in score_grouped_embedding_configs:
                embedding_dims.extend(grouped_config.embedding_dims())
        return embedding_dims

    def embedding_names(self) -> List[str]:
        embedding_names = []
        for grouped_embedding_configs, score_grouped_embedding_configs in zip(
            self._grouped_embedding_configs_per_node,
            self._score_grouped_embedding_configs_per_node,
        ):
            for grouped_config in grouped_embedding_configs:
                embedding_names.extend(grouped_config.embedding_names())
            for grouped_config in score_grouped_embedding_configs:
                embedding_names.extend(grouped_config.embedding_names())
        return embedding_names

    def id_list_feature_names(self) -> List[str]:
        id_list_feature_names = []
        for grouped_config in self._grouped_embedding_configs_per_node:
            for config in grouped_config:
                id_list_feature_names.extend(config.feature_names())
        return id_list_feature_names

    def id_score_list_feature_names(self) -> List[str]:
        id_score_list_feature_names = []
        for grouped_config in self._score_grouped_embedding_configs_per_node:
            for config in grouped_config:
                id_score_list_feature_names.extend(config.feature_names())
        return id_score_list_feature_names

    def _get_id_list_features_num(self) -> int:
        id_list_features_num: int = 0
        for grouped_config in self._grouped_embedding_configs_per_node:
            for config in grouped_config:
                id_list_features_num += config.num_features()
        return id_list_features_num

    def _get_id_score_list_features_num(self) -> int:
        id_score_list_features_num: int = 0
        for grouped_config in self._score_grouped_embedding_configs_per_node:
            for config in grouped_config:
                id_score_list_features_num += config.num_features()
        return id_score_list_features_num

    def _get_id_list_features_hash_sizes(self) -> List[int]:
        id_list_feature_hash_sizes: List[int] = []
        for grouped_config in self._grouped_embedding_configs_per_node:
            for config in grouped_config:
                id_list_feature_hash_sizes.extend(config.feature_hash_sizes())
        return id_list_feature_hash_sizes

    def _get_id_score_list_features_hash_sizes(self) -> List[int]:
        id_score_list_feature_hash_sizes: List[int] = []
        for grouped_config in self._score_grouped_embedding_configs_per_node:
            for config in grouped_config:
                id_score_list_feature_hash_sizes.extend(config.feature_hash_sizes())
        return id_score_list_feature_hash_sizes

    def _dim_sum_per_node(self) -> List[int]:
        dim_sum_per_rank = []
        for grouped_embedding_configs, score_grouped_embedding_configs in zip(
            self._grouped_embedding_configs_per_node,
            self._score_grouped_embedding_configs_per_node,
        ):
            dim_sum = 0
            for grouped_config in grouped_embedding_configs:
                dim_sum += grouped_config.dim_sum()
            for grouped_config in score_grouped_embedding_configs:
                dim_sum += grouped_config.dim_sum()
            dim_sum_per_rank.append(dim_sum)
        return dim_sum_per_rank

    def _features_per_rank(
        self, group: List[List[GroupedEmbeddingConfig]]
    ) -> List[int]:
        features_per_rank = []
        for grouped_embedding_configs in group:
            num_features = 0
            for grouped_config in grouped_embedding_configs:
                num_features += grouped_config.num_features()
            features_per_rank.append(num_features)
        return features_per_rank
