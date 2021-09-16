#!/usr/bin/env python3

from typing import List, Optional, Any, Dict

import torch
import torch.distributed as dist
from torch.distributed._sharding_spec import ShardMetadata
from torchrec.distributed.dist_data import (
    PooledEmbeddingsAllToAll,
    SequenceEmbeddingAllToAll,
)
from torchrec.distributed.embedding_lookup import (
    GroupedPooledEmbeddingsLookup,
    GroupedEmbeddingsLookup,
)
from torchrec.distributed.embedding_sharding import (
    EmbeddingSharding,
    SparseFeaturesAllToAll,
    group_tables,
    BasePooledEmbeddingDist,
    BaseSequenceEmbeddingDist,
    BaseSparseFeaturesDist,
    SequenceShardingContext,
    BaseEmbeddingLookup,
)
from torchrec.distributed.embedding_types import (
    GroupedEmbeddingConfig,
    SparseFeatures,
    ShardedEmbeddingTable,
    ShardedEmbeddingTableShard,
)
from torchrec.distributed.types import (
    ShardedTensorMetadata,
    Awaitable,
)


class TwSparseFeaturesDist(BaseSparseFeaturesDist):
    def __init__(
        self,
        pg: dist.ProcessGroup,
        id_list_features_per_rank: List[int],
        id_score_list_features_per_rank: List[int],
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self._dist = SparseFeaturesAllToAll(
            pg,
            id_list_features_per_rank,
            id_score_list_features_per_rank,
            device,
        )

    def forward(
        self,
        sparse_features: SparseFeatures,
    ) -> Awaitable[SparseFeatures]:
        return self._dist(sparse_features)


class TwPooledEmbeddingDist(BasePooledEmbeddingDist):
    def __init__(
        self,
        pg: dist.ProcessGroup,
        dim_sum_per_rank: List[int],
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self._dist = PooledEmbeddingsAllToAll(
            pg,
            dim_sum_per_rank,
            device,
        )

    def forward(self, local_embs: torch.Tensor) -> Awaitable[torch.Tensor]:
        return self._dist(local_embs)


class TwSequenceEmbeddingDist(BaseSequenceEmbeddingDist):
    def __init__(
        self,
        pg: dist.ProcessGroup,
        features_per_rank: List[int],
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self._dist = SequenceEmbeddingAllToAll(pg, features_per_rank, device)

    def forward(
        self, sharding_ctx: SequenceShardingContext, local_embs: torch.Tensor
    ) -> Awaitable[torch.Tensor]:
        return self._dist(
            local_embs=local_embs,
            lengths=sharding_ctx.lengths_after_input_dist,
            input_splits=sharding_ctx.input_splits,
            output_splits=sharding_ctx.output_splits,
            unbucketize_permute_tensor=None,
        )


class TwEmbeddingSharding(EmbeddingSharding):
    """
    Shards embedding bags table-wise, i.e.. a given embedding table is entirely placed on a selected rank.
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
    ) -> List[List[ShardedEmbeddingTableShard]]:
        world_size = self._pg.size()
        tables_per_rank: List[List[ShardedEmbeddingTableShard]] = [
            [] for i in range(world_size)
        ]
        for table in tables:
            # pyre-fixme [16]
            rank = table.ranks[0]
            # pyre-fixme [16]
            shards = table.sharding_spec.shards

            # construct the global sharded_tensor_metadata
            global_metadata = ShardedTensorMetadata(
                shards_metadata=shards,
                size=torch.Size([table.num_embeddings, table.embedding_dim]),
            )

            tables_per_rank[rank].append(
                ShardedEmbeddingTableShard(
                    num_embeddings=table.num_embeddings,
                    embedding_dim=table.embedding_dim,
                    name=table.name,
                    embedding_names=table.embedding_names,
                    data_type=table.data_type,
                    feature_names=table.feature_names,
                    pooling=table.pooling,
                    compute_kernel=table.compute_kernel,
                    is_weighted=table.is_weighted,
                    local_rows=table.num_embeddings,
                    local_cols=table.embedding_dim,
                    local_metadata=shards[0],
                    global_metadata=global_metadata,
                )
            )
        return tables_per_rank

    def create_input_dist(self) -> BaseSparseFeaturesDist:
        return TwSparseFeaturesDist(
            self._pg,
            self._id_list_features_per_rank(),
            self._id_score_list_features_per_rank(),
            self._device,
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

    def create_pooled_output_dist(self) -> TwPooledEmbeddingDist:
        return TwPooledEmbeddingDist(
            self._pg,
            self._dim_sum_per_rank(),
            self._device,
        )

    def create_sequence_output_dist(
        self,
    ) -> BaseSequenceEmbeddingDist:
        return TwSequenceEmbeddingDist(
            self._pg,
            self._id_list_features_per_rank(),
            self._device,
        )

    def _dim_sum_per_rank(self) -> List[int]:
        dim_sum_per_rank = []
        for grouped_embedding_configs, score_grouped_embedding_configs in zip(
            self._grouped_embedding_configs_per_rank,
            self._score_grouped_embedding_configs_per_rank,
        ):
            dim_sum = 0
            for grouped_config in grouped_embedding_configs:
                dim_sum += grouped_config.dim_sum()
            for grouped_config in score_grouped_embedding_configs:
                dim_sum += grouped_config.dim_sum()
            dim_sum_per_rank.append(dim_sum)
        return dim_sum_per_rank

    def embedding_dims(self) -> List[int]:
        embedding_dims = []
        for grouped_embedding_configs, score_grouped_embedding_configs in zip(
            self._grouped_embedding_configs_per_rank,
            self._score_grouped_embedding_configs_per_rank,
        ):
            for grouped_config in grouped_embedding_configs:
                embedding_dims.extend(grouped_config.embedding_dims())
            for grouped_config in score_grouped_embedding_configs:
                embedding_dims.extend(grouped_config.embedding_dims())
        return embedding_dims

    def embedding_names(self) -> List[str]:
        embedding_names = []
        for grouped_embedding_configs, score_grouped_embedding_configs in zip(
            self._grouped_embedding_configs_per_rank,
            self._score_grouped_embedding_configs_per_rank,
        ):
            for grouped_config in grouped_embedding_configs:
                embedding_names.extend(grouped_config.embedding_names())
            for grouped_config in score_grouped_embedding_configs:
                embedding_names.extend(grouped_config.embedding_names())
        return embedding_names

    def embedding_metadata(self) -> List[Optional[ShardMetadata]]:
        embedding_metadata = []
        for grouped_embedding_configs, score_grouped_embedding_configs in zip(
            self._grouped_embedding_configs_per_rank,
            self._score_grouped_embedding_configs_per_rank,
        ):
            for grouped_config in grouped_embedding_configs:
                embedding_metadata.extend(grouped_config.embedding_metadata())
            for grouped_config in score_grouped_embedding_configs:
                embedding_metadata.extend(grouped_config.embedding_metadata())
        return embedding_metadata

    def id_list_feature_names(self) -> List[str]:
        id_list_feature_names = []
        for grouped_embedding_configs in self._grouped_embedding_configs_per_rank:
            for grouped_config in grouped_embedding_configs:
                id_list_feature_names.extend(grouped_config.feature_names())
        return id_list_feature_names

    def id_score_list_feature_names(self) -> List[str]:
        id_score_list_feature_names = []
        for (
            score_grouped_embedding_configs
        ) in self._score_grouped_embedding_configs_per_rank:
            for grouped_config in score_grouped_embedding_configs:
                id_score_list_feature_names.extend(grouped_config.feature_names())
        return id_score_list_feature_names

    def _id_list_features_per_rank(self) -> List[int]:
        id_list_features_per_rank = []
        for grouped_embedding_configs in self._grouped_embedding_configs_per_rank:
            num_features = 0
            for grouped_config in grouped_embedding_configs:
                num_features += grouped_config.num_features()
            id_list_features_per_rank.append(num_features)
        return id_list_features_per_rank

    def _id_score_list_features_per_rank(self) -> List[int]:
        id_score_list_features_per_rank = []
        for (
            score_grouped_embedding_configs
        ) in self._score_grouped_embedding_configs_per_rank:
            num_features = 0
            for grouped_config in score_grouped_embedding_configs:
                num_features += grouped_config.num_features()
            id_score_list_features_per_rank.append(num_features)
        return id_score_list_features_per_rank
