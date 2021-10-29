#!/usr/bin/env python3

from typing import List, Optional, Any, Dict, Tuple

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
    EmbeddingComputeKernel,
    BaseGroupedFeatureProcessor,
)
from torchrec.distributed.types import (
    ShardedTensorMetadata,
    Awaitable,
    ParameterSharding,
)
from torchrec.modules.embedding_configs import EmbeddingTableConfig


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
        embedding_configs: List[
            Tuple[EmbeddingTableConfig, ParameterSharding, torch.Tensor]
        ],
        pg: dist.ProcessGroup,
        device: Optional[torch.device] = None,
        is_sequence: bool = False,
    ) -> None:
        super().__init__()
        self._pg = pg
        self._device = device
        self._is_sequence = is_sequence
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
        ] = self._grouped_embedding_configs_per_rank[dist.get_rank(pg)]
        self._score_grouped_embedding_configs: List[
            GroupedEmbeddingConfig
        ] = self._score_grouped_embedding_configs_per_rank[dist.get_rank(pg)]

    def _shard(
        self,
        embedding_configs: List[
            Tuple[EmbeddingTableConfig, ParameterSharding, torch.Tensor]
        ],
    ) -> List[List[ShardedEmbeddingTable]]:
        world_size = self._pg.size()
        tables_per_rank: List[List[ShardedEmbeddingTable]] = [
            [] for i in range(world_size)
        ]
        for config in embedding_configs:
            # pyre-fixme [16]
            shards = config[1].sharding_spec.shards

            # construct the global sharded_tensor_metadata
            global_metadata = ShardedTensorMetadata(
                shards_metadata=shards,
                size=torch.Size([config[0].num_embeddings, config[0].embedding_dim]),
            )

            # pyre-fixme [16]
            tables_per_rank[config[1].ranks[0]].append(
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
                    local_rows=config[0].num_embeddings,
                    local_cols=config[0].embedding_dim,
                    compute_kernel=EmbeddingComputeKernel(config[1].compute_kernel),
                    local_metadata=shards[0],
                    global_metadata=global_metadata,
                    weight_init_max=config[0].weight_init_max,
                    weight_init_min=config[0].weight_init_min,
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
        feature_processor: Optional[BaseGroupedFeatureProcessor] = None,
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
                feature_processor=feature_processor,
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

    def embedding_shard_metadata(self) -> List[Optional[ShardMetadata]]:
        embedding_shard_metadata = []
        for grouped_embedding_configs, score_grouped_embedding_configs in zip(
            self._grouped_embedding_configs_per_rank,
            self._score_grouped_embedding_configs_per_rank,
        ):
            for grouped_config in grouped_embedding_configs:
                embedding_shard_metadata.extend(
                    grouped_config.embedding_shard_metadata()
                )
            for grouped_config in score_grouped_embedding_configs:
                embedding_shard_metadata.extend(
                    grouped_config.embedding_shard_metadata()
                )
        return embedding_shard_metadata

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
