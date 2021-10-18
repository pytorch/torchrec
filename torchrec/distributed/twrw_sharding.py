#!/usr/bin/env python3

import itertools
import math
from typing import List, Optional, Dict, Any, Tuple

import torch
import torch.distributed as dist
from torch.distributed._sharding_spec import ShardMetadata
from torchrec.distributed.comm import intra_and_cross_node_pg
from torchrec.distributed.dist_data import (
    PooledEmbeddingsReduceScatter,
    PooledEmbeddingsAllToAll,
)
from torchrec.distributed.embedding_lookup import GroupedPooledEmbeddingsLookup
from torchrec.distributed.embedding_sharding import (
    group_tables,
    SparseFeaturesAllToAll,
    BasePooledEmbeddingDist,
    BaseSequenceEmbeddingDist,
    BaseSparseFeaturesDist,
    EmbeddingSharding,
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
        self._has_feature_processor = has_feature_processor

    def forward(
        self,
        sparse_features: SparseFeatures,
    ) -> Awaitable[SparseFeatures]:
        bucketized_sparse_features = SparseFeatures(
            id_list_features=sparse_features.id_list_features.bucketize(
                num_buckets=self._local_size,
                block_sizes=self._id_list_feature_block_sizes_tensor,
                output_permute=False,
                bucketize_pos=self._has_feature_processor,
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
        embedding_configs: List[
            Tuple[EmbeddingTableConfig, ParameterSharding, torch.Tensor]
        ],
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

        sharded_tables_per_rank = self._shard(embedding_configs)
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
        self._has_feature_processor: bool = False
        for group_config in self._score_grouped_embedding_configs_per_node[
            self._my_rank // self._local_size
        ]:
            if group_config.has_feature_processor:
                self._has_feature_processor = True

    def _shard(
        self,
        embedding_configs: List[
            Tuple[EmbeddingTableConfig, ParameterSharding, torch.Tensor]
        ],
    ) -> List[List[ShardedEmbeddingTable]]:
        world_size = self._world_size
        local_size = self._local_size
        tables_per_rank: List[List[ShardedEmbeddingTable]] = [
            [] for i in range(world_size)
        ]
        for config in embedding_configs:
            # pyre-ignore [16]
            table_node = config[1].ranks[0] // local_size
            # pyre-fixme [16]
            shards = config[1].sharding_spec.shards

            # construct the global sharded_tensor_metadata
            global_metadata = ShardedTensorMetadata(
                shards_metadata=shards,
                size=torch.Size([config[0].num_embeddings, config[0].embedding_dim]),
            )

            for rank in range(
                table_node * local_size,
                (table_node + 1) * local_size,
            ):
                rank_idx = rank - (table_node * local_size)
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
                        local_rows=shards[rank_idx].shard_lengths[0],
                        local_cols=config[0].embedding_dim,
                        compute_kernel=EmbeddingComputeKernel(config[1].compute_kernel),
                        local_metadata=shards[rank_idx],
                        global_metadata=global_metadata,
                        weight_init_max=config[0].weight_init_max,
                        weight_init_min=config[0].weight_init_min,
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
            has_feature_processor=self._has_feature_processor,
        )

    def create_lookup(
        self,
        fused_params: Optional[Dict[str, Any]],
        feature_processor: Optional[BaseGroupedFeatureProcessor] = None,
    ) -> BaseEmbeddingLookup:
        return GroupedPooledEmbeddingsLookup(
            grouped_configs=self._grouped_embedding_configs_per_rank[self._my_rank],
            grouped_score_configs=self._score_grouped_embedding_configs_per_rank[
                self._my_rank
            ],
            fused_params=fused_params,
            device=self._device,
            feature_processor=feature_processor,
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

    def embedding_shard_metadata(self) -> List[Optional[ShardMetadata]]:
        embedding_shard_metadata = []
        for grouped_config in self._grouped_embedding_configs_per_node:
            for config in grouped_config:
                embedding_shard_metadata.extend(config.embedding_shard_metadata())
        for grouped_config in self._score_grouped_embedding_configs_per_node:
            for config in grouped_config:
                embedding_shard_metadata.extend(config.embedding_shard_metadata())
        return embedding_shard_metadata

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
