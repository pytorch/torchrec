#!/usr/bin/env python3

from typing import List, Optional, cast, Dict, Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torchrec.distributed.embedding_lookup import (
    GroupedPooledEmbeddingsLookup,
    GroupedEmbeddingsLookup,
)
from torchrec.distributed.embedding_sharding import (
    EmbeddingSharding,
    ShardedEmbeddingTable,
    group_tables,
    BasePooledEmbeddingDist,
    BaseSequenceEmbeddingDist,
    BaseSparseFeaturesDist,
    SequenceShardingContext,
    BaseEmbeddingLookup,
)
from torchrec.distributed.embedding_types import GroupedEmbeddingConfig, SparseFeatures
from torchrec.distributed.types import Awaitable, NoWait


class DpSparseFeaturesDist(BaseSparseFeaturesDist):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        sparse_features: SparseFeatures,
    ) -> Awaitable[SparseFeatures]:
        return NoWait(sparse_features)


class DpPooledEmbeddingDist(BasePooledEmbeddingDist):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, local_embs: torch.Tensor) -> Awaitable[torch.Tensor]:
        return NoWait(local_embs)


class DpSequenceEmbeddingDist(BaseSequenceEmbeddingDist):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, sharding_ctx: SequenceShardingContext, local_embs: torch.Tensor
    ) -> Awaitable[torch.Tensor]:
        return NoWait(local_embs)


class DpEmbeddingSharding(EmbeddingSharding):
    """
    Use data-parallel, no table sharding
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
        for table in tables:
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
                        local_rows=table.num_embeddings,
                        local_cols=table.embedding_dim,
                    )
                )
        return tables_per_rank

    def create_input_dist(self) -> DpSparseFeaturesDist:
        return DpSparseFeaturesDist()

    def create_lookup(
        self,
        fused_params: Optional[Dict[str, Any]],
    ) -> BaseEmbeddingLookup:
        if self._is_sequence:
            module = GroupedEmbeddingsLookup(
                grouped_configs=self._grouped_embedding_configs,
                fused_params=fused_params,
                device=self._device,
            )
        else:
            module = GroupedPooledEmbeddingsLookup(
                grouped_configs=self._grouped_embedding_configs,
                grouped_score_configs=self._score_grouped_embedding_configs,
                fused_params=fused_params,
                device=self._device,
            )
        # TODO: pass ddp_bucket_cap_mb
        return cast(
            BaseEmbeddingLookup,
            DistributedDataParallel(
                module,
                # pyre-ignore [16]
                device_ids=None if self._device.type == "cpu" else [self._device],
                process_group=self._pg,
                gradient_as_bucket_view=True,
                broadcast_buffers=False,
            ),
        )

    def create_pooled_output_dist(self) -> DpPooledEmbeddingDist:
        return DpPooledEmbeddingDist()

    def create_sequence_output_dist(self) -> DpSequenceEmbeddingDist:
        return DpSequenceEmbeddingDist()

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
