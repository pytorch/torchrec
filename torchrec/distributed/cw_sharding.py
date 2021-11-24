#!/usr/bin/env python3

from typing import List, Optional, Tuple, cast

import torch
import torch.distributed as dist
from torchrec.distributed.embedding_types import (
    ShardedEmbeddingTable,
    EmbeddingComputeKernel,
)
from torchrec.distributed.tw_sharding import TwEmbeddingSharding
from torchrec.distributed.types import (
    ShardMetadata,
    ParameterSharding,
)
from torchrec.modules.embedding_configs import EmbeddingTableConfig


class CwEmbeddingSharding(TwEmbeddingSharding):
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
    ) -> None:
        super().__init__(embedding_configs, pg, device)

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
            shards: List[ShardMetadata] = config[1].sharding_spec.shards
            placed_ranks = cast(List[int], config[1].ranks)

            for rank in range(world_size):
                table = ShardedEmbeddingTable(
                    num_embeddings=config[0].num_embeddings,
                    embedding_dim=config[0].embedding_dim,
                    name=config[0].name,
                    embedding_names=[],
                    data_type=config[0].data_type,
                    feature_names=[],
                    pooling=config[0].pooling,
                    is_weighted=config[0].is_weighted,
                    has_feature_processor=config[0].has_feature_processor,
                    block_size=config[1].block_size,
                    num_shards=len(placed_ranks),
                    compute_kernel=EmbeddingComputeKernel(config[1].compute_kernel),
                )

                if rank in placed_ranks:
                    shard_idx = placed_ranks.index(rank)
                    table.embedding_names = config[0].embedding_names
                    table.feature_names = config[0].feature_names
                    table.local_rows = config[0].num_embeddings
                    table.local_cols = shards[shard_idx].shard_sizes[1]
                    table.local_metadata = shards[shard_idx]
                    table.shard_idx = shard_idx

                tables_per_rank[rank].append(table)

        return tables_per_rank
