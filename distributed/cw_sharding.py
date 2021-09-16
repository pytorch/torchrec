#!/usr/bin/env python3

from typing import List, Optional

import torch
import torch.distributed as dist
from torchrec.distributed.embedding_types import (
    ShardedEmbeddingTable,
    ShardedEmbeddingTableShard,
)
from torchrec.distributed.tw_sharding import TwEmbeddingSharding
from torchrec.distributed.types import (
    ShardedTensorMetadata,
    ShardMetadata,
)


class CwEmbeddingSharding(TwEmbeddingSharding):
    """
    Shards embedding bags table-wise, i.e.. a given embedding table is entirely placed on a selected rank.
    """

    def __init__(
        self,
        sharded_tables: List[ShardedEmbeddingTable],
        pg: dist.ProcessGroup,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(sharded_tables, pg, device)

    def _shard(
        self, tables: List[ShardedEmbeddingTable]
    ) -> List[List[ShardedEmbeddingTableShard]]:
        world_size = self._pg.size()
        tables_per_rank: List[List[ShardedEmbeddingTableShard]] = [
            [] for i in range(world_size)
        ]
        for table in tables:
            # pyre-fixme [16]
            shards: List[ShardMetadata] = table.sharding_spec.shards

            # construct the global sharded_tensor_metadata
            global_metadata = ShardedTensorMetadata(
                shards_metadata=shards,
                size=torch.Size([table.num_embeddings, table.embedding_dim]),
            )
            # pyre-fixme [6]
            for i, rank in enumerate(table.ranks):
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
                        local_cols=shards[i].shard_lengths[1],
                        local_metadata=shards[i],
                        global_metadata=global_metadata,
                    )
                )
        return tables_per_rank
