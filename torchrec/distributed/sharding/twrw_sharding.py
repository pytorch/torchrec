#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import math
from typing import List, Optional, Dict, Any, Tuple, cast, TypeVar

import torch
import torch.distributed as dist
from torchrec.distributed.comm import (
    intra_and_cross_node_pg,
    get_local_size,
)
from torchrec.distributed.dist_data import (
    PooledEmbeddingsReduceScatter,
    PooledEmbeddingsAllToAll,
)
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
    GroupedEmbeddingConfig,
    SparseFeatures,
    ShardedEmbeddingTable,
    EmbeddingComputeKernel,
    BaseGroupedFeatureProcessor,
)
from torchrec.distributed.types import (
    ShardingEnv,
    ShardedTensorMetadata,
    Awaitable,
    ParameterSharding,
    ShardMetadata,
)
from torchrec.modules.embedding_configs import EmbeddingTableConfig
from torchrec.streamable import Multistreamable

F = TypeVar("F", bound=Multistreamable)
T = TypeVar("T")


class BaseTwRwEmbeddingSharding(EmbeddingSharding[F, T]):
    """
    base class for table-wise-row-wise sharding
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
        self._device = device
        intra_pg, cross_pg = intra_and_cross_node_pg(device)
        self._intra_pg: Optional[dist.ProcessGroup] = intra_pg
        self._cross_pg: Optional[dist.ProcessGroup] = cross_pg
        self._local_size: int = (
            intra_pg.size() if intra_pg else get_local_size(self._world_size)
        )

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
            self._rank // self._local_size
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
                        local_rows=shards[rank_idx].shard_sizes[0],
                        local_cols=config[0].embedding_dim,
                        compute_kernel=EmbeddingComputeKernel(config[1].compute_kernel),
                        local_metadata=shards[rank_idx],
                        global_metadata=global_metadata,
                        weight_init_max=config[0].weight_init_max,
                        weight_init_min=config[0].weight_init_min,
                    )
                )

        return tables_per_rank

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


class TwRwSparseFeaturesDist(BaseSparseFeaturesDist[SparseFeatures]):
    """
    Bucketizes sparse features in TWRW fashion and then redistributes with an AlltoAll
    collective operation.

    Args:
        pg (dist.ProcessGroup): ProcessGroup for AlltoAll communication.
        intra_pg (dist.ProcessGroup): ProcessGroup within single host group for AlltoAll
        communication.
        id_list_features_per_rank (List[int]): number of id list features to send to
        each rank.
        id_score_list_features_per_rank (List[int]): number of id score list features to
        send to each rank
        id_list_feature_hash_sizes (List[int]): hash sizes of id list features.
        id_score_list_feature_hash_sizes (List[int]): hash sizes of id score list features.
        device (Optional[torch.device]): device on which buffers will be allocated.
        has_feature_processor (bool): existence of feature processor (ie. position
        weighted features).

    Example::

        3 features
        2 hosts with 2 devices each

        Bucketize each feature into 2 buckets
        Staggered shuffle with feature splits [2, 1]
        AlltoAll operation

        NOTE: result of staggered shuffle and AlltoAll operation look the same after
        reordering in AlltoAll

        Result:
            host 0 device 0:
                feature 0 bucket 0
                feature 1 bucket 0

            host 0 device 1:
                feature 0 bucket 1
                feature 1 bucket 1

            host 1 device 0:
                feature 2 bucket 0

            host 1 device 1:
                feature 2 bucket 1
    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
        intra_pg: dist.ProcessGroup,
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
            "_id_list_sf_staggered_shuffle_tensor",
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
            pg=pg,
            id_list_features_per_rank=id_list_features_per_rank,
            id_score_list_features_per_rank=id_score_list_features_per_rank,
            device=device,
            stagger=self._num_cross_nodes,
        )
        self._has_feature_processor = has_feature_processor

    def forward(
        self,
        sparse_features: SparseFeatures,
    ) -> Awaitable[Awaitable[SparseFeatures]]:
        """
        Bucketizes sparse feature values into local world size number of buckets,
        performs staggered shuffle on the sparse features, and then performs AlltoAll
        operation.

        Call Args:
            sparse_features (SparseFeatures): sparse features to bucketize and
                redistribute.

        Returns:
            Awaitable[SparseFeatures]: awaitable of SparseFeatures.
        """

        bucketized_sparse_features = SparseFeatures(
            id_list_features=bucketize_kjt_before_all2all(
                sparse_features.id_list_features,
                num_buckets=self._local_size,
                block_sizes=self._id_list_feature_block_sizes_tensor,
                output_permute=False,
                bucketize_pos=self._has_feature_processor,
            )[0].permute(
                self._id_list_sf_staggered_shuffle,
                self._id_list_sf_staggered_shuffle_tensor,
            )
            if sparse_features.id_list_features is not None
            else None,
            id_score_list_features=bucketize_kjt_before_all2all(
                sparse_features.id_score_list_features,
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
        """
        Reorders sparse data such that data is in contiguous blocks and correctly ordered
        for global TWRW layout.
        """

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


class TwRwPooledEmbeddingDist(BaseEmbeddingDist[torch.Tensor]):
    """
    Redistributes pooled embedding tensor in TWRW fashion by performing a reduce-scatter
    operation row wise on the host level and then an AlltoAll operation table wise on
    the global level.

    Args:
        cross_pg (dist.ProcessGroup): global level ProcessGroup for AlltoAll
            communication.
        intra_pg (dist.ProcessGroup): host level ProcessGroup for reduce-scatter
            communication.
        dim_sum_per_node (List[int]): number of features (sum of dimensions) of the
            embedding for each host.
        device (Optional[torch.device]): device on which buffers will be allocated.
    """

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
        """
        Performs reduce-scatter pooled operation on pooled embeddings tensor followed by
        AlltoAll pooled operation.

        Call Args:
            local_embs (torch.Tensor): pooled embeddings tensor to distribute.

        Returns:
            Awaitable[torch.Tensor]: awaitable of pooled embeddings tensor.
        """

        return self._cross_dist(self._intra_dist(local_embs).wait())


class TwRwPooledEmbeddingSharding(
    BaseTwRwEmbeddingSharding[SparseFeatures, torch.Tensor]
):
    """
    Shards embedding bags table-wise then row-wise.
    """

    def create_input_dist(
        self, device: Optional[torch.device] = None
    ) -> BaseSparseFeaturesDist[SparseFeatures]:
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
            intra_pg=cast(dist.ProcessGroup, self._intra_pg),
            id_list_features_per_rank=id_list_features_per_rank,
            id_score_list_features_per_rank=id_score_list_features_per_rank,
            id_list_feature_hash_sizes=id_list_feature_hash_sizes,
            id_score_list_feature_hash_sizes=id_score_list_feature_hash_sizes,
            device=device if device is not None else self._device,
            has_feature_processor=self._has_feature_processor,
        )

    def create_lookup(
        self,
        device: Optional[torch.device] = None,
        fused_params: Optional[Dict[str, Any]] = None,
        feature_processor: Optional[BaseGroupedFeatureProcessor] = None,
    ) -> BaseEmbeddingLookup:
        return GroupedPooledEmbeddingsLookup(
            grouped_configs=self._grouped_embedding_configs_per_rank[self._rank],
            grouped_score_configs=self._score_grouped_embedding_configs_per_rank[
                self._rank
            ],
            fused_params=fused_params,
            pg=self._pg,
            device=device if device is not None else self._device,
            feature_processor=feature_processor,
        )

    def create_output_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseEmbeddingDist[torch.Tensor]:
        return TwRwPooledEmbeddingDist(
            cross_pg=cast(dist.ProcessGroup, self._cross_pg),
            intra_pg=cast(dist.ProcessGroup, self._intra_pg),
            dim_sum_per_node=self._dim_sum_per_node(),
            device=device if device is not None else self._device,
        )
