#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, List, Optional, Any, Dict, Tuple, TypeVar

import torch
import torch.distributed as dist
from torchrec.distributed.dist_data import (
    PooledEmbeddingsAllToOne,
    PooledEmbeddingsAllToAll,
)
from torchrec.distributed.embedding_lookup import (
    GroupedPooledEmbeddingsLookup,
    InferGroupedPooledEmbeddingsLookup,
)
from torchrec.distributed.embedding_sharding import (
    EmbeddingSharding,
    SparseFeaturesAllToAll,
    SparseFeaturesOneToAll,
    group_tables,
    BaseEmbeddingDist,
    BaseSparseFeaturesDist,
    BaseEmbeddingLookup,
)
from torchrec.distributed.embedding_types import (
    SparseFeaturesList,
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
    NoWait,
    ParameterSharding,
    ShardMetadata,
)
from torchrec.modules.embedding_configs import EmbeddingTableConfig
from torchrec.streamable import Multistreamable


F = TypeVar("F", bound=Multistreamable)
T = TypeVar("T")


class BaseTwEmbeddingSharding(EmbeddingSharding[F, T]):
    """
    base class for table-wise sharding
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
        self._device = device
        # pyre-ignore[11]
        self._pg: Optional[dist.ProcessGroup] = self._env.process_group
        self._world_size: int = self._env.world_size
        self._rank: int = self._env.rank
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

    def _shard(
        self,
        embedding_configs: List[
            Tuple[EmbeddingTableConfig, ParameterSharding, torch.Tensor]
        ],
    ) -> List[List[ShardedEmbeddingTable]]:
        world_size = self._world_size
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


class TwSparseFeaturesDist(BaseSparseFeaturesDist[SparseFeatures]):
    """
    Redistributes sparse features in hierarchical fashion with an AlltoAll collective
    operation.

    Constructor Args:
        pg (dist.ProcessGroup): ProcessGroup for AlltoAll communication.
        id_list_features_per_rank (List[int]): number of id list features to send to
            each rank.
        id_score_list_features_per_rank (List[int]): number of id score list features to
            send to each rank
        device (Optional[torch.device]): device on which buffers will be allocated.
    """

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
    ) -> Awaitable[Awaitable[SparseFeatures]]:
        """
        Performs AlltoAll operation on sparse features.

        Call Args:
            sparse_features (SparseFeatures): sparse features to redistribute.

        Returns:
            Awaitable[Awaitable[SparseFeatures]]: awaitable of awaitable of
                SparseFeatures.
        """

        return self._dist(sparse_features)


class TwPooledEmbeddingDist(BaseEmbeddingDist[torch.Tensor]):
    """
    Redistributes pooled embedding tensor in hierarchical fashion with an AlltoAll
    collective operation.

    Constructor Args:
        pg (dist.ProcessGroup): ProcessGroup for AlltoAll communication.
        dim_sum_per_rank (List[int]): number of features (sum of dimensions) of the
            embedding in each rank.
        device (Optional[torch.device]): device on which buffers will be allocated.
    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
        dim_sum_per_rank: List[int],
        device: Optional[torch.device] = None,
        callbacks: Optional[List[Callable[[torch.Tensor], torch.Tensor]]] = None,
    ) -> None:
        super().__init__()
        self._dist = PooledEmbeddingsAllToAll(pg, dim_sum_per_rank, device, callbacks)

    def forward(
        self,
        local_embs: torch.Tensor,
    ) -> Awaitable[torch.Tensor]:
        """
        Performs AlltoAll operation on pooled embeddings tensor.

        Call Args:
            local_embs (torch.Tensor): tensor of values to distribute.

        Returns:
            Awaitable[torch.Tensor]: awaitable of pooled embeddings.
        """

        return self._dist(local_embs)


class TwPooledEmbeddingSharding(BaseTwEmbeddingSharding[SparseFeatures, torch.Tensor]):
    """
    Shards embedding bags table-wise, i.e.. a given embedding table is entirely placed
    on a selected rank.
    """

    def create_input_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseSparseFeaturesDist[SparseFeatures]:
        return TwSparseFeaturesDist(
            self._pg,
            self._id_list_features_per_rank(),
            self._id_score_list_features_per_rank(),
            device if device is not None else self._device,
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
        return TwPooledEmbeddingDist(
            self._pg,
            self._dim_sum_per_rank(),
            device if device is not None else self._device,
        )


class InferTwSparseFeaturesDist(BaseSparseFeaturesDist[SparseFeaturesList]):
    """
    Redistributes sparse features to all devices for inference.

    Constructor Args:
        id_list_features_per_rank (List[int]): number of id list features to send to
            each rank.
        id_score_list_features_per_rank (List[int]): number of id score list features to
            send to each rank
        world_size (int): number of devices in the topology.
    """

    def __init__(
        self,
        id_list_features_per_rank: List[int],
        id_score_list_features_per_rank: List[int],
        world_size: int,
    ) -> None:
        super().__init__()
        self._dist: SparseFeaturesOneToAll = SparseFeaturesOneToAll(
            id_list_features_per_rank,
            id_score_list_features_per_rank,
            world_size,
        )

    def forward(
        self,
        sparse_features: SparseFeatures,
    ) -> Awaitable[Awaitable[SparseFeaturesList]]:
        """
        Performs OnetoAll operation on sparse features.

        Call Args:
            sparse_features (SparseFeatures): sparse features to redistribute.

        Returns:
            Awaitable[Awaitable[SparseFeatures]]: awaitable of awaitable of
                SparseFeatures.
        """

        return NoWait(self._dist.forward(sparse_features))


class InferTwPooledEmbeddingDist(BaseEmbeddingDist[List[torch.Tensor]]):
    """
    Merges pooled embedding tensor from each device for inference.

    Constructor Args:
        device (Optional[torch.device]): device on which buffer will be allocated.
        world_size (int): number of devices in the topology.
    """

    def __init__(
        self,
        device: torch.device,
        world_size: int,
    ) -> None:
        super().__init__()
        self._dist: PooledEmbeddingsAllToOne = PooledEmbeddingsAllToOne(
            device,
            world_size,
        )

    def forward(
        self,
        local_embs: List[torch.Tensor],
    ) -> Awaitable[torch.Tensor]:
        """
        Performs AlltoOne operation on pooled embedding tensors.

        Call Args:
            local_embs (List[torch.Tensor]): pooled embedding tensors with
                len(local_embs) == world_size.

        Returns:
            Awaitable[torch.Tensor]: awaitable of merged pooled embedding tensor.
        """

        return self._dist.forward(local_embs)


class InferTwEmbeddingSharding(
    BaseTwEmbeddingSharding[SparseFeaturesList, List[torch.Tensor]]
):
    """
    Shards embedding bags table-wise for inference
    """

    def create_input_dist(
        self, device: Optional[torch.device] = None
    ) -> BaseSparseFeaturesDist[SparseFeaturesList]:
        return InferTwSparseFeaturesDist(
            self._id_list_features_per_rank(),
            self._id_score_list_features_per_rank(),
            self._world_size,
        )

    def create_lookup(
        self,
        device: Optional[torch.device] = None,
        fused_params: Optional[Dict[str, Any]] = None,
        feature_processor: Optional[BaseGroupedFeatureProcessor] = None,
    ) -> BaseEmbeddingLookup[SparseFeaturesList, List[torch.Tensor]]:
        return InferGroupedPooledEmbeddingsLookup(
            grouped_configs_per_rank=self._grouped_embedding_configs_per_rank,
            grouped_score_configs_per_rank=self._score_grouped_embedding_configs_per_rank,
            world_size=self._world_size,
            fused_params=fused_params,
        )

    def create_output_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseEmbeddingDist[List[torch.Tensor]]:
        device = device if device is not None else self._device
        return InferTwPooledEmbeddingDist(
            # pyre-fixme [6]
            device,
            self._world_size,
        )
