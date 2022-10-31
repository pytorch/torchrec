#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, List, Optional, TypeVar

import torch
import torch.distributed as dist
from torchrec.distributed.dist_data import EmbeddingsAllToOne, PooledEmbeddingsAllToAll
from torchrec.distributed.embedding_lookup import (
    GroupedPooledEmbeddingsLookup,
    InferGroupedPooledEmbeddingsLookup,
)
from torchrec.distributed.embedding_sharding import (
    BaseEmbeddingDist,
    BaseEmbeddingLookup,
    BaseSparseFeaturesDist,
    EmbeddingSharding,
    EmbeddingShardingContext,
    EmbeddingShardingInfo,
    group_tables,
    NullShardingContext,
    SparseFeaturesAllToAll,
    SparseFeaturesOneToAll,
)
from torchrec.distributed.embedding_types import (
    BaseGroupedFeatureProcessor,
    EmbeddingComputeKernel,
    GroupedEmbeddingConfig,
    ShardedEmbeddingTable,
    SparseFeatures,
    SparseFeaturesList,
)
from torchrec.distributed.types import (
    Awaitable,
    CommOp,
    NoWait,
    QuantizedCommCodecs,
    ShardedTensorMetadata,
    ShardingEnv,
    ShardMetadata,
)
from torchrec.streamable import Multistreamable


C = TypeVar("C", bound=Multistreamable)
F = TypeVar("F", bound=Multistreamable)
T = TypeVar("T")
W = TypeVar("W")


class BaseTwEmbeddingSharding(EmbeddingSharding[C, F, T, W]):
    """
    Base class for table wise sharding.
    """

    def __init__(
        self,
        sharding_infos: List[EmbeddingShardingInfo],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
        variable_batch_size: bool = False,
    ) -> None:
        super().__init__(qcomm_codecs_registry=qcomm_codecs_registry)
        self._env = env
        self._device = device
        self._pg: Optional[dist.ProcessGroup] = self._env.process_group
        self._world_size: int = self._env.world_size
        self._rank: int = self._env.rank
        sharded_tables_per_rank = self._shard(sharding_infos)
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
        self._variable_batch_size = variable_batch_size

    def _shard(
        self,
        sharding_infos: List[EmbeddingShardingInfo],
    ) -> List[List[ShardedEmbeddingTable]]:
        world_size = self._world_size
        tables_per_rank: List[List[ShardedEmbeddingTable]] = [
            [] for i in range(world_size)
        ]
        for info in sharding_infos:
            # pyre-fixme [16]
            shards = info.param_sharding.sharding_spec.shards
            # construct the global sharded_tensor_metadata
            global_metadata = ShardedTensorMetadata(
                shards_metadata=shards,
                size=torch.Size(
                    [
                        info.embedding_config.num_embeddings,
                        info.embedding_config.embedding_dim,
                    ]
                ),
            )

            # pyre-fixme [16]
            tables_per_rank[info.param_sharding.ranks[0]].append(
                ShardedEmbeddingTable(
                    num_embeddings=info.embedding_config.num_embeddings,
                    embedding_dim=info.embedding_config.embedding_dim,
                    name=info.embedding_config.name,
                    embedding_names=info.embedding_config.embedding_names,
                    data_type=info.embedding_config.data_type,
                    feature_names=info.embedding_config.feature_names,
                    pooling=info.embedding_config.pooling,
                    is_weighted=info.embedding_config.is_weighted,
                    has_feature_processor=info.embedding_config.has_feature_processor,
                    local_rows=info.embedding_config.num_embeddings,
                    local_cols=info.embedding_config.embedding_dim,
                    compute_kernel=EmbeddingComputeKernel(
                        info.param_sharding.compute_kernel
                    ),
                    local_metadata=shards[0],
                    global_metadata=global_metadata,
                    weight_init_max=info.embedding_config.weight_init_max,
                    weight_init_min=info.embedding_config.weight_init_min,
                    fused_params=info.fused_params,
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

    def embedding_names_per_rank(self) -> List[List[str]]:
        embedding_names = []
        for grouped_embedding_configs, score_grouped_embedding_configs in zip(
            self._grouped_embedding_configs_per_rank,
            self._score_grouped_embedding_configs_per_rank,
        ):
            embedding_names_per_rank = []
            for grouped_config in grouped_embedding_configs:
                embedding_names_per_rank.extend(grouped_config.embedding_names())
            for grouped_config in score_grouped_embedding_configs:
                embedding_names_per_rank.extend(grouped_config.embedding_names())
            embedding_names.append(embedding_names_per_rank)
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

    def id_list_feature_names_per_rank(self) -> List[List[str]]:
        id_list_feature_names = []
        for grouped_embedding_configs in self._grouped_embedding_configs_per_rank:
            id_list_feature_names_per_rank = []
            for grouped_config in grouped_embedding_configs:
                id_list_feature_names_per_rank.extend(grouped_config.feature_names())
            id_list_feature_names.append(id_list_feature_names_per_rank)
        return id_list_feature_names

    def id_score_list_feature_names_per_rank(self) -> List[List[str]]:
        id_score_list_feature_names = []
        for (
            score_grouped_embedding_configs
        ) in self._score_grouped_embedding_configs_per_rank:
            id_score_list_feature_names_per_rank = []
            for grouped_config in score_grouped_embedding_configs:
                id_score_list_feature_names_per_rank.extend(
                    grouped_config.feature_names()
                )
            id_score_list_feature_names.append(id_score_list_feature_names_per_rank)
        return id_score_list_feature_names

    def id_list_features_per_rank(self) -> List[int]:
        id_list_features_per_rank = []
        for grouped_embedding_configs in self._grouped_embedding_configs_per_rank:
            num_features = 0
            for grouped_config in grouped_embedding_configs:
                num_features += grouped_config.num_features()
            id_list_features_per_rank.append(num_features)
        return id_list_features_per_rank

    def id_score_list_features_per_rank(self) -> List[int]:
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
    Redistributes sparse features with an AlltoAll collective operation for table wise
    sharding.

    Args:
        pg (dist.ProcessGroup): ProcessGroup for AlltoAll communication.
        id_list_features_per_rank (List[int]): number of id list features to send to
            each rank.
        id_score_list_features_per_rank (List[int]): number of id score list features to
            send to each rank.
        device (Optional[torch.device]): device on which buffers will be allocated.
    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
        id_list_features_per_rank: List[int],
        id_score_list_features_per_rank: List[int],
        device: Optional[torch.device] = None,
        variable_batch_size: bool = False,
    ) -> None:
        super().__init__()
        self._dist = SparseFeaturesAllToAll(
            pg=pg,
            id_list_features_per_rank=id_list_features_per_rank,
            id_score_list_features_per_rank=id_score_list_features_per_rank,
            device=device,
            variable_batch_size=variable_batch_size,
        )

    def forward(
        self,
        sparse_features: SparseFeatures,
    ) -> Awaitable[Awaitable[SparseFeatures]]:
        """
        Performs AlltoAll operation on sparse features.

        Args:
            sparse_features (SparseFeatures): sparse features to redistribute.

        Returns:
            Awaitable[Awaitable[SparseFeatures]]: awaitable of awaitable of SparseFeatures.
        """

        return self._dist(sparse_features)


class TwPooledEmbeddingDist(
    BaseEmbeddingDist[EmbeddingShardingContext, torch.Tensor, torch.Tensor]
):
    """
    Redistributes pooled embedding tensor with an AlltoAll collective operation for
    table wise sharding.

    Args:
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
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__()
        self._dist = PooledEmbeddingsAllToAll(
            pg=pg,
            dim_sum_per_rank=dim_sum_per_rank,
            device=device,
            callbacks=callbacks,
            codecs=qcomm_codecs_registry.get(
                CommOp.POOLED_EMBEDDINGS_ALL_TO_ALL.name, None
            )
            if qcomm_codecs_registry
            else None,
        )

    def forward(
        self,
        local_embs: torch.Tensor,
        sharding_ctx: Optional[EmbeddingShardingContext] = None,
    ) -> Awaitable[torch.Tensor]:
        """
        Performs AlltoAll operation on pooled embeddings tensor.

        Args:
            local_embs (torch.Tensor): tensor of values to distribute.

        Returns:
            Awaitable[torch.Tensor]: awaitable of pooled embeddings.
        """
        if sharding_ctx is None:
            return self._dist(local_embs)
        else:
            return self._dist(
                local_embs, batch_size_per_rank=sharding_ctx.batch_size_per_rank
            )


class TwPooledEmbeddingSharding(
    BaseTwEmbeddingSharding[
        EmbeddingShardingContext, SparseFeatures, torch.Tensor, torch.Tensor
    ]
):
    """
    Shards embedding bags table-wise, i.e.. a given embedding table is entirely placed
    on a selected rank.
    """

    def create_input_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseSparseFeaturesDist[SparseFeatures]:
        assert self._pg is not None
        return TwSparseFeaturesDist(
            self._pg,
            self.id_list_features_per_rank(),
            self.id_score_list_features_per_rank(),
            device if device is not None else self._device,
            self._variable_batch_size,
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
            pg=self._pg,
            device=device if device is not None else self._device,
            feature_processor=feature_processor,
        )

    def create_output_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseEmbeddingDist[EmbeddingShardingContext, torch.Tensor, torch.Tensor]:
        assert self._pg is not None
        return TwPooledEmbeddingDist(
            self._pg,
            self._dim_sum_per_rank(),
            device if device is not None else self._device,
            qcomm_codecs_registry=self.qcomm_codecs_registry,
        )


class InferTwSparseFeaturesDist(BaseSparseFeaturesDist[SparseFeaturesList]):
    """
    Redistributes sparse features to all devices for inference.

    Args:
        id_list_features_per_rank (List[int]): number of id list features to send
            to each rank.
        id_score_list_features_per_rank (List[int]): number of id score list features
            to send to each rank.
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

        Args:
            sparse_features (SparseFeatures): sparse features to redistribute.

        Returns:
            Awaitable[Awaitable[SparseFeatures]]: awaitable of awaitable of SparseFeatures.
        """

        return NoWait(self._dist.forward(sparse_features))


class InferTwPooledEmbeddingDist(
    BaseEmbeddingDist[NullShardingContext, List[torch.Tensor], torch.Tensor]
):
    """
    Merges pooled embedding tensor from each device for inference.

    Args:
        device (Optional[torch.device]): device on which buffer will be allocated.
        world_size (int): number of devices in the topology.
    """

    def __init__(
        self,
        device: torch.device,
        world_size: int,
    ) -> None:
        super().__init__()
        self._dist: EmbeddingsAllToOne = EmbeddingsAllToOne(device, world_size, 1)

    def forward(
        self,
        local_embs: List[torch.Tensor],
        sharding_ctx: Optional[NullShardingContext] = None,
    ) -> Awaitable[torch.Tensor]:
        """
        Performs AlltoOne operation on pooled embedding tensors.

        Args:
            local_embs (List[torch.Tensor]): pooled embedding tensors with
                `len(local_embs) == world_size`.

        Returns:
            Awaitable[torch.Tensor]: awaitable of merged pooled embedding tensor.
        """

        return self._dist.forward(local_embs)


class InferTwEmbeddingSharding(
    BaseTwEmbeddingSharding[
        NullShardingContext, SparseFeaturesList, List[torch.Tensor], torch.Tensor
    ]
):
    """
    Shards embedding bags table-wise for inference
    """

    def create_input_dist(
        self, device: Optional[torch.device] = None
    ) -> BaseSparseFeaturesDist[SparseFeaturesList]:
        return InferTwSparseFeaturesDist(
            self.id_list_features_per_rank(),
            self.id_score_list_features_per_rank(),
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
    ) -> BaseEmbeddingDist[NullShardingContext, List[torch.Tensor], torch.Tensor]:
        device = device if device is not None else self._device
        assert device is not None
        return InferTwPooledEmbeddingDist(
            device,
            self._world_size,
        )
