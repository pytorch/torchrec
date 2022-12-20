#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, TypeVar

import torch
import torch.distributed as dist
from torchrec.distributed.dist_data import KJTAllToAll, PooledEmbeddingsReduceScatter
from torchrec.distributed.embedding_lookup import GroupedPooledEmbeddingsLookup
from torchrec.distributed.embedding_sharding import (
    BaseEmbeddingDist,
    BaseEmbeddingLookup,
    BaseSparseFeaturesDist,
    bucketize_kjt_before_all2all,
    EmbeddingSharding,
    EmbeddingShardingContext,
    EmbeddingShardingInfo,
    group_tables,
)
from torchrec.distributed.embedding_types import (
    BaseGroupedFeatureProcessor,
    EmbeddingComputeKernel,
    GroupedEmbeddingConfig,
    ShardedEmbeddingTable,
)
from torchrec.distributed.types import (
    Awaitable,
    CommOp,
    QuantizedCommCodecs,
    ShardedTensorMetadata,
    ShardingEnv,
    ShardMetadata,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.streamable import Multistreamable


C = TypeVar("C", bound=Multistreamable)
F = TypeVar("F", bound=Multistreamable)
T = TypeVar("T")
W = TypeVar("W")


class BaseRwEmbeddingSharding(EmbeddingSharding[C, F, T, W]):
    """
    Base class for row-wise sharding.
    """

    def __init__(
        self,
        sharding_infos: List[EmbeddingShardingInfo],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        need_pos: bool = False,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__(
            qcomm_codecs_registry=qcomm_codecs_registry,
        )

        self._env = env
        self._pg: Optional[dist.ProcessGroup] = self._env.process_group
        self._world_size: int = self._env.world_size
        self._rank: int = self._env.rank
        if device is None:
            device = torch.device("cpu")
        self._device = device
        sharded_tables_per_rank = self._shard(sharding_infos)
        self._need_pos = need_pos
        self._grouped_embedding_configs_per_rank: List[
            List[GroupedEmbeddingConfig]
        ] = []
        self._grouped_embedding_configs_per_rank = group_tables(sharded_tables_per_rank)
        self._grouped_embedding_configs: List[
            GroupedEmbeddingConfig
        ] = self._grouped_embedding_configs_per_rank[self._rank]

        self._has_feature_processor: bool = False
        for group_config in self._grouped_embedding_configs:
            if group_config.has_feature_processor:
                self._has_feature_processor = True

    def _shard(
        self,
        sharding_infos: List[EmbeddingShardingInfo],
    ) -> List[List[ShardedEmbeddingTable]]:
        tables_per_rank: List[List[ShardedEmbeddingTable]] = [
            [] for i in range(self._world_size)
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

            for rank in range(self._world_size):
                tables_per_rank[rank].append(
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
                        local_rows=shards[rank].shard_sizes[0],
                        local_cols=info.embedding_config.embedding_dim,
                        compute_kernel=EmbeddingComputeKernel(
                            info.param_sharding.compute_kernel
                        ),
                        local_metadata=shards[rank],
                        global_metadata=global_metadata,
                        weight_init_max=info.embedding_config.weight_init_max,
                        weight_init_min=info.embedding_config.weight_init_min,
                        fused_params=info.fused_params,
                    )
                )
        return tables_per_rank

    def embedding_dims(self) -> List[int]:
        embedding_dims = []
        for grouped_config in self._grouped_embedding_configs:
            embedding_dims.extend(grouped_config.embedding_dims())
        return embedding_dims

    def embedding_names(self) -> List[str]:
        embedding_names = []
        for grouped_config in self._grouped_embedding_configs:
            embedding_names.extend(grouped_config.embedding_names())
        return embedding_names

    def embedding_names_per_rank(self) -> List[List[str]]:
        raise NotImplementedError

    def embedding_shard_metadata(self) -> List[Optional[ShardMetadata]]:
        embedding_shard_metadata = []
        for grouped_config in self._grouped_embedding_configs:
            embedding_shard_metadata.extend(grouped_config.embedding_shard_metadata())
        return embedding_shard_metadata

    def feature_names(self) -> List[str]:
        feature_names = []
        for grouped_config in self._grouped_embedding_configs:
            feature_names.extend(grouped_config.feature_names())
        return feature_names

    def _get_num_features(self) -> int:
        return sum(
            group_config.num_features()
            for group_config in self._grouped_embedding_configs
        )

    def _get_feature_hash_sizes(self) -> List[int]:
        feature_hash_sizes: List[int] = []
        for group_config in self._grouped_embedding_configs:
            feature_hash_sizes.extend(group_config.feature_hash_sizes())
        return feature_hash_sizes


class RwSparseFeaturesDist(BaseSparseFeaturesDist[KeyedJaggedTensor]):
    """
    Bucketizes sparse features in RW fashion and then redistributes with an AlltoAll
    collective operation.

    Args:
        pg (dist.ProcessGroup): ProcessGroup for AlltoAll communication.
        intra_pg (dist.ProcessGroup): ProcessGroup within single host group for AlltoAll
            communication.
        num_features (int): total number of features.
        feature_hash_sizes (List[int]): hash sizes of features.
        device (Optional[torch.device]): device on which buffers will be allocated.
        is_sequence (bool): if this is for a sequence embedding.
        has_feature_processor (bool): existence of feature processor (ie. position
            weighted features).

    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
        num_features: int,
        feature_hash_sizes: List[int],
        device: Optional[torch.device] = None,
        is_sequence: bool = False,
        has_feature_processor: bool = False,
        need_pos: bool = False,
    ) -> None:
        super().__init__()
        self._world_size: int = pg.size()
        self._num_features = num_features
        feature_block_sizes = [
            (hash_size + self._world_size - 1) // self._world_size
            for hash_size in feature_hash_sizes
        ]
        self.register_buffer(
            "_feature_block_sizes_tensor",
            torch.tensor(
                feature_block_sizes,
                device=device,
                dtype=torch.int32,
            ),
        )
        self._dist = KJTAllToAll(
            pg=pg,
            splits=self._world_size * [self._num_features],
            device=device,
        )
        self._is_sequence = is_sequence
        self._has_feature_processor = has_feature_processor
        self._need_pos = need_pos
        self.unbucketize_permute_tensor: Optional[torch.Tensor] = None

    def forward(
        self,
        sparse_features: KeyedJaggedTensor,
    ) -> Awaitable[Awaitable[KeyedJaggedTensor]]:
        """
        Bucketizes sparse feature values into world size number of buckets and then
        performs AlltoAll operation.

        Args:
            sparse_features (KeyedJaggedTensor): sparse features to bucketize and
                redistribute.

        Returns:
            Awaitable[Awaitable[KeyedJaggedTensor]]: awaitable of awaitable of KeyedJaggedTensor.
        """

        (
            bucketized_features,
            self.unbucketize_permute_tensor,
        ) = bucketize_kjt_before_all2all(
            sparse_features,
            num_buckets=self._world_size,
            block_sizes=self._feature_block_sizes_tensor,
            output_permute=self._is_sequence,
            bucketize_pos=self._has_feature_processor
            if sparse_features.weights_or_none() is None
            else self._need_pos,
        )

        return self._dist(bucketized_features)


class RwPooledEmbeddingDist(
    BaseEmbeddingDist[EmbeddingShardingContext, torch.Tensor, torch.Tensor]
):
    """
    Redistributes pooled embedding tensor in RW fashion by performing a reduce-scatter
    operation.

    Args:
        pg (dist.ProcessGroup): ProcessGroup for reduce-scatter communication.
    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__()

        self._dist = PooledEmbeddingsReduceScatter(
            pg,
            codecs=qcomm_codecs_registry.get(
                CommOp.POOLED_EMBEDDINGS_REDUCE_SCATTER.name, None
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
        Performs reduce-scatter pooled operation on pooled embeddings tensor.

        Args:
            local_embs (torch.Tensor): pooled embeddings tensor to distribute.

        Returns:
            Awaitable[torch.Tensor]: awaitable of pooled embeddings tensor.
        """

        if sharding_ctx is None:
            return self._dist(local_embs)
        else:
            return self._dist(local_embs, input_splits=sharding_ctx.batch_size_per_rank)


class RwPooledEmbeddingSharding(
    BaseRwEmbeddingSharding[
        EmbeddingShardingContext, KeyedJaggedTensor, torch.Tensor, torch.Tensor
    ]
):
    """
    Shards embedding bags row-wise, i.e.. a given embedding table is evenly distributed
    by rows and table slices are placed on all ranks.
    """

    def create_input_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseSparseFeaturesDist[KeyedJaggedTensor]:
        num_features = self._get_num_features()
        feature_hash_sizes = self._get_feature_hash_sizes()
        return RwSparseFeaturesDist(
            # pyre-fixme[6]: For 1st param expected `ProcessGroup` but got
            #  `Optional[ProcessGroup]`.
            pg=self._pg,
            num_features=num_features,
            feature_hash_sizes=feature_hash_sizes,
            device=device if device is not None else self._device,
            is_sequence=False,
            has_feature_processor=self._has_feature_processor,
            need_pos=self._need_pos,
        )

    def create_lookup(
        self,
        device: Optional[torch.device] = None,
        fused_params: Optional[Dict[str, Any]] = None,
        feature_processor: Optional[BaseGroupedFeatureProcessor] = None,
    ) -> BaseEmbeddingLookup:
        return GroupedPooledEmbeddingsLookup(
            grouped_configs=self._grouped_embedding_configs,
            pg=self._pg,
            device=device if device is not None else self._device,
            feature_processor=feature_processor,
        )

    def create_output_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseEmbeddingDist[EmbeddingShardingContext, torch.Tensor, torch.Tensor]:
        return RwPooledEmbeddingDist(
            # pyre-fixme[6]: For 1st param expected `ProcessGroup` but got
            #  `Optional[ProcessGroup]`.
            self._pg,
            qcomm_codecs_registry=self.qcomm_codecs_registry,
        )
