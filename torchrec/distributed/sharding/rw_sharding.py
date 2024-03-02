#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math
from typing import Any, cast, Dict, List, Optional, Tuple, TypeVar, Union

import torch
import torch.distributed as dist
from torchrec.distributed.dist_data import (
    EmbeddingsAllToOneReduce,
    KJTAllToAll,
    KJTOneToAll,
    PooledEmbeddingsReduceScatter,
    VariableBatchPooledEmbeddingsReduceScatter,
)
from torchrec.distributed.embedding_lookup import (
    GroupedPooledEmbeddingsLookup,
    InferGroupedPooledEmbeddingsLookup,
)
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
    KJTList,
    ShardedEmbeddingTable,
)
from torchrec.distributed.types import (
    Awaitable,
    CommOp,
    NullShardingContext,
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


def get_embedding_shard_metadata(
    grouped_embedding_configs_per_rank: List[List[GroupedEmbeddingConfig]],
) -> Tuple[List[List[int]], bool]:
    is_even_sharding: bool = True
    world_size = len(grouped_embedding_configs_per_rank)

    def get_even_shard_sizes(hash_size: int, world_size: int) -> List[int]:
        block_size: int = math.ceil(hash_size / world_size)
        last_rank: int = hash_size // block_size

        expected_even_shard_sizes = [block_size] * last_rank
        if hash_size % world_size != 0:
            expected_even_shard_sizes.append(hash_size - sum(expected_even_shard_sizes))
        return expected_even_shard_sizes

    embed_sharding = []
    for table in grouped_embedding_configs_per_rank[0][0].embedding_tables:
        embed_sharding_per_feature = []
        total_rows = 0
        sizes = []
        # pyre-ignore [16]: `Optional` has no attribute `shards_metadata`
        for metadata in table.global_metadata.shards_metadata:
            embed_sharding_per_feature.append(metadata.shard_offsets[0])
            total_rows += metadata.shard_sizes[0]
            sizes.append(metadata.shard_sizes[0])
        embed_sharding_per_feature.append(total_rows)
        embed_sharding.extend([embed_sharding_per_feature] * len(table.embedding_names))
        expected_even_sizes = get_even_shard_sizes(total_rows, world_size)
        if sizes != expected_even_sizes:
            is_even_sharding = False

    return (embed_sharding, is_even_sharding)


@torch.fx.wrap
def _fx_wrap_block_bucketize_row_pos(
    block_bucketize_row_pos: List[torch.Tensor],
) -> Optional[List[torch.Tensor]]:
    return block_bucketize_row_pos if block_bucketize_row_pos else None


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
        self._device: torch.device = device
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
        embedding_names = []
        for grouped_embedding_configs in self._grouped_embedding_configs_per_rank:
            embedding_names_per_rank = []
            for grouped_config in grouped_embedding_configs:
                embedding_names_per_rank.extend(grouped_config.embedding_names())
            embedding_names.append(embedding_names_per_rank)
        return embedding_names

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

    def embedding_tables(self) -> List[ShardedEmbeddingTable]:
        embedding_tables = []
        for grouped_config in self._grouped_embedding_configs:
            embedding_tables.extend(grouped_config.embedding_tables)
        return embedding_tables

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
                dtype=torch.int64,
            ),
        )
        self._dist = KJTAllToAll(
            pg=pg,
            splits=[self._num_features] * self._world_size,
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
        embedding_dims: List[int],
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__()

        self._dist: Optional[
            Union[
                PooledEmbeddingsReduceScatter,
                VariableBatchPooledEmbeddingsReduceScatter,
            ]
        ] = None
        self._pg = pg
        self._qcomm_codecs_registry = qcomm_codecs_registry
        self._codecs: Optional[QuantizedCommCodecs] = (
            qcomm_codecs_registry.get(
                CommOp.POOLED_EMBEDDINGS_REDUCE_SCATTER.name, None
            )
            if qcomm_codecs_registry
            else None
        )
        self._embedding_dims = embedding_dims

    def forward(
        self,
        local_embs: torch.Tensor,
        sharding_ctx: Optional[EmbeddingShardingContext] = None,
    ) -> Awaitable[torch.Tensor]:
        """
        Performs reduce-scatter pooled operation on pooled embeddings tensor.

        Args:
            local_embs (torch.Tensor): pooled embeddings tensor to distribute.
            sharding_ctx (Optional[EmbeddingShardingContext]): shared context from
                KJTAllToAll operation.

        Returns:
            Awaitable[torch.Tensor]: awaitable of pooled embeddings tensor.
        """
        if self._dist is None:
            self._create_output_dist_module(sharding_ctx)

        if sharding_ctx is None:
            return cast(PooledEmbeddingsReduceScatter, self._dist)(local_embs)
        elif sharding_ctx.variable_batch_per_feature:
            return cast(VariableBatchPooledEmbeddingsReduceScatter, self._dist)(
                local_embs,
                batch_size_per_rank_per_feature=sharding_ctx.batch_size_per_rank_per_feature,
                embedding_dims=self._embedding_dims,
            )
        else:
            return cast(PooledEmbeddingsReduceScatter, self._dist)(
                local_embs,
                input_splits=sharding_ctx.batch_size_per_rank,
            )

    def _create_output_dist_module(
        self, sharding_ctx: Optional[EmbeddingShardingContext] = None
    ) -> None:
        if sharding_ctx is not None and sharding_ctx.variable_batch_per_feature:
            self._dist = VariableBatchPooledEmbeddingsReduceScatter(
                pg=self._pg,
                codecs=self._codecs,
            )
        else:
            self._dist = PooledEmbeddingsReduceScatter(
                pg=self._pg,
                codecs=self._codecs,
            )


class InferRwPooledEmbeddingDist(
    BaseEmbeddingDist[NullShardingContext, List[torch.Tensor], torch.Tensor]
):
    """
    Redistributes pooled embedding tensor in RW fashion with an AlltoOne operation.

    Args:
        device (torch.device): device on which the tensors will be communicated to.
        world_size (int): number of devices in the topology.
    """

    def __init__(
        self,
        device: torch.device,
        world_size: int,
    ) -> None:
        super().__init__()
        self._dist: EmbeddingsAllToOneReduce = EmbeddingsAllToOneReduce(
            device=device,
            world_size=world_size,
        )

    def forward(
        self,
        local_embs: List[torch.Tensor],
        sharding_ctx: Optional[NullShardingContext] = None,
    ) -> torch.Tensor:
        """
        Performs AlltoOne operation on sequence embeddings tensor.

        Args:
            local_embs (torch.Tensor): tensor of values to distribute.

        Returns:
            Awaitable[torch.Tensor]: awaitable of sequence embeddings.
        """

        return self._dist(local_embs)


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
            embedding_dims=self.embedding_dims(),
        )


@torch.fx.wrap
def get_block_sizes_runtime_device(
    block_sizes: List[int],
    runtime_device: torch.device,
    tensor_cache: Dict[str, Tuple[torch.Tensor, List[torch.Tensor]]],
    embedding_shard_metadata: Optional[List[List[int]]] = None,
    dtype: torch.dtype = torch.int32,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    cache_key: str = "__block_sizes"
    if cache_key not in tensor_cache:
        tensor_cache[cache_key] = (
            torch.tensor(
                block_sizes,
                device=runtime_device,
                dtype=dtype,
            ),
            []
            if embedding_shard_metadata is None
            else [
                torch.tensor(
                    row_pos,
                    device=runtime_device,
                    dtype=dtype,
                )
                for row_pos in embedding_shard_metadata
            ],
        )

    return tensor_cache[cache_key]


class InferRwSparseFeaturesDist(BaseSparseFeaturesDist[KJTList]):
    def __init__(
        self,
        world_size: int,
        num_features: int,
        feature_hash_sizes: List[int],
        device: Optional[torch.device] = None,
        is_sequence: bool = False,
        has_feature_processor: bool = False,
        need_pos: bool = False,
        embedding_shard_metadata: Optional[List[List[int]]] = None,
    ) -> None:
        super().__init__()
        self._world_size: int = world_size
        self._num_features = num_features
        self.feature_block_sizes: List[int] = [
            (hash_size + self._world_size - 1) // self._world_size
            for hash_size in feature_hash_sizes
        ]
        self.tensor_cache: Dict[
            str, Tuple[torch.Tensor, Optional[List[torch.Tensor]]]
        ] = {}

        self._dist = KJTOneToAll(
            splits=self._world_size * [self._num_features],
            world_size=world_size,
            device=device,
        )
        self._is_sequence = is_sequence
        self._has_feature_processor = has_feature_processor
        self._need_pos = need_pos
        self.unbucketize_permute_tensor: Optional[torch.Tensor] = None

        self._embedding_shard_metadata: Optional[
            List[List[int]]
        ] = embedding_shard_metadata

    def forward(
        self,
        sparse_features: KeyedJaggedTensor,
    ) -> KJTList:
        block_sizes, block_bucketize_row_pos = get_block_sizes_runtime_device(
            self.feature_block_sizes,
            sparse_features.device(),
            self.tensor_cache,
            self._embedding_shard_metadata,
        )
        (
            bucketized_features,
            self.unbucketize_permute_tensor,
        ) = bucketize_kjt_before_all2all(
            sparse_features,
            num_buckets=self._world_size,
            block_sizes=block_sizes,
            output_permute=self._is_sequence,
            bucketize_pos=self._has_feature_processor
            if sparse_features.weights_or_none() is None
            else self._need_pos,
            block_bucketize_row_pos=_fx_wrap_block_bucketize_row_pos(
                block_bucketize_row_pos
            ),
        )
        return self._dist.forward(bucketized_features)


class InferRwPooledEmbeddingSharding(
    BaseRwEmbeddingSharding[
        NullShardingContext, KJTList, List[torch.Tensor], torch.Tensor
    ]
):
    def create_input_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseSparseFeaturesDist[KJTList]:
        num_features = self._get_num_features()
        feature_hash_sizes = self._get_feature_hash_sizes()

        (embed_sharding, is_even_sharding) = get_embedding_shard_metadata(
            self._grouped_embedding_configs_per_rank
        )

        return InferRwSparseFeaturesDist(
            world_size=self._world_size,
            num_features=num_features,
            feature_hash_sizes=feature_hash_sizes,
            device=device if device is not None else self._device,
            embedding_shard_metadata=embed_sharding if not is_even_sharding else None,
        )

    def create_lookup(
        self,
        device: Optional[torch.device] = None,
        fused_params: Optional[Dict[str, Any]] = None,
        feature_processor: Optional[BaseGroupedFeatureProcessor] = None,
    ) -> BaseEmbeddingLookup[KJTList, List[torch.Tensor]]:
        return InferGroupedPooledEmbeddingsLookup(
            grouped_configs_per_rank=self._grouped_embedding_configs_per_rank,
            world_size=self._world_size,
            fused_params=fused_params,
            device=device if device is not None else self._device,
        )

    def create_output_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseEmbeddingDist[NullShardingContext, List[torch.Tensor], torch.Tensor]:
        assert device is not None
        return InferRwPooledEmbeddingDist(
            device=device,
            world_size=self._world_size,
        )
