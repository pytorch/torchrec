#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar

import torch
import torch.distributed as dist  # noqa
from fbgemm_gpu.permute_pooled_embedding_modules_split import (
    PermutePooledEmbeddingsSplit,
)
from torchrec.distributed.embedding_lookup import GroupedPooledEmbeddingsLookup
from torchrec.distributed.embedding_sharding import (
    BaseEmbeddingDist,
    BaseEmbeddingLookup,
    BaseSparseFeaturesDist,
    EmbeddingShardingContext,
    EmbeddingShardingInfo,
)
from torchrec.distributed.embedding_types import (
    BaseGroupedFeatureProcessor,
    EmbeddingComputeKernel,
    ShardedEmbeddingTable,
    SparseFeatures,
)
from torchrec.distributed.sharding.tw_sharding import (
    BaseTwEmbeddingSharding,
    TwPooledEmbeddingDist,
    TwSparseFeaturesDist,
)
from torchrec.distributed.types import (
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


class BaseCwEmbeddingSharding(BaseTwEmbeddingSharding[C, F, T, W]):
    """
    Base class for column-wise sharding.
    """

    def __init__(
        self,
        sharding_infos: List[EmbeddingShardingInfo],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        permute_embeddings: bool = False,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
        variable_batch_size: bool = False,
    ) -> None:
        super().__init__(
            sharding_infos,
            env,
            device,
            qcomm_codecs_registry=qcomm_codecs_registry,
            variable_batch_size=variable_batch_size,
        )
        self._permute_embeddings = permute_embeddings
        if self._permute_embeddings:
            self._init_combined_embeddings()

    def _init_combined_embeddings(self) -> None:
        """
        Grabs the embedding names and dims from TwEmbeddingSharder.

        NOTE:
            This could have duplications if there are multiple shards from the same
            table on a rank. Later on we process these to combine shards together.
        """

        embedding_names: List[str] = super().embedding_names()
        embedding_dims: List[int] = super().embedding_dims()

        embedding_shard_metadata: List[
            Optional[ShardMetadata]
        ] = super().embedding_shard_metadata()

        embedding_name_to_index_offset_tuples: Dict[str, List[Tuple[int, int]]] = {}
        for i, (name, metadata) in enumerate(
            zip(embedding_names, embedding_shard_metadata)
        ):
            if name not in embedding_name_to_index_offset_tuples:
                embedding_name_to_index_offset_tuples[name] = []
            embedding_name_to_index_offset_tuples[name].append(
                (i, metadata.shard_offsets[1] if metadata is not None else 0)
            )

        embedding_name_to_index: Dict[str, List[int]] = {}
        for name, index_offset_tuples in embedding_name_to_index_offset_tuples.items():
            embedding_name_to_index[name] = [
                idx_off_tuple[0]
                for idx_off_tuple in sorted(
                    index_offset_tuples,
                    key=lambda idx_off_tuple: idx_off_tuple[1],
                )
            ]

        combined_embedding_names: List[str] = []
        seen_embedding_names: Set[str] = set()

        for name in embedding_names:
            if name not in seen_embedding_names:
                combined_embedding_names.append(name)
                seen_embedding_names.add(name)

        combined_embedding_dims: List[int] = []

        embedding_order: List[int] = []
        for name in combined_embedding_names:
            combined_embedding_dims.append(
                sum([embedding_dims[idx] for idx in embedding_name_to_index[name]])
            )
            embedding_order.extend(embedding_name_to_index[name])

        self._embedding_names: List[str] = embedding_names
        self._embedding_dims: List[int] = embedding_dims
        self._embedding_order: List[int] = embedding_order

        self._combined_embedding_names: List[str] = combined_embedding_names
        self._combined_embedding_dims: List[int] = combined_embedding_dims

    def _shard(
        self,
        sharding_infos: List[EmbeddingShardingInfo],
    ) -> List[List[ShardedEmbeddingTable]]:
        # pyre-fixme[16]: `Optional` has no attribute `size`.
        world_size = self._pg.size()
        tables_per_rank: List[List[ShardedEmbeddingTable]] = [
            [] for i in range(world_size)
        ]
        for info in sharding_infos:
            # pyre-fixme [16]
            shards: List[ShardMetadata] = info.param_sharding.sharding_spec.shards

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

            # pyre-fixme [6]
            for i, rank in enumerate(info.param_sharding.ranks):
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
                        local_rows=info.embedding_config.num_embeddings,
                        local_cols=shards[i].shard_sizes[1],
                        compute_kernel=EmbeddingComputeKernel(
                            info.param_sharding.compute_kernel
                        ),
                        local_metadata=shards[i],
                        global_metadata=global_metadata,
                        fused_params=info.fused_params,
                        weight_init_max=info.embedding_config.weight_init_max,
                        weight_init_min=info.embedding_config.weight_init_min,
                    )
                )

        return tables_per_rank

    def embedding_dims(self) -> List[int]:
        return (
            self._combined_embedding_dims
            if self._permute_embeddings
            else super().embedding_dims()
        )

    def embedding_names(self) -> List[str]:
        return (
            self._combined_embedding_names
            if self._permute_embeddings
            else super().embedding_names()
        )


class CwPooledEmbeddingSharding(
    BaseCwEmbeddingSharding[
        EmbeddingShardingContext, SparseFeatures, torch.Tensor, torch.Tensor
    ]
):
    """
    Shards embedding bags column-wise, i.e.. a given embedding table is partitioned
    along its columns and placed on specified ranks.
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
        device = device if device is not None else self._device
        embedding_permute_op: Optional[PermutePooledEmbeddingsSplit] = None
        callbacks: Optional[List[Callable[[torch.Tensor], torch.Tensor]]] = None
        if self._permute_embeddings and self._embedding_order != list(
            range(len(self._embedding_order))
        ):
            assert len(self._embedding_order) == len(self._embedding_dims)
            embedding_permute_op = PermutePooledEmbeddingsSplit(
                self._embedding_dims,
                self._embedding_order,
            ).to(device=device)
            callbacks = [embedding_permute_op]
        assert self._pg is not None
        return TwPooledEmbeddingDist(
            self._pg,
            self._dim_sum_per_rank(),
            device,
            callbacks,
            qcomm_codecs_registry=self.qcomm_codecs_registry,
        )
