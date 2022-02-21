#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
from typing import TypeVar, Generic, List, Tuple, Optional, Dict, Any

import torch
import torch.distributed as dist
from torch import nn
from torchrec.distributed.dist_data import (
    KJTAllToAll,
    KJTOneToAll,
    KJTAllToAllIndicesAwaitable,
)
from torchrec.distributed.embedding_types import (
    GroupedEmbeddingConfig,
    BaseEmbeddingLookup,
    SparseFeatures,
    EmbeddingComputeKernel,
    ShardedEmbeddingTable,
    BaseGroupedFeatureProcessor,
    SparseFeaturesList,
    ListOfSparseFeaturesList,
)
from torchrec.distributed.types import NoWait, Awaitable, ShardMetadata
from torchrec.modules.embedding_configs import (
    PoolingType,
    DataType,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.streamable import Multistreamable


class SparseFeaturesIndices(Awaitable[SparseFeatures]):
    """
    Awaitable of sparse features redistributed with AlltoAll collective.

    Args:
        id_list_features_awaitable (Optional[Awaitable[KeyedJaggedTensor]]): awaitable
            of sharded id list features.
        id_score_list_features_awaitable (Optional[Awaitable[KeyedJaggedTensor]]):
            awaitable of sharded id score list features.
    """

    def __init__(
        self,
        id_list_features_awaitable: Optional[Awaitable[KeyedJaggedTensor]],
        id_score_list_features_awaitable: Optional[Awaitable[KeyedJaggedTensor]],
    ) -> None:
        super().__init__()
        self._id_list_features_awaitable = id_list_features_awaitable
        self._id_score_list_features_awaitable = id_score_list_features_awaitable

    def _wait_impl(self) -> SparseFeatures:
        """
        Syncs sparse features after AlltoAll.

        Returns:
            SparseFeatures: synced sparse features.
        """

        return SparseFeatures(
            id_list_features=self._id_list_features_awaitable.wait()
            if self._id_list_features_awaitable is not None
            else None,
            id_score_list_features=self._id_score_list_features_awaitable.wait()
            if self._id_score_list_features_awaitable is not None
            else None,
        )


class SparseFeaturesLengths(Awaitable[SparseFeaturesIndices]):
    """
    Awaitable of sparse features indices distribution.

    Args:
        id_list_features_awaitable (Optional[Awaitable[KJTAllToAllIndicesAwaitable]]):
            awaitable of sharded id list features indices AlltoAll. Waiting on this
            value will trigger indices AlltoAll (waiting again will yield final AlltoAll
            results).
        id_score_list_features_awaitable
            (Optional[Awaitable[KJTAllToAllIndicesAwaitable]]):
            awaitable of sharded id score list features indices AlltoAll. Waiting on
            this value will trigger indices AlltoAll (waiting again will yield the final
            AlltoAll results).
    """

    def __init__(
        self,
        id_list_features_awaitable: Optional[Awaitable[KJTAllToAllIndicesAwaitable]],
        id_score_list_features_awaitable: Optional[
            Awaitable[KJTAllToAllIndicesAwaitable]
        ],
    ) -> None:
        super().__init__()
        self._id_list_features_awaitable = id_list_features_awaitable
        self._id_score_list_features_awaitable = id_score_list_features_awaitable

    def _wait_impl(self) -> SparseFeaturesIndices:
        """
        Gets lengths of AlltoAll results, instantiates `SparseFeaturesIndices` for
        indices AlltoAll.

        Returns:
            SparseFeaturesIndices.
        """
        return SparseFeaturesIndices(
            id_list_features_awaitable=self._id_list_features_awaitable.wait()
            if self._id_list_features_awaitable is not None
            else None,
            id_score_list_features_awaitable=self._id_score_list_features_awaitable.wait()
            if self._id_score_list_features_awaitable is not None
            else None,
        )


def bucketize_kjt_before_all2all(
    kjt: KeyedJaggedTensor,
    num_buckets: int,
    block_sizes: torch.Tensor,
    output_permute: bool = False,
    bucketize_pos: bool = False,
) -> Tuple[KeyedJaggedTensor, Optional[torch.Tensor]]:
    """
    Bucketizes the `values` in KeyedJaggedTensor into `num_buckets` buckets,
    `lengths` are readjusted based on the bucketization results.

    Note: This function should be used only for row-wise sharding before calling
    `SparseFeaturesAllToAll`.

    Args:
        num_buckets (int): number of buckets to bucketize the values into.
        block_sizes: (torch.Tensor): bucket sizes for the keyed dimension.
        output_permute (bool): output the memory location mapping from the unbucketized
            values to bucketized values or not.
        bucketize_pos (bool): output the changed position of the bucketized values or
            not.

    Returns:
        Tuple[KeyedJaggedTensor, Optional[torch.Tensor]]: the bucketized `KeyedJaggedTensor` and the optional permute mapping from the unbucketized values to bucketized value.
    """

    num_features = len(kjt.keys())
    assert (
        block_sizes.numel() == num_features
    ), f"Expecting block sizes for {num_features} features, but {block_sizes.numel()} received."

    # kernel expects them to be same type, cast to avoid type mismatch
    block_sizes_new_type = block_sizes.type(kjt.values().type())
    (
        bucketized_lengths,
        bucketized_indices,
        bucketized_weights,
        pos,
        unbucketize_permute,
    ) = torch.ops.fbgemm.block_bucketize_sparse_features(
        kjt.lengths().view(-1),
        kjt.values(),
        bucketize_pos=bucketize_pos,
        sequence=output_permute,
        block_sizes=block_sizes_new_type,
        my_size=num_buckets,
        weights=kjt.weights_or_none(),
    )

    return (
        KeyedJaggedTensor(
            # duplicate keys will be resolved by AllToAll
            keys=kjt.keys() * num_buckets,
            values=bucketized_indices,
            weights=pos if bucketize_pos else bucketized_weights,
            lengths=bucketized_lengths.view(-1),
            offsets=None,
            stride=kjt.stride(),
            length_per_key=None,
            offset_per_key=None,
            index_per_key=None,
        ),
        unbucketize_permute,
    )


class SparseFeaturesAllToAll(nn.Module):
    """
    Redistributes sparse features to a `ProcessGroup` utilizing an AlltoAll collective.

    Args:
        pg (dist.ProcessGroup): process group for AlltoAll communication.
        id_list_features_per_rank (List[int]): number of id list features to send to
            each rank.
        id_score_list_features_per_rank (List[int]): number of id score list features to
            send to each rank
        device (Optional[torch.device]): device on which buffers will be allocated.
        stagger (int): stagger value to apply to recat tensor, see `_recat` function for
            more detail.

    Example:
        >>> id_list_features_per_rank = [2, 1]
        >>> id_score_list_features_per_rank = [1, 3]
        >>> sfa2a = SparseFeaturesAllToAll(
                pg,
                id_list_features_per_rank,
                id_score_list_features_per_rank
            )
        >>> awaitable = sfa2a(rank0_input: SparseFeatures)

        >>> # where:
        >>> #     rank0_input.id_list_features is KeyedJaggedTensor holding

        >>> #             0           1           2
        >>> #     'A'    [A.V0]       None        [A.V1, A.V2]
        >>> #     'B'    None         [B.V0]      [B.V1]
        >>> #     'C'    [C.V0]       [C.V1]      None

        >>> #     rank1_input.id_list_features is KeyedJaggedTensor holding

        >>> #             0           1           2
        >>> #     'A'     [A.V3]      [A.V4]      None
        >>> #     'B'     None        [B.V2]      [B.V3, B.V4]
        >>> #     'C'     [C.V2]      [C.V3]      None

        >>> #     rank0_input.id_score_list_features is KeyedJaggedTensor holding

        >>> #             0           1           2
        >>> #     'A'    [A.V0]       None        [A.V1, A.V2]
        >>> #     'B'    None         [B.V0]      [B.V1]
        >>> #     'C'    [C.V0]       [C.V1]      None
        >>> #     'D'    None         [D.V0]      None

        >>> #     rank1_input.id_score_list_features is KeyedJaggedTensor holding

        >>> #             0           1           2
        >>> #     'A'     [A.V3]      [A.V4]      None
        >>> #     'B'     None        [B.V2]      [B.V3, B.V4]
        >>> #     'C'     [C.V2]      [C.V3]      None
        >>> #     'D'     [D.V1]      [D.V2]      [D.V3, D.V4]

        >>> rank0_output: SparseFeatures = awaitable.wait()

        >>> # rank0_output.id_list_features is KeyedJaggedTensor holding

        >>> #         0           1           2           3           4           5
        >>> # 'A'     [A.V0]      None      [A.V1, A.V2]  [A.V3]      [A.V4]      None
        >>> # 'B'     None        [B.V0]    [B.V1]        None        [B.V2]     [B.V3, B.V4]

        >>> # rank1_output.id_list_features is KeyedJaggedTensor holding
        >>> #         0           1           2           3           4           5
        >>> # 'C'     [C.V0]      [C.V1]      None        [C.V2]      [C.V3]      None

        >>> # rank0_output.id_score_list_features is KeyedJaggedTensor holding

        >>> #         0           1           2           3           4           5
        >>> # 'A'     [A.V0]      None      [A.V1, A.V2]  [A.V3]      [A.V4]      None

        >>> # rank1_output.id_score_list_features is KeyedJaggedTensor holding

        >>> #         0           1           2           3           4           5
        >>> # 'B'     None        [B.V0]      [B.V1]      None        [B.V2]      [B.V3, B.V4]
        >>> # 'C'     [C.V0]       [C.V1]      None       [C.V2]      [C.V3]      None
        >>> # 'D      None         [D.V0]      None       [D.V1]      [D.V2]      [D.V3, D.V4]
    """

    def __init__(
        self,
        # pyre-fixme[11]: Annotation `ProcessGroup` is not defined as a type.
        pg: dist.ProcessGroup,
        id_list_features_per_rank: List[int],
        id_score_list_features_per_rank: List[int],
        device: Optional[torch.device] = None,
        stagger: int = 1,
    ) -> None:
        super().__init__()
        self._id_list_features_all2all: KJTAllToAll = KJTAllToAll(
            pg, id_list_features_per_rank, device, stagger
        )
        self._id_score_list_features_all2all: KJTAllToAll = KJTAllToAll(
            pg, id_score_list_features_per_rank, device, stagger
        )

    def forward(
        self,
        sparse_features: SparseFeatures,
    ) -> Awaitable[SparseFeaturesIndices]:
        """
        Sends sparse features to relevant ProcessGroup ranks. Instantiates lengths
        AlltoAll.
        First wait will get lengths AlltoAll results, then issues indices AlltoAll.
        Second wait will get indices AlltoAll results.

        Args:
            sparse_features (SparseFeatures): sparse features to redistribute.

        Returns:
            Awaitable[SparseFeatures]: awaitable of SparseFeatures.
        """

        return SparseFeaturesLengths(
            id_list_features_awaitable=self._id_list_features_all2all.forward(
                sparse_features.id_list_features
            )
            if sparse_features.id_list_features is not None
            else None,
            id_score_list_features_awaitable=self._id_score_list_features_all2all.forward(
                sparse_features.id_score_list_features
            )
            if sparse_features.id_score_list_features is not None
            else None,
        )


class SparseFeaturesOneToAll(nn.Module):
    """
    Redistributes sparse features to all devices.

    Args:
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
        self._world_size = world_size
        self._id_list_features_one2all: KJTOneToAll = KJTOneToAll(
            id_list_features_per_rank,
            world_size,
        )
        self._id_score_list_features_one2all: KJTOneToAll = KJTOneToAll(
            id_score_list_features_per_rank, world_size
        )

    def forward(
        self,
        sparse_features: SparseFeatures,
    ) -> Awaitable[SparseFeaturesList]:
        """
        Performs OnetoAll operation on sparse features.

        Args:
            sparse_features (SparseFeatures): sparse features to redistribute.

        Returns:
            Awaitable[SparseFeatures]: awaitable of SparseFeatures.
        """

        return NoWait(
            SparseFeaturesList(
                [
                    SparseFeatures(
                        id_list_features=id_list_features,
                        id_score_list_features=id_score_list_features,
                    )
                    for id_list_features, id_score_list_features in zip(
                        self._id_list_features_one2all.forward(
                            sparse_features.id_list_features
                        ).wait()
                        if sparse_features.id_list_features is not None
                        else [None] * self._world_size,
                        self._id_score_list_features_one2all.forward(
                            sparse_features.id_score_list_features
                        ).wait()
                        if sparse_features.id_score_list_features is not None
                        else [None] * self._world_size,
                    )
                ]
            )
        )


# group tables by DataType, PoolingType, Weighted, and EmbeddingComputeKernel.
def group_tables(
    tables_per_rank: List[List[ShardedEmbeddingTable]],
) -> Tuple[List[List[GroupedEmbeddingConfig]], List[List[GroupedEmbeddingConfig]]]:
    """
    Groups tables by `DataType`, `PoolingType`, `Weighted`, and `EmbeddingComputeKernel`.

    Args:
        tables_per_rank (List[List[ShardedEmbeddingTable]]): list of sharding embedding
            tables per rank.

    Returns:
        Tuple[List[List[GroupedEmbeddingConfig]], List[List[GroupedEmbeddingConfig]]]: per rank list of GroupedEmbeddingConfig for unscored and scored features.
    """

    def _group_tables_per_rank(
        embedding_tables: List[ShardedEmbeddingTable],
    ) -> Tuple[List[GroupedEmbeddingConfig], List[GroupedEmbeddingConfig]]:
        grouped_embedding_configs: List[GroupedEmbeddingConfig] = []
        score_grouped_embedding_configs: List[GroupedEmbeddingConfig] = []
        for data_type in DataType:
            for pooling in PoolingType:
                for is_weighted in [True, False]:
                    for has_feature_processor in [True, False]:
                        for compute_kernel in [
                            EmbeddingComputeKernel.DENSE,
                            EmbeddingComputeKernel.SPARSE,
                            EmbeddingComputeKernel.BATCHED_DENSE,
                            EmbeddingComputeKernel.BATCHED_FUSED,
                            EmbeddingComputeKernel.BATCHED_QUANT,
                        ]:
                            grouped_tables: List[ShardedEmbeddingTable] = []
                            grouped_score_tables: List[ShardedEmbeddingTable] = []
                            for table in embedding_tables:
                                if table.compute_kernel in [
                                    EmbeddingComputeKernel.BATCHED_FUSED_UVM,
                                    EmbeddingComputeKernel.BATCHED_FUSED_UVM_CACHING,
                                ]:
                                    compute_kernel_type = (
                                        EmbeddingComputeKernel.BATCHED_FUSED
                                    )
                                else:
                                    compute_kernel_type = table.compute_kernel
                                if (
                                    table.data_type == data_type
                                    and table.pooling == pooling
                                    and table.is_weighted == is_weighted
                                    and table.has_feature_processor
                                    == has_feature_processor
                                    and compute_kernel_type == compute_kernel
                                ):
                                    if table.is_weighted:
                                        grouped_score_tables.append(table)
                                    else:
                                        grouped_tables.append(table)
                            if grouped_tables:
                                grouped_embedding_configs.append(
                                    GroupedEmbeddingConfig(
                                        data_type=data_type,
                                        pooling=pooling,
                                        is_weighted=is_weighted,
                                        has_feature_processor=has_feature_processor,
                                        compute_kernel=compute_kernel,
                                        embedding_tables=grouped_tables,
                                    )
                                )
                            if grouped_score_tables:
                                score_grouped_embedding_configs.append(
                                    GroupedEmbeddingConfig(
                                        data_type=data_type,
                                        pooling=pooling,
                                        is_weighted=is_weighted,
                                        has_feature_processor=has_feature_processor,
                                        compute_kernel=compute_kernel,
                                        embedding_tables=grouped_score_tables,
                                    )
                                )
        return grouped_embedding_configs, score_grouped_embedding_configs

    grouped_embedding_configs_by_rank: List[List[GroupedEmbeddingConfig]] = []
    score_grouped_embedding_configs_by_rank: List[List[GroupedEmbeddingConfig]] = []
    for tables in tables_per_rank:
        (
            grouped_embedding_configs,
            score_grouped_embedding_configs,
        ) = _group_tables_per_rank(tables)
        grouped_embedding_configs_by_rank.append(grouped_embedding_configs)
        score_grouped_embedding_configs_by_rank.append(score_grouped_embedding_configs)
    return (
        grouped_embedding_configs_by_rank,
        score_grouped_embedding_configs_by_rank,
    )


class SparseFeaturesListAwaitable(Awaitable[SparseFeaturesList]):
    """
    Awaitable of SparseFeaturesList.

    Args:
        awaitables (List[Awaitable[SparseFeatures]]): list of `Awaitable` of sparse
            features.
    """

    def __init__(
        self,
        awaitables: List[Awaitable[SparseFeatures]],
    ) -> None:
        super().__init__()
        self.awaitables = awaitables

    def _wait_impl(self) -> SparseFeaturesList:
        """
        Syncs sparse features in `SparseFeaturesList`.

        Returns:
            SparseFeaturesList: synced `SparseFeaturesList`.
        """

        return SparseFeaturesList([w.wait() for w in self.awaitables])


class SparseFeaturesListIndicesAwaitable(Awaitable[List[Awaitable[SparseFeatures]]]):
    """
    Handles the first wait for a list of two-layer awaitables of `SparseFeatures`.
    Wait on this module will get lengths AlltoAll results for each `SparseFeatures`, and
    instantiate its indices AlltoAll.

    Args:
        awaitables (List[Awaitable[Awaitable[SparseFeatures]]]): list of `Awaitable` of
            `Awaitable` sparse features.
    """

    def __init__(
        self,
        awaitables: List[Awaitable[Awaitable[SparseFeatures]]],
    ) -> None:
        super().__init__()
        self.awaitables = awaitables

    def _wait_impl(self) -> List[Awaitable[SparseFeatures]]:
        """
        Syncs sparse features in SparseFeaturesList.

        Returns:
            List[Awaitable[SparseFeatures]]
        """

        return [m.wait() for m in self.awaitables]


class ListOfSparseFeaturesListAwaitable(Awaitable[ListOfSparseFeaturesList]):
    """
    This module handles the tables-wise sharding input features distribution for inference.
    For inference, we currently do not separate lengths from indices.

    Args:
        awaitables (List[Awaitable[SparseFeaturesList]]): list of `Awaitable` of
            `SparseFeaturesList`.
    """

    def __init__(
        self,
        awaitables: List[Awaitable[SparseFeaturesList]],
    ) -> None:
        super().__init__()
        self.awaitables = awaitables

    def _wait_impl(self) -> ListOfSparseFeaturesList:
        """
        Syncs sparse features in List of SparseFeaturesList.

        Returns:
            ListOfSparseFeaturesList: synced `ListOfSparseFeaturesList`.

        """
        return ListOfSparseFeaturesList([w.wait() for w in self.awaitables])


F = TypeVar("F", bound=Multistreamable)
T = TypeVar("T")


class BaseSparseFeaturesDist(abc.ABC, nn.Module, Generic[F]):
    """
    Converts input from data-parallel to model-parallel.
    """

    @abc.abstractmethod
    def forward(
        self,
        sparse_features: SparseFeatures,
    ) -> Awaitable[Awaitable[F]]:
        pass


class BaseEmbeddingDist(abc.ABC, nn.Module, Generic[T]):
    """
    Converts output of EmbeddingLookup from model-parallel to data-parallel.
    """

    @abc.abstractmethod
    def forward(
        self,
        local_embs: T,
    ) -> Awaitable[torch.Tensor]:
        pass


class EmbeddingSharding(abc.ABC, Generic[F, T]):
    """
    Used to implement different sharding types for `EmbeddingBagCollection`, e.g.
    table_wise.
    """

    @abc.abstractmethod
    def create_input_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseSparseFeaturesDist[F]:
        pass

    @abc.abstractmethod
    def create_output_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> BaseEmbeddingDist[T]:
        pass

    @abc.abstractmethod
    def create_lookup(
        self,
        device: Optional[torch.device] = None,
        fused_params: Optional[Dict[str, Any]] = None,
        feature_processor: Optional[BaseGroupedFeatureProcessor] = None,
    ) -> BaseEmbeddingLookup[F, T]:
        pass

    @abc.abstractmethod
    def embedding_dims(self) -> List[int]:
        pass

    @abc.abstractmethod
    def embedding_shard_metadata(self) -> List[Optional[ShardMetadata]]:
        pass

    @abc.abstractmethod
    def embedding_names(self) -> List[str]:
        pass

    @abc.abstractmethod
    def id_list_feature_names(self) -> List[str]:
        pass

    @abc.abstractmethod
    def id_score_list_feature_names(self) -> List[str]:
        pass
