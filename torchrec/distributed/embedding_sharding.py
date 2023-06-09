#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar, Union

import torch
from torch import distributed as dist, nn
from torchrec.distributed.dist_data import (
    KJTAllToAllTensorsAwaitable,
    SplitsAllToAllAwaitable,
)
from torchrec.distributed.embedding_types import (
    BaseEmbeddingLookup,
    BaseGroupedFeatureProcessor,
    EmbeddingComputeKernel,
    FeatureShardingMixIn,
    GroupedEmbeddingConfig,
    KJTList,
    ListOfKJTList,
    ShardedEmbeddingTable,
)
from torchrec.distributed.types import (
    Awaitable,
    ParameterSharding,
    QuantizedCommCodecs,
    ShardMetadata,
)
from torchrec.modules.embedding_configs import (
    DataType,
    EmbeddingTableConfig,
    PoolingType,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.streamable import Multistreamable


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
    `KJTAllToAll`.

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


# group tables by `DataType`, `PoolingType`, and `EmbeddingComputeKernel`.
def group_tables(
    tables_per_rank: List[List[ShardedEmbeddingTable]],
) -> List[List[GroupedEmbeddingConfig]]:
    """
    Groups tables by `DataType`, `PoolingType`, and `EmbeddingComputeKernel`.

    Args:
        tables_per_rank (List[List[ShardedEmbeddingTable]]): list of sharded embedding
            tables per rank with consistent weightedness.

    Returns:
        List[List[GroupedEmbeddingConfig]]: per rank list of GroupedEmbeddingConfig for features.
    """

    def _group_tables_per_rank(
        embedding_tables: List[ShardedEmbeddingTable],
    ) -> List[GroupedEmbeddingConfig]:
        grouped_embedding_configs: List[GroupedEmbeddingConfig] = []

        # add fused params:
        fused_params_groups = []
        for table in embedding_tables:
            if table.fused_params is None:
                table.fused_params = {}
            if table.fused_params not in fused_params_groups:
                fused_params_groups.append(table.fused_params)

        compute_kernels = [
            EmbeddingComputeKernel.DENSE,
            EmbeddingComputeKernel.FUSED,
            EmbeddingComputeKernel.QUANT,
        ]

        for data_type in DataType:
            for pooling in PoolingType:
                # remove this when finishing migration
                for has_feature_processor in [False, True]:
                    for fused_params_group in fused_params_groups:
                        for compute_kernel in compute_kernels:
                            grouped_tables: List[ShardedEmbeddingTable] = []
                            is_weighted = False
                            for table in embedding_tables:
                                compute_kernel_type = table.compute_kernel
                                is_weighted = table.is_weighted
                                if table.compute_kernel in [
                                    EmbeddingComputeKernel.FUSED_UVM,
                                    EmbeddingComputeKernel.FUSED_UVM_CACHING,
                                ]:
                                    compute_kernel_type = EmbeddingComputeKernel.FUSED
                                elif table.compute_kernel in [
                                    EmbeddingComputeKernel.QUANT_UVM,
                                    EmbeddingComputeKernel.QUANT_UVM_CACHING,
                                ]:
                                    compute_kernel_type = EmbeddingComputeKernel.QUANT
                                if (
                                    table.data_type == data_type
                                    and table.pooling.value == pooling.value
                                    and table.has_feature_processor
                                    == has_feature_processor
                                    and compute_kernel_type == compute_kernel
                                    and table.fused_params == fused_params_group
                                ):
                                    grouped_tables.append(table)

                            if fused_params_group is None:
                                fused_params_group = {}

                            if grouped_tables:
                                grouped_embedding_configs.append(
                                    GroupedEmbeddingConfig(
                                        data_type=data_type,
                                        pooling=pooling,
                                        is_weighted=is_weighted,
                                        has_feature_processor=has_feature_processor,
                                        compute_kernel=compute_kernel,
                                        embedding_tables=grouped_tables,
                                        fused_params={
                                            k: v
                                            for k, v in fused_params_group.items()
                                            if k
                                            not in [
                                                "_batch_key"
                                            ]  # drop '_batch_key' not a native fused param
                                        },
                                    )
                                )
        return grouped_embedding_configs

    table_weightedness = [
        table.is_weighted for tables in tables_per_rank for table in tables
    ]
    assert all(table_weightedness) or not any(table_weightedness)

    grouped_embedding_configs_by_rank: List[List[GroupedEmbeddingConfig]] = []
    for tables in tables_per_rank:
        grouped_embedding_configs = _group_tables_per_rank(tables)
        grouped_embedding_configs_by_rank.append(grouped_embedding_configs)

    return grouped_embedding_configs_by_rank


class KJTListAwaitable(Awaitable[KJTList]):
    """
    Awaitable of KJTList.

    Args:
        awaitables (List[Awaitable[KeyedJaggedTensor]]): list of `Awaitable` of sparse
            features.
    """

    def __init__(
        self,
        awaitables: List[Awaitable[KeyedJaggedTensor]],
    ) -> None:
        super().__init__()
        self.awaitables = awaitables

    def _wait_impl(self) -> KJTList:
        """
        Syncs KJTs in `KJTList`.

        Returns:
            KJTList: synced `KJTList`.
        """

        return KJTList([w.wait() for w in self.awaitables])


C = TypeVar("C", bound=Multistreamable)
T = TypeVar("T")


def _set_sharding_context(
    tensors_awaitables: List[Awaitable[KeyedJaggedTensor]],
    ctx: C,
) -> None:
    for awaitable, sharding_context in zip(
        tensors_awaitables,
        getattr(ctx, "sharding_contexts", []),
    ):
        if isinstance(awaitable, KJTAllToAllTensorsAwaitable):
            if hasattr(sharding_context, "batch_size_per_rank"):
                sharding_context.batch_size_per_rank = awaitable._batch_size_per_rank
            if hasattr(sharding_context, "input_splits"):
                sharding_context.input_splits = awaitable._input_splits["values"]
            if hasattr(sharding_context, "output_splits"):
                sharding_context.output_splits = awaitable._output_splits["values"]
            if hasattr(sharding_context, "sparse_features_recat"):
                sharding_context.sparse_features_recat = awaitable._recat


def _split(flat_list: List[T], splits: List[int]) -> List[List[T]]:
    return [
        flat_list[sum(splits[:i]) : sum(splits[:i]) + n] for i, n in enumerate(splits)
    ]


class KJTListSplitsAwaitable(Awaitable[Awaitable[KJTList]], Generic[C]):
    """
    Awaitable of Awaitable of KJTList.

    Args:
        awaitables (List[Awaitable[Awaitable[KeyedJaggedTensor]]]): result from calling
            forward on `KJTAllToAll` with sparse features to redistribute.
        ctx (C): sharding context to save the metadata from the input dist to for the
            embedding AlltoAll.
    """

    def __init__(
        self,
        awaitables: List[Awaitable[Awaitable[KeyedJaggedTensor]]],
        ctx: C,
    ) -> None:
        super().__init__()
        self.awaitables = awaitables
        self.ctx = ctx

    def _wait_impl(self) -> KJTListAwaitable:
        """
        Calls first wait on the awaitable of awaitable of sparse features and updates
        the context with metadata from the tensors awaitable.

        The first wait gets the result of splits AlltoAll and returns the tensors
        awaitable.

        Returns:
            KJTListAwaitable: awaitables for tensors of the sparse features.
        """
        tensors_awaitables = [w.wait() for w in self.awaitables]
        _set_sharding_context(tensors_awaitables, self.ctx)
        return KJTListAwaitable(tensors_awaitables)


@dataclass
class KJTSplitsAllToAllMeta:
    pg: dist.ProcessGroup
    input: KeyedJaggedTensor
    splits: List[int]
    splits_tensors: List[torch.Tensor]
    input_splits: List[List[int]]
    input_tensors: List[torch.Tensor]
    labels: List[str]
    keys: List[str]
    device: torch.device
    stagger: int


class FusedKJTListSplitsAwaitable(Awaitable[List[KJTListAwaitable]]):
    def __init__(
        self,
        requests: List[KJTListSplitsAwaitable[C]],
        contexts: List[C],
        pg: Optional[dist.ProcessGroup],
    ) -> None:
        super().__init__()
        self._contexts = contexts
        self._awaitables: List[
            Union[KJTSplitsAllToAllMeta, Awaitable[Awaitable[KeyedJaggedTensor]]]
        ] = [awaitable for request in requests for awaitable in request.awaitables]
        self._output_lengths: List[int] = [
            len(request.awaitables) for request in requests
        ]
        self._lengths: List[int] = [
            len(awaitable.splits_tensors)
            if isinstance(awaitable, KJTSplitsAllToAllMeta)
            else 0
            for awaitable in self._awaitables
        ]
        splits_tensors = [
            splits_tensor
            for awaitable in self._awaitables
            for splits_tensor in (
                awaitable.splits_tensors
                if isinstance(awaitable, KJTSplitsAllToAllMeta)
                else []
            )
        ]
        self._splits_awaitable: Optional[SplitsAllToAllAwaitable] = (
            SplitsAllToAllAwaitable(
                input_tensors=splits_tensors,
                pg=pg,
            )
            if splits_tensors and pg
            else None
        )

    def _wait_impl(self) -> List[KJTListAwaitable]:
        if self._splits_awaitable:
            splits_list = self._splits_awaitable.wait()
            splits_per_awaitable = _split(splits_list, self._lengths)
        else:
            splits_per_awaitable = [[] for _ in range(len(self._lengths))]
        tensors_awaitables = []
        for splits, awaitable in zip(splits_per_awaitable, self._awaitables):
            if not splits:  # NoWait
                # pyre-fixme[16]: Item `KJTSplitsAllToAllMeta` of
                #  `Union[Awaitable[Awaitable[KeyedJaggedTensor]],
                #  KJTSplitsAllToAllMeta]` has no attribute `wait`.
                tensors_awaitables.append(awaitable.wait())
                continue
            output_splits = splits[:-1]
            batch_size_per_rank = splits[-1]
            tensors_awaitables.append(
                KJTAllToAllTensorsAwaitable(
                    # pyre-fixme[16]: Item `Awaitable` of
                    #  `Union[Awaitable[Awaitable[KeyedJaggedTensor]],
                    #  KJTSplitsAllToAllMeta]` has no attribute `pg`.
                    pg=awaitable.pg,
                    # pyre-fixme[16]: Item `Awaitable` of
                    #  `Union[Awaitable[Awaitable[KeyedJaggedTensor]],
                    #  KJTSplitsAllToAllMeta]` has no attribute `input`.
                    input=awaitable.input,
                    # pyre-fixme[16]: Item `Awaitable` of
                    #  `Union[Awaitable[Awaitable[KeyedJaggedTensor]],
                    #  KJTSplitsAllToAllMeta]` has no attribute `splits`.
                    splits=awaitable.splits,
                    # pyre-fixme[16]: Item `Awaitable` of
                    #  `Union[Awaitable[Awaitable[KeyedJaggedTensor]],
                    #  KJTSplitsAllToAllMeta]` has no attribute `input_splits`.
                    input_splits=awaitable.input_splits,
                    output_splits=output_splits,
                    # pyre-fixme[16]: Item `Awaitable` of
                    #  `Union[Awaitable[Awaitable[KeyedJaggedTensor]],
                    #  KJTSplitsAllToAllMeta]` has no attribute `input_tensors`.
                    input_tensors=awaitable.input_tensors,
                    # pyre-fixme[16]: Item `Awaitable` of
                    #  `Union[Awaitable[Awaitable[KeyedJaggedTensor]],
                    #  KJTSplitsAllToAllMeta]` has no attribute `labels`.
                    labels=awaitable.labels,
                    batch_size_per_rank=batch_size_per_rank,
                    # pyre-fixme[16]: Item `Awaitable` of
                    #  `Union[Awaitable[Awaitable[KeyedJaggedTensor]],
                    #  KJTSplitsAllToAllMeta]` has no attribute `keys`.
                    keys=awaitable.keys,
                    # pyre-fixme[16]: Item `Awaitable` of
                    #  `Union[Awaitable[Awaitable[KeyedJaggedTensor]],
                    #  KJTSplitsAllToAllMeta]` has no attribute `device`.
                    device=awaitable.device,
                    # pyre-fixme[16]: Item `Awaitable` of
                    #  `Union[Awaitable[Awaitable[KeyedJaggedTensor]],
                    #  KJTSplitsAllToAllMeta]` has no attribute `stagger`.
                    stagger=awaitable.stagger,
                )
            )
        output = []
        awaitables_per_output = _split(tensors_awaitables, self._output_lengths)
        for awaitables, ctx in zip(awaitables_per_output, self._contexts):
            _set_sharding_context(awaitables, ctx)
            output.append(KJTListAwaitable(awaitables))
        return output


class ListOfKJTListAwaitable(Awaitable[ListOfKJTList]):
    """
    This module handles the tables-wise sharding input features distribution for
    inference.

    Args:
        awaitables (List[Awaitable[KJTList]]): list of `Awaitable` of `KJTList`.
    """

    def __init__(
        self,
        awaitables: List[Awaitable[KJTList]],
    ) -> None:
        super().__init__()
        self.awaitables = awaitables

    def _wait_impl(self) -> ListOfKJTList:
        """
        Syncs sparse features in list of KJTList.

        Returns:
            ListOfKJTList: synced `ListOfKJTList`.

        """
        return ListOfKJTList([w.wait() for w in self.awaitables])


class ListOfKJTListSplitsAwaitable(Awaitable[Awaitable[ListOfKJTList]]):
    """
    Awaitable of Awaitable of ListOfKJTList.

    Args:
        awaitables (List[Awaitable[Awaitable[KJTList]]]): list of `Awaitable`
            of `Awaitable` of sparse features list.
    """

    def __init__(
        self,
        awaitables: List[Awaitable[Awaitable[KJTList]]],
    ) -> None:
        super().__init__()
        self.awaitables = awaitables

    def _wait_impl(self) -> Awaitable[ListOfKJTList]:
        """
        Calls first wait on the awaitable of awaitable of ListOfKJTList.

        Returns:
            Awaitable[ListOfKJTList]: awaitable of `ListOfKJTList`.

        """
        return ListOfKJTListAwaitable([w.wait() for w in self.awaitables])


F = TypeVar("F", bound=Multistreamable)
T = TypeVar("T")
W = TypeVar("W")


@dataclass
class EmbeddingShardingContext(Multistreamable):
    batch_size_per_rank: List[int] = field(default_factory=list)

    def record_stream(self, stream: torch.cuda.streams.Stream) -> None:
        pass


class BaseSparseFeaturesDist(abc.ABC, nn.Module, Generic[F]):
    """
    Converts input from data-parallel to model-parallel.
    """

    @abc.abstractmethod
    def forward(
        self,
        sparse_features: KeyedJaggedTensor,
    ) -> Union[Awaitable[Awaitable[F]], F]:
        pass


class BaseEmbeddingDist(abc.ABC, nn.Module, Generic[C, T, W]):
    """
    Converts output of EmbeddingLookup from model-parallel to data-parallel.
    """

    @abc.abstractmethod
    def forward(
        self,
        local_embs: T,
        sharding_ctx: Optional[C] = None,
    ) -> Union[Awaitable[W], W]:
        pass


class EmbeddingSharding(abc.ABC, Generic[C, F, T, W], FeatureShardingMixIn):
    """
    Used to implement different sharding types for `EmbeddingBagCollection`, e.g.
    table_wise.
    """

    def __init__(
        self, qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None
    ) -> None:

        self._qcomm_codecs_registry = qcomm_codecs_registry

    @property
    def qcomm_codecs_registry(self) -> Optional[Dict[str, QuantizedCommCodecs]]:
        return self._qcomm_codecs_registry

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
    ) -> BaseEmbeddingDist[C, T, W]:
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
    def embedding_names_per_rank(self) -> List[List[str]]:
        pass


@dataclass
class EmbeddingShardingInfo:
    embedding_config: EmbeddingTableConfig
    param_sharding: ParameterSharding
    param: torch.Tensor
    fused_params: Optional[Dict[str, Any]] = None
