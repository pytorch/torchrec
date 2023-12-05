#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar, Union

import torch
from torch import distributed as dist, nn
from torchrec.distributed.dist_data import (
    KJTAllToAllTensorsAwaitable,
    SplitsAllToAllAwaitable,
)
from torchrec.distributed.embedding_dim_bucketer import (
    EmbDimBucketer,
    EmbDimBucketerPolicy,
    should_do_dim_bucketing,
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
from torchrec.fx.utils import assert_fx_safe
from torchrec.modules.embedding_configs import EmbeddingTableConfig
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.streamable import Multistreamable

torch.fx.wrap("len")

CACHE_LOAD_FACTOR_STR: str = "cache_load_factor"


# torch.Tensor.to can not be fx symbolic traced as it does not go through __torch_dispatch__ => fx.wrap it
@torch.fx.wrap
def _fx_wrap_tensor_to_device_dtype(
    t: torch.Tensor, tensor_device_dtype: torch.Tensor
) -> torch.Tensor:
    return t.to(device=tensor_device_dtype.device, dtype=tensor_device_dtype.dtype)


@torch.fx.wrap
def _fx_wrap_batch_size_per_feature(kjt: KeyedJaggedTensor) -> Optional[torch.Tensor]:
    return (
        torch.tensor(kjt.stride_per_key(), device=kjt.device())
        if kjt.variable_stride_per_key()
        else None
    )


@torch.fx.wrap
def _fx_wrap_max_B(kjt: KeyedJaggedTensor) -> int:
    return max(kjt.stride_per_key()) if kjt.variable_stride_per_key() else -1


@torch.fx.wrap
def _fx_wrap_stride(kjt: KeyedJaggedTensor) -> Optional[int]:
    return None if kjt.variable_stride_per_key() else kjt.stride()


@torch.fx.wrap
def _fx_wrap_stride_per_key_per_rank(
    kjt: KeyedJaggedTensor, num_buckets: int
) -> Optional[List[List[int]]]:
    return (
        kjt.stride_per_key_per_rank() * num_buckets
        if kjt.variable_stride_per_key()
        else None
    )


def bucketize_kjt_before_all2all(
    kjt: KeyedJaggedTensor,
    num_buckets: int,
    block_sizes: torch.Tensor,
    output_permute: bool = False,
    bucketize_pos: bool = False,
    block_bucketize_row_pos: Optional[List[torch.Tensor]] = None,
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
        block_bucketize_row_pos (Optional[List[torch.Tensor]]): The offsets of shard size for each feature.

    Returns:
        Tuple[KeyedJaggedTensor, Optional[torch.Tensor]]: the bucketized `KeyedJaggedTensor` and the optional permute mapping from the unbucketized values to bucketized value.
    """

    num_features = len(kjt.keys())
    assert_fx_safe(
        block_sizes.numel() == num_features,
        f"Expecting block sizes for {num_features} features, but {block_sizes.numel()} received.",
    )
    block_sizes_new_type = _fx_wrap_tensor_to_device_dtype(block_sizes, kjt.values())
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
        batch_size_per_feature=_fx_wrap_batch_size_per_feature(kjt),
        max_B=_fx_wrap_max_B(kjt),
        block_bucketize_pos=block_bucketize_row_pos,
    )

    return (
        KeyedJaggedTensor(
            # duplicate keys will be resolved by AllToAll
            keys=kjt.keys() * num_buckets,
            values=bucketized_indices,
            weights=pos if bucketize_pos else bucketized_weights,
            lengths=bucketized_lengths.view(-1),
            offsets=None,
            stride=_fx_wrap_stride(kjt),
            stride_per_key_per_rank=_fx_wrap_stride_per_key_per_rank(kjt, num_buckets),
            length_per_key=None,
            offset_per_key=None,
            index_per_key=None,
        ),
        unbucketize_permute,
    )


def _get_weighted_avg_cache_load_factor(
    embedding_tables: List[ShardedEmbeddingTable],
) -> Optional[float]:
    """
    Calculate the weighted average cache load factor of all tables. The cache
    load factors are weighted by the hash size of each table.
    """
    cache_load_factor_sum: float = 0.0
    weight: int = 0

    for table in embedding_tables:
        if (
            table.compute_kernel == EmbeddingComputeKernel.FUSED_UVM_CACHING
            and table.fused_params
            and CACHE_LOAD_FACTOR_STR in table.fused_params
        ):
            cache_load_factor_sum += (
                table.fused_params[CACHE_LOAD_FACTOR_STR] * table.num_embeddings
            )
            weight += table.num_embeddings

    # if no fused_uvm_caching tables, return default cache load factor
    if weight == 0:
        return None

    return cache_load_factor_sum / weight


def _get_grouping_fused_params(
    fused_params: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    Only shallow copy the fused params we need for grouping tables into TBEs. In
    particular, we do not copy cache_load_factor.
    """
    grouping_fused_params: Optional[Dict[str, Any]] = copy.copy(fused_params)

    if not grouping_fused_params:
        return grouping_fused_params

    if CACHE_LOAD_FACTOR_STR in grouping_fused_params:
        del grouping_fused_params[CACHE_LOAD_FACTOR_STR]

    return grouping_fused_params


def _get_compute_kernel_type(
    compute_kernel: EmbeddingComputeKernel,
) -> EmbeddingComputeKernel:
    """
    Return the compute kernel type for the given compute kernel.
    """
    compute_kernel_type = compute_kernel
    if compute_kernel_type in [
        EmbeddingComputeKernel.FUSED_UVM,
        EmbeddingComputeKernel.FUSED_UVM_CACHING,
    ]:
        compute_kernel_type = EmbeddingComputeKernel.FUSED
    elif compute_kernel_type in [
        EmbeddingComputeKernel.QUANT_UVM,
        EmbeddingComputeKernel.QUANT_UVM_CACHING,
    ]:
        compute_kernel_type = EmbeddingComputeKernel.QUANT
    return compute_kernel_type


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

        emb_dim_bucketer_policy = (
            EmbDimBucketerPolicy.ALL_BUCKETS
            if should_do_dim_bucketing(embedding_tables)
            else EmbDimBucketerPolicy.SINGLE_BUCKET
        )
        emb_dim_bucketer = EmbDimBucketer(embedding_tables, emb_dim_bucketer_policy)
        logging.info(f"bucket count {emb_dim_bucketer.bucket_count()}")

        # populate grouping keys
        grouping_keys = []
        for table in embedding_tables:
            if table.fused_params is None:
                table.fused_params = {}

            grouping_key = (
                table.data_type,
                table.pooling,
                table.has_feature_processor,
                _get_grouping_fused_params(table.fused_params),
                _get_compute_kernel_type(table.compute_kernel),
                emb_dim_bucketer.get_bucket(table.embedding_dim, table.data_type),
            )
            if grouping_key not in grouping_keys:
                grouping_keys.append(grouping_key)

        for grouping_key in grouping_keys:
            (
                data_type,
                pooling,
                has_feature_processor,
                fused_params_group,
                compute_kernel_type,
                dim_bucket,
            ) = grouping_key
            grouped_tables: List[ShardedEmbeddingTable] = []
            is_weighted = False
            for table in embedding_tables:
                is_weighted = table.is_weighted
                if (
                    table.data_type == data_type
                    and table.pooling.value == pooling.value
                    and table.has_feature_processor == has_feature_processor
                    and fused_params_group
                    == _get_grouping_fused_params(table.fused_params)
                    and compute_kernel_type
                    == _get_compute_kernel_type(table.compute_kernel)
                    and (
                        emb_dim_bucketer.get_bucket(
                            table.embedding_dim, table.data_type
                        )
                        == dim_bucket
                    )
                ):
                    grouped_tables.append(table)

            if fused_params_group is None:
                fused_params_group = {}

            if grouped_tables:
                logging.info(
                    f"{len(grouped_tables)} tables are grouped for bucket: {dim_bucket}."
                )
                cache_load_factor = _get_weighted_avg_cache_load_factor(grouped_tables)
                per_tbe_fused_params = copy.copy(fused_params_group)
                if cache_load_factor is not None:
                    per_tbe_fused_params[CACHE_LOAD_FACTOR_STR] = cache_load_factor

                grouped_embedding_configs.append(
                    GroupedEmbeddingConfig(
                        data_type=data_type,
                        pooling=pooling,
                        is_weighted=is_weighted,
                        has_feature_processor=has_feature_processor,
                        compute_kernel=compute_kernel_type,
                        embedding_tables=grouped_tables,
                        fused_params={
                            k: v
                            for k, v in per_tbe_fused_params.items()
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


C = TypeVar("C", bound=Multistreamable)
T = TypeVar("T")


class KJTListAwaitable(Awaitable[KJTList]):
    """
    Awaitable of KJTList.

    Args:
        awaitables (List[Awaitable[KeyedJaggedTensor]]): list of `Awaitable` of sparse
            features.
        ctx (C): sharding context to save the batch size info from the KJT for the
            embedding AlltoAll.
    """

    def __init__(
        self,
        awaitables: List[Awaitable[KeyedJaggedTensor]],
        ctx: C,
    ) -> None:
        super().__init__()
        self.awaitables = awaitables
        self.ctx = ctx

    def _wait_impl(self) -> KJTList:
        """
        Syncs KJTs in `KJTList`.

        Returns:
            KJTList: synced `KJTList`.
        """
        kjts = [w.wait() for w in self.awaitables]
        _set_sharding_context_post_a2a(kjts, self.ctx)
        return KJTList(kjts)


def _set_sharding_context_post_a2a(
    kjts: List[KeyedJaggedTensor],
    ctx: C,
) -> None:
    for kjt, sharding_context in zip(kjts, getattr(ctx, "sharding_contexts", [])):
        if (
            hasattr(sharding_context, "batch_size_per_rank_per_feature")
            and kjt.variable_stride_per_key()
            and kjt.stride_per_key_per_rank()
        ):
            sharding_context.batch_size_per_rank_per_feature = [
                [
                    kjt.stride_per_key_per_rank()[i][j]
                    for i in range(len(kjt.stride_per_key_per_rank()))
                ]
                for j in range(len(kjt.stride_per_key_per_rank()[0]))
            ]


def _set_sharding_context_intra_a2a(
    tensors_awaitables: List[Awaitable[KeyedJaggedTensor]],
    ctx: C,
) -> None:
    for awaitable, sharding_context in zip(
        tensors_awaitables,
        getattr(ctx, "sharding_contexts", []),
    ):
        if isinstance(awaitable, KJTAllToAllTensorsAwaitable):
            if hasattr(sharding_context, "input_splits"):
                sharding_context.input_splits = awaitable._input_splits["values"]
            if hasattr(sharding_context, "output_splits"):
                sharding_context.output_splits = awaitable._output_splits["values"]
            if hasattr(sharding_context, "sparse_features_recat"):
                sharding_context.sparse_features_recat = awaitable._recat
            if (
                hasattr(sharding_context, "batch_size_per_rank")
                and awaitable._stride_per_rank is not None
            ):
                sharding_context.batch_size_per_rank = awaitable._stride_per_rank


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
        _set_sharding_context_intra_a2a(tensors_awaitables, self.ctx)
        return KJTListAwaitable(tensors_awaitables, self.ctx)


@dataclass
class KJTSplitsAllToAllMeta:
    pg: dist.ProcessGroup
    _input: KeyedJaggedTensor
    splits: List[int]
    splits_tensors: List[torch.Tensor]
    input_splits: List[List[int]]
    input_tensors: List[torch.Tensor]
    labels: List[str]
    keys: List[str]
    device: torch.device
    stagger: int
    splits_cumsum: List[int]


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
            if splits_tensors and pg is not None
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
                assert isinstance(awaitable, Awaitable)
                tensors_awaitables.append(awaitable.wait())
                continue
            assert isinstance(awaitable, KJTSplitsAllToAllMeta)
            if awaitable._input.variable_stride_per_key():
                output_splits = splits
                stride_per_rank = None
            else:
                output_splits = splits[:-1]
                stride_per_rank = splits[-1]
            tensors_awaitables.append(
                KJTAllToAllTensorsAwaitable(
                    pg=awaitable.pg,
                    input=awaitable._input,
                    splits=awaitable.splits,
                    input_splits=awaitable.input_splits,
                    output_splits=output_splits,
                    input_tensors=awaitable.input_tensors,
                    labels=awaitable.labels,
                    keys=awaitable.keys,
                    device=awaitable.device,
                    stagger=awaitable.stagger,
                    stride_per_rank=stride_per_rank,
                )
            )
        output = []
        awaitables_per_output = _split(tensors_awaitables, self._output_lengths)
        for awaitables, ctx in zip(awaitables_per_output, self._contexts):
            _set_sharding_context_intra_a2a(awaitables, ctx)
            output.append(KJTListAwaitable(awaitables, ctx))
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
    batch_size_per_rank_per_feature: List[List[int]] = field(default_factory=list)
    batch_size_per_feature_pre_a2a: List[int] = field(default_factory=list)
    variable_batch_per_feature: bool = False

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

    def embedding_tables(self) -> List[ShardedEmbeddingTable]:
        raise NotImplementedError

    def uncombined_embedding_dims(self) -> List[int]:
        return self.embedding_dims()

    def uncombined_embedding_names(self) -> List[str]:
        return self.embedding_names()


@dataclass
class EmbeddingShardingInfo:
    embedding_config: EmbeddingTableConfig
    param_sharding: ParameterSharding
    param: torch.Tensor
    fused_params: Optional[Dict[str, Any]] = None
