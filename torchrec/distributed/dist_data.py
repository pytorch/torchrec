#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from typing import List, Optional, Callable

import torch
import torch.distributed as dist
from torch import nn
from torch.autograd.profiler import record_function
from torchrec.distributed.comm_ops import (
    alltoall_pooled,
    alltoall_sequence,
    reduce_scatter_pooled,
)
from torchrec.distributed.types import Awaitable, NoWait
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

try:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:merge_pooled_embeddings")
    torch.ops.load_library(
        "//deeplearning/fbgemm/fbgemm_gpu:merge_pooled_embeddings_cpu"
    )
except OSError:
    pass

# OSS
try:
    import fbgemm_gpu  # @manual # noqa
except ImportError:
    pass


def _recat(local_split: int, num_splits: int, stagger: int = 1) -> List[int]:
    """
    Calculates relevant recat indices required to reorder AlltoAll collective.

    Args:
        local_split: how many features in local split.
        num_splits: how many splits (typically WORLD_SIZE).
        stagger: secondary reordering, (typically 1, but WORLD_SIZE/LOCAL_WORLD_SIZE
            for TWRW).

    Returns:
        List[int]

    Example:
        >>> _recat(2, 4, 1)
            [0, 2, 4, 6, 1, 3, 5, 7]
        >>> _recat(2, 4, 2)
            [0, 4, 2, 6, 1, 5, 3, 7]
    """

    recat: List[int] = []

    feature_order: List[int] = [
        x + num_splits // stagger * y
        for x in range(num_splits // stagger)
        for y in range(stagger)
    ]

    for i in range(local_split):
        for j in feature_order:  # range(num_splits):
            recat.append(i + j * local_split)
    return recat


def _split_lengths(
    splits: List[int], keys: List[str], offset_per_key: List[int]
) -> List[int]:
    # Calculates lengths [x1, x2, x3, ..., y1, y2], splits [3, ..., 2]
    #   -> [x1+x2+x3, ..., y1+y2]
    length_per_split: List[int] = []
    i = 0
    offset = 0
    for split in splits:
        new_offset = offset_per_key[i + split]
        length_per_split.append(new_offset - offset)
        i += split
        offset = new_offset
    return length_per_split


class KJTAllToAllIndicesAwaitable(Awaitable[KeyedJaggedTensor]):
    """
    Awaitable for KJT indices and weights All2All.

    Args:
        pg  (dist.ProcessGroup): ProcessGroup for AlltoAll communication.
        input (KeyedJaggedTensor): Input KJT tensor.
        splits (List[int]): List of len(pg.size()) which indicates how many features to
            send to each pg.rank(). It is assumed the KeyedJaggedTensor is ordered by
            destination rank. Same for all ranks.
        keys (List[str]): KJT keys after AlltoAll.
        recat (torch.Tensor): recat tensor for reordering tensor order after AlltoAll.
        in_lengths_per_worker (List[str]): indices number of indices each rank will get.
    """

    def __init__(
        self,
        # pyre-fixme[11]: Annotation `ProcessGroup` is not defined as a type.
        pg: dist.ProcessGroup,
        input: KeyedJaggedTensor,
        splits: List[int],
        keys: List[str],
        recat: torch.Tensor,
        in_lengths_per_worker: List[int],
    ) -> None:
        super().__init__()
        self._workers: int = pg.size()
        self._device: torch.device = input.values().device
        self._recat = recat
        self._splits = splits
        self._pg: dist.ProcessGroup = pg
        self._keys = keys
        self._in_lengths_per_worker: List[int] = []
        self._out_lengths_per_worker: List[int] = []
        self._input = input
        if self._workers == 1:
            return

        self._lengths: torch.Tensor = input.lengths()

        self._in_lengths_per_worker = in_lengths_per_worker
        self._out_lengths_per_worker = (
            self._lengths.view(self._workers, -1).sum(dim=1).cpu().tolist()
        )

        in_values = self._input.values().view(-1)
        out_values = torch.empty(
            sum(self._out_lengths_per_worker),
            device=self._device,
            dtype=in_values.dtype,
        )
        with record_function("## all2all_data:indices ##"):
            # pyre-fixme[11]: Annotation `Work` is not defined as a type.
            self._values_awaitable: dist.Work = dist.all_to_all_single(
                output=out_values,
                input=in_values,
                output_split_sizes=self._out_lengths_per_worker,
                input_split_sizes=self._in_lengths_per_worker,
                group=self._pg,
                async_op=True,
            )

        self._values: torch.Tensor = out_values

        self._weights_awaitable: Optional[dist.Work] = None
        self._weights: Optional[torch.Tensor] = None

        if self._input.weights_or_none() is not None:
            in_weights = self._input.weights().view(-1)
            out_weights = torch.empty(
                sum(self._out_lengths_per_worker),
                device=self._device,
                dtype=in_weights.dtype,
            )
            with record_function("## all2all_data:weights ##"):
                self._weights_awaitable: dist.Work = dist.all_to_all_single(
                    output=out_weights,
                    input=in_weights,
                    output_split_sizes=self._out_lengths_per_worker,
                    input_split_sizes=self._in_lengths_per_worker,
                    group=self._pg,
                    async_op=True,
                )
            self._weights: torch.Tensor = out_weights

    def _wait_impl(self) -> KeyedJaggedTensor:
        """
        Overwrites wait function as we don't handle callbacks here.

        Returns:
            KeyedJaggedTensor: Synced KJT after AlltoAll.
        """

        if self._workers == 1:
            self._input.sync()
            return self._input

        self._values_awaitable.wait()

        if self._weights_awaitable:
            self._weights_awaitable.wait()

        keys = self._keys
        lengths = self._lengths
        values = self._values
        weights = self._weights

        with record_function("## all2all_data:recat_values ##"):
            if self._recat.numel():
                lengths, values, weights = torch.ops.fbgemm.permute_2D_sparse_data(
                    self._recat,
                    lengths.view(self._workers * self._splits[self._pg.rank()], -1),
                    values,
                    weights,
                    values.numel(),
                )
                lengths = lengths.view(-1)

        ret = KeyedJaggedTensor.from_lengths_sync(
            keys=keys,
            values=values,
            weights=weights,
            lengths=lengths,
            stride=self._workers * self._input.stride(),
        )
        return ret


class KJTAllToAllLengthsAwaitable(Awaitable[KJTAllToAllIndicesAwaitable]):
    """
    Awaitable for KJT's lengths AlltoAll.

    wait() waits on lengths AlltoAll, then instantiates `KJTAllToAllIndicesAwaitable`
    awaitable where indices and weights AlltoAll will be issued.

    Args:
        pg  (dist.ProcessGroup): ProcessGroup for AlltoAll communication.
        input (KeyedJaggedTensor): Input KJT tensor
        splits (List[int]): List of len(pg.size()) which indicates how many features to
            send to each pg.rank(). It is assumed the KeyedJaggedTensor is ordered by
            destination rank. Same for all ranks.
        keys (List[str]): KJT keys after AlltoAll
        recat (torch.Tensor): recat tensor for reordering tensor order after AlltoAll.
    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
        input: KeyedJaggedTensor,
        splits: List[int],
        keys: List[str],
        recat: torch.Tensor,
    ) -> None:
        super().__init__()
        self._workers: int = pg.size()
        self._input = input
        self._in_lengths_per_worker: List[int] = []
        self._recat = recat
        self._splits = splits
        self._pg: dist.ProcessGroup = pg
        self._device: torch.device = input.values().device
        self._keys = keys
        if self._workers == 1:
            return
        dim_0 = splits[pg.rank()]
        dim_1 = input.stride()
        in_lengths = input.lengths().view(-1)
        out_lengths = torch.empty(
            dim_0 * dim_1 * self._workers,
            device=self._device,
            dtype=in_lengths.dtype,
        )
        self._in_lengths_per_worker = _split_lengths(
            splits, input.keys(), input.offset_per_key()
        )

        with record_function("## all2all_data:lengths ##"):
            self._lengths_awaitable: dist.Work = dist.all_to_all_single(
                output=out_lengths,
                input=in_lengths,
                output_split_sizes=[dim_0 * dim_1] * self._workers,
                input_split_sizes=[split * dim_1 for split in self._splits],
                group=self._pg,
                async_op=True,
            )
            self._lengths: torch.Tensor = out_lengths

    def _wait_impl(self) -> KJTAllToAllIndicesAwaitable:
        """
        Overwrites wait function as we don't handle callbacks here.

        Returns:
            KJTAllToAllIndicesAwaitable.
        """

        if self._workers == 1:
            kjt = self._input
        else:
            self._lengths_awaitable.wait()
            kjt = KeyedJaggedTensor(
                keys=self._keys,
                values=self._input.values(),
                weights=self._input.weights_or_none(),
                lengths=self._lengths,
                stride=self._input.stride(),
            )

        ret = KJTAllToAllIndicesAwaitable(
            self._pg,
            kjt,
            self._splits,
            self._keys,
            self._recat,
            self._in_lengths_per_worker,
        )
        return ret


class KJTAllToAll(nn.Module):
    """
    Redistributes `KeyedJaggedTensor` to a `ProcessGroup` according to splits.

    Implementation utilizes AlltoAll collective as part of torch.distributed.
    Requires two collective calls, one to transmit final tensor lengths (to allocate
    correct space), and one to transmit actual sparse values.

    Args:
        pg (dist.ProcessGroup): ProcessGroup for AlltoAll communication.
        splits (List[int]): List of len(pg.size()) which indicates how many features to
            send to each pg.rank(). It is assumed the KeyedJaggedTensor is ordered by
            destination rank. Same for all ranks.
        device (Optional[torch.device]): device on which buffers will be allocated.
        stagger (int): stagger value to apply to recat tensor, see _recat function for
            more detail.

    Example:
        >>> keys=['A','B','C']
        >>> splits=[2,1]
        >>> kjtA2A = KJTAllToAll(pg, splits, device)
        >>> awaitable = kjtA2A(rank0_input)

        where:
            rank0_input is KeyedJaggedTensor holding

                    0           1           2
            'A'    [A.V0]       None        [A.V1, A.V2]
            'B'    None         [B.V0]      [B.V1]
            'C'    [C.V0]       [C.V1]      None

            rank1_input is KeyedJaggedTensor holding

                    0           1           2
            'A'     [A.V3]      [A.V4]      None
            'B'     None        [B.V2]      [B.V3, B.V4]
            'C'     [C.V2]      [C.V3]      None

        >>> rank0_output = awaitable.wait()

        where:
            rank0_output is KeyedJaggedTensor holding

                    0           1           2           3           4           5
            'A'     [A.V0]      None      [A.V1, A.V2]  [A.V3]      [A.V4]      None
            'B'     None        [B.V0]    [B.V1]        None        [B.V2]      [B.V3, B.V4]

            rank1_output is KeyedJaggedTensor holding
                    0           1           2           3           4           5
            'C'     [C.V0]      [C.V1]      None        [C.V2]      [C.V3]      None
    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
        splits: List[int],
        device: Optional[torch.device] = None,
        stagger: int = 1,
    ) -> None:
        super().__init__()
        assert len(splits) == pg.size()
        self._pg: dist.ProcessGroup = pg
        self._splits = splits
        self._no_dist: bool = all(s == 0 for s in splits)
        self._splits_cumsum: List[int] = [0] + list(itertools.accumulate(splits))
        self.register_buffer(
            "_recat",
            torch.tensor(
                _recat(
                    local_split=splits[pg.rank()],
                    num_splits=len(splits),
                    stagger=stagger,
                ),
                device=device,
                dtype=torch.int,
            ),
        )

    def forward(
        self, input: KeyedJaggedTensor
    ) -> Awaitable[KJTAllToAllIndicesAwaitable]:
        """
        Sends input to relevant `ProcessGroup` ranks.
        First wait will have lengths results and issue indices/weights AlltoAll.
        Second wait will have indices/weights results.

        Args:
            input (KeyedJaggedTensor): input KeyedJaggedTensor of values to distribute.

        Returns:
            Awaitable[KeyedJaggedTensor]: awaitable of a KeyedJaggedTensor.
        """

        with torch.no_grad():
            assert len(input.keys()) == sum(self._splits)
            rank = dist.get_rank(self._pg)
            local_keys = input.keys()[
                self._splits_cumsum[rank] : self._splits_cumsum[rank + 1]
            ]

            return KJTAllToAllLengthsAwaitable(
                pg=self._pg,
                input=input,
                splits=self._splits,
                keys=local_keys,
                recat=self._recat,
            )


class KJTOneToAll(nn.Module):
    """
    Redistributes `KeyedJaggedTensor` to all devices.

    Implementation utilizes OnetoAll function, which essentially P2P copies the feature
    to the devices.

    Args:
        splits (List[int]): lengths of features to split the KeyJaggedTensor features
            into before copying them.
        world_size (int): number of devices in the topology.
        recat (torch.Tensor): recat tensor for reordering tensor order after AlltoAll.
    """

    def __init__(
        self,
        splits: List[int],
        world_size: int,
    ) -> None:
        super().__init__()
        self._splits = splits
        self._world_size = world_size
        assert self._world_size == len(splits)

    def forward(self, kjt: KeyedJaggedTensor) -> Awaitable[List[KeyedJaggedTensor]]:
        """
        Splits features first and then sends the slices to the corresponding devices.

        Args:
            kjt (KeyedJaggedTensor): the input features.

        Returns:
            Awaitable[List[KeyedJaggedTensor]]: awaitable of KeyedJaggedTensor splits.
        """

        kjts: List[KeyedJaggedTensor] = kjt.split(self._splits)
        dist_kjts = [
            split_kjt.to(torch.device("cuda", rank), non_blocking=True)
            for rank, split_kjt in enumerate(kjts)
        ]
        return NoWait(dist_kjts)


class PooledEmbeddingsAwaitable(Awaitable[torch.Tensor]):
    """
    Awaitable for pooled embeddings after collective operation.

    Args:
        tensor_awaitable (Awaitable[torch.Tensor]): awaitable of concatenated tensors
            from all the processes in the group after collective.
    """

    def __init__(
        self,
        tensor_awaitable: Awaitable[torch.Tensor],
    ) -> None:
        super().__init__()
        self._tensor_awaitable = tensor_awaitable

    def _wait_impl(self) -> torch.Tensor:
        """
        Syncs pooled embeddings after collective operation.

        Returns:
            torch.Tensor: synced pooled embeddings.
        """

        ret = self._tensor_awaitable.wait()
        return ret

    @property
    def callbacks(self) -> List[Callable[[torch.Tensor], torch.Tensor]]:
        return self._callbacks


class PooledEmbeddingsAllToAll(nn.Module):
    # TODO: potentially refactor to take KT instead of torch.Tensor: D29174501
    """
    Shards batches and collects keys of tensor with a `ProcessGroup` according to
    `dim_sum_per_rank`.

    Implementation utilizes `alltoall_pooled` operation.

    Args:
        pg (dist.ProcessGroup): ProcessGroup for AlltoAll communication.
        dim_sum_per_rank (List[int]): number of features (sum of dimensions) of the
            embedding in each rank.
        device (Optional[torch.device]): device on which buffers will be allocated.
        callbacks (Optional[List[Callable[[torch.Tensor], torch.Tensor]]])

    Example:
        >>> dim_sum_per_rank = [2, 1]
        >>> a2a = PooledEmbeddingsAllToAll(pg, dim_sum_per_rank, device)

        >>> t0 = torch.rand((6, 2))
        >>> t1 = torch.rand((6, 1))
        >>> rank0_output = a2a(t0).wait()
        >>> rank1_output = a2a(t1).wait()
        >>> print(rank0_output.size())
            torch.Size([3, 3])
        >>> print(rank1_output.size())
            torch.Size([3, 3])
    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
        dim_sum_per_rank: List[int],
        device: Optional[torch.device] = None,
        callbacks: Optional[List[Callable[[torch.Tensor], torch.Tensor]]] = None,
    ) -> None:
        super().__init__()
        # pyre-fixme[4]: Attribute must be annotated.
        self._pg = pg
        self._callbacks: List[Callable[[torch.Tensor], torch.Tensor]] = []
        if callbacks is not None:
            self._callbacks = callbacks

        self._dim_sum_per_rank = dim_sum_per_rank
        self.register_buffer(
            "_dim_sum_per_rank_tensor",
            torch.tensor(dim_sum_per_rank, device=device, dtype=torch.int),
        )
        cumsum_dim_sum_per_rank = list(itertools.accumulate(dim_sum_per_rank))
        self.register_buffer(
            "_cumsum_dim_sum_per_rank_tensor",
            torch.tensor(cumsum_dim_sum_per_rank, device=device, dtype=torch.int),
        )

    def forward(
        self, local_embs: torch.Tensor, batch_size_per_rank: Optional[List[int]] = None
    ) -> PooledEmbeddingsAwaitable:
        """
        Performs AlltoAll pooled operation on pooled embeddings tensor.

        Args:
            local_embs (torch.Tensor): tensor of values to distribute.

        Returns:
            PooledEmbeddingsAwaitable: awaitable of pooled embeddings.
        """

        if local_embs.numel() == 0:
            local_embs.view(local_embs.size(0) * self._pg.size(), 0)
        if batch_size_per_rank is None:
            B_global = local_embs.size(0)
            assert (
                B_global % self._pg.size() == 0
            ), f"num of ranks {self._pg.size()} doesn't divide global batch size {B_global}"
            B_local = B_global // self._pg.size()
            batch_size_per_rank = [B_local] * self._pg.size()
        tensor_awaitable = alltoall_pooled(
            a2a_pooled_embs_tensor=local_embs,
            batch_size_per_rank=batch_size_per_rank,
            dim_sum_per_rank=self._dim_sum_per_rank,
            dim_sum_per_rank_tensor=self._dim_sum_per_rank_tensor,
            cumsum_dim_sum_per_rank_tensor=self._cumsum_dim_sum_per_rank_tensor,
            group=self._pg,
        )

        pooled_embedding_awaitable = PooledEmbeddingsAwaitable(
            tensor_awaitable=tensor_awaitable
        )
        pooled_embedding_awaitable.callbacks.extend(self._callbacks)

        return pooled_embedding_awaitable

    @property
    def callbacks(self) -> List[Callable[[torch.Tensor], torch.Tensor]]:
        return self._callbacks


class PooledEmbeddingsAllToOne(nn.Module):
    """
    Merges the pooled embedding tensor on each device into single tensor.

    Args:
        device (torch.device): device on which buffer will be allocated
        world_size (int): number of devices in the topology.
    """

    def __init__(
        self,
        device: torch.device,
        world_size: int,
    ) -> None:
        super().__init__()
        self._device = device
        self._world_size = world_size

    def forward(self, tensors: List[torch.Tensor]) -> Awaitable[torch.Tensor]:
        """
        Performs AlltoOne operation on pooled embeddings tensors.

        Args:
            tensors (List[torch.Tensor]): list of pooled embedding tensors.

        Returns:
            Awaitable[torch.Tensor]: awaitable of the merged pooled embeddings.
        """

        assert len(tensors) == self._world_size
        return NoWait(
            torch.ops.fbgemm.merge_pooled_embeddings(
                tensors,
                tensors[0].size(0),
                self._device,
            )
        )


class PooledEmbeddingsReduceScatter(nn.Module):
    """
    The module class that wraps reduce-scatter communication primitive for pooled
    embedding communication in row-wise and twrw sharding.

    For pooled embeddings, we have a local model-parallel output tensor with a layout of
    [num_buckets x batch_size, dimension]. We need to sum over num_buckets dimension
    across batches. We split tensor along the first dimension into equal chunks (tensor
    slices of different buckets) and reduce them into the output tensor and scatter the
    results for corresponding ranks.

    The class returns the async `Awaitable` handle for pooled embeddings tensor.
    The reduce-scatter is only available for NCCL backend.

    Args:
        pg (dist.ProcessGroup): The process group that the reduce-scatter communication
            happens within.

    Example:
        >>> init_distributed(rank=rank, size=2, backend="nccl")
        >>> pg = dist.new_group(backend="nccl")
        >>> input = torch.randn(2 * 2, 2)
        >>> m = PooledEmbeddingsReduceScatter(pg)
        >>> output = m(input)
        >>> tensor = output.wait()
    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
    ) -> None:
        super().__init__()
        # pyre-fixme[4]: Attribute must be annotated.
        self._pg = pg

    def forward(self, local_embs: torch.Tensor) -> PooledEmbeddingsAwaitable:
        """
        Performs reduce scatter operation on pooled embeddings tensor.

        Args:
            local_embs (torch.Tensor): tensor of shape [num_buckets x batch_size, dimension].

        Returns:
            PooledEmbeddingsAwaitable: awaitable of pooled embeddings of tensor of shape [batch_size, dimension].
        """

        tensor_awaitable = reduce_scatter_pooled(
            list(torch.chunk(local_embs, self._pg.size(), dim=0)), self._pg
        )
        return PooledEmbeddingsAwaitable(tensor_awaitable=tensor_awaitable)


class SequenceEmbeddingsAwaitable(Awaitable[torch.Tensor]):
    """
    Awaitable for sequence embeddings after collective operation.

    Args:
        tensor_awaitable (Awaitable[torch.Tensor]): awaitable of concatenated tensors
            from all the processes in the group after collective.
        unbucketize_permute_tensor (Optional[torch.Tensor]): stores the permute order of
            KJT bucketize (for row-wise sharding only).
    """

    def __init__(
        self,
        tensor_awaitable: Awaitable[torch.Tensor],
        unbucketize_permute_tensor: Optional[torch.Tensor],
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self._tensor_awaitable = tensor_awaitable
        self._unbucketize_permute_tensor = unbucketize_permute_tensor
        self._embedding_dim = embedding_dim

        if self._unbucketize_permute_tensor is not None:
            self.callbacks.append(
                lambda ret: torch.index_select(
                    ret.view(-1, self._embedding_dim),
                    0,
                    self._unbucketize_permute_tensor,
                )
            )

    def _wait_impl(self) -> torch.Tensor:
        """
        Syncs sequence embeddings after collective operation.

        Returns:
            torch.Tensor: synced pooled embeddings.
        """

        ret = self._tensor_awaitable.wait()
        return ret


class SequenceEmbeddingAllToAll(nn.Module):
    """
    Redistributes sequence embedding to a `ProcessGroup` according to splits.

    Args:
        pg (dist.ProcessGroup): the process group that the AlltoAll communication
            happens within.
        features_per_rank (List[int]): List of number of features per rank.
        device (Optional[torch.device]): device on which buffers will be allocated.

    Example:
        >>> init_distributed(rank=rank, size=2, backend="nccl")
        >>> pg = dist.new_group(backend="nccl")
        >>> features_per_rank = [4, 4]
        >>> m = SequenceEmbeddingAllToAll(pg, features_per_rank)
        >>> local_embs = torch.rand((6, 2))
        >>> sharding_ctx: SequenceShardingContext
        >>> output = m(
            local_embs=local_embs,
            lengths=sharding_ctx.lengths_after_input_dist,
            input_splits=sharding_ctx.input_splits,
            output_splits=sharding_ctx.output_splits,
            unbucketize_permute_tensor=None,
        )
        >>> tensor = output.wait()
    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
        features_per_rank: List[int],
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        # pyre-fixme[4]: Attribute must be annotated.
        self._pg = pg

        forward_recat = []
        for j in range(self._pg.size()):
            for i in range(features_per_rank[self._pg.rank()]):
                forward_recat.append(j + i * self._pg.size())
        self.register_buffer(
            "_forward_recat_tensor",
            torch.tensor(forward_recat, device=device, dtype=torch.int),
        )
        backward_recat = []
        for i in range(features_per_rank[self._pg.rank()]):
            for j in range(self._pg.size()):
                backward_recat.append(i + j * features_per_rank[self._pg.rank()])
        self.register_buffer(
            "_backward_recat_tensor",
            torch.tensor(backward_recat, device=device, dtype=torch.int),
        )

    def forward(
        self,
        local_embs: torch.Tensor,
        lengths: torch.Tensor,
        input_splits: List[int],
        output_splits: List[int],
        unbucketize_permute_tensor: Optional[torch.Tensor] = None,
    ) -> SequenceEmbeddingsAwaitable:
        """
        Performs AlltoAll operation on sequence embeddings tensor.

        Args:
            local_embs (torch.Tensor): input embeddings tensor.
            lengths (torch.Tensor): lengths of sparse features after AlltoAll.
            input_splits (List[int]): input splits of AlltoAll.
            output_splits (List[int]): output splits of AlltoAll.
            unbucketize_permute_tensor (Optional[torch.Tensor]): stores the permute order
                of the KJT bucketize (for row-wise sharding only).

        Returns:
            SequenceEmbeddingsAwaitable
        """

        tensor_awaitable = alltoall_sequence(
            a2a_sequence_embs_tensor=local_embs,
            forward_recat_tensor=self._forward_recat_tensor,
            backward_recat_tensor=self._backward_recat_tensor,
            lengths_after_sparse_data_all2all=lengths,
            input_splits=input_splits,
            output_splits=output_splits,
            group=self._pg,
        )
        return SequenceEmbeddingsAwaitable(
            tensor_awaitable=tensor_awaitable,
            unbucketize_permute_tensor=unbucketize_permute_tensor,
            embedding_dim=local_embs.shape[1],
        )
