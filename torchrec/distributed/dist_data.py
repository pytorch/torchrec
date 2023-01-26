#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
from typing import Callable, Dict, List, Optional

import torch
import torch.distributed as dist
from torch import nn
from torch.autograd.profiler import record_function
from torchrec.distributed.comm_ops import (
    all_gather_base_pooled,
    alltoall_pooled,
    alltoall_sequence,
    reduce_scatter_base_pooled,
    reduce_scatter_v_pooled,
)
from torchrec.distributed.embedding_types import KJTList
from torchrec.distributed.types import Awaitable, NoWait, QuantizedCommCodecs
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
    import fbgemm_gpu  # @manual  # noqa
except ImportError:
    pass

logger: logging.Logger = logging.getLogger()


def _get_recat(
    local_split: int,
    num_splits: int,
    stagger: int = 1,
    device: Optional[torch.device] = None,
    batch_size_per_rank: Optional[List[int]] = None,
) -> Optional[torch.Tensor]:
    """
    Calculates relevant recat indices required to reorder AlltoAll collective.

    Args:
        local_split (int): number of features in local split.
        num_splits (int): number of splits (typically WORLD_SIZE).
        stagger (int): secondary reordering, (typically 1, but
            `WORLD_SIZE/LOCAL_WORLD_SIZE` for TWRW).
        device (Optional[torch.device]): device on which buffer will be allocated.
        batch_size_per_rank (Optional[List[int]]): batch size per rank, needed for
            variable batch size.

    Returns:
        Optional[torch.Tensor]: recat tensor, None if local rank is empty.

    Example::

        _recat(2, 4, 1)
            # [0, 2, 4, 6, 1, 3, 5, 7]
        _recat(2, 4, 2)
            # [0, 4, 2, 6, 1, 5, 3, 7]
        _recat(0, 4, 2)
            # None
    """
    with record_function("## all2all_data:recat_permute_gen ##"):
        if local_split == 0:
            return None

        recat: List[int] = []

        feature_order: List[int] = [
            x + num_splits // stagger * y
            for x in range(num_splits // stagger)
            for y in range(stagger)
        ]

        for i in range(local_split):
            for j in feature_order:  # range(num_splits):
                recat.append(i + j * local_split)

        # variable batch size
        if batch_size_per_rank is not None:
            batch_size_per_feature = list(
                itertools.chain.from_iterable(
                    itertools.repeat(x, local_split) for x in batch_size_per_rank
                )
            )
            permuted_batch_size_per_feature = [batch_size_per_feature[r] for r in recat]
            input_offset = [0] + list(itertools.accumulate(batch_size_per_feature))
            output_offset = [0] + list(
                itertools.accumulate(permuted_batch_size_per_feature)
            )
            recat_tensor = torch.tensor(
                recat,
                device=device,
                dtype=torch.int32,
            )
            input_offset_tensor = torch.tensor(
                input_offset,
                device=device,
                dtype=torch.int32,
            )
            output_offset_tensor = torch.tensor(
                output_offset,
                device=device,
                dtype=torch.int32,
            )
            recat = torch.ops.fbgemm.expand_into_jagged_permute(
                recat_tensor,
                input_offset_tensor,
                output_offset_tensor,
                output_offset[-1],
            )
            return recat
        else:
            return torch.tensor(recat, device=device, dtype=torch.int32)


class SplitsAllToAllAwaitable(Awaitable[List[List[int]]]):
    """
    Awaitable for splits AlltoAll.

    Args:
        input_tensors (List[torch.Tensor]): tensor of splits to redistribute.
        num_workers (int): total ranks.
        device (torch.device): device on which buffers are allocated.
        pg (dist.ProcessGroup): ProcessGroup for AlltoAll communication.
    """

    def __init__(
        self,
        input_tensors: List[torch.Tensor],
        num_workers: int,
        device: torch.device,
        pg: dist.ProcessGroup,
    ) -> None:
        super().__init__()
        self.num_workers: int = num_workers

        with record_function("## all2all_data:kjt splits ##"):
            self._output_tensor: torch.Tensor = torch.empty(
                [num_workers * len(input_tensors)],
                device=device,
                dtype=input_tensors[0].dtype,
            )
            input_tensor = torch.stack(input_tensors, dim=1).flatten()
            self._splits_awaitable: dist.Work = dist.all_to_all_single(
                output=self._output_tensor,
                input=input_tensor,
                group=pg,
                async_op=True,
            )

    def _wait_impl(self) -> List[List[int]]:
        self._splits_awaitable.wait()
        return self._output_tensor.view(self.num_workers, -1).T.tolist()


class KJTAllToAllTensorsAwaitable(Awaitable[KeyedJaggedTensor]):
    """
    Awaitable for KJT tensors AlltoAll.

    Args:
        pg (dist.ProcessGroup): ProcessGroup for AlltoAll communication.
        input (KeyedJaggedTensor): input KJT.
        splits (List[int]): list of len(pg.size()) which indicates how many features to
            send to each pg.rank(). It is assumed the `KeyedJaggedTensor` is ordered by
            destination rank. Same for all ranks.
        input_splits (List[List[int]]): input splits (number of values each rank will
            get) for each tensor in AlltoAll.
        output_splits (List[List[int]]): output splits (number of values per rank in
            output) for each tensor in AlltoAll.
        input_tensors (List[torch.Tensor]): provided KJT tensors (ie. lengths, values)
            to redistribute according to splits.
        labels (List[str]): labels for each provided tensor.
        batch_size_per_rank (List[int]): batch size per rank, to support variable batch
            size.
        keys (List[str]): KJT keys after AlltoAll.
        device (torch.device): device on which buffers will be allocated.
        stagger (int): stagger value to apply to recat tensor.
    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
        input: KeyedJaggedTensor,
        splits: List[int],
        input_splits: List[List[int]],
        output_splits: List[List[int]],
        input_tensors: List[torch.Tensor],
        labels: List[str],
        batch_size_per_rank: List[int],
        keys: List[str],
        device: torch.device,
        stagger: int,
    ) -> None:
        super().__init__()
        self._workers: int = pg.size()
        self._pg: dist.ProcessGroup = pg
        self._device: torch.device = device
        self._input = input
        self._splits = splits
        self._input_splits: Dict[str, List[int]] = dict(zip(labels, input_splits))
        self._output_splits: Dict[str, List[int]] = dict(zip(labels, output_splits))
        self._batch_size_per_rank = batch_size_per_rank
        self._keys = keys
        self._stagger = stagger
        self._recat: Optional[torch.Tensor] = _get_recat(
            local_split=splits[pg.rank()],
            num_splits=len(splits),
            stagger=stagger,
            device=device,
            batch_size_per_rank=batch_size_per_rank
            if len(set(batch_size_per_rank)) > 1
            else None,
        )
        if self._workers == 1:
            return

        self._output_tensors: List[torch.Tensor] = []
        self._awaitables: List[dist.Work] = []

        for input_split, output_split, input_tensor, label in zip(
            input_splits,
            output_splits,
            input_tensors,
            labels,
        ):
            output_tensor = torch.empty(
                sum(output_split), device=self._device, dtype=input_tensor.dtype
            )
            with record_function(f"## all2all_data:kjt {label} ##"):
                awaitable = dist.all_to_all_single(
                    output=output_tensor,
                    input=input_tensor,
                    output_split_sizes=output_split,
                    input_split_sizes=input_split,
                    group=self._pg,
                    async_op=True,
                )

            self._output_tensors.append(output_tensor)
            self._awaitables.append(awaitable)

    def _wait_impl(self) -> KeyedJaggedTensor:
        """
        Overwrites wait function as we don't handle callbacks here.

        Returns:
            KeyedJaggedTensor: Synced KJT after AlltoAll.
        """

        if self._workers == 1:
            self._input.sync()
            return self._input

        for awaitable in self._awaitables:
            awaitable.wait()

        return type(self._input).dist_init(
            keys=self._keys,
            tensors=self._output_tensors,
            batch_size_per_rank=self._batch_size_per_rank,
            recat=self._recat,
        )


class KJTAllToAllSplitsAwaitable(Awaitable[KJTAllToAllTensorsAwaitable]):
    """
    Awaitable for KJT tensors splits AlltoAll.

    Args:
        pg (dist.ProcessGroup): ProcessGroup for AlltoAll communication.
        input (KeyedJaggedTensor): input KJT.
        splits (List[int]): list of len(pg.size()) which indicates how many features to
            send to each pg.rank(). It is assumed the `KeyedJaggedTensor` is ordered by
            destination rank. Same for all ranks.
        tensor_splits (Dict[str, List[int]]): tensor splits provided by input KJT.
        input_tensors (List[torch.Tensor]): provided KJT tensors (ie. lengths, values)
            to redistribute according to splits.
        keys (List[str]): KJT keys after AlltoAll.
        device (torch.device): device on which buffers will be allocated.
        stagger (int): stagger value to apply to recat tensor.
    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
        input: KeyedJaggedTensor,
        splits: List[int],
        labels: List[str],
        tensor_splits: List[List[int]],
        input_tensors: List[torch.Tensor],
        keys: List[str],
        device: torch.device,
        stagger: int,
    ) -> None:
        super().__init__()
        self._workers: int = pg.size()
        self._pg: dist.ProcessGroup = pg
        self._device: torch.device = device
        self._input = input
        self._splits = splits
        self._labels = labels
        self._input_splits = tensor_splits
        self._input_tensors = input_tensors
        self._keys = keys
        self._stagger = stagger
        self._output_splits: List[List[int]] = self._input_splits
        self._batch_size_per_rank: List[int] = [input.stride()]
        if self._workers == 1:
            return

        input_tensors = [
            torch.tensor(split, device=device) for split in self._input_splits
        ]
        batch_size_tensor = torch.tensor(
            [input.stride()] * self._workers, device=device
        )
        input_tensors.append(batch_size_tensor)

        self._splits_awaitable = SplitsAllToAllAwaitable(
            input_tensors, self._workers, self._device, self._pg
        )

    def _wait_impl(self) -> KJTAllToAllTensorsAwaitable:
        """
        Overwrites wait function as we don't handle callbacks here.

        Returns:
            KJTAllToAllTensorsAwaitable.
        """

        if self._workers > 1:
            output_list = self._splits_awaitable.wait()
            self._output_splits = output_list[:-1]
            self._batch_size_per_rank = output_list[-1]

        return KJTAllToAllTensorsAwaitable(
            pg=self._pg,
            input=self._input,
            splits=self._splits,
            input_splits=self._input_splits,
            output_splits=self._output_splits,
            input_tensors=self._input_tensors,
            labels=self._labels,
            batch_size_per_rank=self._batch_size_per_rank,
            keys=self._keys,
            device=self._device,
            stagger=self._stagger,
        )


class KJTAllToAll(nn.Module):
    """
    Redistributes `KeyedJaggedTensor` to a `ProcessGroup` according to splits.

    Implementation utilizes AlltoAll collective as part of torch.distributed.

    The input provides the necessary tensors and input splits to distribute.
    The first collective call in `KJTAllToAllSplitsAwaitable` will transmit output
    splits (to allocate correct space for tensors) and batch size per rank. The
    following collective calls in `KJTAllToAllTensorsAwaitable` will transmit the actual
    tensors asynchronously.

    Args:
        pg (dist.ProcessGroup): ProcessGroup for AlltoAll communication.
        splits (List[int]): List of len(pg.size()) which indicates how many features to
            send to each pg.rank(). It is assumed the `KeyedJaggedTensor` is ordered by
            destination rank. Same for all ranks.
        device (Optional[torch.device]): device on which buffers will be allocated.
        stagger (int): stagger value to apply to recat tensor, see `_get_recat` function
            for more detail.

    Example::

        keys=['A','B','C']
        splits=[2,1]
        kjtA2A = KJTAllToAll(pg, splits, device)
        awaitable = kjtA2A(rank0_input)

        # where:
        # rank0_input is KeyedJaggedTensor holding

        #         0           1           2
        # 'A'    [A.V0]       None        [A.V1, A.V2]
        # 'B'    None         [B.V0]      [B.V1]
        # 'C'    [C.V0]       [C.V1]      None

        # rank1_input is KeyedJaggedTensor holding

        #         0           1           2
        # 'A'     [A.V3]      [A.V4]      None
        # 'B'     None        [B.V2]      [B.V3, B.V4]
        # 'C'     [C.V2]      [C.V3]      None

        rank0_output = awaitable.wait()

        # where:
        # rank0_output is KeyedJaggedTensor holding

        #         0           1           2           3           4           5
        # 'A'     [A.V0]      None      [A.V1, A.V2]  [A.V3]      [A.V4]      None
        # 'B'     None        [B.V0]    [B.V1]        None        [B.V2]      [B.V3, B.V4]

        # rank1_output is KeyedJaggedTensor holding
        #         0           1           2           3           4           5
        # 'C'     [C.V0]      [C.V1]      None        [C.V2]      [C.V3]      None
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
        self._workers: int = pg.size()
        self._splits = splits
        self._no_dist: bool = all(s == 0 for s in splits)
        self._splits_cumsum: List[int] = [0] + list(itertools.accumulate(splits))
        self._stagger = stagger

    def forward(
        self, input: KeyedJaggedTensor
    ) -> Awaitable[KJTAllToAllTensorsAwaitable]:
        """
        Sends input to relevant `ProcessGroup` ranks.

        The first wait will get the output splits for the provided tensors and issue
        tensors AlltoAll. The second wait will get the tensors.

        Args:
            input (KeyedJaggedTensor): `KeyedJaggedTensor` of values to distribute.

        Returns:
            Awaitable[KJTAllToAllTensorsAwaitable]: awaitable of a `KJTAllToAllTensorsAwaitable`.
        """

        device = input.values().device

        with torch.no_grad():
            assert len(input.keys()) == sum(self._splits)
            rank = dist.get_rank(self._pg)
            local_keys = input.keys()[
                self._splits_cumsum[rank] : self._splits_cumsum[rank + 1]
            ]

            return KJTAllToAllSplitsAwaitable(
                pg=self._pg,
                input=input,
                splits=self._splits,
                labels=input.dist_labels(),
                tensor_splits=input.dist_splits(self._splits),
                input_tensors=input.dist_tensors(),
                keys=local_keys,
                device=device,
                stagger=self._stagger,
            )


class KJTOneToAll(nn.Module):
    """
    Redistributes `KeyedJaggedTensor` to all devices.

    Implementation utilizes OnetoAll function, which essentially P2P copies the feature
    to the devices.

    Args:
        splits (List[int]): lengths of features to split the `KeyJaggedTensor` features
            into before copying them.
        world_size (int): number of devices in the topology.
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

    def forward(self, kjt: KeyedJaggedTensor) -> Awaitable[KJTList]:
        """
        Splits features first and then sends the slices to the corresponding devices.

        Args:
            kjt (KeyedJaggedTensor): the input features.

        Returns:
            Awaitable[List[KeyedJaggedTensor]]: awaitable of `KeyedJaggedTensor` splits.
        """

        kjts: List[KeyedJaggedTensor] = kjt.split(self._splits)
        dist_kjts = [
            kjts[rank].to(torch.device("cuda", rank), non_blocking=True)
            for rank in range(self._world_size)
        ]
        return NoWait(KJTList(dist_kjts))


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
        callbacks (Optional[List[Callable[[torch.Tensor], torch.Tensor]]]): callback
            functions.

    Example::

        dim_sum_per_rank = [2, 1]
        a2a = PooledEmbeddingsAllToAll(pg, dim_sum_per_rank, device)

        t0 = torch.rand((6, 2))
        t1 = torch.rand((6, 1))
        rank0_output = a2a(t0).wait()
        rank1_output = a2a(t1).wait()
        print(rank0_output.size())
            # torch.Size([3, 3])
        print(rank1_output.size())
            # torch.Size([3, 3])
    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
        dim_sum_per_rank: List[int],
        device: Optional[torch.device] = None,
        callbacks: Optional[List[Callable[[torch.Tensor], torch.Tensor]]] = None,
        codecs: Optional[QuantizedCommCodecs] = None,
    ) -> None:
        super().__init__()
        self._pg = pg
        self._callbacks: List[Callable[[torch.Tensor], torch.Tensor]] = []
        if callbacks is not None:
            self._callbacks = callbacks
        self._dim_sum_per_rank = dim_sum_per_rank
        self._codecs = codecs
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
            batch_size_per_rank (Optional[List[int]]): batch size per rank, to support
                variable batch size.

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
            codecs=self._codecs,
        )

        pooled_embedding_awaitable = PooledEmbeddingsAwaitable(
            tensor_awaitable=tensor_awaitable,
        )
        pooled_embedding_awaitable.callbacks.extend(self._callbacks)

        return pooled_embedding_awaitable

    @property
    def callbacks(self) -> List[Callable[[torch.Tensor], torch.Tensor]]:
        return self._callbacks


class EmbeddingsAllToOne(nn.Module):
    """
    Merges the pooled/sequence embedding tensor on each device into single tensor.

    Args:
        device (torch.device): device on which buffer will be allocated.
        world_size (int): number of devices in the topology.
        cat_dim (int): which dimension you would like to concatenate on.
            For pooled embedding it is 1; for sequence embedding it is 0.
    """

    def __init__(
        self,
        device: torch.device,
        world_size: int,
        cat_dim: int,
    ) -> None:
        super().__init__()
        self._device = device
        self._world_size = world_size
        self._cat_dim = cat_dim

    def forward(self, tensors: List[torch.Tensor]) -> Awaitable[torch.Tensor]:
        """
        Performs AlltoOne operation on pooled/sequence embeddings tensors.

        Args:
            tensors (List[torch.Tensor]): list of embedding tensors.

        Returns:
            Awaitable[torch.Tensor]: awaitable of the merged embeddings.
        """
        assert len(tensors) == self._world_size
        non_cat_size = tensors[0].size(1 - self._cat_dim)
        return NoWait(
            torch.ops.fbgemm.merge_pooled_embeddings(
                tensors,
                non_cat_size,
                # syntax for torchscript
                str(self._device),
                self._cat_dim,
            )
        )


class SeqEmbeddingsAllToOne(nn.Module):
    """
    Merges the pooled/sequence embedding tensor on each device into single tensor.

    Args:
        device (torch.device): device on which buffer will be allocated
        world_size (int): number of devices in the topology.
        cat_dim (int): which dimension you like to concate on.
            For pooled embedding it is 1; for sequence embedding it is 0.
    """

    def __init__(
        self,
        device: torch.device,
        world_size: int,
    ) -> None:
        super().__init__()
        self._device = device
        self._world_size = world_size

    def forward(self, tensors: List[torch.Tensor]) -> Awaitable[List[torch.Tensor]]:
        """
        Performs AlltoOne operation on pooled embeddings tensors.

        Args:
            tensors (List[torch.Tensor]): list of pooled embedding tensors.

        Returns:
            Awaitable[torch.Tensor]: awaitable of the merged pooled embeddings.
        """

        assert len(tensors) == self._world_size
        return NoWait(
            torch.ops.fbgemm.all_to_one_device(
                tensors,
                self._device,
            )
        )


class PooledEmbeddingsReduceScatter(nn.Module):
    """
    The module class that wraps reduce-scatter communication primitives for pooled
    embedding communication in row-wise and twrw sharding.

    For pooled embeddings, we have a local model-parallel output tensor with a layout of
    [num_buckets x batch_size, dimension]. We need to sum over num_buckets dimension
    across batches. We split tensor along the first dimension into unequal chunks (tensor
    slices of different buckets) according to input_splits and reduce them into the output
    tensor and scatter the results for corresponding ranks.

    The class returns the async `Awaitable` handle for pooled embeddings tensor.
    The reduce-scatter-v is only available for NCCL backend.

    Args:
        pg (dist.ProcessGroup): The process group that the reduce-scatter communication
            happens within.
        codecs (Optional[QuantizedCommCodecs]): Quantization codec

     Example::

        init_distributed(rank=rank, size=2, backend="nccl")
        pg = dist.new_group(backend="nccl")
        input = torch.randn(2 * 2, 2)
        input_splits = [1,3]
        m = PooledEmbeddingsReduceScatter(pg)
        output = m(input, input_splits)
        tensor = output.wait()
    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
        codecs: Optional[QuantizedCommCodecs] = None,
    ) -> None:
        super().__init__()
        self._pg = pg
        self._codecs = codecs

    def forward(
        self, local_embs: torch.Tensor, input_splits: Optional[List[int]] = None
    ) -> PooledEmbeddingsAwaitable:
        """
        Performs reduce scatter operation on pooled embeddings tensor.

        Args:
            local_embs (torch.Tensor): tensor of shape [num_buckets x batch_size, dimension].
            input_splits (Optional[List[int]]): list of splits for local_embs dim0.

        Returns:
            PooledEmbeddingsAwaitable: awaitable of pooled embeddings of tensor of shape [batch_size, dimension].
        """

        if input_splits and len(set(input_splits)) > 1:
            tensor_awaitable = reduce_scatter_v_pooled(
                local_embs, input_splits, self._pg, codecs=self._codecs
            )
        else:
            tensor_awaitable = reduce_scatter_base_pooled(
                local_embs, self._pg, codecs=self._codecs
            )
        return PooledEmbeddingsAwaitable(tensor_awaitable=tensor_awaitable)


class PooledEmbeddingsAllGather(nn.Module):
    """
    The module class that wraps all-gather communication primitive for pooled
    embedding communication

    We have a local input tensor with a layout of
    [batch_size, dimension]. We need to gather input tensors from all ranks into a flatten output tensor.

    The class returns the async `Awaitable` handle for pooled embeddings tensor.
    The all-gather is only available for NCCL backend.

    Args:
        pg (dist.ProcessGroup): The process group that the all-gather communication
            happens within.

    Example::

        init_distributed(rank=rank, size=2, backend="nccl")
        pg = dist.new_group(backend="nccl")
        input = torch.randn(2, 2)
        m = PooledEmbeddingsAllGather(pg)
        output = m(input)
        tensor = output.wait()
    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
        codecs: Optional[QuantizedCommCodecs] = None,
    ) -> None:
        super().__init__()
        self._pg = pg
        self._codecs = codecs

    def forward(self, local_emb: torch.Tensor) -> PooledEmbeddingsAwaitable:
        """
        Performs reduce scatter operation on pooled embeddings tensor.

        Args:
            local_emb (torch.Tensor): tensor of shape [num_buckets x batch_size, dimension].

        Returns:
            PooledEmbeddingsAwaitable: awaitable of pooled embeddings of tensor of shape [batch_size, dimension].
        """

        tensor_awaitable = all_gather_base_pooled(
            local_emb, self._pg, codecs=self._codecs
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
        embedding_dim (int): embedding dimension.
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
            torch.Tensor: synced sequence embeddings.
        """

        ret = self._tensor_awaitable.wait()
        return ret


class SequenceEmbeddingsAllToAll(nn.Module):
    """
    Redistributes sequence embedding to a `ProcessGroup` according to splits.

    Args:
        pg (dist.ProcessGroup): the process group that the AlltoAll communication
            happens within.
        features_per_rank (List[int]): list of number of features per rank.
        device (Optional[torch.device]): device on which buffers will be allocated.

    Example::

        init_distributed(rank=rank, size=2, backend="nccl")
        pg = dist.new_group(backend="nccl")
        features_per_rank = [4, 4]
        m = SequenceEmbeddingsAllToAll(pg, features_per_rank)
        local_embs = torch.rand((6, 2))
        sharding_ctx: SequenceShardingContext
        output = m(
            local_embs=local_embs,
            lengths=sharding_ctx.lengths_after_input_dist,
            input_splits=sharding_ctx.input_splits,
            output_splits=sharding_ctx.output_splits,
            unbucketize_permute_tensor=None,
        )
        tensor = output.wait()
    """

    def __init__(
        self,
        pg: dist.ProcessGroup,
        features_per_rank: List[int],
        device: Optional[torch.device] = None,
        codecs: Optional[QuantizedCommCodecs] = None,
    ) -> None:
        super().__init__()
        self._pg = pg
        self._local_split: int = features_per_rank[self._pg.rank()]
        self._num_splits: int = self._pg.size()

        forward_recat = []
        for j in range(self._num_splits):
            for i in range(self._local_split):
                forward_recat.append(j + i * self._num_splits)
        self.register_buffer(
            "_forward_recat_tensor",
            torch.tensor(forward_recat, device=device, dtype=torch.int),
        )
        backward_recat = []
        for i in range(self._local_split):
            for j in range(self._num_splits):
                backward_recat.append(i + j * self._local_split)
        self.register_buffer(
            "_backward_recat_tensor",
            torch.tensor(backward_recat, device=device, dtype=torch.int),
        )
        self._codecs = codecs

    def forward(
        self,
        local_embs: torch.Tensor,
        lengths: torch.Tensor,
        input_splits: List[int],
        output_splits: List[int],
        unbucketize_permute_tensor: Optional[torch.Tensor] = None,
        batch_size_per_rank: Optional[List[int]] = None,
        sparse_features_recat: Optional[torch.Tensor] = None,
    ) -> SequenceEmbeddingsAwaitable:
        """
        Performs AlltoAll operation on sequence embeddings tensor.

        Args:
            local_embs (torch.Tensor): input embeddings tensor.
            lengths (torch.Tensor): lengths of sparse features after AlltoAll.
            input_splits (List[int]): input splits of AlltoAll.
            output_splits (List[int]): output splits of AlltoAll.
            unbucketize_permute_tensor (Optional[torch.Tensor]): stores the permute
                order of the KJT bucketize (for row-wise sharding only).
            batch_size_per_rank: (Optional[List[int]]): batch size per rank.
            sparse_features_recat (Optional[torch.Tensor]): recat tensor used for sparse
                feature input dist.

        Returns:
            SequenceEmbeddingsAwaitable: awaitable of sequence embeddings.
        """

        variable_batch_size = (
            batch_size_per_rank is not None and len(set(batch_size_per_rank)) > 1
        )

        if sparse_features_recat is not None:
            forward_recat_tensor = torch.ops.fbgemm.invert_permute(
                sparse_features_recat
            )
            backward_recat_tensor = sparse_features_recat
        else:
            forward_recat_tensor = self._forward_recat_tensor
            backward_recat_tensor = self._backward_recat_tensor

        tensor_awaitable = alltoall_sequence(
            a2a_sequence_embs_tensor=local_embs,
            forward_recat_tensor=forward_recat_tensor,
            backward_recat_tensor=backward_recat_tensor,
            lengths_after_sparse_data_all2all=lengths,
            input_splits=input_splits,
            output_splits=output_splits,
            variable_batch_size=variable_batch_size,
            group=self._pg,
            codecs=self._codecs,
        )
        return SequenceEmbeddingsAwaitable(
            tensor_awaitable=tensor_awaitable,
            unbucketize_permute_tensor=unbucketize_permute_tensor,
            embedding_dim=local_embs.shape[1],
        )
