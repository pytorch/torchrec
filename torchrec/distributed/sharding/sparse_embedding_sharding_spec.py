#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Type

import torch
import torch.distributed._shard.sharded_tensor.metadata as sharded_tensor_meta
from torch.distributed import ProcessGroup
from torch.distributed._shard._utils import narrow_tensor
from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed._shard.partial_tensor import _PartialTensor
from torch.distributed._shard.replicated_tensor import ReplicatedTensor
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._shard.sharding_spec import ShardingSpec
from torch.distributed._shard.sharding_spec.api import custom_sharding_spec_op
from torch.distributed._shard.sharding_spec.chunk_sharding_spec_ops.softmax import (
    sharded_softmax as chunk_sharding_spec_softmax,
)


@dataclass
class SparseEmbeddingShardingSpec(ShardingSpec):
    """
    Custom ShardingSpec to represent sharded output of sequence
    embedding lookups. This is typically a tensor of shape (B, N, D) where B
    is batch size, N is the sequence length and D is the embedding dim. The
    Tensor is typically sharded on dimension N, since embedding lookups
    can be sharded arbitrarily based on input, the Tensor is not sharded
    uniformly on N and is pretty arbitrary. As an example if N is 10 and we
    have 4 shards, the indices along dimension N could be sharded as follows
    in a non-contiguous manner: ``[0,3,5], [1,2,4,7], [6,9], [8]``

    In addition to this, it is
    important to keep track of the ordering of elements along dimension N
    since that is important in terms of applying positional encoding
    information.

    Args:
        dim (int):
            The dimension to shard on
        indices (torch.LongTensor):
            The indices along the sharded dimension representing the order of
            elements along that dimension. Should have the same size as the
            sharded dimension.
        lengths (torch.LongTensor):
            The lengths of each shard along the sharded dimension. Should
            have the same size as ``placements``.
        placement(List[torch.distributed._remote_device]):
            Specifies the placement of each shard of the Tensor. The size of
            the list represents the number of shards to be created. This is a
            list of :class:`torch.distributed._remote_device`'s.
    """

    dim: int
    indices: torch.LongTensor
    lengths: torch.LongTensor
    placements: List[torch.distributed._remote_device]

    def build_metadata(
        self,
        tensor_sizes: torch.Size,
        tensor_properties: sharded_tensor_meta.TensorProperties,
    ) -> sharded_tensor_meta.ShardedTensorMetadata:
        shard_metadatas: List[ShardMetadata] = []
        offset = 0
        for idx, placement in enumerate(self.placements):
            shard_offsets = [0 for _ in tensor_sizes]
            shard_offsets[self.dim] = offset
            offset += self.lengths[idx].item()

            shard_lengths = list(copy.deepcopy(tensor_sizes))
            shard_lengths[self.dim] = self.lengths[idx].item()

            shard_metadatas.append(
                ShardMetadata(shard_offsets, shard_lengths, placement)
            )

        return sharded_tensor_meta.ShardedTensorMetadata(
            shard_metadatas, tensor_sizes, tensor_properties
        )

    def shard(
        self,
        tensor: torch.Tensor,
        src_rank: int = 0,
        # pyre-ignore[11]
        process_group: ProcessGroup = None,
    ) -> "ShardedTensor":
        raise NotImplementedError("Not Supported!")


# pyre-ignore[56]
@custom_sharding_spec_op(SparseEmbeddingShardingSpec, torch.nn.functional.linear)
def handle_linear(
    types: Tuple[Type[torch.Tensor], ...],
    # pyre-ignore[2]
    args: Sequence[Any] = (),
    kwargs: Optional[Mapping[str, Any]] = None,
) -> ShardedTensor:
    input = args[0]
    weight = args[1]
    bias = args[2]

    if (
        isinstance(input, ShardedTensor)
        and isinstance(weight, torch.Tensor)
        and bias is None
    ):
        # pyre-ignore[16]
        sharding_dim = input.sharding_spec().dim
        # pyre-ignore[6]
        if sharding_dim != len(input.size()) - 1:
            # pyre-ignore[6]
            size = list(input.size())
            size[-1] = weight.size()[0]
            return ShardedTensor._init_from_local_tensor(
                input.local_tensor().matmul(weight.t()),
                input.sharding_spec(),
                size,
                process_group=input._process_group,
            )
        else:
            raise RuntimeError(
                "Sharding on last dim not supported for torch.nn.functional.linear"
            )
    else:
        raise RuntimeError(
            f"torch.nn.functional.linear not supported for args: {args} and kwargs: {kwargs}"
        )


def _dispatch_handle_add(lhs: ShardedTensor, rhs: ReplicatedTensor) -> ShardedTensor:
    # pyre-ignore[16]
    reordered_rhs = rhs.index_select(
        # pyre-ignore[16]
        lhs.sharding_spec().dim,
        # pyre-ignore[16]
        lhs.sharding_spec().indices,
    )
    narrowed_rhs = narrow_tensor(reordered_rhs, lhs.local_shards()[0].metadata)
    return ShardedTensor._init_from_local_tensor(
        lhs.local_tensor() + narrowed_rhs,
        lhs.sharding_spec(),
        # pyre-ignore[6]
        lhs.size(),
        process_group=lhs._process_group,
    )


# pyre-ignore[56]
@custom_sharding_spec_op(SparseEmbeddingShardingSpec, torch.Tensor.__add__)
def handle_add(
    types: Tuple[Type[torch.Tensor], ...],
    # pyre-ignore[2]
    args: Sequence[Any] = (),
    kwargs: Optional[Mapping[str, Any]] = None,
) -> ShardedTensor:
    lhs = args[0]
    rhs = args[1]

    # Adding ReplicatedTensor(B, 1, D) to ShardedTensor(B, N, D) sharded on dimension N.
    if isinstance(lhs, ShardedTensor) and isinstance(rhs, ReplicatedTensor):
        return _dispatch_handle_add(lhs, rhs)
    elif isinstance(lhs, ReplicatedTensor) and isinstance(rhs, ShardedTensor):
        return _dispatch_handle_add(rhs, lhs)
    elif isinstance(lhs, ShardedTensor) and isinstance(rhs, ShardedTensor):
        if lhs.size() == rhs.size() and lhs.sharding_spec() == rhs.sharding_spec():
            return ShardedTensor._init_from_local_tensor(
                lhs.local_tensor() + rhs.local_tensor(),
                lhs.sharding_spec(),
                # pyre-ignore[6]
                lhs.size(),
                process_group=lhs._process_group,
            )
    raise RuntimeError(f"torch.add not supported for args: {args} and kwargs: {kwargs}")


# pyre-ignore[56]
@custom_sharding_spec_op(SparseEmbeddingShardingSpec, torch.Tensor.__getitem__)
def handle_getitem(
    types: Tuple[Type[torch.Tensor], ...],
    # pyre-ignore[2]
    args: Sequence[Any] = (),
    kwargs: Optional[Mapping[str, Any]] = None,
) -> ShardedTensor:
    st = args[0]
    key = args[1]
    sharding_dim = st.sharding_spec().dim
    if isinstance(key, int) or isinstance(key, slice):
        if sharding_dim != 0:
            new_size = list(st.size())
            if isinstance(key, int):
                new_size.pop(0)
            else:
                step = 1 if key.step is None else key.step
                new_size[0] = (key.stop - key.start - 1) // step + 1
            return ShardedTensor._init_from_local_tensor(
                st.local_tensor()[key],
                st.sharding_spec(),
                new_size,
                process_group=st._process_group,
            )
    if isinstance(key, tuple):
        local_tensor = st.local_tensor()
        new_size = list(st.size())
        for dim, elem in enumerate(key):
            if not isinstance(elem, slice):
                raise RuntimeError(
                    f"Only slices supported in __getitem__ for tuples, found: {elem}"
                )

            if elem.start is None and elem.stop is None and elem.step is None:
                # Preserve this dim and continue
                continue
            elif dim == sharding_dim:
                raise RuntimeError("Slicing on sharding dim not supported!")
            else:
                step = 1 if elem.step is None else elem.step
                indices = torch.LongTensor(list(range(elem.start, elem.stop, step))).to(
                    local_tensor.device
                )
                local_tensor = local_tensor.index_select(dim, indices)
                new_size[dim] = (elem.stop - elem.start - 1) // step + 1

        return ShardedTensor._init_from_local_tensor(
            local_tensor, st.sharding_spec(), new_size, process_group=st._process_group
        )

    raise RuntimeError(
        f"__getitem__ not supported for args: {args} and kwargs: {kwargs}"
    )


# pyre-ignore[56]
@custom_sharding_spec_op(SparseEmbeddingShardingSpec, torch.Tensor.transpose)
def sharded_transpose(
    types: Tuple[Type[torch.Tensor], ...],
    # pyre-ignore[2]
    args: Sequence[Any] = (),
    kwargs: Optional[Mapping[str, Any]] = None,
) -> ShardedTensor:
    input = args[0]
    dim0 = args[1]
    dim1 = args[2]

    res_size = list(input.size())
    tmp = res_size[dim0]
    res_size[dim0] = res_size[dim1]
    res_size[dim1] = tmp
    sharding_spec = copy.deepcopy(input.sharding_spec())

    local_shard = torch.transpose(input.local_tensor(), dim0, dim1)
    if sharding_spec.dim == dim0:
        sharding_spec.dim = dim1
    elif sharding_spec.dim == dim1:
        sharding_spec.dim = dim0

    return ShardedTensor._init_from_local_tensor(
        local_shard.contiguous(),
        sharding_spec,
        res_size,
        process_group=input._process_group,
    )


# pyre-ignore[56]
@custom_sharding_spec_op(SparseEmbeddingShardingSpec, torch.bmm)
def sharded_bmm(
    types: Tuple[Type[torch.Tensor], ...],
    # pyre-ignore[2]
    args: Sequence[Any] = (),
    kwargs: Optional[Mapping[str, Any]] = None,
) -> _PartialTensor:
    st1 = args[0]
    st2 = args[1]
    spec1 = st1.sharding_spec()
    spec2 = st2.sharding_spec()
    if (
        torch.equal(spec1.indices, spec2.indices)
        and torch.equal(spec1.lengths, spec2.lengths)
        and spec1.placements == spec2.placements
        and spec1.dim == 2
        and spec2.dim == 1
    ):
        return _PartialTensor(
            torch.bmm(st1.local_tensor(), st2.local_tensor()),
            st1._process_group,
            torch.distributed.distributed_c10d.ReduceOp.SUM,
        )

    raise RuntimeError(f"torch.bmm not supported for args: {args} and kwargs: {kwargs}")


# pyre-ignore[56]
@custom_sharding_spec_op(SparseEmbeddingShardingSpec, torch.nn.functional.softmax)
def sharded_softmax(
    types: Tuple[Type[torch.Tensor], ...],
    # pyre-ignore[2]
    args: Sequence[Any] = (),
    kwargs: Optional[Mapping[str, Any]] = None,
) -> ShardedTensor:
    # Reuse ChunkShardingSpec softmax
    return chunk_sharding_spec_softmax(types, args, kwargs)
