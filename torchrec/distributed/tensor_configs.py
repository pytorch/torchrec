#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple

import torch
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    MetadataIndex,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import (
    TensorWriteData,
    WriteItem,
    WriteItemType,
)

aten = torch.ops.aten


# pyre-ignore
class LocalShardsWrapper(torch.Tensor):
    __slots__ = ["_local_shards", "_storage_meta"]
    _local_shards: List[torch.Tensor]
    _storage_meta: TensorStorageMetadata

    @staticmethod
    def __new__(
        cls, local_shards: List[torch.Tensor], local_offsets: List[Tuple[int, ...]]
    ) -> "LocalShardsWrapper":
        assert len(local_shards) > 0
        assert len(local_shards) == len(local_offsets)
        assert all(
            tensor.device == local_shards[0].device for tensor in local_shards[1:]
        )

        # we calculate the total tensor size by "concat" on second tensor dimension
        cat_tensor_shape = list(local_shards[0].size())
        if len(local_shards) > 1:  # column-wise sharding
            for shard in local_shards[1:]:
                cat_tensor_shape[1] += shard.size()[1]

        wrapper_properties = TensorProperties.create_from_tensor(local_shards[0])
        wrapper_shape = torch.Size(cat_tensor_shape)
        chunks_meta = [
            ChunkStorageMetadata(
                offsets=torch.Size(offset),
                sizes=shard.size(),
            )
            for shard, offset in zip(local_shards, local_offsets)
        ]

        r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            torch.Size(cat_tensor_shape),
        )
        r._local_shards = local_shards
        r._storage_meta = TensorStorageMetadata(
            properties=wrapper_properties,
            size=wrapper_shape,
            chunks=chunks_meta,
        )

        return r

    # necessary for ops dispatching from this subclass to its local shards
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        dispatcher = {
            torch.ops._c10d_functional.all_gather_into_tensor.default: cls.handle_all_gather,
            aten._to_copy.default: cls.handle_to_copy,
            aten.view.default: cls.handle_view,
            aten.equal.default: cls.handle_equal,
        }

        if func in dispatcher:
            return dispatcher[func](args, kwargs)
        else:
            raise NotImplementedError(
                f"{func} is not supported for LocalShardsWrapper!"
            )

    @staticmethod
    def handle_all_gather(args, kwargs):
        dim = args[0].local_sizes()[0][1]
        cat_tensor = torch.cat(
            [t.view(-1) for t in args[0].local_shards()], dim=0
        ).view(-1, dim)
        return torch.ops._c10d_functional.all_gather_into_tensor.default(
            cat_tensor, *args[1:], **kwargs
        )

    @staticmethod
    def handle_to_copy(args, kwargs):
        res_shards_list = [
            aten._to_copy.default(shard, *args[1:], **kwargs)
            for shard in args[0].local_shards()
        ]
        return LocalShardsWrapper(res_shards_list, args[0].local_offsets())

    @staticmethod
    def handle_view(args, kwargs):
        # TODO - fix, it needs to respect the view called on it. args[1:]
        res_shards_list = [
            aten.view.default(shard, shard.shape, **kwargs)
            for shard in args[0].local_shards()
        ]
        return LocalShardsWrapper(res_shards_list, args[0].local_offsets())

    @staticmethod
    def handle_equal(args, kwargs):
        """
        LocalShardsWrapper equal impl also checks for equality of storage metadata
        and the order of the shards
        """
        a, b = args[0], args[1]
        if len(a.local_shards()) != len(b.local_shards()):
            return False
        if not all(
            aten.equal.default(x, y) for x, y in zip(a.local_shards(), b.local_shards())
        ):
            return False
        if not a.storage_metadata() == b.storage_metadata():
            return False
        return True

    @property
    def device(self):
        return self._local_shards[0].device

    @property
    def is_meta(self):
        return self._local_shards[0].is_meta

    def is_pinned(self):
        return self._storage_meta.properties.pin_memory

    def detach(self):
        return LocalShardsWrapper(
            [t.detach() for t in self._local_shards], self.local_offsets()
        )

    def local_shards(self) -> List[torch.Tensor]:
        """
        Returns a list of :class:`torch.Tensor' corresponding to the
        local shards for this rank. Returns an empty list if the current rank
        does not host any shards for this Tensor.
        """
        return self._local_shards

    def local_sizes(self) -> List[torch.Size]:
        """
        Returns a list of :class:`torch.Size' corresponding to the
        local sizes for the shards on this rank. Returns an empty list if the current rank
        does not host any shards for this Tensor.
        """
        return [chunk.sizes for chunk in self._storage_meta.chunks]

    def local_offsets(self) -> List[torch.Size]:
        """
        Returns a list of :class:`torch.Size' corresponding to the
        local offsets for the shards on this rank. Returns an empty list if the current rank
        does not host any shards for this Tensor.
        """
        return [chunk.offsets for chunk in self._storage_meta.chunks]

    @property
    def local_chunks(self) -> List[ChunkStorageMetadata]:
        """
        Returns a :class:`List[ChunkStorageMetadata]` object corresponding to the
        metadata for each tensor shard
        """
        return self._storage_meta.chunks

    def storage_metadata(self) -> TensorStorageMetadata:
        """
        Returns a :class:`TensorStorageMetadata` object corresponding to the
        metadata for the local tensor on current rank
        """
        return self._storage_meta

    def create_write_items(self, fqn) -> List[WriteItem]:
        """
        For compatibility with DCP, we support creation of WriteItems
        such that they can be saved properly.
        """
        return [
            WriteItem(
                index=MetadataIndex(fqn, chunks.offsets),
                type=WriteItemType.SHARD,
                tensor_data=TensorWriteData(
                    chunk=ChunkStorageMetadata(
                        offsets=chunks.offsets,
                        sizes=chunks.sizes,
                    ),
                    properties=self._storage_meta.properties,
                    size=tensor.size(),
                ),
            )
            for tensor, chunks in zip(self.local_shards(), self.local_chunks)
        ]

    def __repr__(self):
        return f"LocalShardsWrapper:{self._local_shards} {self._storage_meta}"

    def __str__(self):
        return f"LocalShardsWrapper:{self._local_shards} {self._storage_meta}"
