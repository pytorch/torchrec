#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Any, Callable, cast, List, Optional, Union

import torch
from torch.autograd import Function
from torch.distributed._functional_collectives import _expand_group, wait_tensor
from torch.distributed._functional_collectives_impl import _register_tensor_wrapper
from torch.utils._pytree import tree_leaves, tree_map_only


class StreamSyncTensor(torch.Tensor):
    __slots__ = ["elem", "stream"]
    # pyre-ignore
    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def __new__(cls, elem, stream) -> "StreamSyncTensor":
        r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            elem.size(),
            strides=elem.stride(),
            storage_offset=elem.storage_offset(),
            dtype=elem.dtype,
            layout=elem.layout,
            device=elem.device,
            requires_grad=True,
            # elem.requires_grad,
        )
        r.stream = stream
        r.elem = elem
        r.retain_grad()
        return r

    # pyre-ignore
    def __repr__(self, *, tensor_contents=None) -> str:
        return f"StreamSyncTensor({self.stream}, {self.elem}"


    @classmethod
    def __torch_dispatch__(
        cls,
        func,
        types,
        args=(),
        kwargs=None,
    ) -> Any:
        kwargs = kwargs or {}

        stream_sync_tensor_stream = None
        non_stream_sync_tensors = []
        leaves = tree_leaves(args)
        for leaf in leaves:
            if isinstance(leaf, StreamSyncTensor):
                stream_sync_tensor_stream = leaf.stream
                continue

            if isinstance(leaf, torch.Tensor):
                non_stream_sync_tensors.append(leaf)

        def unwrap(e):
            return e.elem

        args = tree_map_only(StreamSyncTensor, unwrap, args)
        kwargs = tree_map_only(StreamSyncTensor, unwrap, kwargs)

        if not non_stream_sync_tensors:
            with torch.cuda.stream(stream_sync_tensor_stream):
                out = func(*args, **kwargs)
                out.requires_grad = True
                out.retain_grad()
                return StreamSyncTensor(
                    out,
                    stream_sync_tensor_stream,
                )

        print("SYNCING THE STREAMS!")
        for non_stream_sync_tensor in non_stream_sync_tensors:
            torch.cuda.current_stream(device=non_stream_sync_tensor.device).wait_stream(
                stream_sync_tensor_stream
            )
        return func(*args, **kwargs)

    def split(self, *args, **kwargs) -> List["StreamSyncTensor"]:
        splits = self.elem.split(*args, **kwargs)
        return [StreamSyncTensor(split, self.stream) for split in splits]

    # pyre-ignore
    def view(self, *args, **kwargs) -> "StreamSyncTensor":
        out = StreamSyncTensor(self.elem.view(*args, **kwargs), self.stream)
        out.retain_grad()
        out.requires_grad = True
        return out

    # pyre-ignore
    def reshape(self, *args, **kwargs) -> "StreamSyncTensor":
        return StreamSyncTensor(self.elem.reshape(*args, **kwargs), self.stream)

    def narrow(self, *args, **kwargs) -> "StreamSyncTensor":
        return StreamSyncTensor(self.elem.narrow(*args, **kwargs), self.stream)

    def transpose(self, *args, **kwargs) -> "StreamSyncTensor":
        return StreamSyncTensor(self.elem.transpose(*args, **kwargs), self.stream)

from torch.distributed._functional_collectives import RANK_TYPES


def all_to_all_single(
    self: torch.Tensor,
    output_split_sizes: Optional[List[int]],
    input_split_sizes: Optional[List[int]],
    group: RANK_TYPES,
    tag: str = "",
    stream: Optional[torch.cuda.Stream] = None,
) -> StreamSyncTensor:
    if stream is None:
        stream = torch.cuda.Stream()  # Create a new stream.
    return StreamSyncTensor(
        _AlltoAllSingle.apply(
            stream, group, output_split_sizes, input_split_sizes, self
        ),
        stream,
    )


class _AlltoAllSingle(Function):
    @staticmethod
    # pyre-ignore
    def forward(ctx, stream, group, output_split_sizes, input_split_sizes, input):
        ctx.group = group
        ctx.input_size = input.size()
        ctx.output_split_sizes = input_split_sizes
        ctx.input_split_sizes = output_split_sizes
        ctx.stream = stream

        tag, rankset, group_size = _expand_group(group, "")
        with torch.cuda.stream(stream):
            tensor = torch.ops.c10d_functional.all_to_all_single(input, output_split_sizes, input_split_sizes, tag, rankset, group_size)  # type: ignore[attr-defined]
            tensor.requires_grad = True
            tensor.retain_grad()
            wait_tensor(tensor)
        return tensor

    @staticmethod
    # pyre-ignore
    def backward(ctx, grad_output):
        ret = (None, None, None, None) + (
            _AlltoAllSingle.apply(
                ctx.stream,
                ctx.group,
                ctx.output_split_sizes,
                ctx.input_split_sizes,
                grad_output.contiguous(),
            ),
        )
        return ret
