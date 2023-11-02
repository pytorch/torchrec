#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Any, List, Optional

import torch
from torch.autograd import Function
from torch.distributed._functional_collectives import _expand_group, wait_tensor
from torch.utils._pytree import tree_leaves, tree_map_only
from torch.utils._python_dispatch import return_and_correct_aliasing


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
            requires_grad=elem.requires_grad,
        )
        return r

    def __init__(self, elem, stream):
        self.elem = elem
        self.stream = stream

    # pyre-ignore
    def __repr__(self, *, tensor_contents=None) -> str:
        torch.cuda.current_stream().wait_stream(self.stream)
        return f"StreamSyncTensor({repr(self.elem)})"

    @classmethod
    def __torch_dispatch__(
        cls,
        func,
        types,
        args=(),
        kwargs=None,
    ) -> Any:
        kwargs = kwargs or {}

        if func == torch.ops.aten.view.default:
            # view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
            # in backwards call
            func = torch.ops.aten.reshape.default

        # TODO maybe support multiple stream sync tensors from different a2a calls
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

        def wrap(e, stream):    
            return StreamSyncTensor(e, stream=stream)

        unwrapped_args = tree_map_only(StreamSyncTensor, unwrap, args)
        unwrapped_kwargs = tree_map_only(StreamSyncTensor, unwrap, kwargs)

        if not non_stream_sync_tensors:
            with torch.cuda.stream(stream_sync_tensor_stream):
                out = func(*unwrapped_args, **unwrapped_kwargs)
            out = tree_map_only(torch.Tensor, partial(wrap, stream=stream_sync_tensor_stream), out)
            return return_and_correct_aliasing(func, args, kwargs, out)
        
        # print("syncing")
        for non_stream_sync_tensor in non_stream_sync_tensors:
            torch.cuda.current_stream(device=non_stream_sync_tensor.device).wait_stream(
                stream_sync_tensor_stream
            )
        torch.cuda.current_stream().wait_stream(stream_sync_tensor_stream)
        return func(*unwrapped_args, **unwrapped_kwargs)

from torch.distributed._functional_collectives import RANK_TYPES

def all_to_all_single(
    input: torch.Tensor,
    output_split_sizes: Optional[List[int]],
    input_split_sizes: Optional[List[int]],
    group: RANK_TYPES,
    tag: str = "",
    stream: Optional[torch.cuda.Stream] = None,
) -> StreamSyncTensor:
    if stream is None:
        stream = torch.cuda.Stream()
    return _AlltoAllSingle.apply(stream, group, output_split_sizes, input_split_sizes, input)

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
        # import torch.distributed as dist
        # my_rank = dist.get_rank(ctx.group)
        # if my_rank == 2:
        #     import time
        #     # print("LET ME SLEEP")
        #     time.sleep(.02)

        input.record_stream(stream)
        with torch.cuda.stream(stream):
            tensor = torch.ops.c10d_functional.all_to_all_single(input, output_split_sizes, input_split_sizes, tag, rankset, group_size)  # type: ignore[attr-defined]
            wait_tensor(tensor)
        return StreamSyncTensor(tensor, ctx.stream)

    @staticmethod
    # pyre-ignore
    def backward(ctx, grad_output):
        # import torch.distributed as dist
        # my_rank = dist.get_rank(ctx.group)
        # if my_rank == 2:
        #     import time
        #     # print("LET ME SLEEP")
        #     time.sleep(.02)

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
