#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Any, Tuple

import torch
from torch.autograd import Function
from torch.utils._python_dispatch import return_and_correct_aliasing
from torch.utils._pytree import tree_leaves, tree_map_only


class StreamSyncTensor(torch.Tensor):
    """
    A Tensor wrapper subclass that is used to manage tensors from different streams.
    This is particularly useful to cleanly overlap collective call operations with compute ops.

    There are two main modes of the dispatch function.

    1. If all the incoming tensors are StreamSyncTensors on the same steream, then no stream synchronizations are requires, and we just perform the op on the same stream.
    2. Otherwise, we assume that all the tensors are on different streams, and we synchronize them before performing the operation.

    args::
        elem: torch.Tensor - torch.Tensor that may have been created from a separate stream.
        stream: torch.cuda.Stream - stream that the tensor was created on.
    Example::

        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            x = torch.randn(10, device="cuda")
            y = 2*x
        y_stream = StreamSyncTensor(y, stream=s)

        z = torch.randn(10, device="cuda")

        # here y_stream will manage the syncing of stream s and current cuda stream, as well as any record_stream semantics.
        a = y_stream + z
    """

    __slots__ = ["elem", "stream"]
    # pyre-ignore
    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def __new__(
        cls, elem: torch.Tensor, stream: torch.cuda.Stream
    ) -> "StreamSyncTensor":
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

    def __init__(self, elem: torch.Tensor, stream: torch.cuda.Stream) -> None:
        self.elem: torch.Tensor = elem
        self.stream: torch.cuda.Stream = stream

    # pyre-ignore
    def __repr__(self, *, tensor_contents=None) -> str:
        return f"StreamSyncTensor({repr(self.elem.shape)})"

    @classmethod
    # pyre-ignore
    def __torch_dispatch__(
        cls,
        # pyre-ignore
        func,
        # pyre-ignore
        types,
        # pyre-ignore
        args=(),
        # pyre-ignore
        kwargs=None,
    ) -> Any:
        kwargs = kwargs or {}

        non_stream_sync_tensors = []
        stream_sync_tensors = []
        leaves = tree_leaves(args)
        for leaf in leaves:
            if isinstance(leaf, StreamSyncTensor):
                stream_sync_tensors.append(leaf)
            elif isinstance(leaf, torch.Tensor):
                non_stream_sync_tensors.append(leaf)

        def unwrap(e: StreamSyncTensor) -> torch.Tensor:
            return e.elem

        def wrap(
            e: torch.Tensor,
            stream: torch.cuda.Stream,
        ) -> StreamSyncTensor:
            return StreamSyncTensor(e, stream=stream)

        unwrapped_args = tree_map_only(StreamSyncTensor, unwrap, args)
        unwrapped_kwargs = tree_map_only(StreamSyncTensor, unwrap, kwargs)

        # Assume for now that all the streams are the same.
        # TODO handle case where we have multiple StreamSyncTensors on multiple streams
        stream_sync_tensor_stream = stream_sync_tensors[0].stream
        if not non_stream_sync_tensors:
            # Handle the case where all tensors involved in torch op is a StreamSyncTensor. We continue operating on the side stream
            # even if should_sync_to_main is set to True

            with torch.cuda.stream(stream_sync_tensor_stream):
                out = func(*unwrapped_args, **unwrapped_kwargs)
                out = tree_map_only(
                    torch.Tensor,
                    partial(
                        wrap,
                        stream=stream_sync_tensor_stream,
                    ),
                    out,
                )
                return return_and_correct_aliasing(func, args, kwargs, out)

        # We sync the streams and go back into main stream for all following ops
        for stream_sync_tensor in stream_sync_tensors:
            torch.cuda.current_stream().wait_stream(stream_sync_tensor.stream)
            stream_sync_tensor.elem.record_stream(torch.cuda.current_stream())

        unwrapped_args = tree_map_only(StreamSyncTensor, unwrap, args)
        unwrapped_kwargs = tree_map_only(StreamSyncTensor, unwrap, kwargs)

        out = func(*unwrapped_args, **unwrapped_kwargs)
        return out


class WrapInStreamSyncTensorFunc(Function):
    @staticmethod
    # pyre-fixme[14]: `forward` overrides method defined in `Function` inconsistently.
    def forward(
        # pyre-ignore
        ctx,
        t: torch.Tensor,
        stream: torch.cuda.Stream,
    ) -> StreamSyncTensor:
        return StreamSyncTensor(t, stream=stream)

    @staticmethod
    # pyre-fixme[14]: `backward` overrides method defined in `Function` inconsistently.
    def backward(
        # pyre-ignore
        ctx,
        grad_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        return grad_output, None