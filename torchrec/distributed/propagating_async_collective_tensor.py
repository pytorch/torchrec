#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Any, Callable, cast, List, Optional, Union

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.autograd import Function
from torch.distributed._functional_collectives import _expand_group, wait_tensor
from torch.distributed._functional_collectives_impl import _register_tensor_wrapper
from torch.utils._pytree import tree_leaves, tree_map_only
from torch.autograd.profiler import record_function



class PropagatingAsyncCollectiveTensor(torch.Tensor):
    r"""
    A Tensor wrapper subclass that is used to trigger a call to wait
    prior to first use of the underlying tensor.
    Use it inside functional collective pytorch wrappers like the following:
    def functional_collective(self, group, tag):
        tag, rankset, group_size = _expand_group(group, tag)
        tensor = torch.ops.c10d_functional.{collective}(self, tag, rankset, group_size)
        return _maybe_wrap_tensor(tensor)
    """

    __slots__ = ["manifest_elem", "fake_elem", "manifested_thunk"]

    # pyre-ignore
    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def __new__(
        cls,
        manifest_elem: Callable[[], torch.Tensor],
        fake_elem: torch.Tensor,
        device: torch.device,
    ) -> "PropagatingAsyncCollectiveTensor":
        r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            fake_elem.size(),
            strides=fake_elem.stride(),
            storage_offset=fake_elem.storage_offset(),
            dtype=fake_elem.dtype,
            layout=fake_elem.layout,
            device=device,
            requires_grad=True,
        )

        r.retain_grad()
        r.manifest_elem = manifest_elem
        r.fake_elem = fake_elem
        r.manifested_thunk = None
        return r

    # pyre-ignore
    def tolist(self):
        return self.manifest().tolist()

    # pyre-ignore
    def __repr__(self, *, tensor_contents=None) -> str:
        return f"PropagatingAsyncCollectiveTensor({self.manifest() if self.manifested_thunk is not None else 'Not manifested'} fake {self.fake_elem})"

    def manifest(self) -> torch.Tensor:
        if self.manifested_thunk is None:
            self.manifested_thunk = self.manifest_elem()
        return self.manifested_thunk

    @classmethod
    # pyre-ignore
    def __torch_dispatch__(
        cls,
        #  pyre-ignore
        func,
        #  pyre-ignore
        types,
        #  pyre-ignore
        args=(),
        #  pyre-ignore
        kwargs=None,
    ) -> Any:
        kwargs = kwargs or {}

        # print(
        #     "INSIDE DISPATCHER args",
        #     args,
        #     "kwargs",
        #     kwargs,
        #     "func",
        #     func,
        #     "types",
        #     types,
        # )

        def unwrap(e: PropagatingAsyncCollectiveTensor) -> torch.Tensor:
            return e.manifest()

        def fake_unwrap(
            e: torch.Tensor,
            # PropagatingAsyncCollectiveTensor,
        ) -> Union[FakeTensor, torch.Tensor]:
            if isinstance(e, PropagatingAsyncCollectiveTensor):
                return e.fake_elem
            return e.to("meta")
            # return e.fake_elem

        # TODO have some logic that can let us delay, or manifest immediately if a non-Async tensor is participating
        # we don't wrap the result as it doesn't need to be waited on.
        # pyre-ignore
        def do(func, args, kwargs):
            # print("unwrapping in ", func, args, kwargs)
            with record_function(f"## Manifesting {func}"):
                _args = tree_map_only(PropagatingAsyncCollectiveTensor, unwrap, args)
                _kwargs = tree_map_only(PropagatingAsyncCollectiveTensor, unwrap, kwargs)
                ret = func(*_args, **_kwargs)
                # print("returning for func", func, ret)
                return ret

        device = None
        leaves = tree_leaves(args)
        for leaf in leaves:
            if device is not None:
                continue
            if hasattr(leaf, "device"):
                # print("leaf with device", leaf, )
                device = getattr(leaf, "device")
        # print("YING USING DEVICE", device)

        with record_function(f"## Fake Tensor ops {func}"):
            fake_unwrapped_args = tree_map_only(
                torch.Tensor, fake_unwrap, args
            )
            fake_unwrapped_kwargs = tree_map_only(
                torch.Tensor, fake_unwrap, kwargs
            )
        # print("calling fake mode func", func, "fake unwrapped args ", *fake_unwrapped_args, "kwargs", **fake_unwrapped_kwargs)
        return PropagatingAsyncCollectiveTensor(
            partial(do, func=func, args=args, kwargs=kwargs),
            func(*fake_unwrapped_args, **fake_unwrapped_kwargs),
            device,
        )

    # Doing these in __dispatch__ complains about Attempted to make a tensor into a differentiable view, but the tensor already had autograd metadata associated with it.  If you are using a __torch_dispatch__ mode, the most common cause for this problem is that you used torch.overrides.enable_reentrant_dispatch() improperly; tensors created within the extent of reentrant dispatch MUST NOT be directly returned from __torch_dispatch__; instead, they must be wrapped into fresh tensors that serve as the output.  If you are not using wrappers, you probably don't need reentrant dispatch.  If this doesn't seem applicable, please file a bug to PyTorch.
    # and I don't know what's happening, so just putting them here for now ahhhh
    def split(self, *args, **kwargs) -> List["PropagatingAsyncCollectiveTensor"]:
        def do_split_and_index(index: int, did) -> torch.Tensor:
            with record_function(f"## Manifesting split"):
            # TODO thunk the initial split
                if did is None:
                    did = self.manifest().split(*args, **kwargs)
                return did[index]

        list_of_prop_async = []
        with record_function(f"## Fake split"):
            fake_split = self.fake_elem.split(*args, **kwargs)

        did = None
        for index, fake_elem in enumerate(fake_split):
            list_of_prop_async.append(
                PropagatingAsyncCollectiveTensor(
                    partial(
                        do_split_and_index,
                        index=index,
                        did = did
                    ),
                    fake_elem,
                    self.device,
                )
            )
        return list_of_prop_async

    # pyre-ignore
    def reshape(self, *args, **kwargs) -> "PropagatingAsyncCollectiveTensor":
        def do_reshape() -> torch.Tensor:
            with record_function(f"## Manifesting reshape"):
                did = self.manifest().reshape(*args, **kwargs)
                did.retain_grad()
                return did
            

        with record_function(f"## fake reshape"):
            fake_reshape = self.fake_elem.reshape(*args, **kwargs)
        return PropagatingAsyncCollectiveTensor(
            do_reshape,
            fake_reshape,
            self.device,
        )

    # pyre-ignore
    def view(self, *args, **kwargs) -> "PropagatingAsyncCollectiveTensor":
        def do_view() -> torch.Tensor:
            with record_function(f"## Manifesting view"):
                # print("doing custom view")
                m = self.manifest()
                did = m.view(*args, **kwargs)
                return did

        with record_function(f"## fake view"):
            fake_view = self.fake_elem.view(*args, **kwargs)
        return PropagatingAsyncCollectiveTensor(
            do_view,
            fake_view,
            self.device,
        )

    # pyre-ignore
    def narrow(self, *args, **kwargs) -> "PropagatingAsyncCollectiveTensor":
        def do() -> torch.Tensor:
            # print("doing custom view")
            with record_function(f"## Manifesting narrow"):
                m = self.manifest()
                did = m.narrow(*args, **kwargs)
                return did

        with record_function(f"## Fake narrow"):
            fake = self.fake_elem.narrow(*args, **kwargs)
        return PropagatingAsyncCollectiveTensor(
            do,
            fake,
            self.device,
        )

    # pyre-ignore
    def transpose(self, *args, **kwargs) -> "PropagatingAsyncCollectiveTensor":
        def do() -> torch.Tensor:
            # print("doing custom view")
            with record_function(f"## Manifesting transpose"):
                m = self.manifest()
                did = m.transpose(*args, **kwargs)
                return did
            
        with record_function(f"## Fake transpose"):
            fake = self.fake_elem.transpose(*args, **kwargs)
        return PropagatingAsyncCollectiveTensor(
            do,
            fake,
            self.device,
        )

    def backward(self, *args, **kwargs) -> None:
        # return
        # print("in custom backward", self.manifest())
        # super().backward(*args, **kwargs)
        self.manifest().backward(*args, **kwargs)

    # TODO +=/*= don't seem to work correctly


from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._functional_collectives import RANK_TYPES

fake_mode = FakeTensorMode(allow_non_fake_inputs=True)


def all_to_all_single(
    self: torch.Tensor,
    output_split_sizes: Optional[List[int]],
    input_split_sizes: Optional[List[int]],
    group: RANK_TYPES,
    tag: str = "",
) -> PropagatingAsyncCollectiveTensor:
    return _maybe_wrap_tensor(
        _AlltoAllSingle.apply(group, output_split_sizes, input_split_sizes, self)
    )


def _maybe_wrap_tensor(t: torch.Tensor) -> torch.Tensor:
    def do():
        wt = wait_tensor(t)
        wt.retain_grad()
        return wt

    res = PropagatingAsyncCollectiveTensor(do, t.to(torch.device("meta")), t.device)
    return cast(torch.Tensor, res)


class _AlltoAllSingle(Function):
    @staticmethod
    # pyre-ignore
    def forward(ctx, group, output_split_sizes, input_split_sizes, input):
        ctx.group = group
        ctx.input_size = input.size()
        ctx.output_split_sizes = input_split_sizes
        ctx.input_split_sizes = output_split_sizes
        tag, rankset, group_size = _expand_group(group, "")
        tensor = torch.ops.c10d_functional.all_to_all_single(input, output_split_sizes, input_split_sizes, tag, rankset, group_size)  # type: ignore[attr-defined]
        tensor.requires_grad = True
        tensor.retain_grad()
        return tensor

    @staticmethod
    # pyre-ignore
    def backward(ctx, grad_output):
        # print("HELLO IN BACKWARD")
        # backward is sync for now TODO maybe fix this
        ret = (None, None, None) + (
            _AlltoAllSingle.apply(
                ctx.group,
                ctx.output_split_sizes,
                ctx.input_split_sizes,
                grad_output.contiguous(),
            ),
        )
        return ret
