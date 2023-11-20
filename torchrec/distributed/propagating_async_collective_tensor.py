#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Any, Callable, Generic, List, Optional, TypeVar

import torch
from torch.autograd import Function
from torch.utils._pytree import tree_leaves, tree_map_only

T = TypeVar("T")


class _Thunk(Generic[T]):
    __slots__ = ["value"]

    def __init__(self, value: Optional[T] = None) -> None:
        self.value: Optional[T] = None


HIGH_PRIORITY_COLLECTIVE_CALL_SEQ_NUM: int = 2**61  # being a little bit conservative


class UnwrapPact(Function):
    @staticmethod
    # pyre-ignore
    def forward(ctx, tensor: "PropagatingAsyncCollectiveTensor") -> torch.Tensor:
        if tensor.manifested_thunk.value is None:
            tensor.manifested_thunk.value = tensor.manifest_elem()
        return tensor.manifested_thunk.value

    @staticmethod
    # pyre-ignore
    def backward(ctx, grad_output) -> torch.Tensor:
        return grad_output


from torch.autograd import Function

# pyre-ignore
def walk_up_and_set_seq_nr(node: "BackwardCFunc") -> None:
    if node is None:
        return
    if node._sequence_nr() < HIGH_PRIORITY_COLLECTIVE_CALL_SEQ_NUM:
        # TODO enable profiler to understand when thread_id and sequence id deviate
        node._set_sequence_nr(
            HIGH_PRIORITY_COLLECTIVE_CALL_SEQ_NUM + node._sequence_nr()
        )
    if not getattr(node, "_stop_priority_sequence", False):
        for g in node.next_functions:
            walk_up_and_set_seq_nr(g[0])


class PropagatingAsyncCollectiveTensor(torch.Tensor):
    r"""
    A Tensor wrapper subclass that is used to manage torch ops on a tensor from a collective call.
    This is particularly useful to cleanly overlap collective call operations with compute ops.

    args::
        manifest_elem: Callable[[], torch.Tensor] - a function that manifests all data dependent PACTs.
        A common use case is for the first PACT to block and wait for a collective call to finish.

        fake_elem: torch.Tensor - a meta tensor that can be used as a placeholder for the real tensor.
        This allows you to query properties of the underlying tensor (shape, device, etc),
        without manifesting the actual underlying tensor.

        device: torch.device - the device that the tensor will be placed on. This is needed because we cannot
        derive it from fake_elem. NB: This can be solved in the future by using FakeTensors, but currently
        the CPU overhead of using FakeTensors is too large.

        manifest_on_op_with_non_prop_async_tensor : bool - whether to manifest the PACT when an op consumes both
        a PACT and a regular Tensor. If this is set to true, the callback structure will be invoked and the torch.op
        will return a regular Tensor. Otherwise the input regular Tensor will be added to the callback structure and
        this op will be actually performed on manifestation. For the TorchRec specific case, manifest_on_op_with_non_prop_async_tensor
        is set to False to record posta2a ops such as embedding permutations, and set to True in the output of KeyedTensor.

    attrs::
        manifested_thunk: Optional[torch.Tensor] - the tensor that is returned after callbacks are manifested. This is so that if a PACT has already been manifested once, it doesn't need to be again.

    Example::
        See detailed usage in torchrec.distributed.comm_ops.all_to_all_single for a collective call example.

        pact = PropagatingAsyncCollectiveTensor(
            lambda: torch.ones(10, device=torch.device("cuda)), (1)
            fake_elem: torch.ones(10).to("meta"),
            device: torch.device("cuda"),
            manifest_on_op_with_non_prop_async_tensor: bool = False,
        )

        type(pact) == PropagatingAsyncCollectiveTensor
        pact.shape == (10,)

        pact_twice = pact + pact (2)
        type(pact_twice) == PropagatingAsyncCollectiveTensor

        another_tensor = torch.randn(10, device=torch.device("cuda"))
        pact_twice_another_tensor = pact_twice + another_tensor (3)

        type(pact_twice) == PropagatingAsyncCollectiveTensor

        pact_twice_another_tensor.manifest_on_op_with_non_prop_async_tensor = True

        # Here we will manifest the PACT and run the recorded ops

        manifested_pact_twice_another_tensor = pact_twice_another_tensor + another_tensor

        type(manifested_pact_twice_another_tensor) == torch.Tensor
        manifested_pact_twice_another_tensor == torch.ones(10) + torch.ones(10) + another_tensor + another_tensor.

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
        manifest_on_op_with_non_prop_async_tensor: bool = False,
    ) -> "PropagatingAsyncCollectiveTensor":
        return torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            fake_elem.size(),
            strides=fake_elem.stride(),
            storage_offset=fake_elem.storage_offset(),
            dtype=fake_elem.dtype,
            layout=fake_elem.layout,
            device=device,
            requires_grad=fake_elem.requires_grad,
        )

    def __init__(
        self,
        manifest_elem: Callable[[], torch.Tensor],
        fake_elem: torch.Tensor,
        device: torch.device,
        manifest_on_op_with_non_prop_async_tensor: bool = False,
    ) -> None:
        super().__init__()
        self.manifest_elem: Callable[[], torch.Tensor] = manifest_elem
        self.fake_elem: torch.Tensor = fake_elem
        self.manifested_thunk: _Thunk[torch.Tensor] = _Thunk()
        self.manifest_on_op_with_non_prop_async_tensor: bool = (
            manifest_on_op_with_non_prop_async_tensor
        )

    # pyre-ignore
    def __repr__(self, *, tensor_contents=None) -> str:
        return f"PropagatingAsyncCollectiveTensor ({self.manifested_thunk.value if self.manifested_thunk.value is not None else 'Not manifested'})"

    def manifest(self) -> torch.Tensor:
        # maintain grad if someone explicitly calls manifest()
        return UnwrapPact.apply(self)

    # pyre-ignore
    def backward(self, *args, **kwargs) -> None:
        # In the event we ever run a backwards on a PACT, this means that we've never manifested and the entire graph is over a tensor entirely data dependent on the PACT Tensors.
        # In this case manifest immediately
        self.manifest().backward(*args, **kwargs)

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

        non_pact_tensors = []
        pact_tensors = []
        leaves = tree_leaves(args)
        for leaf in leaves:
            if isinstance(leaf, PropagatingAsyncCollectiveTensor):
                pact_tensors.append(leaf)
            elif isinstance(leaf, torch.Tensor):
                non_pact_tensors.append(leaf)

        manifest_on_op_with_non_prop_async_tensor = any(
            t.manifest_on_op_with_non_prop_async_tensor for t in pact_tensors
        )
        device = pact_tensors[0].device

        def unwrap(e: PropagatingAsyncCollectiveTensor) -> torch.Tensor:
            return e.manifest()

        if manifest_on_op_with_non_prop_async_tensor and non_pact_tensors:
            manifested_args = tree_map_only(
                PropagatingAsyncCollectiveTensor, unwrap, args
            )
            manifested_kwargs = tree_map_only(
                PropagatingAsyncCollectiveTensor, unwrap, kwargs
            )
            return func(*manifested_args, **manifested_kwargs)

        def fake_unwrap(e: torch.Tensor) -> torch.Tensor:
            if isinstance(e, PropagatingAsyncCollectiveTensor):
                return e.fake_elem
            return e.to("meta")

        fake_unwrapped_args = tree_map_only(torch.Tensor, fake_unwrap, args)
        fake_unwrapped_kwargs = tree_map_only(torch.Tensor, fake_unwrap, kwargs)
        fake_out = func(*fake_unwrapped_args, **fake_unwrapped_kwargs)

        if isinstance(fake_out, list):

            def manifest_list_and_index(
                # pyre-ignore
                func,
                # pyre-ignore
                args,
                # pyre-ignore
                kwargs,
                thunk: _Thunk[List[torch.Tensor]],
                index: int,
            ) -> torch.Tensor:
                if thunk.value is None:
                    manifested_args = tree_map_only(
                        PropagatingAsyncCollectiveTensor, unwrap, args
                    )
                    manifested_kwargs = tree_map_only(
                        PropagatingAsyncCollectiveTensor, unwrap, kwargs
                    )

                    thunk.value = func(*manifested_args, **manifested_kwargs)
                return thunk.value[index]

            thunk = _Thunk[List[torch.Tensor]]()
            out = []
            for index, fake_elem in enumerate(fake_out):
                out.append(
                    PropagatingAsyncCollectiveTensor(
                        partial(
                            manifest_list_and_index,
                            func=func,
                            args=args,
                            kwargs=kwargs,
                            thunk=thunk,
                            index=index,
                        ),
                        fake_elem,
                        device,
                        manifest_on_op_with_non_prop_async_tensor=manifest_on_op_with_non_prop_async_tensor,
                    )
                )
            return out
        elif isinstance(fake_out, torch.Tensor):
            # pyre-ignore
            def manifest_elem(func, args, kwargs) -> torch.Tensor:
                manifested_args = tree_map_only(
                    PropagatingAsyncCollectiveTensor, unwrap, args
                )
                manifested_kwargs = tree_map_only(
                    PropagatingAsyncCollectiveTensor, unwrap, kwargs
                )

                return func(*manifested_args, **manifested_kwargs)

            return PropagatingAsyncCollectiveTensor(
                partial(manifest_elem, func=func, args=args, kwargs=kwargs),
                fake_out,
                device,
                manifest_on_op_with_non_prop_async_tensor,
            )
        else:
            raise RuntimeError(
                f"Expected output from {func} to be either a tensor or a list of tensors, got {type(fake_out)} instead."
            )
