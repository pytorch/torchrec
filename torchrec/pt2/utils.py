#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import functools
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor

"""
Prepares KJT for PT2 tracing.

KJT contains caching/lazy compute logic.
For tracing we need to drop all caches to have all compute logic in the graph.
This is done by recreation of KJT with minimal specified data.

convert_to_vb - If True recreates KJT as Variable Batch.
"""


def kjt_for_pt2_tracing(
    kjt: KeyedJaggedTensor,
    convert_to_vb: bool = False,
    mark_length: bool = False,
) -> KeyedJaggedTensor:
    # Breaking dependency cycle
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

    is_vb = kjt.variable_stride_per_key()
    if convert_to_vb and not is_vb:
        stride: int = kjt.stride()
        n = len(kjt.keys())

        inverse_indices_tensor = (
            torch.arange(stride).expand(n, stride).contiguous().to(device=kjt.device())
        )
        torch._dynamo.decorators.mark_static(inverse_indices_tensor, 0)
        torch._dynamo.decorators.mark_static(inverse_indices_tensor, 1)

        lengths = kjt.lengths().long()
        # We can mark static lengths dimension as we have fixed batch_size, but using VB path for tracing
        torch._dynamo.decorators.mark_static(lengths, 0)
        values = kjt.values().long()
        torch._dynamo.decorators.mark_unbacked(values, 0)

        return KeyedJaggedTensor(
            keys=kjt.keys(),
            values=values,
            lengths=lengths,
            weights=kjt.weights_or_none(),
            stride_per_key_per_rank=[[stride]] * n,
            inverse_indices=(kjt.keys(), inverse_indices_tensor),
        )

    inverse_indices = None
    stride = None

    if is_vb:
        inverse_indices = kjt.inverse_indices_or_none()

        if inverse_indices is not None:
            inverse_indices_tensor = inverse_indices[1]
            torch._dynamo.decorators.mark_static(inverse_indices_tensor, 0)
            torch._dynamo.decorators.mark_static(inverse_indices_tensor, 1)

    lengths = kjt.lengths().long()

    stride = kjt.stride()

    values = kjt.values().long()
    torch._dynamo.decorators.mark_unbacked(values, 0)
    weights = kjt.weights_or_none()
    if weights is not None:
        torch._dynamo.decorators.mark_unbacked(weights, 0)
    if mark_length:
        torch._dynamo.decorators.mark_unbacked(lengths, 0)

    length_per_key_marked_dynamic = []

    for length in kjt.length_per_key():
        length_per_key_marked_dynamic.append(length)

    return PT2KeyedJaggedTensor(
        keys=kjt.keys(),
        values=values,
        lengths=lengths,
        weights=weights,
        stride=stride if not is_vb else None,
        stride_per_key_per_rank=kjt.stride_per_key_per_rank() if is_vb else None,
        inverse_indices=inverse_indices,
        length_per_key=(length_per_key_marked_dynamic if is_vb else None),
    )


class PT2KeyedJaggedTensor(KeyedJaggedTensor):
    """
    This subclass of KeyedJaggedTensor is used to support PT2 tracing.
    We can apply some modifications to make KJT friendly for PT2 tracing.
    """

    def __init__(
        self,
        keys: List[str],
        values: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
        stride: Optional[int] = None,
        stride_per_key_per_rank: Optional[List[List[int]]] = None,
        stride_per_key: Optional[List[int]] = None,
        length_per_key: Optional[List[int]] = None,
        lengths_offset_per_key: Optional[List[int]] = None,
        offset_per_key: Optional[List[int]] = None,
        index_per_key: Optional[Dict[str, int]] = None,
        jt_dict: Optional[Dict[str, JaggedTensor]] = None,
        inverse_indices: Optional[Tuple[List[str], torch.Tensor]] = None,
    ) -> None:
        super().__init__(
            keys=keys,
            values=values,
            weights=weights,
            lengths=lengths,
            offsets=offsets,
            stride=stride,
            stride_per_key_per_rank=stride_per_key_per_rank,
            stride_per_key=stride_per_key,
            length_per_key=None,
            lengths_offset_per_key=lengths_offset_per_key,
            offset_per_key=offset_per_key,
            index_per_key=index_per_key,
            jt_dict=jt_dict,
            inverse_indices=inverse_indices,
        )
        self.length_per_key_tensors: List[torch.Tensor] = []
        for length in length_per_key or []:
            # dynamo does not support directly mark integers as dynamic, we thus apply a trick to embed the integer into a tensor's size and mark the size as dynamic
            t = torch.empty((length, 0))
            torch._dynamo.mark_dynamic(t, 0)
            self.length_per_key_tensors.append(t)

        self.stride_per_key_per_rank_tensor: List[List[torch.Tensor]] = []
        for strides_per_key in stride_per_key_per_rank or []:
            strides_per_key_list: List[torch.Tensor] = []
            for s in strides_per_key:
                t = torch.empty((s, 0))
                torch._dynamo.mark_dynamic(t, 0)
                strides_per_key_list.append(t)
            self.stride_per_key_per_rank_tensor.append(strides_per_key_list)

    def length_per_key(self) -> List[int]:
        if len(self.length_per_key_tensors) > 0:
            # since size has been marked as dynamic, we get a list of dynamic integers
            self._length_per_key = [t.size(0) for t in self.length_per_key_tensors]
        else:
            self._length_per_key = super().length_per_key()
        return self._length_per_key

    def stride_per_key_per_rank(self) -> List[List[int]]:
        if len(self.stride_per_key_per_rank_tensor) > 0:
            self._stride_per_key_per_rank = [
                [t.size(0) for t in strides_per_key_list]
                for strides_per_key_list in self.stride_per_key_per_rank_tensor
            ]
        else:
            self._stride_per_key_per_rank = super().stride_per_key_per_rank()
        return self._stride_per_key_per_rank


# pyre-ignore
def default_pipeline_input_transformer(inp):
    # different input items need different handlings
    for attr_name in ["id_list_features", "id_score_list_features"]:
        if hasattr(inp, attr_name):
            attr = getattr(inp, attr_name)
            if isinstance(attr, KeyedJaggedTensor):
                setattr(inp, attr_name, kjt_for_pt2_tracing(attr))
    for attr_name in [
        "uhm_history_timestamps",
        "raw_uhm_history_timestamps",
        "event_id_list_feature_invert_indexes",
    ]:
        if hasattr(inp, attr_name):
            attr = getattr(inp, attr_name)
            if isinstance(attr, dict):
                for key in attr:
                    torch._dynamo.decorators.mark_dynamic(attr[key], 0)
    if hasattr(inp, "supervision_label"):
        torch._dynamo.decorators.mark_dynamic(inp.supervision_label["keys"], 0)
        torch._dynamo.decorators.mark_dynamic(inp.supervision_label["values"], 0)

    for attr_name in ["event_id_list_features_seqs"]:
        if hasattr(inp, attr_name):
            attr = getattr(inp, attr_name)
            if isinstance(attr, dict):
                for key in attr:
                    if isinstance(attr[key], KeyedJaggedTensor):
                        attr[key] = kjt_for_pt2_tracing(attr[key], mark_length=True)

                setattr(inp, attr_name, attr)

    return inp


def register_fake_classes() -> None:
    @torch._library.register_fake_class("fbgemm::AtomicCounter")
    class FakeAtomicCounter:
        def __init__(self, counter_):
            self.counter_ = counter_

        @classmethod
        def __obj_unflatten__(cls, flat_obj):
            return cls(**dict(flat_obj))

        def increment(self) -> int:
            self.counter_ += 1
            return self.counter_

        def decrement(self) -> int:
            self.counter_ -= 1
            return self.counter_

        def reset(self):
            self.counter_ = 0

        def get(self) -> int:
            return self.counter_

        def set(self, val):
            self.counter_ = val

    @torch._library.register_fake_class("fbgemm::TensorQueue")
    class FakeTensorQueue:
        def __init__(self, queue, init_tensor):
            self.queue = queue
            self.init_tensor = init_tensor

        @classmethod
        def __obj_unflatten__(cls, flattened_ctx):
            return cls(**dict(flattened_ctx))

        def push(self, x):
            self.queue.append(x)

        def pop(self):
            if len(self.queue) == 0:
                return self.init_tensor
            return self.queue.pop(0)

        def top(self):
            if len(self.queue) == 0:
                return self.init_tensor
            return self.queue[0]

        def size(self):
            return len(self.queue)


def deregister_fake_classes() -> None:
    torch._library.fake_class_registry.deregister_fake_class("fbgemm::AtomicCounter")
    torch._library.fake_class_registry.deregister_fake_class("fbgemm::TensorQueue")


# pyre-ignore[24]
def pt2_compile_callable(f: Callable) -> Callable:
    """
    This method is used to decorate the update and compute methods of a metric computation class.
    If the metric computation class has enable_pt2_compile attribute set to True,
    then the update and compute methods will be compiled using torch.compile.
    """

    @functools.wraps(f)
    # pyre-ignore[3]
    def inner_forward(
        ref: torch.nn.Module,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if hasattr(ref, "enable_pt2_compile") and ref.enable_pt2_compile:
            pt2_compiled_attr_name = f"_{f.__name__}_pt2_compiled"
            if not hasattr(ref, pt2_compiled_attr_name):
                setattr(ref, pt2_compiled_attr_name, torch.compile(f))
            return getattr(ref, pt2_compiled_attr_name)(ref, *args, **kwargs)

        return f(ref, *args, **kwargs)

    return inner_forward
