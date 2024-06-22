#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import torch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

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

    return KeyedJaggedTensor(
        keys=kjt.keys(),
        values=values,
        lengths=lengths,
        weights=kjt.weights_or_none(),
        stride=stride if not is_vb else None,
        stride_per_key_per_rank=kjt.stride_per_key_per_rank() if is_vb else None,
        inverse_indices=inverse_indices,
    )


# pyre-ignore
def default_pipeline_input_transformer(inp):
    for attr_name in ["id_list_features", "id_score_list_features"]:
        if hasattr(inp, attr_name):
            attr = getattr(inp, attr_name)
            if isinstance(attr, KeyedJaggedTensor):
                setattr(inp, attr_name, kjt_for_pt2_tracing(attr))
    return inp
