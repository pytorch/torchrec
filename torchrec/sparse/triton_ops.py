#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Dict, List, Tuple

import torch

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl


def triton_permute_pooled_embs(
    values: List[torch.Tensor],
    keys: List[List[str]],
    lengths: List[List[int]],
    groups: List[List[str]],
) -> Tuple[torch.Tensor, List[int]]:
    """
    Permute the values of a KeyedTensor based on the groups.
    """
    assert len(values) == len(keys)
    assert len(values) == len(lengths)
    P = sum(len(g) for g in groups)
    B = values[0].shape[0]
    device = values[0].device
    in_length: int = 0
    out_length: int = 0
    splits: List[int] = [0] * len(groups)

    # permute: [in_offset, out_offset, length, next]
    permutes: List[List[int]] = [[0] * 4 for _ in range(P)]
    # key -> (in_tensor, in_offset, length)
    lookup: Dict[str, Tuple[int, int, int]] = {}
    for i, (key, length) in enumerate(zip(keys, lengths)):
        for k, l in zip(key, length):
            lookup[k] = (i, in_length, l)
            in_length += l

    curr = 0
    for j, group in enumerate(groups):
        for k in group:
            in_tensor, in_offset, length = lookup[k]
            permutes[curr][:] = [in_offset, out_length, length, 0]
            out_length += length
            splits[j] += length
            curr += 1

    permute_tensor = torch.tensor(permutes, dtype=torch.int32).to(
        device, non_blocking=True
    )
    output: torch.Tensor = torch.empty(B, out_length, device=device)
    permute_pooled_embeddings_kernel[(B, P)](
        torch.concat(values, dim=1),
        output,
        permute_tensor,
        in_length,
        out_length,
    )
    return output, splits


@triton.jit
def permute_pooled_embeddings_kernel(
    values,
    outputs,
    permutes,
    in_length,
    out_length,
):
    batch_id = tl.program_id(0)
    pid = tl.program_id(1)
    in_offset = tl.load(permutes + 4 * pid)
    out_offset = tl.load(permutes + 4 * pid + 1)
    length = tl.load(permutes + 4 * pid + 2)
    BLOCK_SIZE: tl.constexpr = 32

    idx = tl.arange(0, BLOCK_SIZE)
    in_ptr = values + batch_id * in_length + in_offset + idx
    out_ptr = outputs + batch_id * out_length + out_offset + idx

    for k in range(0, length, BLOCK_SIZE):
        inputs = tl.load(in_ptr + k, mask=idx < length - k)
        tl.store(out_ptr + k, inputs, mask=idx < length - k)


def triton_permute_multi_embs(
    values: List[torch.Tensor],
    keys: List[List[str]],
    lengths: List[List[int]],
    groups: List[List[str]],
) -> List[torch.Tensor]:
    """
    Permute the values of a KeyedTensor based on the groups.
    """
    assert len(values) == len(keys)
    assert len(values) == len(lengths)
    P = sum(len(g) for g in groups)
    B = values[0].shape[0]
    device = values[0].device
    in_lengths: List[int] = [0] * len(values)
    out_lengths: List[int] = [0] * len(groups)

    inputs: torch.Tensor = torch.tensor(
        [v.data_ptr() for v in values], dtype=torch.int64
    ).to(device, non_blocking=True)

    # permute: [in_tensor, out_tensor, in_offset, out_offset, length, next]
    permutes: List[List[int]] = [[0] * 6 for _ in range(P)]
    # key -> (in_tensor, in_offset, length)
    lookup: Dict[str, Tuple[int, int, int]] = {}
    for i, (key, length) in enumerate(zip(keys, lengths)):
        for k, l in zip(key, length):
            lookup[k] = (i, in_lengths[i], l)
            in_lengths[i] += l

    curr = 0
    for out_tensor, group in enumerate(groups):
        for k in group:
            in_tensor, in_offset, length = lookup[k]
            permutes[curr][:] = [
                in_tensor,
                out_tensor,
                in_offset,
                out_lengths[out_tensor],
                length,
                0,
            ]
            out_lengths[out_tensor] += length
            curr += 1

    permute_tensor = torch.tensor(permutes, dtype=torch.int64).to(
        device, non_blocking=True
    )
    outputs: List[torch.Tensor] = [
        torch.empty(B, L, device=device) for L in out_lengths
    ]
    output: torch.Tensor = torch.tensor(
        [o.data_ptr() for o in outputs], dtype=torch.int64
    ).to(device, non_blocking=True)
    in_lengths_ptr: torch.Tensor = torch.tensor(in_lengths, dtype=torch.int64).to(
        device, non_blocking=True
    )
    out_lengths_ptr: torch.Tensor = torch.tensor(out_lengths, dtype=torch.int64).to(
        device, non_blocking=True
    )
    permute_multi_embeddings_kernel[(B, P)](
        values[0],
        inputs,
        output,
        permute_tensor,
        in_lengths_ptr,
        out_lengths_ptr,
    )
    return outputs


@triton.jit
def permute_multi_embeddings_kernel(
    example,
    inputs,
    output,
    permutes,
    in_lengths,
    out_lengths,
):
    batch_id = tl.program_id(0)
    pid = tl.program_id(1)
    in_tensor = tl.load(permutes + 6 * pid)
    out_tensor = tl.load(permutes + 6 * pid + 1)
    in_offset = tl.load(permutes + 6 * pid + 2)
    out_offset = tl.load(permutes + 6 * pid + 3)
    length = tl.load(permutes + 6 * pid + 4)

    in_length = tl.load(in_lengths + in_tensor)
    out_length = tl.load(out_lengths + out_tensor)

    BLOCK_SIZE: tl.constexpr = 32
    idx = tl.arange(0, BLOCK_SIZE)

    in_ptr = (
        tl.load(inputs + in_tensor).to(example.dtype, bitcast=True)
        + batch_id * in_length
        + in_offset
        + idx
    )
    out_ptr = (
        tl.load(output + out_tensor).to(example.dtype, bitcast=True)
        + batch_id * out_length
        + out_offset
        + idx
    )

    for k in range(0, length, BLOCK_SIZE):
        in_data = tl.load(in_ptr + k, mask=idx < length - k)
        tl.store(out_ptr + k, in_data, mask=idx < length - k)


# @custom_impl("torchrec::permute_multi_embeddings", "CUDA")
# @custom_impl("torchrec::permute_multi_embeddings", "AutogradCUDA")
# def permute_multi_embeddings(
#     values: List[torch.Tensor],
#     keys: List[List[str]],
#     lengths: List[List[int]],
#     groups: List[List[str]],
# ) -> List[torch.Tensor]:
#     """
#     Permute the values of a KeyedTensor based on the groups.
#     """
#     assert len(values) == len(keys)
#     assert len(values) == len(lengths)
