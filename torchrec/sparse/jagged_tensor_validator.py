#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


def validate_keyed_jagged_tensor(
    kjt: KeyedJaggedTensor,
) -> None:
    """
    Validates the inputs that construct a KeyedJaggedTensor.

    This function ensures that:
    - At least one of lengths or offsets is provided
    - If both are provided, they are consistent with each other
    - The dimensions of these tensors align with the values tensor

    Any invalid input will result in a ValueError being thrown.
    """
    # TODO: Add validation checks on keys, values, weights
    _validate_lengths_and_offsets(kjt)


def _validate_lengths_and_offsets(kjt: KeyedJaggedTensor) -> None:
    lengths = kjt.lengths_or_none()
    offsets = kjt.offsets_or_none()
    if lengths is None and offsets is None:
        raise ValueError(
            "lengths and offsets cannot be both empty in KeyedJaggedTensor"
        )
    elif lengths is not None and offsets is not None:
        _validate_lengths_and_offsets_consistency(lengths, offsets, kjt.values())
    elif lengths is not None:
        _validate_lengths(lengths, kjt.values())
    elif offsets is not None:
        _validate_offsets(offsets, kjt.values())


def _validate_lengths_and_offsets_consistency(
    lengths: torch.Tensor, offsets: torch.Tensor, values: torch.Tensor
) -> None:
    _validate_lengths(lengths, values)
    _validate_offsets(offsets, values)

    if lengths.numel() != offsets.numel() - 1:
        raise ValueError(
            f"Expected lengths size to be 1 more than offsets size, but got lengths size: {lengths.numel()} and offsets size: {offsets.numel()}"
        )

    if not lengths.equal(torch.diff(offsets)):
        raise ValueError("offsets is not equal to the cumulative sum of lengths")


def _validate_lengths(lengths: torch.Tensor, values: torch.Tensor) -> None:
    if lengths.sum().item() != values.numel():
        raise ValueError(
            f"Sum of lengths must equal the number of values, but got {lengths.sum().item()} and {values.numel()}"
        )


def _validate_offsets(offsets: torch.Tensor, values: torch.Tensor) -> None:
    if offsets.numel() == 0:
        raise ValueError("offsets cannot be empty")

    if offsets[0] != 0:
        raise ValueError(f"Expected first offset to be 0, but got {offsets[0]} instead")

    if offsets[-1] != values.numel():
        raise ValueError(
            f"The last element of offsets must equal to the number of values, but got {offsets[-1]} and {values.numel()}"
        )
