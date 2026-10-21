#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools

import jax
from torchrec.experimental.torch_tpu.pallas.permute_1d import permute_1d_sparse_data_sc


@functools.partial(jax.jit, static_argnames=["permuted_lengths_sum"])
def permute_2d_sparse_data_sc(
    permute: jax.Array,
    lengths: jax.Array,
    values: jax.Array,
    weights: jax.Array | None,
    permuted_lengths_sum: int | None,
) -> tuple[jax.Array, jax.Array]:
    """Permute feature rows and their values, preserving the bags within each row."""
    if lengths.ndim != 2:
        raise ValueError("lengths must be two-dimensional")

    feature_sizes = lengths.sum(axis=1)
    _, output_values = permute_1d_sparse_data_sc(
        permute=permute,
        lengths=feature_sizes,
        indices=values,
        weights=weights,
        permuted_lengths_sum=permuted_lengths_sum,
    )
    return lengths[permute], output_values
