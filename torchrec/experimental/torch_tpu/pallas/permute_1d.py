#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools

import jax
import jax.numpy as jnp
from torchrec.experimental.torch_tpu.pallas import associative_scan


@functools.partial(jax.jit, static_argnames=("permuted_lengths_sum",))
def permute_1d_sparse_data_indices_jax(
    permute: jax.Array,
    lengths: jax.Array,
    permuted_lengths_sum: int,  # Total length of the output
) -> tuple[jax.Array, jax.Array]:
    permute = permute.astype(jnp.int32)
    permuted_lengths = lengths[permute]
    input_offsets = associative_scan.complete_cumsum_jax(lengths)
    output_offsets = associative_scan.complete_cumsum_jax(permuted_lengths)
    source_offsets = input_offsets[permute] - output_offsets[:-1]
    source_indices = jnp.arange(permuted_lengths_sum, dtype=jnp.int32) + jnp.repeat(
        source_offsets,
        permuted_lengths,
        total_repeat_length=permuted_lengths_sum,
    )
    return permuted_lengths, source_indices


@functools.partial(jax.jit, static_argnames=("permuted_lengths_sum",))
def permute_1d_sparse_data_jax(
    permute: jax.Array,
    lengths: jax.Array,
    values: jax.Array,
    weights: jax.Array | None,
    permuted_lengths_sum: int,
) -> tuple[jax.Array, jax.Array, jax.Array | None]:
    """Permute variable-length segments of values and optional weights."""
    permuted_lengths, source_indices = permute_1d_sparse_data_indices_jax(
        permute, lengths, permuted_lengths_sum
    )
    return (
        permuted_lengths,
        values[source_indices],
        None if weights is None else weights[source_indices],
    )
