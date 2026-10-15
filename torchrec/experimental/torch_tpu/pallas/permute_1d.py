#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu, tpu_sc as plsc
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


@functools.partial(jax.jit, static_argnames=["permuted_lengths_sum"])
def permute_1d_sparse_data_sc(
    permute: jax.Array,
    lengths: jax.Array,
    indices: jax.Array,
    weights: jax.Array | None,
    permuted_lengths_sum: int | None,
) -> tuple[jax.Array, jax.Array]:
    """
    Permute variable-length segments of values.

    Args:
        permute (jax.Array): input segment indices in output order. Indices
            may be repeated to duplicate segments.
        lengths (jax.Array): one-dimensional array of input segment lengths.
        indices (jax.Array): one-dimensional array of concatenated input values.
        weights (Optional[jax.Array]): optional weights. Only ``None`` is
            currently supported.
        permuted_lengths_sum (Optional[int]): total number of output values.
            Must be supplied as a static integer.

    Returns:
        Tuple[jax.Array, jax.Array]: permuted segment lengths and concatenated
            values.

    TODO: Consider pingponging the offset calculation between subcores
    TODO: Consider doing aligned DMA access rather than indirect DMA for faster gathers
    """
    if weights is not None:
        raise NotImplementedError("Permuting weights is not added yet")
    if permuted_lengths_sum is None:
        raise ValueError("Pallas requires a static permuted_lengths_sum")

    # Typical sparse-core subcore information
    sc = pltpu.get_tpu_info().sparse_core  # pyrefly: ignore [missing-attribute]
    mesh = plsc.VectorSubcoreMesh(
        core_axis_name="c",
        subcore_axis_name="s",
        num_cores=sc.num_cores,
        num_subcores=sc.num_subcores,  # pyrefly: ignore [unexpected-keyword]
    )
    num_subcores = sc.num_cores * sc.num_subcores
    NL = 16
    ID_TILE = 16

    # Get the offsets for the input and outputs
    # TODO: Check if this is done on sparse-core and latency from it
    permuted_lengths = lengths[permute]
    input_offsets = associative_scan.complete_cumsum_jax(lengths)
    output_offsets = associative_scan.complete_cumsum_jax(permuted_lengths)

    # Get Output Sizes
    numb_bags: int = permute.size
    numb_perm_inds = permuted_lengths_sum

    # Parallelism strategy: assign each output_offset to a subcore
    # Assign each indices to a lane
    n_blocks = (numb_bags + num_subcores - 1) // num_subcores

    # Make the offsets match the size of the tile
    permute = jnp.repeat(permute, NL)
    input_offsets = jnp.repeat(input_offsets, NL)
    output_offsets = jnp.repeat(output_offsets, NL)

    # Pad indices to be a multiple of `NL`` size
    pad_right = (-indices.size) % NL + ID_TILE
    indices = jnp.pad(indices, (0, pad_right), constant_values=-1)

    @pl.kernel(  # pyrefly: ignore [missing-attribute]
        out_type=jax.ShapeDtypeStruct((numb_perm_inds,), indices.dtype),
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            use_tc_tiling_on_sc=False,  # pyrefly: ignore [unexpected-keyword]
            needs_layout_passes=False,  # pyrefly: ignore [unexpected-keyword]
        ),
        scratch_types=[
            pltpu.VMEM((NL,), permute.dtype),
            pltpu.VMEM((NL,), indices.dtype),
            pltpu.VMEM((NL,), input_offsets.dtype),
            pltpu.VMEM((NL,), output_offsets.dtype),
        ],
        name="permute_1d_sparse_data_sc",
        mesh=mesh,
    )
    def _kernel(
        permuted_hbm,
        indices_hbm,
        input_offsets_hbm,
        output_offsets_hbm,
        permuted_indices_out,
        permuted_vmem,
        indices_vmem,
        input_offsets_vmem,
        output_offsets_vmem,
    ):

        @functools.partial(pltpu.emit_pipeline, grid=(n_blocks,))
        def _inner():
            # Grab the global subcore ID across all sparse cores.
            core_index = jax.lax.axis_index("c")
            subcore_in_core = jax.lax.axis_index("s")
            subcore_id = core_index * sc.num_subcores + subcore_in_core
            # Grab the tile and feature/batch_tile assigned to this subcore.
            permute_id = pl.program_id(0) * num_subcores + subcore_id

            @pl.when(permute_id < numb_bags)
            def _valid_permute_id():
                permute_id_tile = pl.multiple_of(permute_id * NL, NL)

                pltpu.sync_copy(
                    permuted_hbm.at[pl.ds(permute_id_tile, NL)],
                    permuted_vmem,
                )
                output_id = permuted_vmem[...][0]
                output_id_tile = pl.multiple_of(output_id * NL, NL)

                # Grab the inputs positions
                pltpu.sync_copy(
                    input_offsets_hbm.at[pl.ds(output_id_tile, NL)],
                    input_offsets_vmem,
                )
                input_start_id = input_offsets_vmem[...][0]
                output_id_tile2 = pl.multiple_of((output_id + 1) * NL, NL)
                pltpu.sync_copy(
                    input_offsets_hbm.at[pl.ds(output_id_tile2, NL)],
                    input_offsets_vmem,
                )
                input_end_id = input_offsets_vmem[...][0]

                # Grab the outputs positions
                pltpu.sync_copy(
                    output_offsets_hbm.at[pl.ds(permute_id_tile, NL)],
                    output_offsets_vmem,
                )
                output_start_id = output_offsets_vmem[...][0]

                # Grab `ID_TILE` number of indices to transfer
                numb_indices = input_end_id - input_start_id
                numb_chunks = (numb_indices + ID_TILE - 1) // ID_TILE

                @pl.loop(0, numb_chunks)
                def _permute_indices(i):
                    lanes = jnp.arange(ID_TILE)
                    local_ind = lanes + i * ID_TILE
                    valid = local_ind < numb_indices

                    output_offsets_vmem[...] = jnp.where(
                        valid,
                        input_start_id + local_ind,
                        -1,
                    )
                    pltpu.sync_copy(
                        indices_hbm.at[
                            plsc.Indices(output_offsets_vmem, ignored_value=-1)
                        ],
                        indices_vmem,
                    )

                    output_offsets_vmem[...] = jnp.where(
                        valid,
                        output_start_id + local_ind,
                        -1,
                    )
                    pltpu.sync_copy(
                        indices_vmem,
                        permuted_indices_out.at[
                            plsc.Indices(output_offsets_vmem, ignored_value=-1)
                        ],
                    )

        _inner()

    return permuted_lengths, _kernel(permute, indices, input_offsets, output_offsets)
