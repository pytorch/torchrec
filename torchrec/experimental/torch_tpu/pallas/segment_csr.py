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


@functools.partial(jax.jit, static_argnames=["batch_size"])
def segment_sum_csr_sc(
    batch_size: int,
    csr_seg: jax.Array,
    values: jax.Array,
) -> jax.Array:
    """
    Sum each segment/component inside an array.

    Args:
        batch_size (int): The batch-size to multiply the indices with
        csr_seg (jax.Array): one-dimensional array of input segment offsets.
        values (jax.Array): one-dimensional array of values to be summed.

    Returns:
        jax.Array: Summation of each segment of values, same size of csr_seg

    TODO: Improve this by removing the grab by plsc.Indices, and just do a separate tail handling component
          Consider doing an aligned DMA with closet multiple then doing plsc.Indices.
    TODO: Compute/memory optimization: While summing, grab the next set of indices
    """
    if values.shape[0] == 0:
        return jnp.zeros((len(csr_seg) - 1,), dtype=values.dtype)
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
    SEGMENT_GATHER_LEN = NL

    # Parallelism strategy:assign each segment to a sub-core and sum through with the lanes
    numb_segments = len(csr_seg) - 1
    n_blocks = (numb_segments + num_subcores - 1) // num_subcores

    # Pad the arrays, or make them match the tile size
    batch_size_arr = jnp.full((NL,), batch_size, dtype=jnp.int32)
    csr_seg = jnp.repeat(csr_seg, NL)

    # Pad csr_seg till the end
    pad_right = (-values.shape[0]) % NL
    values = jnp.pad(values, (0, pad_right), constant_values=0)

    @pl.kernel(  # pyrefly: ignore [missing-attribute]
        out_type=jax.ShapeDtypeStruct((numb_segments,), values.dtype),
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            use_tc_tiling_on_sc=False,  # pyrefly: ignore [unexpected-keyword]
            needs_layout_passes=False,  # pyrefly: ignore [unexpected-keyword]
        ),
        scratch_types=[
            pltpu.VMEM((NL,), jnp.int32),
            pltpu.VMEM((NL,), jnp.int32),
            pltpu.VMEM((SEGMENT_GATHER_LEN,), values.dtype),
            pltpu.VMEM((SEGMENT_GATHER_LEN,), jnp.int32),
            pltpu.VMEM((NL,), values.dtype),
        ],
        name="segment_sum_csr_sc",
        mesh=mesh,
    )
    def _kernel(
        batch_size_hbm,
        csr_seg_hbm,
        values_hbm,
        sum_arr_out,
        batch_size_vmem,
        csr_seg_vmem,
        values_vmem,
        indices_vmem,
        summation_buffer_vmem,
    ):

        @functools.partial(
            pltpu.emit_pipeline,
            grid=(n_blocks,),
        )
        def _inner():
            # Grab the global subcore ID across all sparse cores.
            core_index = jax.lax.axis_index("c")
            subcore_in_core = jax.lax.axis_index("s")
            subcore_id = core_index * sc.num_subcores + subcore_in_core
            # Grab the segment-id
            segment_id = pl.program_id(0) * num_subcores + subcore_id

            @pl.when(segment_id < numb_segments)
            def _valid_segment():
                # Grab the start and end indices
                segment_id_tile = pl.multiple_of(segment_id * NL, NL)
                pltpu.sync_copy(
                    csr_seg_hbm.at[pl.ds(segment_id_tile, NL)], csr_seg_vmem
                )
                start_id = csr_seg_vmem[...][0]

                segment_id_tile = pl.multiple_of((segment_id + 1) * NL, NL)
                pltpu.sync_copy(
                    csr_seg_hbm.at[pl.ds(segment_id_tile, NL)], csr_seg_vmem
                )
                end_id = csr_seg_vmem[...][0]

                # Grab batch-size
                batch_size_tile = pl.multiple_of(0, NL)
                pltpu.sync_copy(
                    batch_size_hbm.at[pl.ds(batch_size_tile, NL)], batch_size_vmem
                )
                batch_size = batch_size_vmem[...][0]

                # Grab the segments in term of chunks
                segment_length = (end_id - start_id) * batch_size
                numb_chunks = (
                    segment_length + SEGMENT_GATHER_LEN - 1
                ) // SEGMENT_GATHER_LEN

                # Float value
                summation_buffer_vmem[...] = jnp.zeros(
                    (NL,), dtype=summation_buffer_vmem.dtype
                )

                @pl.loop(0, numb_chunks)
                def _gather_segment_chunk(i):
                    # Might have to check valid chunk as well
                    @pl.when(i * SEGMENT_GATHER_LEN < segment_length)
                    def _valid():
                        chunk_start = start_id * batch_size + i * SEGMENT_GATHER_LEN
                        positions = chunk_start + jnp.arange(
                            SEGMENT_GATHER_LEN, dtype=jnp.int32
                        )
                        valid = positions < end_id * batch_size
                        indices_vmem[...] = jnp.where(valid, positions, -1)

                        # Grab the values  values
                        pltpu.sync_copy(
                            values_hbm.at[plsc.Indices(indices_vmem, ignored_value=-1)],
                            values_vmem,
                        )

                        @pl.loop(0, SEGMENT_GATHER_LEN, step=NL)
                        def _summation(j):
                            segment_pos = (
                                chunk_start + jnp.arange(NL, dtype=jnp.int32) + j
                            )
                            segment_valid = segment_pos < batch_size * end_id

                            values_lane = values_vmem[pl.ds(j, NL)]
                            summation_buffer_vmem[...] += jnp.where(
                                segment_valid, values_lane, 0
                            )

                output = jnp.sum(summation_buffer_vmem[...])
                # Need to broadcast this output to a vmem reference to use sync_copy
                summation_buffer_vmem[...] = jnp.broadcast_to(output, (NL,))
                csr_seg_vmem[...] = jnp.where(
                    jnp.arange(NL, dtype=jnp.int32) == 0, segment_id, -1
                )
                pltpu.sync_copy(
                    summation_buffer_vmem,
                    sum_arr_out.at[plsc.Indices(csr_seg_vmem, ignored_value=-1)],
                )

        _inner()

    return _kernel(batch_size_arr, csr_seg, values)
