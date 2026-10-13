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


@functools.partial(jax.jit, static_argnames=["total_columns"])
def batch_index_select_sc(
    inputs: jax.Array,
    indices: jax.Array,
    input_rows: jax.Array,
    input_columns: jax.Array,
    total_columns: int,
) -> jax.Array:
    """Expand deduplicated VBE rows into batch-major output on SparseCore.

    inputs: flattened concatenated embedding tables
    indices: flattened reverse_indices, conceptually shaped [K, B]
    input_rows: deduped batch size for each uncombined feature
    input_columns: embedding dimension for each uncombined feature
    total_columns: combined embedding dimension of the output

    VBE feature-segment parallelism splits each feature reverse_indices into
    NL-sized segments. Each subcore gathers one segment of deduped embedding
    rows and writes it into the corresponding embedding output rows.
    """
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
    COL_TILE = 16
    # The kernel views the output as [batch, combined embedding dim] for a
    # rectangular output DMA, then flattens it to match the FBGEMM contract.
    num_features = input_columns.shape[0]

    batch_size = indices.shape[0] // num_features
    if indices.shape[0] % num_features != 0:
        raise ValueError(
            "Total number of elements in `indices` should be a multiple of `num_features`"
        )
    if batch_size % NL != 0:
        raise NotImplementedError(f"batch size must be divisible by {NL}")
    if total_columns % COL_TILE != 0 or inputs.shape[0] % COL_TILE != 0:
        raise NotImplementedError(
            f"embedding dimensions must be divisible by {COL_TILE}"
        )

    # Parallelism scheme: each subcore handles one NL-sized segment of
    # one feature reverse_indices and reconstructs that embedding output segment.
    tiles_per_feature = batch_size // NL
    total_tiles = num_features * tiles_per_feature
    n_blocks = (total_tiles + num_subcores - 1) // num_subcores

    # Materialize three separate offset arrays on TPU.
    rows = jnp.asarray(input_rows, dtype=jnp.int32)
    columns = jnp.asarray(input_columns, dtype=jnp.int32)
    zero = jnp.zeros((1,), dtype=jnp.int32)
    input_offsets = jnp.concatenate((zero, jnp.cumsum(rows * columns)[:-1]))
    column_offsets = jnp.concatenate((zero, jnp.cumsum(columns)[:-1]))
    # Repeat each feature scalar across all lanes so the kernel can read lane 0
    # statically; SparseCore does not lower a dynamic VMEM lane selection.
    input_offsets = jnp.repeat(input_offsets, NL)
    column_offsets = jnp.repeat(column_offsets, NL)
    columns = jnp.repeat(columns, NL)
    # View the flat input as COL_TILE-wide rows so plsc.Indices can gather one
    # column chunk from all NL selected embedding rows in a single indexed DMA.
    input_tiles = inputs.reshape((-1, COL_TILE))
    indices = indices.astype(jnp.int32)

    @pl.kernel(  # pyrefly: ignore [missing-attribute]
        out_type=jax.ShapeDtypeStruct((batch_size, total_columns), inputs.dtype),
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            use_tc_tiling_on_sc=False,  # pyrefly: ignore [unexpected-keyword]
            needs_layout_passes=False,  # pyrefly: ignore [unexpected-keyword]
        ),
        scratch_types=[
            pltpu.VMEM((NL,), jnp.int32),
            pltpu.VMEM((NL, COL_TILE), inputs.dtype),
            pltpu.VMEM((NL,), jnp.int32),
            pltpu.VMEM((NL,), jnp.int32),
            pltpu.VMEM((NL,), jnp.int32),
            pltpu.SemaphoreType.DMA(()),
        ],
        name="batch_index_select_sc",
        mesh=mesh,
    )
    def _kernel(
        input_tiles_hbm,
        indices_hbm,
        input_offsets_hbm,
        columns_hbm,
        column_offsets_hbm,
        out_hbm,
        indices_vmem,
        inputs_vmem,
        input_offsets_vmem,
        columns_vmem,
        column_offsets_vmem,
        gather_sem,
    ):
        @functools.partial(pltpu.emit_pipeline, grid=(n_blocks,))
        def _inner():
            # Grab the global subcore ID across all sparse cores.
            core_index = jax.lax.axis_index("c")
            subcore_in_core = jax.lax.axis_index("s")
            subcore_id = core_index * sc.num_subcores + subcore_in_core
            # Grab the tile and feature/batch_tile assigned to this subcore.
            tile = pl.program_id(0) * num_subcores + subcore_id

            # Only work on tiles that are less than the logical tile count.
            @pl.when(tile < total_tiles)
            def _valid_tile():
                feature = tile // tiles_per_feature
                feature_block_start = pl.multiple_of(feature * NL, NL)
                # Load the feature metadata into VMEM.
                pltpu.sync_copy(
                    columns_hbm.at[pl.ds(feature_block_start, NL)], columns_vmem
                )
                pltpu.sync_copy(
                    column_offsets_hbm.at[pl.ds(feature_block_start, NL)],
                    column_offsets_vmem,
                )
                pltpu.sync_copy(
                    input_offsets_hbm.at[pl.ds(feature_block_start, NL)],
                    input_offsets_vmem,
                )
                col_length = columns_vmem[...][0]
                col_start = column_offsets_vmem[...][0]
                input_start = input_offsets_vmem[...][0]

                # Stage this feature segment reverse_indices in VMEM.
                batch_tile = tile % tiles_per_feature
                batch_start = batch_tile * NL
                index_start = pl.multiple_of(feature * batch_size + batch_start, 8)
                pltpu.sync_copy(indices_hbm.at[pl.ds(index_start, NL)], indices_vmem)
                # Convert selected embedding rows to COL_TILE-row indices in the
                # reshaped input. Incrementing them advances one column chunk.
                chunks_per_row = col_length // COL_TILE
                indices_vmem[...] = (
                    input_start // COL_TILE + indices_vmem[...] * chunks_per_row
                )

                # Gather NL noncontiguous input rows, then write one rectangular
                # [NL, COL_TILE] output block for each feature-column chunk.
                @pl.loop(0, chunks_per_row)
                def _copy_chunk(chunk):
                    # Grab the embeddings using this segment reverse_indices.
                    # TODO remove this unnecessary async copy.
                    gather = pltpu.make_async_copy(
                        input_tiles_hbm.at[plsc.Indices(indices_vmem)],
                        inputs_vmem,
                        gather_sem,
                    )
                    gather.start()
                    gather.wait()
                    output_col = pl.multiple_of(col_start + chunk * COL_TILE, COL_TILE)
                    # Write the gathered embedding segment into emb_output.
                    pltpu.sync_copy(
                        inputs_vmem,
                        out_hbm.at[pl.ds(batch_start, NL), pl.ds(output_col, COL_TILE)],
                    )
                    indices_vmem[...] = indices_vmem[...] + 1

        _inner()

    # FBGEMM returns a flat tensor; this reshape is metadata-only.
    return _kernel(
        input_tiles, indices, input_offsets, columns, column_offsets
    ).reshape(-1)
