#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple

import torch

from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
)
from torchrec.distributed.quant_embedding import ShardedQuantEmbeddingCollection

from torchrec.distributed.quant_embeddingbag import ShardedQuantEmbeddingBagCollection


def get_tbes_from_sharded_module(
    module: torch.nn.Module,
) -> List[IntNBitTableBatchedEmbeddingBagsCodegen]:
    assert type(module) in [
        ShardedQuantEmbeddingBagCollection,
        ShardedQuantEmbeddingCollection,
    ], "Only support ShardedQuantEmbeddingBagCollection and ShardedQuantEmbeddingCollection for get TBEs"
    tbes = []
    for lookup in module._lookups:
        for lookup_per_rank in lookup._embedding_lookups_per_rank:
            for emb_module in lookup_per_rank._emb_modules:
                tbes.append(emb_module._emb_module)
    return tbes


def get_tbe_specs_from_sharded_module(
    module: torch.nn.Module,
) -> List[
    Tuple[str, int, int, str, str]
]:  # # tuple of (feature_names, rows, dims, str(SparseType), str(EmbeddingLocation/placement))
    assert type(module) in [
        ShardedQuantEmbeddingBagCollection,
        ShardedQuantEmbeddingCollection,
    ], "Only support ShardedQuantEmbeddingBagCollection and ShardedQuantEmbeddingCollection for get TBE specs"
    tbe_specs = []
    tbes = get_tbes_from_sharded_module(module)
    for tbe in tbes:
        for spec in tbe.embedding_specs:
            tbe_specs.append(
                (
                    spec[0],
                    spec[1],
                    spec[2],
                    str(spec[3]),
                    str(spec[4]),
                )
            )
    return tbe_specs
