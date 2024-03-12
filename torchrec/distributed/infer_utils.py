#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple

from torchrec.distributed.quant_embeddingbag import ShardedQuantEmbeddingBagCollection


def get_tbe_specs_from_sqebc(
    sqebc: ShardedQuantEmbeddingBagCollection,
) -> List[
    Tuple[str, int, int, str, str]
]:  # # tuple of (feature_names, rows, dims, str(SparseType), str(EmbeddingLocation/placement))
    tbe_specs = []
    for lookup in sqebc._lookups:
        for lookup_per_rank in lookup._embedding_lookups_per_rank:
            for emb_module in lookup_per_rank._emb_modules:
                for spec in emb_module._emb_module.embedding_specs:
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
