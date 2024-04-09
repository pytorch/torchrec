#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import time
from typing import cast, List

import click

from torchrec.distributed.test_utils.test_model import ModelInput
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


def prepare_benchmark(
    sparse_features: int, batch_size: int = 10
) -> List[KeyedJaggedTensor]:
    tables = [
        EmbeddingBagConfig(
            num_embeddings=100,
            embedding_dim=20,
            name=f"table_{i}",
            feature_names=[f"feature_{i}"],
        )
        for i in range(sparse_features)
    ]
    kjt_lists = []
    for _ in range(batch_size):
        raw_value = ModelInput.generate(
            batch_size=100,
            world_size=1,
            num_float_features=0,
            tables=tables,
            weighted_tables=tables,
        )
        kjt_lists.append(raw_value[0].idlist_features)

    return kjt_lists


def bench(
    n_sparse: int,
    warmup_step: int = 10,
) -> None:
    batch_size = 100
    input_data = prepare_benchmark(n_sparse, batch_size + warmup_step)
    start = time.perf_counter()
    for i in range(warmup_step):
        input_data[i].to_dict()
    end = time.perf_counter()
    print(f"warmup time {(end-start)*1000/warmup_step:.1f}ms")
    start = time.perf_counter()
    for i in range(warmup_step, warmup_step + batch_size):
        input_data[i].to_dict()
    end = time.perf_counter()
    print(f"benmark avarge time {(end-start)*1000/batch_size:.1f}ms")


@click.command()
@click.option(
    "--n_sparse",
    default=100,
    help="Total number of sparse embeddings to be used.",
)
def main(n_sparse: int) -> None:
    bench(n_sparse)


if __name__ == "__main__":
    main()
