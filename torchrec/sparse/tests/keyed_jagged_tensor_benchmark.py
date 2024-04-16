#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import copy
import multiprocessing
import time
from typing import Callable, Dict, List

import click

from torchrec.distributed.test_utils.test_model import ModelInput
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor


def generate_kjt(
    table_configs: List[EmbeddingBagConfig],
    sparse_features_per_kjt: int,
) -> KeyedJaggedTensor:
    raw_value = ModelInput.generate(
        batch_size=1,
        world_size=1,
        num_float_features=0,
        tables=table_configs,
        weighted_tables=[],
    )
    return raw_value[0].idlist_features


def prepare_benchmark(
    sparse_features_per_kjt: int,
    test_size: int,
    num_embeddings: int = 1000,
    embedding_dim: int = 50,
    in_parallel: bool = False,
) -> List[KeyedJaggedTensor]:
    tables: List[EmbeddingBagConfig] = [
        EmbeddingBagConfig(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            name=f"table_{i}",
            feature_names=[f"feature_{i}"],
        )
        for i in range(sparse_features_per_kjt)
    ]

    kjt_lists: List[KeyedJaggedTensor] = []

    if in_parallel:
        # TODO Make this parallel version performance efficient
        with multiprocessing.Pool() as pool:
            kjt_lists: List[KeyedJaggedTensor] = pool.starmap(
                generate_kjt,
                [(tables, sparse_features_per_kjt) for _ in range(test_size)],
            )
    else:
        for _ in range(test_size):
            kjt_lists.append(generate_kjt(tables, sparse_features_per_kjt))
    return list(kjt_lists)


def kjt_to_dict(kjt: KeyedJaggedTensor) -> Dict[str, JaggedTensor]:
    return kjt.to_dict()


def benchmark(
    input_data: List[KeyedJaggedTensor],
    test_name: str,
    warmup_size: int,
    test_size: int,
    test_func: Callable[..., object],
) -> None:
    start = time.perf_counter()
    for i in range(warmup_size):
        test_func(input_data[i])
    end = time.perf_counter()
    print(f"warmup time for {test_name} {(end-start)*1000/warmup_size:.1f}ms")
    start = time.perf_counter()
    for i in range(warmup_size, warmup_size + test_size):
        test_func(input_data[i])
    end = time.perf_counter()
    print(f"benmark avarge time {test_name} {(end-start)*1000/test_size:.1f}ms")


def bench(
    feature_per_kjt: int,
    test_size: int,
    warmup_size: int,
    test_jitscripted: bool = True,
    parallel: bool = False,
) -> None:
    generated_input_data = prepare_benchmark(
        feature_per_kjt, test_size + warmup_size, in_parallel=parallel
    )
    assert len(generated_input_data) == test_size + warmup_size

    test_sets = [generated_input_data]
    test_names = ["eager"]
    if test_jitscripted:
        test_sets.append(copy.deepcopy(generated_input_data))
        test_names.append("jitscripted")

    benchmark(
        input_data=test_sets[0],
        test_name=test_names[0],
        warmup_size=warmup_size,
        test_size=test_size,
        test_func=lambda x: x.to_dict(),
    )
    benchmark(
        input_data=test_sets[1],
        test_name=test_names[1],
        warmup_size=warmup_size,
        test_size=test_size,
        test_func=lambda x: kjt_to_dict(x),
    )


@click.command()
@click.option(
    "--feature_per_kjt",
    default=50,
    help="Total number of sparse features per KJT. Loosely corresponds to lengths of KJT values.",
)
@click.option(
    "--test_size",
    default=100,
    help="Total number of KJT tested in the benchmark.",
)
@click.option(
    "--warmup_size",
    default=10,
    help="Total warmup number of KJT tested before the formal benchmark.",
)
@click.option(
    "--parallel",
    "-p",
    help="Generate input data in parallel.",
    required=False,
    type=bool,
    is_flag=False,
)
def main(
    feature_per_kjt: int, test_size: int, warmup_size: int, parallel: bool = False
) -> None:
    bench(feature_per_kjt, test_size, warmup_size)


if __name__ == "__main__":
    main()
