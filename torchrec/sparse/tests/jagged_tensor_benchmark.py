#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import time
import timeit
from typing import Callable, List, Tuple

import click

import numpy as np

import torch
from torchrec.sparse.jagged_tensor import _regroup_keyed_tensors, KeyedTensor


def prepare_benchmark(
    dense_features: int, sparse_features: int
) -> Tuple[List["KeyedTensor"], List[List[str]]]:
    key_dim = 1
    tensor_list_1 = [torch.randn(2, 3) for i in range(dense_features)]
    keys_1 = [f"dense_{i}" for i in range(dense_features)]
    kt_1 = KeyedTensor.from_tensor_list(keys_1, tensor_list_1, key_dim)
    tensor_list_2 = [torch.randn(2, 3) for i in range(sparse_features)]
    keys_2 = [f"sparse_{i}" for i in range(sparse_features)]
    kt_2 = KeyedTensor.from_tensor_list(keys_2, tensor_list_2, key_dim)
    return ([kt_1, kt_2], [["dense_0", "sparse_1", "dense_2"], ["dense_1", "sparse_0"]])


def bench(
    name: str,
    fn: Callable[[List["KeyedTensor"], List[List[str]]], List[torch.Tensor]],
    n_dense: int,
    n_sparse: int,
) -> None:
    input_data = prepare_benchmark(n_dense, n_sparse)
    start = time.perf_counter()
    for _ in range(3):
        fn(input_data[0], input_data[1])
    end = time.perf_counter()
    print(f"warmup time {(end-start)*1000:.1f}ms")
    results = timeit.repeat(
        lambda: fn(input_data[0], input_data[1]), number=10, repeat=10
    )
    print(f"{name} {np.median(results)*1000:.1f}us")


@click.command()
@click.option(
    "--n_dense",
    type=int,
    default=2000,
    help="Total number of dense embeddings to be used.",
)
@click.option(
    "--n_sparse",
    default=3000,
    help="Total number of sparse embeddings to be used.",
)
def main(n_dense: int, n_sparse: int) -> None:
    bench("regular ", _regroup_keyed_tensors, n_dense, n_sparse)


if __name__ == "__main__":
    main()
