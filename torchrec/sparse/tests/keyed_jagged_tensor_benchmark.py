#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import random
import time
import timeit
from typing import Any, Callable, Dict, List, Optional, Tuple

import click
import torch
from torchrec.distributed.benchmark.benchmark_utils import BenchmarkResult
from torchrec.distributed.dist_data import _get_recat

from torchrec.distributed.test_utils.test_model import ModelInput
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor


def generate_kjt(
    tables: List[EmbeddingBagConfig],
    batch_size: int,
    mean_pooling_factor: int,
    device: torch.device,
) -> KeyedJaggedTensor:
    global_input = ModelInput.generate(
        batch_size=batch_size,
        world_size=1,  # 1 for cpu
        num_float_features=0,
        tables=tables,
        weighted_tables=[],
        # mean pooling factor per feature
        tables_pooling=[mean_pooling_factor] * len(tables),
        # returns KJTs with values all set to 0
        # we don't care about KJT values for benchmark, and this saves time
        randomize_indices=False,
        device=device,
    )[0]
    return global_input.idlist_features


def build_kjts(
    tables: List[EmbeddingBagConfig],
    batch_size: int,
    mean_pooling_factor: int,
    device: torch.device,
) -> KeyedJaggedTensor:
    start = time.perf_counter()
    print("Starting to build KJTs")

    kjt = generate_kjt(
        tables,
        batch_size,
        mean_pooling_factor,
        device,
    )

    end = time.perf_counter()
    time_taken_s = end - start
    print(f"Took {time_taken_s * 1000:.1f}ms to build KJT\n")
    return kjt


def wrapped_func(
    kjt: KeyedJaggedTensor,
    test_func: Callable[[KeyedJaggedTensor], object],
    fn_kwargs: Dict[str, Any],
    jit_script: bool,
) -> Callable[..., object]:
    def fn() -> object:
        return test_func(kjt, **fn_kwargs)

    return fn if jit_script else torch.jit.script(fn)


def benchmark_kjt(
    test_name: str,
    test_func: Callable[..., object],
    kjt: KeyedJaggedTensor,
    num_repeat: int,
    num_warmup: int,
    num_features: int,
    batch_size: int,
    mean_pooling_factor: int,
    fn_kwargs: Dict[str, Any],
    is_static_method: bool,
    jit_script: bool,
) -> None:

    for _ in range(num_warmup):
        test_func(**fn_kwargs)

    times = []
    for _ in range(num_repeat):
        time_elapsed = timeit.timeit(lambda: test_func(**fn_kwargs), number=1)
        # remove length_per_key and offset_per_key cache for fairer comparison
        kjt.unsync()
        times.append(time_elapsed)

    result = BenchmarkResult(
        short_name=test_name,
        elapsed_time=torch.tensor(times),
        max_mem_allocated=[0],
    )

    print(
        f"  {test_name : <{35}} | JIT Script: {'Yes' if jit_script else 'No' : <{8}} | B: {batch_size : <{8}} | F: {num_features : <{8}} | Mean Pooling Factor: {mean_pooling_factor : <{8}} | Runtime (P50): {result.runtime_percentile(50, interpolation='linear'):5f} ms | Runtime (P90): {result.runtime_percentile(90, interpolation='linear'):5f} ms"
    )


def get_k_splits(n: int, k: int) -> List[int]:
    split_size, _ = divmod(n, k)
    splits = [split_size] * (k - 1) + [n - split_size * (k - 1)]
    return splits


def gen_dist_split_input(
    tables: List[EmbeddingBagConfig],
    batch_size: int,
    num_workers: int,
    num_features: int,
    mean_pooling_factor: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[int], Optional[torch.Tensor]]:
    batch_size_per_rank = get_k_splits(n=batch_size, k=num_workers)
    kjts = [
        generate_kjt(tables, batch_size_rank, mean_pooling_factor, device)
        for batch_size_rank in batch_size_per_rank
    ]
    kjt_lengths = torch.cat([kjt.lengths() for kjt in kjts])
    kjt_values = torch.cat([kjt.values() for kjt in kjts])
    recat = _get_recat(
        local_split=num_features,
        num_splits=num_workers,
        device=device,
        batch_size_per_rank=batch_size_per_rank,
        use_tensor_compute=False,
    )

    return (kjt_lengths, kjt_values, batch_size_per_rank, recat)


@torch.jit.script
def permute(kjt: KeyedJaggedTensor, indices: List[int]) -> KeyedJaggedTensor:
    return kjt.permute(indices)


@torch.jit.script
def todict(kjt: KeyedJaggedTensor) -> Dict[str, JaggedTensor]:
    return kjt.to_dict()


@torch.jit.script
def split(kjt: KeyedJaggedTensor, segments: List[int]) -> List[KeyedJaggedTensor]:
    return kjt.split(segments)


@torch.jit.script
def getitem(kjt: KeyedJaggedTensor, key: str) -> JaggedTensor:
    return kjt[key]


@torch.jit.script
def dist_splits(kjt: KeyedJaggedTensor, key_splits: List[int]) -> List[List[int]]:
    return kjt.dist_splits(key_splits)


def bench(
    num_repeat: int,
    num_warmup: int,
    num_features: int,
    batch_size: int,
    mean_pooling_factor: int,
    num_workers: int,
) -> None:
    # TODO: support CUDA benchmark
    device: torch.device = torch.device("cpu")

    tables: List[EmbeddingBagConfig] = [
        EmbeddingBagConfig(
            num_embeddings=20,  # determines indices range
            embedding_dim=10,  # doesn't matter for benchmark
            name=f"table_{i}",
            feature_names=[f"feature_{i}"],
        )
        for i in range(num_features)
    ]

    kjt = build_kjts(
        tables,
        batch_size,
        mean_pooling_factor,
        device,
    )

    splits = get_k_splits(n=num_features, k=8)
    permute_indices = random.sample(range(num_features), k=num_features)
    key = f"feature_{random.randint(0, num_features - 1)}"

    kjt_lengths, kjt_values, strides_per_rank, recat = gen_dist_split_input(
        tables, batch_size, num_workers, num_features, mean_pooling_factor, device
    )

    # pyre-ignore[33]
    benchmarked_methods: List[Tuple[str, Dict[str, Any], bool, Callable[..., Any]]] = [
        ("permute", {"indices": permute_indices}, False, permute),
        ("to_dict", {}, False, todict),
        ("split", {"segments": splits}, False, split),
        ("__getitem__", {"key": key}, False, getitem),
        ("dist_splits", {"key_splits": splits}, False, dist_splits),
        (
            "dist_init",
            {
                "keys": kjt.keys(),
                "tensors": [
                    # lengths from each rank, should add up to num_features x batch_size in total
                    kjt_lengths,
                    # values from each rank
                    kjt_values,
                ],
                "variable_stride_per_key": False,
                "num_workers": num_workers,
                "recat": recat,
                "stride_per_rank": strides_per_rank,
            },
            True,  # is static method
            torch.jit.script(KeyedJaggedTensor.dist_init),
        ),
    ]

    for method_name, fn_kwargs, is_static_method, jit_func in benchmarked_methods:
        test_func = getattr(KeyedJaggedTensor if is_static_method else kjt, method_name)
        benchmark_kjt(
            test_name=method_name,
            test_func=test_func,
            kjt=kjt,
            num_repeat=num_repeat,
            num_warmup=num_warmup,
            num_features=num_features,
            batch_size=batch_size,
            mean_pooling_factor=mean_pooling_factor,
            fn_kwargs=fn_kwargs,
            is_static_method=is_static_method,
            jit_script=False,
        )

        if not is_static_method:
            # Explicitly pass in KJT for instance methods
            fn_kwargs = {"kjt": kjt, **fn_kwargs}

        benchmark_kjt(
            test_name=method_name,
            test_func=jit_func,
            kjt=kjt,
            num_repeat=num_repeat,
            num_warmup=num_warmup,
            num_features=num_features,
            batch_size=batch_size,
            mean_pooling_factor=mean_pooling_factor,
            fn_kwargs=fn_kwargs,
            is_static_method=is_static_method,
            jit_script=True,
        )


@click.command()
@click.option(
    "--num-repeat",
    default=30,
    help="Number of times method under test is run",
)
@click.option(
    "--num-warmup",
    default=10,
    help="Number of times method under test is run for warmup",
)
@click.option(
    "--num-features",
    default=1280,
    help="Total number of sparse features per KJT",
)
@click.option(
    "--batch-size",
    default=4096,
    help="Batch size per KJT (assumes non-VBE)",
)
@click.option(
    "--mean-pooling-factor",
    default=100,
    help="Avg pooling factor for KJT",
)
@click.option(
    "--num-workers",
    default=4,
    help="World size to simulate for dist_init",
)
def main(
    num_repeat: int,
    num_warmup: int,
    num_features: int,
    batch_size: int,
    mean_pooling_factor: int,
    num_workers: int,
) -> None:
    bench(
        num_repeat,
        num_warmup,
        num_features,
        batch_size,
        mean_pooling_factor,
        num_workers,
    )


if __name__ == "__main__":
    main()
