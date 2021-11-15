#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
import math
import random
import statistics
import time
from typing import Callable, Dict, List, Optional, Tuple

import click
import numpy as np
import torch
from fbgemm_gpu.split_table_batched_embeddings_ops import (
    BoundsCheckMode,
    CacheAlgorithm,
    ComputeDevice,
    DenseTableBatchedEmbeddingBagsCodegen,
    EmbeddingLocation,
    OptimType,
    SparseType,
    SplitTableBatchedEmbeddingBagsCodegen,
    IntNBitTableBatchedEmbeddingBagsCodegen,
)
from numpy.random import default_rng
from torch import Tensor

logging.basicConfig(level=logging.DEBUG)

PRECISION_SIZE_MULTIPLIER: Dict[SparseType, float] = {
    SparseType.FP32: 4,
    SparseType.FP16: 2,
    SparseType.INT8: 1,
    SparseType.INT4: 0.5,
}


def round_up(a: int, b: int) -> int:
    return int((a + b - 1) // b) * b


def get_device() -> torch.device:
    return (
        torch.cuda.current_device()
        if torch.cuda.is_available()
        else torch.device("cpu")
    )


# Merged indices with shape (T, B, L) -> (flattened indices with shape
# (T * B * L), offsets with shape (T * B + 1))
def get_table_batched_offsets_from_dense(
    merged_indices: Tensor,
) -> Tuple[Tensor, Tensor]:
    (T, B, L) = merged_indices.size()
    lengths = np.ones((T, B)) * L
    flat_lengths = lengths.flatten()
    return (
        merged_indices.long().contiguous().view(-1).to(get_device()),
        torch.tensor(([0] + np.cumsum(flat_lengths).tolist())).long().to(get_device()),
    )


def generate_requests(
    iters: int,
    B: int,
    T: int,
    L: int,
    E: int,
    # inter-batch indices reuse rate
    reuse: float = 0.0,
    # alpha <= 1.0: use uniform distribution
    # alpha > 1.0: use zipf distribution
    alpha: float = 1.0,
    weights_precision: SparseType = SparseType.FP32,
    weighted: bool = False,
) -> List[Tuple[torch.IntTensor, torch.IntTensor, Optional[Tensor]]]:
    if alpha <= 1.0:
        all_indices = torch.randint(
            low=0,
            high=E,
            size=(iters, T, B, L),
            device=get_device(),
            dtype=torch.int32,
        )
        # each bag is usually sorted
        (all_indices, _) = torch.sort(all_indices)
        all_indices = all_indices.reshape(iters, T, B * L)
    else:
        assert E >= L, "num-embeddings must be greater than equal to bag-size"
        # oversample and then remove duplicates to obtain sampling without
        # replacement
        all_indices = (np.random.zipf(a=alpha, size=(iters, T, B, 3 * L)) - 1) % E
        for index_tuple in itertools.product(range(iters), range(T), range(B)):
            # sample without replacement from
            # https://stats.stackexchange.com/questions/20590/how-do-i-sample-without-replacement-using-a-sampling-with-replacement-function
            r = set()
            for x in all_indices[index_tuple]:
                if x not in r:
                    r.add(x)
                    if len(r) == L:
                        break
            assert (len(r)) == L, "too skewed distribution (alpha too big)"
            all_indices[index_tuple][:L] = list(r)
        # shuffle indices so we don't have unintended spatial locality
        all_indices = torch.as_tensor(all_indices[:, :, :, :L])
        rng = default_rng()
        permutation = torch.as_tensor(
            rng.choice(E, size=all_indices.max().item() + 1, replace=False)
        )
        all_indices = permutation.gather(0, all_indices.flatten())
        all_indices = all_indices.to(get_device()).int().reshape(iters, T, B * L)
    for it in range(iters - 1):
        for t in range(T):
            reused_indices = torch.randperm(B * L, device=get_device())[
                : int(B * L * reuse)
            ]
            all_indices[it + 1, t, reused_indices] = all_indices[it, t, reused_indices]

    rs = []
    for it in range(iters):
        weights_tensor = (
            None if not weighted else torch.randn(T * B * L, device=get_device())
        )
        rs.append(
            get_table_batched_offsets_from_dense(all_indices[it].view(T, B, L))
            + (weights_tensor,)
        )
    return rs


def benchmark_requests(
    requests: List[Tuple[torch.IntTensor, torch.IntTensor, Optional[Tensor]]],
    func: Callable[[Tensor, Tensor, Optional[Tensor]], Tensor],
    flush_gpu_cache_size_mb: int = 0,
    check_median: bool = False,
) -> float:
    times = []
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
    for (indices, offsets, weights) in requests:
        start_time = time.time()
        if torch.cuda.is_available():
            if flush_gpu_cache_size_mb:
                _ = torch.rand(
                    flush_gpu_cache_size_mb * 1024 * 1024 // 4, dtype=torch.float
                )
                torch.cuda.synchronize()
            start_event.record()
        func(indices, offsets, weights)
        if torch.cuda.is_available():
            end_event.record()
            torch.cuda.synchronize()
            it_time = start_event.elapsed_time(end_event) * 1.0e-3
            times.append(it_time)
        else:
            it_time = time.time() - start_time
            times.append(it_time)
    avg_time = sum(times) / len(requests)
    median_time = statistics.median(times)
    return median_time if check_median else avg_time


def benchmark_pipelined_requests(
    requests: List[Tuple[torch.IntTensor, torch.IntTensor, Optional[Tensor]]],
    func1: Callable[[Tensor, Tensor, Optional[Tensor]], None],
    func2: Callable[[Tensor, Tensor, Optional[Tensor]], None],
    flush_gpu_cache_size_mb: int = 0,
) -> Tuple[float, float]:
    torch.cuda.synchronize()
    start_events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in requests
    ]
    end_events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in requests
    ]
    for ((indices, offsets, indices_weights), start_event, end_event) in zip(
        requests, start_events, end_events
    ):
        if flush_gpu_cache_size_mb:
            _ = torch.rand(
                flush_gpu_cache_size_mb * 1024 * 1024 // 4, dtype=torch.float
            )
            torch.cuda.synchronize()
        start_event[0].record()
        func1(indices, offsets, indices_weights)
        end_event[0].record()
        start_event[1].record()
        func2(indices, offsets, indices_weights)
        end_event[1].record()
    torch.cuda.synchronize()
    return (
        sum(
            start_event[0].elapsed_time(end_event[0]) * 1.0e-3
            for start_event, end_event in zip(start_events, end_events)
        )
        / len(requests),
        sum(
            start_event[1].elapsed_time(end_event[1]) * 1.0e-3
            for start_event, end_event in zip(start_events, end_events)
        )
        / len(requests),
    )


@click.group()
def cli() -> None:
    pass


@cli.command()
# recommended value: alpha=1.15 for training and alpha=1.09 for inference
@click.option("--alpha", default=1.0)
@click.option("--bag-size", default=20)
@click.option("--batch-size", default=512)
@click.option("--embedding-dim", default=128)
@click.option("--weights-precision", type=SparseType, default=SparseType.FP32)
@click.option("--stoc", is_flag=True, default=False)
@click.option("--iters", default=100)
@click.option("--managed", default="device")
@click.option("--mixed", is_flag=True, default=False)
@click.option("--num-embeddings", default=int(1e5))
@click.option("--num-tables", default=32)
@click.option("--reuse", default=0.0)
@click.option("--row-wise/--no-row-wise", default=True)
@click.option("--weighted", is_flag=True, default=False)
@click.option("--weighted-num-requires-grad", type=int, default=None)
@click.option("--flush-gpu-cache-size-mb", default=0)
@click.option("--dense", is_flag=True, default=False)
@click.option("--pooled-embedding-precision", type=SparseType, default=SparseType.FP32)
def device(  # noqa C901
    alpha: float,
    bag_size: int,
    batch_size: int,
    embedding_dim: int,
    weights_precision: SparseType,
    stoc: bool,
    iters: int,
    managed: str,
    mixed: bool,
    num_embeddings: int,
    num_tables: int,
    reuse: float,
    row_wise: bool,
    weighted: bool,
    weighted_num_requires_grad: Optional[int],
    flush_gpu_cache_size_mb: int,
    dense: bool,
    pooled_embedding_precision: SparseType,
) -> None:
    np.random.seed(42)
    torch.manual_seed(42)
    B = batch_size
    D = embedding_dim
    L = bag_size
    E = num_embeddings
    T = num_tables
    if weighted_num_requires_grad:
        assert weighted_num_requires_grad <= T
        weighted_requires_grad_tables = np.random.choice(
            T, replace=False, size=(weighted_num_requires_grad,)
        ).tolist()
        feature_requires_grad = (
            torch.tensor(
                [1 if t in weighted_requires_grad_tables else 0 for t in range(T)]
            )
            .to(get_device())
            .int()
        )
    else:
        feature_requires_grad = None
    if mixed:
        Ds = [
            round_up(np.random.randint(low=int(0.5 * D), high=int(1.5 * D)), 4)
            for _ in range(T)
        ]
        D = np.average(Ds)
    else:
        Ds = [D] * T
    optimizer = OptimType.EXACT_ROWWISE_ADAGRAD if row_wise else OptimType.EXACT_ADAGRAD

    if managed == "device":
        managed_option = (
            EmbeddingLocation.DEVICE
            if torch.cuda.is_available()
            else EmbeddingLocation.HOST
        )
    else:
        managed_option = EmbeddingLocation.MANAGED

    if dense:
        emb = DenseTableBatchedEmbeddingBagsCodegen(
            [
                (
                    E,
                    d,
                )
                for d in Ds
            ],
            use_cpu=not torch.cuda.is_available(),
        )
    else:
        emb = SplitTableBatchedEmbeddingBagsCodegen(
            [
                (
                    E,
                    d,
                    managed_option,
                    ComputeDevice.CUDA
                    if torch.cuda.is_available()
                    else ComputeDevice.CPU,
                )
                for d in Ds
            ],
            optimizer=optimizer,
            learning_rate=0.1,
            eps=0.1,
            weights_precision=weights_precision,
            stochastic_rounding=stoc,
            pooled_output_precision=pooled_embedding_precision,
        )
    emb = emb.to(get_device())

    if weights_precision == SparseType.INT8:
        emb.init_embedding_weights_uniform(-0.0003, 0.0003)

    nparams = sum(w.numel() for w in emb.split_embedding_weights())

    param_size_multiplier = PRECISION_SIZE_MULTIPLIER[weights_precision]

    logging.info(
        f"Embedding parameters: {nparams / 1.0e9: .2f} GParam, "
        f"{nparams * param_size_multiplier / 1.0e9: .2f}GB"
    )
    logging.info(
        f"Accessed weights per batch: {B * sum(Ds) * L * param_size_multiplier / 1.0e6: .2f}MB"
    )

    requests = generate_requests(
        iters,
        B,
        T,
        L,
        E,
        reuse=reuse,
        alpha=alpha,
        weights_precision=weights_precision,
        weighted=weighted,
    )

    # forward
    time_per_iter = benchmark_requests(
        requests,
        lambda indices, offsets, per_sample_weights: emb.forward(
            indices.long(),
            offsets.long(),
            per_sample_weights,
            feature_requires_grad=feature_requires_grad,
        ),
        flush_gpu_cache_size_mb=flush_gpu_cache_size_mb,
    )
    logging.info(
        f"Forward, B: {B}, "
        f"E: {E}, T: {T}, D: {D}, L: {L}, W: {weighted}, "
        f"BW: {param_size_multiplier * B * sum(Ds) * L / time_per_iter / 1.0e9: .2f}GB/s, "  # noqa: B950
        f"T: {time_per_iter * 1.0e6:.0f}us"
    )

    if pooled_embedding_precision == SparseType.INT8:
        # backward bench not representative
        return

    grad_output = torch.randn(B, sum(Ds)).to(get_device())
    # backward
    time_per_iter = benchmark_requests(
        requests,
        lambda indices, offsets, per_sample_weights: emb(
            indices.long(),
            offsets.long(),
            per_sample_weights,
            feature_requires_grad=feature_requires_grad,
        ).backward(grad_output),
        flush_gpu_cache_size_mb=flush_gpu_cache_size_mb,
    )
    logging.info(
        f"ForwardBackward, B: {B}, E: {E}, T: {T}, D: {D}, L: {L}, "
        f"BW: {3 * param_size_multiplier * B * sum(Ds) * L / time_per_iter / 1.0e9: .2f}GB/s, "
        f"T: {time_per_iter * 1.0e6:.0f}us"
    )


@cli.command()
@click.option("--alpha", default=1.0)
@click.option("--bag-size", default=20)
@click.option("--batch-size", default=512)
@click.option("--embedding-dim", default=128)
@click.option("--weights-precision", type=SparseType, default=SparseType.FP32)
@click.option("--stoc", is_flag=True, default=False)
@click.option("--iters", default=100)
@click.option("--mixed", is_flag=True, default=False)
@click.option("--num-embeddings", default=int(1e5))
@click.option("--num-tables", default=32)
@click.option("--reuse", default=0.0)
@click.option("--uvm-tables", default=1)
@click.option("--uvm-bag-size", default=1)
@click.option("--weighted", is_flag=True, default=False)
@click.option("--flush-gpu-cache-size-mb", default=0)
def uvm(
    alpha: bool,
    bag_size: int,
    batch_size: int,
    embedding_dim: int,
    weights_precision: SparseType,
    stoc: bool,
    iters: int,
    mixed: bool,
    num_embeddings: int,
    num_tables: int,
    reuse: float,
    uvm_tables: int,
    uvm_bag_size: int,
    weighted: bool,
    flush_gpu_cache_size_mb: int,
) -> None:
    np.random.seed(42)
    torch.manual_seed(42)
    B = batch_size
    D = embedding_dim
    L = bag_size
    E = num_embeddings
    T = num_tables
    T_uvm = uvm_tables
    assert T_uvm <= T
    assert (
        T_uvm > 0
    ), f"T_uvm specified {T_uvm} <= 0. If not testing UVM, please use device benchmark."
    T_gpu = T - T_uvm
    L_uvm = uvm_bag_size

    if mixed:
        Ds = [
            round_up(np.random.randint(low=int(0.5 * D), high=int(1.5 * D)), 4)
            for _ in range(T)
        ]
        D = np.average(Ds)
    else:
        Ds = [D] * T
    emb_uvm = SplitTableBatchedEmbeddingBagsCodegen(
        [
            (
                E,
                d,
                EmbeddingLocation.MANAGED,
                ComputeDevice.CUDA,
            )
            for d in Ds[:T_uvm]
        ],
        weights_precision=weights_precision,
        stochastic_rounding=stoc,
    ).cuda()

    if weights_precision == SparseType.INT8:
        emb_uvm.init_embedding_weights_uniform(-0.0003, 0.0003)

    if T_gpu > 0:
        emb_gpu = SplitTableBatchedEmbeddingBagsCodegen(
            [
                (
                    E,
                    d,
                    EmbeddingLocation.DEVICE,
                    ComputeDevice.CUDA,
                )
                for d in Ds[T_uvm:]
            ],
            weights_precision=weights_precision,
            stochastic_rounding=stoc,
        ).cuda()

        if weights_precision == SparseType.INT8:
            emb_gpu.init_embedding_weights_uniform(-0.0003, 0.0003)

        emb_mixed = SplitTableBatchedEmbeddingBagsCodegen(
            [
                (
                    E,
                    d,
                    managed_option,
                    ComputeDevice.CUDA,
                )
                for (d, managed_option) in zip(
                    Ds,
                    [EmbeddingLocation.MANAGED] * T_uvm
                    + [EmbeddingLocation.DEVICE] * T_gpu,
                )
            ],
            weights_precision=weights_precision,
            stochastic_rounding=stoc,
        ).cuda()

        if weights_precision == SparseType.INT8:
            emb_mixed.init_embedding_weights_uniform(-0.0003, 0.0003)

    requests_uvm = generate_requests(
        iters,
        B,
        T_uvm,
        L_uvm,
        E,
        reuse=reuse,
        alpha=alpha,
        weights_precision=weights_precision,
        weighted=weighted,
    )

    requests_gpu = None
    if T_gpu > 0:
        requests_gpu = generate_requests(
            iters,
            B,
            T_gpu,
            L,
            E,
            reuse=reuse,
            alpha=alpha,
            weights_precision=weights_precision,
            weighted=False,
        )

    param_size_multiplier = PRECISION_SIZE_MULTIPLIER[weights_precision]

    time_per_iter = benchmark_requests(
        requests_uvm,
        lambda indices, offsets, per_sample_weights: emb_uvm.forward(
            indices.long(),
            offsets.long(),
            per_sample_weights,
        ),
        flush_gpu_cache_size_mb=flush_gpu_cache_size_mb,
    )
    logging.info(
        f"UVM Forward, B: {B}, "
        f"E: {E}, T: {T_uvm}, D: {D}, L: {L_uvm}, W: {weighted}, "
        f"BW: {param_size_multiplier * B * sum(Ds[:T_uvm]) * L_uvm / time_per_iter / 1.0e9: .2f}GB/s, "  # noqa: B950
        f"T: {time_per_iter * 1.0e6:.0f}us"
    )

    if T_gpu > 0:
        requests = []
        assert requests_gpu is not None
        for rs_uvm, rs_gpu in zip(requests_uvm, requests_gpu):
            indices = torch.cat([rs_uvm[0], rs_gpu[0]])
            lengths = [L_uvm] * (T_uvm * B) + [L] * (T_gpu * B)
            offsets = torch.tensor(([0] + np.cumsum(lengths).tolist())).int().cuda()
            per_sample_weights = None
            if weighted:
                assert (this_rs_uvm_weights := rs_uvm[2]) is not None
                assert (this_rs_gpu_weights := rs_gpu[2]) is not None
                per_sample_weights = torch.cat(
                    [this_rs_uvm_weights, this_rs_gpu_weights]
                )
            requests.append((indices, offsets, per_sample_weights))

        # forward
        time_per_iter = benchmark_requests(
            requests_gpu,
            lambda indices, offsets, per_sample_weights: emb_gpu.forward(
                indices.long(),
                offsets.long(),
                per_sample_weights,
            ),
            flush_gpu_cache_size_mb=flush_gpu_cache_size_mb,
        )

        logging.info(
            f"GPU Forward, B: {B}, "
            f"E: {E}, T: {T_gpu}, D: {D}, L: {L}, W: {weighted}, "
            f"BW: {param_size_multiplier * B * sum(Ds[T_uvm:]) * L / time_per_iter / 1.0e9: .2f}GB/s, "  # noqa: B950
            f"T: {time_per_iter * 1.0e6:.0f}us"
        )

        time_per_iter = benchmark_requests(
            requests,
            lambda indices, offsets, per_sample_weights: emb_mixed.forward(
                indices.long(),
                offsets.long(),
                per_sample_weights,
            ),
            flush_gpu_cache_size_mb=flush_gpu_cache_size_mb,
        )
        logging.info(
            f"Mixed Forward, B: {B}, "
            f"E: {E}, T_GPU: {T_gpu}, T_UVM: {T_uvm}, D: {D}, L_GPU: {L}, L_UVM: {L_uvm}, W: {weighted}, "
            f"BW: {((param_size_multiplier * B)*((sum(Ds[:T_uvm]) * L_uvm) + sum(Ds[T_uvm:]) * L)) / time_per_iter / 1.0e9: .2f}GB/s, "  # noqa: B950
            f"T: {time_per_iter * 1.0e6:.0f}us"
        )


@cli.command()
@click.option("--alpha", default=1.0)
@click.option("--bag-size", default=20)
@click.option("--batch-size", default=512)
@click.option("--cache-algorithm", default="lru")
@click.option("--cache-sets", default=1024)
@click.option("--embedding-dim", default=128)
@click.option("--weights-precision", type=SparseType, default=SparseType.FP32)
@click.option("--stoc", is_flag=True, default=False)
@click.option("--long-index", is_flag=True, default=False)
@click.option("--iters", default=100)
@click.option("--mixed", is_flag=True, default=False)
@click.option("--num-embeddings", default=int(1e5))
@click.option("--num-tables", default=32)
@click.option("--reuse", default=0.1)
@click.option("--weighted", is_flag=True, default=False)
@click.option("--flush-gpu-cache-size-mb", default=0)
def cache(  # noqa C901
    alpha: float,
    bag_size: int,
    batch_size: int,
    cache_algorithm: str,
    cache_sets: int,
    embedding_dim: int,
    weights_precision: SparseType,
    stoc: bool,
    iters: int,
    long_index: bool,
    mixed: bool,
    num_embeddings: int,
    num_tables: int,
    reuse: float,
    weighted: bool,
    flush_gpu_cache_size_mb: int,
) -> None:
    np.random.seed(42)
    torch.manual_seed(42)
    optimizer = OptimType.EXACT_ROWWISE_ADAGRAD
    B = batch_size
    D = embedding_dim
    L = bag_size
    E = num_embeddings
    T = num_tables
    cache_alg = CacheAlgorithm.LRU if cache_algorithm == "lru" else CacheAlgorithm.LFU
    if mixed:
        Ds = [
            round_up(np.random.randint(low=int(0.5 * D), high=int(1.5 * D)), 4)
            for _ in range(T)
        ]
        D = np.average(Ds)
    else:
        Ds = [D] * T

    emb_nc = SplitTableBatchedEmbeddingBagsCodegen(
        [
            (
                E,
                d,
                EmbeddingLocation.MANAGED,
                ComputeDevice.CUDA,
            )
            for d in Ds
        ],
        optimizer=optimizer,
        weights_precision=weights_precision,
        stochastic_rounding=stoc,
    ).cuda()

    if weights_precision == SparseType.INT8:
        emb_nc.init_embedding_weights_uniform(-0.0003, 0.0003)

    emb = SplitTableBatchedEmbeddingBagsCodegen(
        [
            (
                E,
                d,
                EmbeddingLocation.MANAGED_CACHING,
                ComputeDevice.CUDA,
            )
            for d in Ds
        ],
        optimizer=optimizer,
        weights_precision=weights_precision,
        stochastic_rounding=stoc,
        cache_sets=cache_sets,
        cache_algorithm=cache_alg,
    ).cuda()

    if weights_precision == SparseType.INT8:
        emb.init_embedding_weights_uniform(-0.0003, 0.0003)

    nparams = sum(w.numel() for w in emb.split_embedding_weights())
    param_size_multiplier = PRECISION_SIZE_MULTIPLIER[weights_precision]
    logging.info(
        f"Embedding tables: {E * T} rows, {nparams / 1.0e9: .2f} GParam, "
        f"{nparams * param_size_multiplier  / 1.0e6: .2f}MB"
    )
    logging.info(
        f"Accessed weights per batch: {B * T * L} rows, "
        f"{B * T * L * D * param_size_multiplier / 1.0e6: .2f}MB"
    )

    requests = generate_requests(
        2 * iters, B, T, L, E, reuse=reuse, alpha=alpha, weighted=weighted
    )
    warmup_requests, requests = requests[:iters], requests[iters:]
    grad_output = torch.randn(B, sum(Ds)).cuda()

    time_per_iter = benchmark_requests(
        requests,
        lambda indices, offsets, per_sample_weights: emb_nc(
            indices.long(), offsets.long(), per_sample_weights
        ).backward(grad_output),
        flush_gpu_cache_size_mb=flush_gpu_cache_size_mb,
    )
    logging.info(
        f"ForwardBackward (UVM), B: {B}, E: {E}, T: {T}, D: {D}, L: {L}, "
        f"BW: {3 * param_size_multiplier * B * sum(Ds) * L / time_per_iter / 1.0e9: .2f}GB/s, "
        f"T: {time_per_iter * 1.0e6:.0f}us"
    )

    # warm up
    for indices, offsets, _ in warmup_requests:
        emb.forward(indices.long(), offsets.long())
    # get cache miss rate (forward and backward) and exchanged cache lines (prefetch)
    cache_misses = []
    exchanged_cache_lines = []
    NOT_FOUND = -1
    for indices, offsets, _ in requests:
        # pyre-fixme[29]:
        #  `Union[BoundMethod[typing.Callable(Tensor.clone)[[Named(self,
        #  Variable[torch._TTensor (bound to Tensor)])], Variable[torch._TTensor (bound
        #  to Tensor)]], Tensor], Tensor, torch.nn.Module]` is not a function.
        old_lxu_cache_state = emb.lxu_cache_state.clone()
        emb.prefetch(indices.long(), offsets.long())
        exchanged_cache_lines.append(
            # pyre-fixme[16]: `bool` has no attribute `sum`.
            (emb.lxu_cache_state != old_lxu_cache_state)
            .sum()
            .item()
        )
        cache_misses.append((emb.lxu_cache_locations_list[0] == NOT_FOUND).sum().item())
        emb.forward(indices.long(), offsets.long())
    logging.info(
        f"Exchanged cache lines -- mean: {sum(exchanged_cache_lines)/len(requests): .2f}, "
        f"max: {max(exchanged_cache_lines)}, min: {min(exchanged_cache_lines)}"
    )
    logging.info(
        f"Cache miss -- mean: {sum(cache_misses)/len(requests)}, "
        f"max: {max(cache_misses)}, min: {min(cache_misses)}"
    )

    # benchmark prefetch
    emb.reset_cache_states()
    for indices, offsets, _ in warmup_requests:
        emb.forward(indices, offsets)
    prefetch_time, forward_backward_time = benchmark_pipelined_requests(
        requests,
        lambda indices, offsets, indices_weights: emb.prefetch(indices, offsets),
        lambda indices, offsets, indices_weights: emb.forward(
            indices, offsets, indices_weights
        ).backward(grad_output),
        flush_gpu_cache_size_mb=flush_gpu_cache_size_mb,
    )
    e2e_time = prefetch_time + forward_backward_time

    logging.info(
        f"ForwardBackward (LXU), reuse: {reuse}, alpha: {alpha}, B: {B}, "
        f"E: {E}, T: {T}, D: {D}, L: {L}, "
        f"BW: {3 * param_size_multiplier * B * sum(Ds) * L / e2e_time / 1.0e9: .2f}GB/s, "
        f"Tprefetch: {prefetch_time * 1.0e6:.0f}us, "
        f"{2 * sum(exchanged_cache_lines) * param_size_multiplier * D / prefetch_time / len(requests) / 1.0e9: .2f} GB/s, "
        f"Tfwdbwd: {forward_backward_time * 1.0e6:.0f}us, "
        f"{3 * param_size_multiplier * B * sum(Ds) * L / forward_backward_time / 1.0e9: .2f} GB/s, "
        f"Te2e: {e2e_time * 1.0e6:.0f}us, "
    )


def benchmark_cpu_requests(
    requests: List[Tuple[torch.IntTensor, torch.IntTensor, Optional[torch.Tensor]]],
    func: Callable[[Tensor, Tensor, Optional[Tensor]], Tensor],
) -> float:
    import time

    start_time = time.perf_counter()
    for (indices, offsets, weights) in requests:
        func(indices, offsets, weights)
    end_time = time.perf_counter()
    return (end_time - start_time) / len(requests)


@cli.command()
@click.option("--alpha", default=1.0)
@click.option("--bag-size", default=20)
@click.option("--batch-size", default=512)
@click.option("--embedding-dim", default=128)
@click.option("--weights-precision", type=SparseType, default=SparseType.INT4)
@click.option("--stoc", is_flag=True, default=False)
@click.option("--iters", default=100)
@click.option("--managed", default="device")
@click.option("--mixed", is_flag=True, default=False)
@click.option("--num-embeddings", default=int(1e5))
@click.option("--num-tables", default=32)
@click.option("--reuse", default=0.0)
@click.option("--row-wise/--no-row-wise", default=True)
@click.option("--weighted", is_flag=True, default=False)
@click.option("--int4", is_flag=True, default=False)
@click.option("--index-remapping", is_flag=True, default=False)
def cpu(  # noqa C901
    alpha: float,
    bag_size: int,
    batch_size: int,
    embedding_dim: int,
    weights_precision: SparseType,
    stoc: bool,
    iters: int,
    managed: str,
    mixed: bool,
    num_embeddings: int,
    num_tables: int,
    reuse: float,
    row_wise: bool,
    weighted: bool,
    int4: bool,
    index_remapping: bool,
) -> None:
    np.random.seed(42)
    torch.manual_seed(42)
    B = batch_size
    D = embedding_dim
    L = bag_size
    E = num_embeddings
    T = num_tables
    if mixed:
        Ds = [
            # int4 table batched emb op can only handle mixed D where D is multiple of 8
            round_up(np.random.randint(low=int(0.5 * D), high=int(1.5 * D)), 8)
            for _ in range(T)
        ]
        D = np.average(Ds)
    else:
        Ds = [D] * T

    emb = IntNBitTableBatchedEmbeddingBagsCodegen(
        [("", E, d, weights_precision, EmbeddingLocation.HOST) for d in Ds],
        device="cpu",
        index_remapping=[torch.arange(E) for _ in Ds] if index_remapping else None,
    ).cpu()
    emb.fill_random_weights()

    nparams = sum(w.numel() for (w, _) in emb.split_embedding_weights())
    logging.info(
        f"Int4 Embedding parameters: {nparams * 2 / 1.0e9: .2f} GParam, "
        f"{nparams / 1.0e9: .2f}GB"
    )
    logging.info(f"Accessed weights per batch: {B * T * L * D * 0.5 / 1.0e6: .2f}MB")

    requests = generate_requests(
        iters,
        B,
        T,
        L,
        E,
        reuse=reuse,
        alpha=alpha,
        weights_precision=weights_precision,
        weighted=weighted,
    )
    requests = [
        (a.cpu().int(), b.cpu().int(), c.cpu() if c else None) for (a, b, c) in requests
    ]

    time_per_iter = benchmark_cpu_requests(
        requests,
        lambda indices, offsets, per_sample_weights: emb.forward(
            indices,
            offsets,
            per_sample_weights,
        ),
    )

    logging.info(
        f"{weights_precision} Forward, B: {B}, "
        f"E: {E}, T: {T}, D: {D}, L: {L}, W: {weighted}, "
        f"BW: {(2 * B * T * D + PRECISION_SIZE_MULTIPLIER[weights_precision] * B * T * L * D) / time_per_iter / 1.0e9: .2f}GB/s, "  # noqa: B950
        f"T: {time_per_iter * 1.0e6:.0f}us"
    )


@cli.command()
@click.option("--alpha", default=1.0)
@click.option("--bag-size", default=20)
@click.option("--batch-size", default=512)
@click.option("--embedding-dim", default=128)
@click.option("--weights-precision", type=SparseType, default=SparseType.INT4)
@click.option("--stoc", is_flag=True, default=False)
@click.option("--managed", default="device")
@click.option("--mixed", is_flag=True, default=False)
@click.option("--num-embeddings", default=int(1e5))
@click.option("--num-tables", default=32)
@click.option("--reuse", default=0.0)
@click.option("--row-wise/--no-row-wise", default=True)
@click.option("--weighted", is_flag=True, default=False)
@click.option("--weighted-num-requires-grad", type=int, default=None)
@click.option("--bounds-check-mode", type=int, default=BoundsCheckMode.WARNING.value)
@click.option("--pruning-ratio", type=float, default=None)
@click.option("--load-factor", default=0.75)
@click.option("--use-array-for-index-remapping", is_flag=True, default=True)
@click.option("--check-median", is_flag=True, default=True)
@click.option("--iters", default=100)
@click.option("--runs-of-iters", default=5)
@click.option("--warmup-runs", default=2)
@click.option("--output-dtype", type=SparseType, default=SparseType.FP16)
def nbit_device(  # noqa C901
    alpha: float,
    bag_size: int,
    batch_size: int,
    embedding_dim: int,
    weights_precision: SparseType,
    stoc: bool,
    managed: str,
    mixed: bool,
    num_embeddings: int,
    num_tables: int,
    reuse: float,
    row_wise: bool,
    weighted: bool,
    weighted_num_requires_grad: Optional[int],
    bounds_check_mode: int,
    pruning_ratio: Optional[float],
    load_factor: float,
    use_array_for_index_remapping: bool,
    check_median: bool,
    iters: int,
    runs_of_iters: int,
    warmup_runs: int,
    output_dtype: SparseType,
) -> None:
    np.random.seed(42)
    torch.manual_seed(42)
    B = batch_size
    D = embedding_dim
    L = bag_size
    E = num_embeddings
    original_E = E
    T = num_tables
    index_remapping = None
    if mixed:
        # int4 table batched emb op can only handle mixed D where D is multiple of 8
        Ds = [
            round_up(np.random.randint(low=int(0.5 * D), high=int(1.5 * D)), 8)
            for _ in range(T)
        ]
        D = np.average(Ds)
    else:
        Ds = [D] * T

    mem_for_pruning = 0
    if pruning_ratio:
        assert pruning_ratio < 1 and pruning_ratio >= 0
        E = math.ceil(E * (1.0 - pruning_ratio))
        index_remapping = []
        for _ in range(T):
            mapping = torch.tensor([-1] * original_E, dtype=torch.int32)
            selected_indices = random.sample(range(original_E), E)
            for i, idx in enumerate(selected_indices):
                mapping[idx] = i
            index_remapping.append(mapping)
            if use_array_for_index_remapping:
                mem_for_pruning += mapping.numel() * 4
            else:
                mem_for_pruning += E / load_factor * 2 * 4

    if managed == "device":
        managed_option = EmbeddingLocation.DEVICE
    else:
        managed_option = EmbeddingLocation.MANAGED

    emb = IntNBitTableBatchedEmbeddingBagsCodegen(
        [("", E, d, weights_precision, managed_option) for d in Ds],
        bounds_check_mode=BoundsCheckMode(bounds_check_mode),
        index_remapping=index_remapping,
        load_factor=load_factor,
        use_array_for_index_remapping=use_array_for_index_remapping,
        output_dtype=output_dtype,
    ).cuda()
    emb.fill_random_weights()

    nparams = sum(w.numel() for (w, _) in emb.split_embedding_weights())
    logging.info(
        f"{weights_precision} Embedding parameters: {nparams * 2 / 1.0e9: .2f} GParam, "
        f"{nparams / 1.0e9: .2f}GB"
    )
    logging.info(f"Accessed weights per batch: {B * T * L * D * 0.5 / 1.0e6: .2f}MB")

    times = []
    for i in range(runs_of_iters):
        requests = generate_requests(
            iters,
            B,
            T,
            L,
            E,
            reuse=reuse,
            alpha=alpha,
            weights_precision=weights_precision,
            weighted=weighted,
        )
        requests = [(a.int(), b.int(), c if c else None) for (a, b, c) in requests]

        # forward
        time_per_iter = benchmark_requests(
            requests,
            lambda indices, offsets, per_sample_weights: emb.forward(
                indices.int(),
                offsets.int(),
                per_sample_weights,
            ),
            check_median=check_median,
        )

        # free up GPU memory
        del requests

        logging.info(
            f"Iteration {i}: "
            f"{weights_precision} Forward, B: {B}, "
            f"E: {E}, T: {T}, D: {D}, L: {L}, W: {weighted}, "
            f"BW: {(2 * B * T * D + PRECISION_SIZE_MULTIPLIER[weights_precision] * B * T * L * D) / time_per_iter / 1.0e9: .2f}GB/s, "  # noqa: B950
            f"T: {time_per_iter * 1.0e6:.0f}us, "
            f"Memory Usage For Pruning: {mem_for_pruning / 1.0e6:.0f}MB"
        )

        if i >= warmup_runs:
            times.append(time_per_iter)

    time_per_iter = statistics.mean(times)
    logging.info(
        f"Average of all iterations: "
        f"{weights_precision} Forward, B: {B}, "
        f"E: {E}, T: {T}, D: {D}, L: {L}, W: {weighted}, "
        f"BW: {(2 * B * T * D + PRECISION_SIZE_MULTIPLIER[weights_precision] * B * T * L * D) / time_per_iter / 1.0e9: .2f}GB/s, "  # noqa: B950
        f"T: {time_per_iter * 1.0e6:.0f}us, "
        f"Memory Usage For Pruning: {mem_for_pruning / 1.0e6:.0f}MB"
    )


@cli.command()
@click.option("--bag-size", default=20)
@click.option("--batch-size", default=2048)
@click.option("--iters", default=10)
@click.option("--num-embeddings", default=int(1e5))
@click.option("--num-tables", default=100)
@click.option("--load-factor", default=0.75)
@click.option("--hit-rate", default=0.9)
@click.option("--use-cpu", is_flag=True, default=False)
def hashtable(  # noqa C901
    bag_size: int,
    batch_size: int,
    iters: int,
    num_embeddings: int,
    num_tables: int,
    load_factor: float,
    hit_rate: float,
    use_cpu: bool,
) -> None:
    B = batch_size
    T = num_tables
    L = bag_size
    E = num_embeddings
    np.random.seed(42)
    torch.manual_seed(42)
    if hit_rate == 1.0:
        chosen_indices = torch.cat([torch.arange(E) for _ in range(T)], dim=0).int()
    else:
        chosen_indices = (
            torch.randint(low=0, high=int(E * 1.0 / hit_rate), size=(E * T,))
            .view(-1)
            .int()
        )
    dense_indices = torch.cat([torch.arange(E) for _ in range(T)], dim=0).int()
    offsets = torch.tensor([E * t for t in range(T + 1)]).int()
    assert offsets[-1] == chosen_indices.numel()
    assert offsets.numel() == T + 1
    assert (offsets.numel() - 1) // T == 1

    capacities = [round_up(int(E / load_factor), 32) for _ in range(T)]

    hash_table = torch.zeros(
        (sum(capacities), 2),
        dtype=torch.int32,
    )
    hash_table_offsets = torch.tensor([0] + np.cumsum(capacities).tolist()).long()

    assert hash_table.numel() * 4 < 2 ** 32
    # initialize
    hash_table[:, :] = -1
    torch.ops.fb.pruned_hashmap_insert(
        chosen_indices, dense_indices, offsets, hash_table, hash_table_offsets
    )

    requests = generate_requests(
        iters,
        B,
        T,
        L,
        E,
    )

    if not use_cpu:
        hash_table = hash_table.cuda()
        hash_table_offsets = hash_table_offsets.cuda()
        requests = [(a.cuda().int(), b.cuda().int(), c) for (a, b, c) in requests]
    else:
        requests = [(a.int().cpu(), b.int().cpu(), c) for (a, b, c) in requests]

    empirical_hit_rate = np.mean(
        [
            torch.ops.fb.pruned_hashmap_lookup(
                indices, offsets, hash_table, hash_table_offsets
            )
            .ne(-1)
            .sum()
            .item()
            / indices.numel()
            for indices, offsets, _ in requests
        ]
    )

    time_per_iter = benchmark_requests(
        requests,
        lambda indices, offsets, _: torch.ops.fb.pruned_hashmap_lookup(
            indices, offsets, hash_table, hash_table_offsets
        ),
    )

    logging.info(
        f"LinearTable: B: {B}, T: {T}, L: {L}, E: {E}, QPS: {B * T * L / time_per_iter / 1.0e9:.2f}B QPS/s, "
        f"T: {time_per_iter * 1.0e6:.0f}us, load factor: {E * T / hash_table.shape[0] * 100:.1f}%, hit rate: {empirical_hit_rate * 100:.2f}%, Table size: {hash_table.numel() * 4 / 1.0e6:.0f}MB"
    )

    if use_cpu:
        ht = torch.classes.fb.PrunedMapCPU()
        ht.insert(chosen_indices, dense_indices, offsets, T)

        time_per_iter = benchmark_requests(
            requests,
            lambda indices, offsets, _: ht.lookup(indices, offsets),
        )

        logging.info(
            f"HashTable: B: {B}, T: {T}, L: {L}, E: {E}, QPS: {B * T * L / time_per_iter / 1.0e9:.2f}B QPS/s, "
            f"T: {time_per_iter * 1.0e6:.0f}us, load factor: {E * T / hash_table.shape[0] * 100:.1f}%, hit rate: {empirical_hit_rate * 100:.2f}%, Table size: {hash_table.numel() * 4 / 1.0e6:.0f}MB"
        )


@cli.command()
@click.option("--bag-size", default=20)
@click.option("--batch-size", default=2048)
@click.option("--iters", default=100)
@click.option("--num-embeddings", default=int(1e5))
@click.option("--num-tables", default=100)
@click.option("--pruning-ratio", default=0.9)
def pruned_array(  # noqa C901
    bag_size: int,
    batch_size: int,
    iters: int,
    num_embeddings: int,
    num_tables: int,
    pruning_ratio: float,
) -> None:
    B = batch_size
    T = num_tables
    L = bag_size
    E = num_embeddings
    np.random.seed(42)
    torch.manual_seed(42)
    assert pruning_ratio > 0 and pruning_ratio <= 1
    original_E = int(E / (1.0 - pruning_ratio))
    index_remappings = torch.tensor(
        [-1] * original_E * T, dtype=torch.int32, device="cuda"
    )
    index_remappings_offsets = torch.empty(T + 1, dtype=torch.int32, device="cuda")
    index_remappings_offsets[0] = 0
    dense_indicies = torch.tensor(range(E), dtype=torch.int32, device="cuda")
    for t in range(T):
        selected_indices = torch.add(
            torch.randperm(original_E, device="cuda"), t * original_E
        )[:E]
        index_remappings[selected_indices] = dense_indicies
        index_remappings_offsets[t + 1] = index_remappings_offsets[t] + original_E

    requests = generate_requests(
        iters,
        B,
        T,
        L,
        E,
    )
    requests = [(a.cuda().int(), b.cuda().int(), c) for (a, b, c) in requests]

    time_per_iter = benchmark_requests(
        requests,
        lambda indices, offsets, _: torch.ops.fb.pruned_array_lookup(
            indices,
            offsets,
            index_remappings,
            index_remappings_offsets,
        ),
    )

    logging.info(
        f"LinearTable: B: {B}, T: {T}, L: {L}, E: {E}, QPS: {B * T * L / time_per_iter / 1.0e9:.2f}B QPS/s, "
        f"T: {time_per_iter * 1.0e6:.0f}us, Pruning Ratio: {pruning_ratio * 100:.2f}%, Table size: {original_E * T * 4 / 1.0e6:.0f}MB"
    )


@cli.command()
@click.option("--bag-size", default=20)
@click.option("--batch-size", default=512)
@click.option("--iters", default=100)
@click.option("--num-embeddings", default=int(1e5))
@click.option("--num-tables", default=32)
@click.option("--bounds-check-mode", type=int, default=BoundsCheckMode.WARNING.value)
def bounds_check_indices(  # noqa C901
    bag_size: int,
    batch_size: int,
    iters: int,
    num_embeddings: int,
    num_tables: int,
    bounds_check_mode: int,
) -> None:
    np.random.seed(42)
    torch.manual_seed(42)
    B = batch_size
    L = bag_size
    E = num_embeddings
    T = num_tables

    requests = generate_requests(
        iters,
        B,
        T,
        L,
        E,
    )
    # requests = [(a.int(), b.int(), c if c else None) for (a, b, c) in requests]

    warning = torch.tensor([0]).long().cuda()
    rows_per_table = torch.tensor([E for _ in range(T)]).long().cuda()
    # forward
    time_per_iter = benchmark_requests(
        requests,
        lambda indices, offsets, _: torch.ops.fb.bounds_check_indices(
            rows_per_table,
            indices,
            offsets,
            BoundsCheckMode(bounds_check_mode),
            warning,
        ),
    )

    logging.info(
        f"Bounds Check Indices:  B: {B}, "
        f"E: {E}, T: {T}, L: {L}, "
        f"BW: {(8 * B * T * L + 8 * (B * T + 1)) / time_per_iter / 1.0e9: .2f}GB/s, "  # noqa: B950
        f"T: {time_per_iter * 1.0e6:.0f}us"
    )


if __name__ == "__main__":
    cli()
