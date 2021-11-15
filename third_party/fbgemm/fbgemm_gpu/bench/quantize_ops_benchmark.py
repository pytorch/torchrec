# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import time
from typing import Callable, Tuple

import click
import torch
from torch import Tensor

logging.basicConfig(level=logging.DEBUG)

try:
    torch.ops.load_library("fbgemm_gpu_py.so")
except Exception:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")

def benchmark_torch_function(
    func: Callable[[Tensor], Tensor],
    input: Tensor,
    flush_gpu_cache_size_mb: int,
    ) -> Tuple[float, Tensor]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        # Flush the cache
        if flush_gpu_cache_size_mb:
            _ = torch.rand(
                flush_gpu_cache_size_mb * 1024 * 1024 // 4, dtype=torch.float
            )
            torch.cuda.synchronize()
        start_event.record()
        # Benchmark code
        output = func(input)
        # Accumulate the time for iters iteration
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) * 1.0e-3
    else:
        start_time = time.time()
        output = func(input)
        elapsed_time = time.time() - start_time
    return float(elapsed_time), output

@click.command()
@click.option("--flush-gpu-cache-size-mb", default=0)
@click.option("--iters", default=100)
@click.option("--num-columns", default=512)
@click.option("--num-rows", default=512)
@click.option("--warmup-runs", default=2)
def main(
    flush_gpu_cache_size_mb: int,
    iters: int,
    num_columns: int,
    num_rows: int,
    warmup_runs: int,
) -> None:

    total_time = {
        "8bit_quant": 0.0,
        "4bit_quant": 0.0,
        "2bit_quant": 0.0,
        "8bit_dequant": 0.0,
        "4bit_dequant": 0.0,
        "2bit_dequant": 0.0,
    }

    input_data = torch.rand(num_rows, num_columns).float()
    if torch.cuda.is_available():
        input_data = input_data.cuda()
    for step in range(iters + warmup_runs):
        time, quant_data_8bit = benchmark_torch_function(
            torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized,
            input_data,
            flush_gpu_cache_size_mb,
        )
        if step >= warmup_runs:
            total_time["8bit_quant"] += time

        time, quant_data_4bit = benchmark_torch_function(
            lambda input : torch.ops.fbgemm.FloatToFusedNBitRowwiseQuantizedSBHalf(input, 4),
            input_data,
            flush_gpu_cache_size_mb,
        )
        if step >= warmup_runs:
            total_time["4bit_quant"] += time

        time, quant_data_2bit = benchmark_torch_function(
            lambda input : torch.ops.fbgemm.FloatToFusedNBitRowwiseQuantizedSBHalf(input, 2),
            input_data,
            flush_gpu_cache_size_mb,
        )
        if step >= warmup_runs:
            total_time["2bit_quant"] += time

        time, _ = benchmark_torch_function(
            torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloat,
            quant_data_8bit,
            flush_gpu_cache_size_mb,
        )
        if step >= warmup_runs:
            total_time["8bit_dequant"] += time

        time, _ = benchmark_torch_function(
            lambda input : torch.ops.fbgemm.FusedNBitRowwiseQuantizedSBHalfToFloat(input, 4),
            quant_data_4bit,
            flush_gpu_cache_size_mb,
        )
        if step >= warmup_runs:
            total_time["4bit_dequant"] += time

        time, _ = benchmark_torch_function(
            lambda input : torch.ops.fbgemm.FusedNBitRowwiseQuantizedSBHalfToFloat(input, 2),
            quant_data_2bit,
            flush_gpu_cache_size_mb,
        )
        if step >= warmup_runs:
            total_time["2bit_dequant"] += time

    logging.info(
        f"-------------- ncols={num_columns}, nrows={num_rows}-------------"
    )
    for k, t_time in total_time.items():
        logging.info(
            f"{k} time per iter: {t_time / iters * 1.0e6:.0f}us"
        )

if __name__ == "__main__":
    main()
