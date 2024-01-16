#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

import argparse
import logging
import os
import time
from typing import List

import torch

from torchrec.distributed.benchmark.benchmark_utils import (
    benchmark_module,
    BenchmarkResult,
    CompileMode,
    get_tables,
)
from torchrec.distributed.embedding_types import EmbeddingComputeKernel, ShardingType
from torchrec.distributed.test_utils.infer_utils import TestQuantEBCSharder
from torchrec.quant.embedding_modules import (
    EmbeddingBagCollection as QuantEmbeddingBagCollection,
)

logger: logging.Logger = logging.getLogger()


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--warmup_iters", type=int, default=20)
    parser.add_argument("--bench_iters", type=int, default=2000)
    parser.add_argument("--prof_iters", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="/var/tmp/torchrec-bench")
    parser.add_argument("--num_benchmarks", type=int, default=9)

    return parser


def write_report(
    benchmark_results: List[BenchmarkResult],
    report_file: str,
    report_str: str,
    num_requests: int,
) -> None:

    for benchmark_res in benchmark_results:
        avg_dur_s = benchmark_res.elapsed_time.mean().item() * 1e-3  # time in seconds
        std_dur_s = benchmark_res.elapsed_time.std().item() * 1e-3  # time in seconds

        qps = int(num_requests / avg_dur_s)

        mem_allocated_by_rank = benchmark_res.max_mem_allocated

        mem_str = ""
        for i, mem_mb in enumerate(mem_allocated_by_rank):
            mem_str += f"Rank {i}: {mem_mb:7} "

        report_str += f"{benchmark_res.short_name:40} Avg QPS:{qps:10} Avg Duration: {int(1000*avg_dur_s):5}"
        report_str += f"ms Standard Dev Duration: {(1000*std_dur_s):.2f}ms\n"
        report_str += f"\tMemory Allocated Per Rank:\n\t{mem_str}\n"

    with open(report_file, "w") as f:
        f.write(report_str)

    logger.info(f"Report written to {report_file}:\n{report_str}")


def benchmark_qebc() -> None:
    parser = init_argparse()
    args = parser.parse_args()

    datetime_sfx: str = time.strftime("%Y%m%dT%H%M%S")

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        # Create output directory if not exist
        os.mkdir(output_dir)

    output_dir += f"/run_{datetime_sfx}"
    if not os.path.exists(output_dir):
        # Place all outputs under the datetime folder
        os.mkdir(output_dir)

    BENCH_SHARDING_TYPES = [
        ShardingType.TABLE_WISE,
        ShardingType.ROW_WISE,
        ShardingType.COLUMN_WISE,
    ]

    BENCH_COMPILE_MODES = [
        CompileMode.EAGER,
        CompileMode.FX_SCRIPT,
    ]

    table_sizes = [
        (40_000_000, 256),
        (4_000_000, 256),
        (1_000_000, 256),
    ]

    tables_info = "\nTABLE SIZES QUANT:"
    for i, (num, dim) in enumerate(table_sizes):
        mb = int(float(num * dim) / 1024 / 1024)
        tables_info += f"\nTABLE[{i}][{num:9}, {dim:4}] u8: {mb:6}Mb"

    report: str = f"REPORT BENCHMARK {datetime_sfx} world_size:{args.world_size} batch_size:{args.batch_size//1000}k\n"
    report += "Module: QuantEmbeddingBagCollection\n"
    report += tables_info
    report += "\n"

    num_requests = args.bench_iters * args.batch_size * args.num_benchmarks
    report += f"num_requests:{num_requests:8}\n"
    report_file: str = f"{output_dir}/run.report"

    tables = get_tables(table_sizes)
    sharder = TestQuantEBCSharder(
        sharding_type="",
        kernel_type=EmbeddingComputeKernel.QUANT.value,
        shardable_params=[table.name for table in tables],
    )

    module = QuantEmbeddingBagCollection(
        tables=tables,
        is_weighted=False,
        device=torch.device("cpu"),
        quant_state_dict_split_scale_bias=True,
    )

    args_kwargs = {
        argname: getattr(args, argname)
        for argname in dir(args)
        # Don't include output_dir since output_dir was modified
        if not argname.startswith("_") and argname != "output_dir"
    }

    benchmark_results = benchmark_module(
        module=module,
        sharder=sharder,
        sharding_types=BENCH_SHARDING_TYPES,
        compile_modes=BENCH_COMPILE_MODES,
        tables=tables,
        output_dir=output_dir,
        **args_kwargs,
    )

    write_report(benchmark_results, report_file, report, num_requests)


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    benchmark_qebc()
