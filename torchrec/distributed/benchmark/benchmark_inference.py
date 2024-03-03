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
from functools import partial
from typing import List, Tuple

import torch

from torchrec.distributed.benchmark.benchmark_utils import (
    benchmark_module,
    BenchmarkResult,
    CompileMode,
    get_tables,
)
from torchrec.distributed.embedding_types import EmbeddingComputeKernel, ShardingType
from torchrec.distributed.test_utils.infer_utils import (
    TestQuantEBCSharder,
    TestQuantECSharder,
)
from torchrec.quant.embedding_modules import (
    EmbeddingBagCollection as QuantEmbeddingBagCollection,
    EmbeddingCollection as QuantEmbeddingCollection,
)

logger: logging.Logger = logging.getLogger()


def init_argparse_and_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--warmup_iters", type=int, default=20)
    parser.add_argument("--bench_iters", type=int, default=500)
    parser.add_argument("--prof_iters", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="/var/tmp/torchrec-bench")
    parser.add_argument("--num_benchmarks", type=int, default=9)

    args = parser.parse_args()
    return args


BENCH_SHARDING_TYPES: List[ShardingType] = [
    ShardingType.TABLE_WISE,
    ShardingType.ROW_WISE,
    ShardingType.COLUMN_WISE,
]

BENCH_COMPILE_MODES: List[CompileMode] = [
    CompileMode.EAGER,
    CompileMode.FX_SCRIPT,
]

TABLE_SIZES: List[Tuple[int, int]] = [
    (40_000_000, 256),
    (4_000_000, 256),
    (1_000_000, 256),
]


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


def benchmark_qec(args: argparse.Namespace, output_dir: str) -> List[BenchmarkResult]:
    tables = get_tables(TABLE_SIZES, is_pooled=False)
    sharder = TestQuantECSharder(
        sharding_type="",
        kernel_type=EmbeddingComputeKernel.QUANT.value,
        shardable_params=[table.name for table in tables],
    )

    module = QuantEmbeddingCollection(
        # pyre-ignore [6]
        tables=tables,
        device=torch.device("cpu"),
        quant_state_dict_split_scale_bias=True,
    )

    args_kwargs = {
        argname: getattr(args, argname)
        for argname in dir(args)
        # Don't include output_dir since output_dir was modified
        if not argname.startswith("_") and argname != "output_dir"
    }

    return benchmark_module(
        module=module,
        sharder=sharder,
        sharding_types=BENCH_SHARDING_TYPES,
        compile_modes=BENCH_COMPILE_MODES,
        tables=tables,
        output_dir=output_dir,
        **args_kwargs,
    )


def benchmark_qebc(args: argparse.Namespace, output_dir: str) -> List[BenchmarkResult]:
    tables = get_tables(TABLE_SIZES)
    sharder = TestQuantEBCSharder(
        sharding_type="",
        kernel_type=EmbeddingComputeKernel.QUANT.value,
        shardable_params=[table.name for table in tables],
    )

    module = QuantEmbeddingBagCollection(
        # pyre-ignore [6]
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

    return benchmark_module(
        module=module,
        sharder=sharder,
        sharding_types=BENCH_SHARDING_TYPES,
        compile_modes=BENCH_COMPILE_MODES,
        tables=tables,
        output_dir=output_dir,
        **args_kwargs,
    )


def main() -> None:
    args: argparse.Namespace = init_argparse_and_args()

    num_requests = args.bench_iters * args.batch_size * args.num_benchmarks
    datetime_sfx: str = time.strftime("%Y%m%dT%H%M%S")

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        # Create output directory if not exist
        os.mkdir(output_dir)

    benchmark_results_per_module = []
    write_report_funcs_per_module = []

    for module_name in ["QuantEmbeddingBagCollection", "QuantEmbeddingCollection"]:
        output_dir = args.output_dir + f"/run_{datetime_sfx}"
        if module_name == "QuantEmbeddingBagCollection":
            output_dir += "_qebc"
            benchmark_func = benchmark_qebc
        else:
            output_dir += "_qec"
            benchmark_func = benchmark_qec

        if not os.path.exists(output_dir):
            # Place all outputs under the datetime folder
            os.mkdir(output_dir)

        tables_info = "\nTABLE SIZES QUANT:"
        for i, (num, dim) in enumerate(TABLE_SIZES):
            mb = int(float(num * dim) / 1024 / 1024)
            tables_info += f"\nTABLE[{i}][{num:9}, {dim:4}] u8: {mb:6}Mb"

        report: str = (
            f"REPORT BENCHMARK {datetime_sfx} world_size:{args.world_size} batch_size:{args.batch_size}\n"
        )
        report += f"Module: {module_name}\n"
        report += tables_info
        report += "\n"

        num_requests = args.bench_iters * args.batch_size * args.num_benchmarks
        report += f"num_requests:{num_requests:8}\n"
        report_file: str = f"{output_dir}/run.report"

        # Save results to output them once benchmarking is all done
        benchmark_results_per_module.append(benchmark_func(args, output_dir))
        write_report_funcs_per_module.append(
            partial(
                write_report,
                report_file=report_file,
                report_str=report,
                num_requests=num_requests,
            )
        )

    for i, write_report_func in enumerate(write_report_funcs_per_module):
        write_report_func(benchmark_results_per_module[i])


def invoke_main() -> None:
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)

    main()


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
