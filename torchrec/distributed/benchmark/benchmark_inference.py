#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

import argparse
import datetime
import gc
import logging
import os
import time
from dataclasses import dataclass

from enum import Enum
from typing import List, Tuple

import torch
from torch.autograd.profiler import record_function

from torchrec.distributed.embedding_types import EmbeddingComputeKernel, ShardingType
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.enumerators import EmbeddingEnumerator
from torchrec.distributed.planner.shard_estimators import (
    EmbeddingPerfEstimator,
    EmbeddingStorageEstimator,
)
from torchrec.distributed.shard import _shard_modules

from torchrec.distributed.test_utils.infer_utils import TestQuantEBCSharder
from torchrec.distributed.test_utils.test_model import ModelInput
from torchrec.distributed.types import DataType, ShardingEnv
from torchrec.fx import symbolic_trace
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.quant.embedding_modules import (
    EmbeddingBagCollection as QuantEmbeddingBagCollection,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor

logger: logging.Logger = logging.getLogger()


# TODO: Workaround for torchscript silent failure issue
# https://fb.workplace.com/groups/1405155842844877/permalink/7840375302656200/
torch._C._jit_override_can_fuse_on_cpu(False)
torch._C._jit_override_can_fuse_on_gpu(False)
torch._C._jit_set_texpr_fuser_enabled(False)
torch._C._jit_set_nvfuser_enabled(False)


class TerminalStage(Enum):
    FP = 0
    QUANT = 1
    FXJIT_QUANT = 2
    SHARDED_QUANT = 3
    FXJIT_SHARDED_QUANT = 4


class EBCWrapper(torch.nn.Module):
    "Wrapper Module for benchmarking TorchRec Inference Modules"
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self._module = module

    def forward(self, input: KeyedJaggedTensor) -> KeyedTensor:
        return self._module.forward(input)


@dataclass
class BenchmarkResult:
    "Class for holding results of benchmark runs"
    short_name: str
    total_duration_sec: float
    max_mem_allocated: List[int]


def print_device_memory_allocated_status(log: str) -> None:
    for di in range(torch.cuda.device_count()):
        device = torch.device(f"cuda:{di}")
        logger.info(
            f"cuda.memory_allocated[{log}][{device}]:{torch.cuda.memory.memory_allocated(device) // 1024 // 1024} Mb"
        )


UNSHARDED_TERMINAL_STAGES: List[TerminalStage] = [
    TerminalStage.QUANT,
    TerminalStage.FXJIT_QUANT,
]

SHARDED_TERMINAL_STAGES: List[TerminalStage] = [
    TerminalStage.SHARDED_QUANT,
    TerminalStage.FXJIT_SHARDED_QUANT,
]


def _model_ebc(
    tables: List[EmbeddingBagConfig],
    quant_device: torch.device,
    device: torch.device,
    quant_state_dict_split_scale_bias: bool,
    sharding_type: ShardingType,
    world_size: int,
    batch_size: int,
    terminal_stage: TerminalStage,
    inputs: List[KeyedJaggedTensor],
) -> torch.nn.Module:
    logging.info(f" _model_ebc.BEGIN[{terminal_stage}]")
    print_device_memory_allocated_status(f"_model_ebc.BEGIN {terminal_stage}")

    wrapped_module = EBCWrapper(
        QuantEmbeddingBagCollection(
            tables=tables,
            is_weighted=False,
            device=quant_device,
            quant_state_dict_split_scale_bias=quant_state_dict_split_scale_bias,
        )
    )

    print_device_memory_allocated_status(f"_model_ebc.WRAPPED_MODULE {terminal_stage}")

    if terminal_stage == TerminalStage.QUANT:
        return wrapped_module

    if terminal_stage == TerminalStage.FXJIT_QUANT:
        wrapped_module(inputs[0])
        graph_module = symbolic_trace(
            wrapped_module, leaf_modules=["IntNBitTableBatchedEmbeddingBagsCodegen"]
        )
        scripted_module = torch.jit.script(graph_module)
        return scripted_module

    print_device_memory_allocated_status(f"_model_ebc.BEFORE_SHARDING {terminal_stage}")
    sharder = TestQuantEBCSharder(
        sharding_type=sharding_type.value,
        kernel_type=EmbeddingComputeKernel.QUANT.value,
        shardable_params=[table.name for table in tables],
    )

    topology: Topology = Topology(world_size=world_size, compute_device="cuda")
    planner = EmbeddingShardingPlanner(
        topology=topology,
        batch_size=batch_size,
        enumerator=EmbeddingEnumerator(
            topology=topology,
            batch_size=batch_size,
            estimator=[
                EmbeddingPerfEstimator(topology=topology, is_inference=True),
                EmbeddingStorageEstimator(topology=topology),
            ],
        ),
    )

    # pyre-ignore [6]
    plan = planner.plan(wrapped_module, [sharder])

    sharded_module = _shard_modules(
        module=wrapped_module,
        # pyre-ignore [6]
        sharders=[sharder],
        device=device,
        plan=plan,
        env=ShardingEnv.from_local(world_size=topology.world_size, rank=0),
    )
    print_device_memory_allocated_status(f"_model_ebc.AFTER_SHARDING {terminal_stage}")

    if terminal_stage == TerminalStage.SHARDED_QUANT:
        return sharded_module

    sharded_module(inputs[0])
    sharded_traced_module = symbolic_trace(
        sharded_module, leaf_modules=["IntNBitTableBatchedEmbeddingBagsCodegen"]
    )

    sharded_scripted_module = torch.jit.script(sharded_traced_module)
    return sharded_scripted_module


def benchmark(
    name: str,
    model: torch.nn.Module,
    warmup_inputs: List[KeyedJaggedTensor],
    bench_inputs: List[KeyedJaggedTensor],
    prof_inputs: List[KeyedJaggedTensor],
    batch_size: int,
    world_size: int,
    output_dir: str,
) -> BenchmarkResult:
    model.training = False
    max_mem_allocated: List[int] = []
    logger.info(f" BENCHMARK_MODEL[{name}]:\n{model}")

    for _input in warmup_inputs:
        model(_input)

    # Reset memory for measurement
    for di in range(torch.cuda.device_count()):
        torch.cuda.reset_max_memory_allocated(torch.device(f"cuda:{di}"))

    # Measure time taken for batches in bench_inputs
    time_start = datetime.datetime.now()

    for _input in bench_inputs:
        model(_input)

    time_end = datetime.datetime.now()

    for di in range(world_size):
        torch.cuda.synchronize(torch.device(f"cuda:{di}"))

    for di in range(world_size):
        b = torch.cuda.max_memory_allocated(torch.device(f"cuda:{di}"))
        max_mem_allocated.append(b // 1024 // 1024)

    total_duration_sec = (time_end - time_start).total_seconds()

    # pyre-ignore[2]
    def trace_handler(prof) -> None:
        total_average = prof.profiler.total_average()
        logger.info(f" TOTAL_AVERAGE:\n{name}\n{total_average}")
        dir_path: str = output_dir
        trace_file: str = f"{dir_path}/trace-{name}.json"
        stacks_cpu_file = f"{dir_path}/stacks-cpu-{name}.stacks"
        stacks_cuda_file = f"{dir_path}/stacks-cuda-{name}.stacks"
        logger.info(f" PROFILE[{name}].chrome_trace:{trace_file}")

        prof.export_chrome_trace(trace_file)
        prof.export_stacks(stacks_cpu_file, "self_cpu_time_total")
        prof.export_stacks(stacks_cuda_file, "self_cuda_time_total")

    # - git clone https://github.com/brendangregg/FlameGraph
    # - cd FlameGraph
    # - ./flamegraph.pl --title "CPU time" --countname "us." profiler.stacks > perf_viz.svg

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        with_modules=True,
        on_trace_ready=trace_handler,
    ) as p:
        for _input in prof_inputs:
            with record_function("## forward ##"):
                model(_input)
                p.step()
        for di in range(torch.cuda.device_count()):
            torch.cuda.synchronize(torch.device(f"cuda:{di}"))

    return BenchmarkResult(
        short_name=name,
        total_duration_sec=total_duration_sec,
        max_mem_allocated=max_mem_allocated,
    )


def benchmark_type_name(
    terminal_stage: TerminalStage, sharding_type: ShardingType
) -> str:
    if terminal_stage in UNSHARDED_TERMINAL_STAGES:
        name = "unsharded-qebc"
        if terminal_stage == TerminalStage.FXJIT_QUANT:
            name += "-fxjit-quant"
    else:
        if sharding_type == ShardingType.ROW_WISE:
            name = "rw-sharded-qebc"
        elif sharding_type == ShardingType.COLUMN_WISE:
            name = "cw-sharded-qebc"
        else:
            name = "tw-sharded-qebc"

        if terminal_stage == TerminalStage.FXJIT_SHARDED_QUANT:
            name += "-fxjit-quant"

    return name


def run_benchmark(
    tables: List[EmbeddingBagConfig],
    quant_device: torch.device,
    device: torch.device,
    sharding_type: ShardingType,
    world_size: int,
    batch_size: int,
    terminal_stage: TerminalStage,
    warmup_inputs: List[KeyedJaggedTensor],
    bench_inputs: List[KeyedJaggedTensor],
    prof_inputs: List[KeyedJaggedTensor],
    output_dir: str,
) -> BenchmarkResult:
    module = _model_ebc(
        terminal_stage=terminal_stage,
        tables=tables,
        quant_device=quant_device,
        device=device,
        quant_state_dict_split_scale_bias=True,
        sharding_type=sharding_type,
        world_size=world_size,
        batch_size=batch_size,
        inputs=bench_inputs,
    )

    name = benchmark_type_name(terminal_stage, sharding_type)

    return benchmark(
        name,
        module,
        warmup_inputs,
        bench_inputs,
        prof_inputs,
        batch_size,
        world_size=world_size,
        output_dir=output_dir,
    )


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--warmup_iters", type=int, default=20)
    parser.add_argument("--bench_iters", type=int, default=2000)
    parser.add_argument("--prof_iters", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="/var/tmp/torchrec-bench")

    return parser


def get_tables_and_input(
    table_sizes: List[Tuple[int, int]],
    num_inputs: int,
    batch_size: int,
    world_size: int,
) -> Tuple[List[EmbeddingBagConfig], List[KeyedJaggedTensor]]:
    tables: List[EmbeddingBagConfig] = [
        EmbeddingBagConfig(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            name="table_" + str(i),
            feature_names=["feature_" + str(i)],
            data_type=DataType.INT8,
        )
        for i, (num_embeddings, embedding_dim) in enumerate(table_sizes)
    ]

    inputs: List[KeyedJaggedTensor] = []
    for _ in range(num_inputs):
        model_input = ModelInput.generate(
            batch_size=batch_size,
            world_size=world_size,
            num_float_features=0,
            tables=tables,
            weighted_tables=[],
            long_indices=False,
        )[1][0]
        inputs.append(model_input.idlist_features.to(torch.device("cuda:0")))

    return tables, inputs


def write_report(
    benchmark_results: List[BenchmarkResult],
    report_file: str,
    report_str: str,
    num_requests: int,
) -> None:
    # Sort by slowest to fastest run
    benchmark_results.sort(key=lambda res: res.total_duration_sec)

    for result in benchmark_results:
        name, dur_s, max_mem_allocated = (
            result.short_name,
            result.total_duration_sec,
            result.max_mem_allocated,
        )
        qps = int(num_requests / dur_s)
        mem_str = ""
        for mem_mb in max_mem_allocated:
            mem_str += f"{mem_mb:7} "

        report_str += f"{name:40} QPS:{qps:10} Duration: {int(1000*dur_s):5}ms Memory Allocated Per Rank:{mem_str}\n"

    with open(report_file, "w") as f:
        f.write(report_str)

    logger.info(f"Report written to {report_file}:\n{report_str}")


def main() -> None:
    parser = init_argparse()
    args = parser.parse_args()

    datetime_sfx: str = time.strftime("%Y%m%dT%H%M%S")

    warmup_iters: int = args.warmup_iters
    bench_iters: int = args.bench_iters
    prof_iters: int = args.prof_iters
    num_inputs_to_gen: int = warmup_iters + bench_iters + prof_iters

    world_size: int = args.world_size
    batch_size: int = args.batch_size

    output_dir: str = args.output_dir

    if not os.path.exists(output_dir):
        # Create output directory if not exist
        os.mkdir(output_dir)

    output_dir += f"/run_{datetime_sfx}"
    if not os.path.exists(output_dir):
        # Place all outputs under the datetime folder
        os.mkdir(output_dir)

    # TODO: ROW_WISE and COLUMN_WISE are not supported yet
    BENCH_SHARDING_TYPES = [
        ShardingType.TABLE_WISE,
        # ShardingType.ROW_WISE,
        # ShardingType.COLUMN_WISE,
    ]

    table_sizes = [
        (96_000_000, 256),
        (4_000_000, 256),
        (1_000_000, 256),
    ]

    tables, inputs = get_tables_and_input(
        table_sizes, num_inputs_to_gen, batch_size, world_size
    )
    print_device_memory_allocated_status("Memory After Generating Inputs")

    warmup_inputs = inputs[:warmup_iters]
    bench_inputs = inputs[warmup_iters : (warmup_iters + bench_iters)]
    prof_inputs = inputs[-prof_iters:]

    tables_info = "\nTABLE SIZES QUANT:"
    for i, (num, dim) in enumerate(table_sizes):
        mb = int(float(num * dim) / 1024 / 1024)
        tables_info += f"\nTABLE[{i}][{num:9}, {dim:4}] u8: {mb:6}Mb"

    report: str = f"REPORT BENCHMARK {datetime_sfx} world_size:{world_size} batch_size:{batch_size//1000}k\n"
    report += tables_info
    report += "\n"

    benchmark_results: List[BenchmarkResult] = []

    # enable_reference_cycle_detector()

    # TEST UNSHARDED
    for terminal_stage in UNSHARDED_TERMINAL_STAGES:
        benchmark_type = benchmark_type_name(terminal_stage, ShardingType.TABLE_WISE)
        logging.info(
            f"\n\n###### Running QEBC Benchmark Type: {benchmark_type} ######\n"
        )

        res = run_benchmark(
            terminal_stage=terminal_stage,
            tables=tables,
            quant_device=torch.device("cuda:0"),
            device=torch.device("cuda:0"),
            sharding_type=ShardingType.TABLE_WISE,
            world_size=world_size,
            batch_size=batch_size,
            warmup_inputs=warmup_inputs,
            bench_inputs=bench_inputs,
            prof_inputs=prof_inputs,
            output_dir=output_dir,
        )

        # Reference cycles present with torch.fx.GraphModule
        gc.collect()
        benchmark_results.append(res)
        print_device_memory_allocated_status("Memory Post Benchmarking")

    # TEST SHARDED
    for sharding_type in BENCH_SHARDING_TYPES:
        for terminal_stage in SHARDED_TERMINAL_STAGES:
            benchmark_type = benchmark_type_name(terminal_stage, sharding_type)
            logging.info(
                f"\n\n###### Running QEBC Benchmark Type: {benchmark_type} ######\n"
            )
            res = run_benchmark(
                terminal_stage=terminal_stage,
                tables=tables,
                quant_device=torch.device("cpu"),
                device=torch.device("cuda:0"),
                sharding_type=sharding_type,
                world_size=world_size,
                batch_size=batch_size,
                warmup_inputs=warmup_inputs,
                bench_inputs=bench_inputs,
                prof_inputs=prof_inputs,
                output_dir=output_dir,
            )

            gc.collect()
            benchmark_results.append(res)
            print_device_memory_allocated_status("Memory Post Benchmarking")

    num_requests = len(bench_inputs) * batch_size
    report += f"num_requests:{num_requests:8}\n"

    report_file: str = f"{output_dir}/run.report"

    write_report(benchmark_results, report_file, report, num_requests)


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    main()
