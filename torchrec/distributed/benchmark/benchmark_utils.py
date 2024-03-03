#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

#!/usr/bin/env python3

import copy
import gc
import logging
from dataclasses import dataclass

from enum import Enum
from typing import Dict, List, Tuple, TypeVar, Union

import torch
from torch.autograd.profiler import record_function
from torchrec.distributed.embedding_types import ShardingType

from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.enumerators import EmbeddingEnumerator
from torchrec.distributed.planner.shard_estimators import (
    EmbeddingPerfEstimator,
    EmbeddingStorageEstimator,
)
from torchrec.distributed.shard import _shard_modules
from torchrec.distributed.test_utils.test_model import ModelInput

from torchrec.distributed.types import DataType, ModuleSharder, ShardingEnv
from torchrec.fx import symbolic_trace
from torchrec.modules.embedding_configs import EmbeddingBagConfig, EmbeddingConfig
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor, KeyedTensor

logger: logging.Logger = logging.getLogger()


class CompileMode(Enum):
    EAGER = "eager"
    FX_SCRIPT = "fx_script"


@dataclass
class BenchmarkResult:
    "Class for holding results of benchmark runs"
    short_name: str
    elapsed_time: torch.Tensor
    max_mem_allocated: List[int]


class ECWrapper(torch.nn.Module):
    """
    Wrapper Module for benchmarking EC Modules

    Args:
        module: module to benchmark

    Call Args:
        input: KeyedJaggedTensor KJT input to module

    Returns:
        output: KT output from module

    Example:
        e1_config = EmbeddingConfig(
            name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1"]
        )
        e2_config = EmbeddingConfig(
            name="t2", embedding_dim=3, num_embeddings=10, feature_names=["f2"]
        )

        ec = EmbeddingCollection(tables=[e1_config, e2_config])

        features = KeyedJaggedTensor(
            keys=["f1", "f2"],
            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
            offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8]),
        )

        ec.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.PlaceholderObserver.with_args(
                dtype=torch.qint8
            ),
            weight=torch.quantization.PlaceholderObserver.with_args(dtype=torch.qint8),
        )

        qec = QuantEmbeddingCollection.from_float(ecc)

        wrapped_module = ECWrapper(qec)
        quantized_embeddings = wrapped_module(features)
    """

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self._module = module

    def forward(self, input: KeyedJaggedTensor) -> Dict[str, JaggedTensor]:
        """
        Args:
            input (KeyedJaggedTensor): KJT of form [F X B X L].

        Returns:
            Dict[str, JaggedTensor]
        """
        return self._module.forward(input)


class EBCWrapper(torch.nn.Module):
    """
    Wrapper Module for benchmarking Modules

    Args:
        module: module to benchmark

    Call Args:
        input: KeyedJaggedTensor KJT input to module

    Returns:
        output: KT output from module

    Example:
        table_0 = EmbeddingBagConfig(
            name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1"]
        )
        table_1 = EmbeddingBagConfig(
            name="t2", embedding_dim=4, num_embeddings=10, feature_names=["f2"]
        )
        ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])

        features = KeyedJaggedTensor(
            keys=["f1", "f2"],
            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
            offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8]),
        )

        ebc.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.PlaceholderObserver.with_args(
                dtype=torch.qint8
            ),
            weight=torch.quantization.PlaceholderObserver.with_args(dtype=torch.qint8),
        )

        qebc = QuantEmbeddingBagCollection.from_float(ebc)

        wrapped_module = EBCWrapper(qebc)
        quantized_embeddings = wrapped_module(features)
    """

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self._module = module

    def forward(self, input: KeyedJaggedTensor) -> KeyedTensor:
        """
        Args:
            input (KeyedJaggedTensor): KJT of form [F X B X L].

        Returns:
            KeyedTensor
        """
        return self._module.forward(input)


T = TypeVar("T", bound=torch.nn.Module)


def get_tables(
    table_sizes: List[Tuple[int, int]], is_pooled: bool = True
) -> Union[List[EmbeddingBagConfig], List[EmbeddingConfig]]:
    if is_pooled:
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
    else:
        tables: List[EmbeddingConfig] = [
            EmbeddingConfig(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
                data_type=DataType.INT8,
            )
            for i, (num_embeddings, embedding_dim) in enumerate(table_sizes)
        ]

    return tables


def get_inputs(
    tables: Union[List[EmbeddingBagConfig], List[EmbeddingConfig]],
    batch_size: int,
    world_size: int,
    num_inputs: int,
) -> List[KeyedJaggedTensor]:
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

    return inputs


def transform_module(
    module: torch.nn.Module,
    device: torch.device,
    inputs: List[KeyedJaggedTensor],
    sharder: ModuleSharder[T],
    sharding_type: ShardingType,
    compile_mode: CompileMode,
    world_size: int,
    batch_size: int,
) -> torch.nn.Module:
    def fx_script_module(eager_module: torch.nn.Module) -> torch.nn.Module:
        eager_module(inputs[0])
        graph_module = symbolic_trace(
            eager_module, leaf_modules=["IntNBitTableBatchedEmbeddingBagsCodegen"]
        )
        scripted_module = torch.jit.script(graph_module)
        return scripted_module

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

    # Don't want to modify the module outright
    # Since module is on cpu, won't cause cuda oom.
    copied_module = copy.deepcopy(module)
    # pyre-ignore [6]
    plan = planner.plan(copied_module, [sharder])

    sharded_module = _shard_modules(
        module=copied_module,
        # pyre-ignore [6]
        sharders=[sharder],
        device=device,
        plan=plan,
        env=ShardingEnv.from_local(world_size=topology.world_size, rank=0),
    )

    if compile_mode == CompileMode.FX_SCRIPT:
        return fx_script_module(sharded_module)
    else:
        return sharded_module


def benchmark(
    name: str,
    model: torch.nn.Module,
    warmup_inputs: List[KeyedJaggedTensor],
    bench_inputs: List[KeyedJaggedTensor],
    prof_inputs: List[KeyedJaggedTensor],
    world_size: int,
    output_dir: str,
    num_benchmarks: int,
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
    start = [torch.cuda.Event(enable_timing=True) for _ in range(num_benchmarks)]
    end = [torch.cuda.Event(enable_timing=True) for _ in range(num_benchmarks)]

    for i in range(num_benchmarks):
        start[i].record()
        for _input in bench_inputs:
            model(_input)
        end[i].record()

    for di in range(torch.cuda.device_count()):
        torch.cuda.synchronize(torch.device(f"cuda:{di}"))

    # TODO: First Benchmark Run for Eager Mode produces outlier
    # Start counting after first as workaround for standard deviation
    elapsed_time = torch.tensor(
        [si.elapsed_time(ei) for si, ei in zip(start[1:], end[1:])]
    )

    for di in range(world_size):
        b = torch.cuda.max_memory_allocated(torch.device(f"cuda:{di}"))
        max_mem_allocated.append(b // 1024 // 1024)

    # pyre-ignore[2]
    def trace_handler(prof) -> None:
        total_average = prof.profiler.total_average()
        logger.info(f" TOTAL_AVERAGE:\n{name}\n{total_average}")
        dir_path: str = output_dir

        if dir_path == "":
            return

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
        elapsed_time=elapsed_time,
        max_mem_allocated=max_mem_allocated,
    )


def benchmark_type_name(compile_mode: CompileMode, sharding_type: ShardingType) -> str:
    if sharding_type == ShardingType.TABLE_WISE:
        name = "tw-sharded"
    elif sharding_type == ShardingType.ROW_WISE:
        name = "rw-sharded"
    elif sharding_type == ShardingType.COLUMN_WISE:
        name = "cw-sharded"
    else:
        raise Exception(f"Unknown sharding type {sharding_type}")

    if compile_mode == CompileMode.EAGER:
        name += "-eager"
    elif compile_mode == CompileMode.FX_SCRIPT:
        name += "-fxjit"

    return name


def init_module_and_run_benchmark(
    module: torch.nn.Module,
    sharder: ModuleSharder[T],
    device: torch.device,
    sharding_type: ShardingType,
    compile_mode: CompileMode,
    world_size: int,
    batch_size: int,
    warmup_inputs: List[KeyedJaggedTensor],
    bench_inputs: List[KeyedJaggedTensor],
    prof_inputs: List[KeyedJaggedTensor],
    output_dir: str,
    num_benchmarks: int,
) -> BenchmarkResult:
    """
    There are a couple of caveats here as to why the module has to be initialized
    here:
    1. Device. To accurately track memory usage, when sharding modules the initial
       placement of the module should be on CPU. This is to avoid double counting
       memory allocations and also to prevent CUDA OOMs.
    2. Garbage Collector. Since torch.fx.GraphModule has circular references,
       garbage collection us funky and can lead to ooms. Since this frame is
       called by the loop through compile modes and sharding types, returning the
       benchmark result will mean that the reference to module is lost instead of
       existing in the loop
    """

    module = transform_module(
        module=module,
        device=device,
        inputs=warmup_inputs,
        sharder=sharder,
        sharding_type=sharding_type,
        compile_mode=compile_mode,
        world_size=world_size,
        batch_size=batch_size,
    )

    name = benchmark_type_name(compile_mode, sharding_type)

    return benchmark(
        name,
        module,
        warmup_inputs,
        bench_inputs,
        prof_inputs,
        world_size=world_size,
        output_dir=output_dir,
        num_benchmarks=num_benchmarks,
    )


def benchmark_module(
    module: torch.nn.Module,
    sharder: ModuleSharder[T],
    sharding_types: List[ShardingType],
    compile_modes: List[CompileMode],
    tables: Union[List[EmbeddingBagConfig], List[EmbeddingConfig]],
    warmup_iters: int = 20,
    bench_iters: int = 2000,
    prof_iters: int = 20,
    batch_size: int = 2048,
    world_size: int = 2,
    num_benchmarks: int = 9,
    output_dir: str = "",
) -> List[BenchmarkResult]:
    """
    Args:
        eager_module: Eager mode module to be benchmarked
        sharding_types: Sharding types to be benchmarked
        compile_modes: Compilation modes to be benchmarked
        warmup_iters: Number of iterations to run before profiling
        bench_iters: Number of iterations to run during profiling
        prof_iters: Number of iterations to run after profiling
        batch_size: Batch size used in the model
        world_size: World size used in the
        num_benchmarks: How many times to run over benchmark inputs for statistics
        output_dir: Directory to output profiler outputs (traces, stacks)

    Returns:
        A list of BenchmarkResults
    """

    # logging.info(f"###### Benchmarking Module: {eager_module} ######\n")
    logging.info(f"Warmup iterations: {warmup_iters}")
    logging.info(f"Benchmark iterations: {bench_iters}")
    logging.info(f"Profile iterations: {prof_iters}")
    logging.info(f"Batch Size: {batch_size}")
    logging.info(f"World Size: {world_size}")
    logging.info(f"Number of Benchmarks: {num_benchmarks}")
    logging.info(f"Output Directory: {output_dir}")

    benchmark_results: List[BenchmarkResult] = []
    num_inputs_to_gen: int = warmup_iters + bench_iters + prof_iters
    inputs = get_inputs(tables, batch_size, world_size, num_inputs_to_gen)

    warmup_inputs = inputs[:warmup_iters]
    bench_inputs = inputs[warmup_iters : (warmup_iters + bench_iters)]
    prof_inputs = inputs[-prof_iters:]

    if isinstance(tables[0], EmbeddingBagConfig):
        wrapped_module = EBCWrapper(module)
    else:
        wrapped_module = ECWrapper(module)

    for sharding_type in sharding_types:
        for compile_mode in compile_modes:
            # Test sharders should have a singular sharding_type
            # pyre-ignore [16]
            sharder._sharding_type = sharding_type.value

            benchmark_type = benchmark_type_name(compile_mode, sharding_type)
            logging.info(
                f"\n\n###### Running Benchmark Type: {benchmark_type} ######\n"
            )
            res = init_module_and_run_benchmark(
                module=wrapped_module,
                sharder=sharder,
                # TODO: GPU hardcode for now, expand if needed for heter hardware
                device=torch.device("cuda:0"),
                sharding_type=sharding_type,
                compile_mode=compile_mode,
                world_size=world_size,
                batch_size=batch_size,
                warmup_inputs=warmup_inputs,
                bench_inputs=bench_inputs,
                prof_inputs=prof_inputs,
                num_benchmarks=num_benchmarks,
                output_dir=output_dir,
            )

            gc.collect()
            benchmark_results.append(res)

    return benchmark_results
