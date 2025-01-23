#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
# pyre-ignore-all-errors[16]

#!/usr/bin/env python3

import argparse
import contextlib
import copy
import gc
import json
import logging
import os
import time
import timeit
from dataclasses import dataclass

from enum import Enum
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import torch
from torch import multiprocessing as mp
from torch.autograd.profiler import record_function
from torchrec.distributed import DistributedModelParallel
from torchrec.distributed.embedding_types import ShardingType
from torchrec.distributed.global_settings import set_propogate_device

from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.enumerators import EmbeddingEnumerator
from torchrec.distributed.planner.shard_estimators import (
    EmbeddingPerfEstimator,
    EmbeddingStorageEstimator,
)
from torchrec.distributed.shard import _shard_modules
from torchrec.distributed.test_utils.multi_process import MultiProcessContext
from torchrec.distributed.test_utils.test_model import ModelInput

from torchrec.distributed.types import DataType, ModuleSharder, ShardingEnv
from torchrec.fx import symbolic_trace
from torchrec.modules.embedding_configs import EmbeddingBagConfig, EmbeddingConfig
from torchrec.quant.embedding_modules import (
    EmbeddingBagCollection as QuantEmbeddingBagCollection,
    EmbeddingCollection as QuantEmbeddingCollection,
)
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor, KeyedTensor
from torchrec.test_utils import get_free_port

logger: logging.Logger = logging.getLogger()

# Reference: https://github.com/facebookresearch/dlrm/blob/main/torchrec_dlrm/README.MD
DLRM_NUM_EMBEDDINGS_PER_FEATURE = [
    4833188,
    36746,
    17245,
    7413,
    20243,
    3,
    7114,
    1441,
    62,
    29275261,
    1572176,
    345138,
    10,
    2209,
    11267,
    128,
    4,
    974,
    14,
    48937457,
    11316796,
    40094537,
    452104,
    12606,
    104,
    35,
]

EMBEDDING_DIM: int = 128


class CompileMode(Enum):
    EAGER = "eager"
    FX_SCRIPT = "fx_script"


@dataclass
class MemoryStats:
    rank: int
    malloc_retries: int
    max_mem_allocated_mbs: int
    max_mem_reserved_mbs: int

    @classmethod
    def for_device(cls, rank: int) -> "MemoryStats":
        stats = torch.cuda.memory_stats(rank)
        alloc_retries = stats.get("num_alloc_retries", 0)
        max_allocated = stats.get("allocated_bytes.all.peak", 0)
        max_reserved = stats.get("reserved_bytes.all.peak", 0)
        return cls(
            rank,
            alloc_retries,
            max_allocated // 1024 // 1024,
            max_reserved // 1024 // 1024,
        )

    def __str__(self) -> str:
        return f"Rank {self.rank}: retries={self.malloc_retries}, allocated={self.max_mem_allocated_mbs:7}mb, reserved={self.max_mem_reserved_mbs:7}mb"


@dataclass
class BenchmarkResult:
    "Class for holding results of benchmark runs"
    short_name: str
    elapsed_time: torch.Tensor  # milliseconds
    mem_stats: List[MemoryStats]  # memory stats per rank
    rank: int = -1

    def __str__(self) -> str:
        runtime = f"Runtime (P90): {self.runtime_percentile(90):.2f} ms"
        mem_alloc = (
            f"Peak Memory alloc (P90): {self.max_mem_alloc_percentile(90)/1000:.2f} GB"
        )
        mem_reserved = f"Peak Memory reserved (P90): {self.max_mem_reserved_percentile(90)/1000:.2f} GB"
        malloc_retries = f"Malloc retries (P50/P90/P100): {self.mem_retries(50) } / {self.mem_retries(90)} / {self.mem_retries(100)}"
        return f"{self.short_name: <{35}} | {malloc_retries} | {runtime} | {mem_alloc} | {mem_reserved}"

    def runtime_percentile(
        self, percentile: int = 50, interpolation: str = "nearest"
    ) -> torch.Tensor:
        return torch.quantile(
            self.elapsed_time,
            percentile / 100.0,
            interpolation=interpolation,
        )

    def max_mem_alloc_percentile(
        self, percentile: int = 50, interpolation: str = "nearest"
    ) -> torch.Tensor:
        return self._mem_percentile(
            lambda m: m.max_mem_allocated_mbs, percentile, interpolation
        )

    def max_mem_reserved_percentile(
        self, percentile: int = 50, interpolation: str = "nearest"
    ) -> torch.Tensor:
        return self._mem_percentile(
            lambda m: m.max_mem_reserved_mbs, percentile, interpolation
        )

    def mem_retries(
        self, percentile: int = 50, interpolation: str = "nearest"
    ) -> torch.Tensor:
        return self._mem_percentile(
            lambda m: m.malloc_retries, percentile, interpolation
        )

    def _mem_percentile(
        self,
        mem_selector: Callable[[MemoryStats], int],
        percentile: int = 50,
        interpolation: str = "nearest",
    ) -> torch.Tensor:
        mem_data = torch.tensor(
            [mem_selector(mem_stat) for mem_stat in self.mem_stats], dtype=torch.float
        )
        return torch.quantile(mem_data, percentile / 100.0, interpolation=interpolation)


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


def default_func_to_benchmark(
    model: torch.nn.Module, bench_inputs: List[KeyedJaggedTensor]
) -> None:
    with torch.inference_mode():
        for bench_input in bench_inputs:
            model(bench_input)


def get_tables(
    table_sizes: List[Tuple[int, int]],
    is_pooled: bool = True,
    data_type: DataType = DataType.INT8,
) -> Union[List[EmbeddingBagConfig], List[EmbeddingConfig]]:
    if is_pooled:
        tables: List[EmbeddingBagConfig] = [
            EmbeddingBagConfig(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
                data_type=data_type,
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
                data_type=data_type,
            )
            for i, (num_embeddings, embedding_dim) in enumerate(table_sizes)
        ]

    return tables


def get_inputs(
    tables: Union[List[EmbeddingBagConfig], List[EmbeddingConfig]],
    batch_size: int,
    world_size: int,
    num_inputs: int,
    train: bool,
    pooling_configs: Optional[List[int]] = None,
    variable_batch_embeddings: bool = False,
) -> List[List[KeyedJaggedTensor]]:
    inputs_batch: List[List[KeyedJaggedTensor]] = []

    if variable_batch_embeddings and not train:
        raise RuntimeError("Variable batch size is only supported in training mode")

    for _ in range(num_inputs):
        if variable_batch_embeddings:
            _, model_input_by_rank = ModelInput.generate_variable_batch_input(
                average_batch_size=batch_size,
                world_size=world_size,
                num_float_features=0,
                tables=tables,
            )
        else:
            _, model_input_by_rank = ModelInput.generate(
                batch_size=batch_size,
                world_size=world_size,
                num_float_features=0,
                tables=tables,
                weighted_tables=[],
                long_indices=False,
                tables_pooling=pooling_configs,
            )

        if train:
            sparse_features_by_rank = [
                model_input.idlist_features for model_input in model_input_by_rank
            ]
            inputs_batch.append(sparse_features_by_rank)
        else:
            sparse_features = model_input_by_rank[0].idlist_features
            inputs_batch.append([sparse_features])

    # Transpose if train, as inputs_by_rank is currently in  [B X R] format
    inputs_by_rank = [
        [sparse_features for sparse_features in sparse_features_rank]
        for sparse_features_rank in zip(*inputs_batch)
    ]

    return inputs_by_rank


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

        mem_str = ""
        for memory_stats in benchmark_res.mem_stats:
            mem_str += f"{memory_stats}\n"

        report_str += f"{benchmark_res.short_name:40} Avg QPS:{qps:10} Avg Duration: {int(1000*avg_dur_s):5}"
        report_str += f"ms Standard Dev Duration: {(1000*std_dur_s):.2f}ms\n"
        report_str += f"\tMemory Allocated Per Rank:\n\t{mem_str}\n"

    with open(report_file, "w") as f:
        f.write(report_str)

    logger.info(f"Report written to {report_file}:\n{report_str}")


def set_embedding_config(
    embedding_config_json: str,
) -> Tuple[List[Tuple[int, int]], List[int]]:
    """
    the config file should follow this pattern: {feature: {num_embeddings: int, embedding_dim: int}}
    """
    embedding_configs = []
    pooling_configs = []
    has_pooling_config = False
    try:
        if os.path.exists(embedding_config_json):
            with open(embedding_config_json, "r") as f:
                embedding_config_json = json.load(f)

            for _, config in embedding_config_json.items():
                embedding_configs.append(
                    (config["num_embeddings"], config["embedding_dim"])
                )
                if "pooling_factor" in config:
                    pooling_configs.append(config["pooling_factor"])
                    has_pooling_config = True
                else:
                    if has_pooling_config:
                        raise RuntimeError(
                            "We cannot handle some features have pooling factor and others don't."
                        )
        else:
            raise RuntimeError(
                f"Could not find embedding config json at path {embedding_config_json}"
            )
    except BaseException as e:
        logger.warning(
            f"Failed to load embedding config because {e}, fallback to DLRM config"
        )
        embedding_configs = [
            (num_embeddings, EMBEDDING_DIM)
            for num_embeddings in DLRM_NUM_EMBEDDINGS_PER_FEATURE
        ]

    return embedding_configs, pooling_configs


def init_argparse_and_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--warmup_iters", type=int, default=20)
    parser.add_argument("--bench_iters", type=int, default=500)
    parser.add_argument("--prof_iters", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--max_num_embeddings", type=int, default=1000000)
    parser.add_argument("--output_dir", type=str, default="/var/tmp/torchrec-bench")
    parser.add_argument("--num_benchmarks", type=int, default=5)
    parser.add_argument("--embedding_config_json", type=str, default="")
    parser.add_argument("--device_type", type=str, default="cuda")

    args = parser.parse_args()

    return args


def transform_module(
    module: torch.nn.Module,
    device: torch.device,
    inputs: List[KeyedJaggedTensor],
    sharder: ModuleSharder[T],
    sharding_type: ShardingType,
    compile_mode: CompileMode,
    world_size: int,
    batch_size: int,
    # pyre-fixme[24]: Generic type `ContextManager` expects 1 type parameter.
    ctx: ContextManager,
    benchmark_unsharded_module: bool = False,
) -> torch.nn.Module:
    def fx_script_module(eager_module: torch.nn.Module) -> torch.nn.Module:
        eager_module(inputs[0])
        graph_module = symbolic_trace(
            eager_module, leaf_modules=["IntNBitTableBatchedEmbeddingBagsCodegen"]
        )
        scripted_module = torch.jit.script(graph_module)
        return scripted_module

    set_propogate_device(True)

    sharded_module = None

    if not benchmark_unsharded_module:
        topology: Topology = Topology(world_size=world_size, compute_device=device.type)
        planner = EmbeddingShardingPlanner(
            topology=topology,
            batch_size=batch_size,
            enumerator=EmbeddingEnumerator(
                topology=topology,
                batch_size=batch_size,
                estimator=[
                    EmbeddingPerfEstimator(topology=topology),
                    EmbeddingStorageEstimator(topology=topology),
                ],
            ),
        )

        # Don't want to modify the module outright
        # Since module is on cpu, won't cause cuda oom.
        copied_module = copy.deepcopy(module)
        # pyre-ignore [6]
        plan = planner.plan(copied_module, [sharder])

        if isinstance(ctx, MultiProcessContext):
            sharded_module = DistributedModelParallel(
                copied_module,
                # pyre-ignore[6]
                env=ShardingEnv.from_process_group(ctx.pg),
                plan=plan,
                # pyre-ignore[6]
                sharders=[sharder],
                device=ctx.device,
            )
        else:
            env = ShardingEnv.from_local(world_size=topology.world_size, rank=0)

            sharded_module = _shard_modules(
                module=copied_module,
                # pyre-fixme[6]: For 2nd argument expected
                #  `Optional[List[ModuleSharder[Module]]]` but got
                #  `List[ModuleSharder[Variable[T (bound to Module)]]]`.
                sharders=[sharder],
                device=device,
                plan=plan,
                env=env,
            )

    if compile_mode == CompileMode.FX_SCRIPT:
        return fx_script_module(
            # pyre-fixme[6]: For 1st argument expected `Module` but got
            #  `Optional[Module]`.
            sharded_module
            if not benchmark_unsharded_module
            else module
        )
    else:
        # pyre-fixme[7]: Expected `Module` but got `Optional[Module]`.
        return sharded_module if not benchmark_unsharded_module else module


def benchmark(
    name: str,
    model: torch.nn.Module,
    warmup_inputs: Union[List[KeyedJaggedTensor], List[Dict[str, Any]]],
    bench_inputs: Union[List[KeyedJaggedTensor], List[Dict[str, Any]]],
    prof_inputs: Union[List[KeyedJaggedTensor], List[Dict[str, Any]]],
    world_size: int,
    output_dir: str,
    num_benchmarks: int,
    # pyre-ignore[2]
    func_to_benchmark: Any,
    benchmark_func_kwargs: Optional[Dict[str, Any]],
    rank: int,
    enable_logging: bool = True,
    device_type: str = "cuda",
    benchmark_unsharded_module: bool = False,
) -> BenchmarkResult:
    memory_stats: List[MemoryStats] = []
    if enable_logging:
        logger.info(f" BENCHMARK_MODEL[{name}]:\n{model}")

    for _input in warmup_inputs:
        model(_input)

    if device_type == "cuda":
        if rank == -1:
            # Reset memory for measurement, no process per rank so do all
            for di in range(world_size):
                torch.cuda.reset_peak_memory_stats(di)
        else:
            torch.cuda.reset_peak_memory_stats(rank)

    start = []
    end = []
    if device_type == "cuda":
        # Measure time taken for batches in bench_inputs
        start = [torch.cuda.Event(enable_timing=True) for _ in range(num_benchmarks)]
        end = [torch.cuda.Event(enable_timing=True) for _ in range(num_benchmarks)]

    if benchmark_func_kwargs is None:
        # Need this to unwrap
        benchmark_func_kwargs = {}

    times = []
    if device_type == "cuda":
        for i in range(num_benchmarks):
            start[i].record()
            func_to_benchmark(model, bench_inputs, **benchmark_func_kwargs)
            end[i].record()
    elif device_type == "cpu":
        times = timeit.repeat(
            lambda: func_to_benchmark(model, bench_inputs, **benchmark_func_kwargs),
            number=1,
            repeat=num_benchmarks,
        )

    if device_type == "cuda":
        if rank == -1:
            for di in range(world_size):
                torch.cuda.synchronize(di)
        else:
            torch.cuda.synchronize(rank)

    # TODO: First Benchmark Run for Eager Mode produces outlier
    # Start counting after first as workaround for standard deviation
    if device_type == "cuda":
        elapsed_time = torch.tensor(
            [si.elapsed_time(ei) for si, ei in zip(start[1:], end[1:])]
        )
    else:
        elapsed_time = torch.tensor(times) * 1e3

    if device_type == "cuda":
        if rank == -1:
            # Add up all memory allocated in inference mode
            for di in range(world_size):
                memory_stats.append(MemoryStats.for_device(di))
        else:
            # Only add up memory allocated for current rank in training mode
            memory_stats.append(MemoryStats.for_device(rank))

    if output_dir != "":
        # Only do profiling if output_dir is set

        # pyre-ignore[2]
        def trace_handler(prof) -> None:
            total_average = prof.profiler.total_average()
            logger.info(f" TOTAL_AVERAGE:\n{name}\n{total_average}")
            dir_path: str = output_dir

            # only 1 rank should output in pg case, rank = 0
            if rank > 0:
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

        if device_type == "cuda":
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

            if rank == -1:
                for di in range(torch.cuda.device_count()):
                    torch.cuda.synchronize(torch.device(f"cuda:{di}"))
            else:
                torch.cuda.synchronize()

    return BenchmarkResult(
        short_name=name,
        elapsed_time=elapsed_time,
        mem_stats=memory_stats,
        rank=rank,
    )


def benchmark_func(
    name: str,
    bench_inputs: List[Dict[str, Any]],
    prof_inputs: List[Dict[str, Any]],
    world_size: int,
    profile_dir: str,
    num_benchmarks: int,
    num_profiles: int,
    # pyre-ignore[2]
    func_to_benchmark: Any,
    benchmark_func_kwargs: Optional[Dict[str, Any]],
    rank: int,
    device_type: str = "cuda",
    pre_gpu_load: int = 0,
) -> BenchmarkResult:
    memory_stats: List[MemoryStats] = []
    if device_type == "cuda":
        if rank == -1:
            # Reset memory for measurement, no process per rank so do all
            for di in range(world_size):
                torch.cuda.reset_peak_memory_stats(di)
                torch.cuda.reset_accumulated_memory_stats(di)
        else:
            torch.cuda.reset_peak_memory_stats(rank)
            torch.cuda.reset_accumulated_memory_stats(rank)

    start = []
    end = []
    if device_type == "cuda":
        # Measure time taken for batches in bench_inputs
        start = [torch.cuda.Event(enable_timing=True) for _ in range(num_benchmarks)]
        end = [torch.cuda.Event(enable_timing=True) for _ in range(num_benchmarks)]

    if benchmark_func_kwargs is None:
        # Need this to unwrap
        benchmark_func_kwargs = {}

    times = []
    if device_type == "cuda":
        a = torch.rand(16384, 16384, device="cuda")
        for _ in range(pre_gpu_load):
            a = a * torch.rand(16384, 16384, device="cuda")
        for i in range(num_benchmarks):
            start[i].record()
            func_to_benchmark(bench_inputs, **benchmark_func_kwargs)
            end[i].record()
    elif device_type == "cpu":
        times = timeit.repeat(
            lambda: func_to_benchmark(bench_inputs, **benchmark_func_kwargs),
            number=1,
            repeat=num_benchmarks,
        )

    if device_type == "cuda":
        if rank == -1:
            for di in range(world_size):
                torch.cuda.synchronize(di)
        else:
            torch.cuda.synchronize(rank)

    # TODO: First Benchmark Run for Eager Mode produces outlier
    # Start counting after first as workaround for standard deviation
    if device_type == "cuda":
        elapsed_time = torch.tensor(
            [si.elapsed_time(ei) for si, ei in zip(start[1:], end[1:])]
        )
    else:
        elapsed_time = torch.tensor(times) * 1e3

    if device_type == "cuda":
        if rank == -1:
            # Add up all memory allocated in inference mode
            for di in range(world_size):
                memory_stats.append(MemoryStats.for_device(di))
        else:
            # Only add up memory allocated for current rank in training mode
            memory_stats.append(MemoryStats.for_device(rank))

    if profile_dir != "":
        # Only do profiling if output_dir is set

        # pyre-ignore[2]
        def trace_handler(prof) -> None:
            total_average = prof.profiler.total_average()
            logger.info(f" TOTAL_AVERAGE:\n{name}\n{total_average}")
            dir_path: str = profile_dir
            if rank == 0:
                trace_file: str = f"{dir_path}/trace-{name}.json"
            else:
                trace_file: str = f"{dir_path}/trace-{name}-{rank}.json"
                return  # only 1 rank should output in pg case, rank = 0
            logger.info(f" PROFILE[{name}].chrome_trace:{trace_file}")
            prof.export_chrome_trace(trace_file)

        if device_type == "cuda":
            a = torch.rand(16384, 16384, device="cuda")
            for _ in range(pre_gpu_load):
                a = a * torch.rand(16384, 16384, device="cuda")
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,
                with_flops=True,
                with_modules=True,
                on_trace_ready=trace_handler,
            ) as p:
                for i in range(num_profiles):
                    with record_function(f"## profile {i} ##"):
                        func_to_benchmark(prof_inputs, **benchmark_func_kwargs)
                        p.step()

            if rank == -1:
                for di in range(torch.cuda.device_count()):
                    torch.cuda.synchronize(torch.device(f"cuda:{di}"))
            else:
                torch.cuda.synchronize()

    return BenchmarkResult(
        short_name=name,
        elapsed_time=elapsed_time,
        mem_stats=memory_stats,
        rank=rank,
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
    warmup_inputs: List[List[KeyedJaggedTensor]],
    bench_inputs: List[List[KeyedJaggedTensor]],
    prof_inputs: List[List[KeyedJaggedTensor]],
    tables: Union[List[EmbeddingBagConfig], List[EmbeddingConfig]],
    output_dir: str,
    num_benchmarks: int,
    # pyre-ignore[2]
    func_to_benchmark: Any,
    benchmark_func_kwargs: Optional[Dict[str, Any]],
    rank: int = -1,
    queue: Optional[mp.Queue] = None,
    pooling_configs: Optional[List[int]] = None,
    benchmark_unsharded_module: bool = False,
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

    if rank >= 0:
        warmup_inputs_cuda = [
            warmup_input.to(torch.device(f"{device.type}:{rank}"))
            for warmup_input in warmup_inputs[rank]
        ]
        bench_inputs_cuda = [
            bench_input.to(torch.device(f"{device.type}:{rank}"))
            for bench_input in bench_inputs[rank]
        ]
        prof_inputs_cuda = [
            prof_input.to(torch.device(f"{device.type}:{rank}"))
            for prof_input in prof_inputs[rank]
        ]
    else:
        warmup_inputs_cuda = [
            warmup_input.to(torch.device(f"{device.type}:0"))
            for warmup_input in warmup_inputs[0]
        ]
        bench_inputs_cuda = [
            bench_input.to(torch.device(f"{device.type}:0"))
            for bench_input in bench_inputs[0]
        ]
        prof_inputs_cuda = [
            prof_input.to(torch.device(f"{device.type}:0"))
            for prof_input in prof_inputs[0]
        ]

    with (
        MultiProcessContext(rank, world_size, "nccl", None)
        if rank != -1
        else contextlib.nullcontext()
    ) as ctx:
        module = transform_module(
            module=module,
            device=device,
            inputs=warmup_inputs_cuda,
            sharder=sharder,
            sharding_type=sharding_type,
            compile_mode=compile_mode,
            world_size=world_size,
            batch_size=batch_size,
            # pyre-ignore[6]
            ctx=ctx,
            benchmark_unsharded_module=benchmark_unsharded_module,
        )

        if benchmark_unsharded_module:
            name = "unsharded" + compile_mode.name
        else:
            name = benchmark_type_name(compile_mode, sharding_type)

        res = benchmark(
            name,
            module,
            warmup_inputs_cuda,
            bench_inputs_cuda,
            prof_inputs_cuda,
            world_size=world_size,
            output_dir=output_dir,
            num_benchmarks=num_benchmarks,
            func_to_benchmark=func_to_benchmark,
            benchmark_func_kwargs=benchmark_func_kwargs,
            rank=rank,
            device_type=device.type,
            benchmark_unsharded_module=benchmark_unsharded_module,
        )

        if queue is not None:
            queue.put(res)

            while not queue.empty():
                time.sleep(1)

    return res


def multi_process_benchmark(
    callable: Callable[
        ...,
        None,
    ],
    # pyre-ignore
    **kwargs,
) -> BenchmarkResult:

    def setUp() -> None:
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = str("localhost")
            os.environ["MASTER_PORT"] = str(get_free_port())

    assert "world_size" in kwargs
    world_size = kwargs["world_size"]

    setUp()
    benchmark_res_per_rank = []
    # kineto has a known problem with fork-server: it'll hang
    # when dumping the trace. Workaround with spawn
    ctx = mp.get_context("spawn")
    qq = ctx.SimpleQueue()
    processes = []

    for rank in range(world_size):
        kwargs["rank"] = rank
        kwargs["world_size"] = world_size
        kwargs["queue"] = qq
        p = ctx.Process(
            target=callable,
            kwargs=kwargs,
        )
        p.start()
        processes.append(p)

    for _ in range(world_size):
        res = qq.get()

        benchmark_res_per_rank.append(res)
        assert len(res.mem_stats) == 1

    for p in processes:
        p.join()
        assert 0 == p.exitcode

    total_benchmark_res = BenchmarkResult(
        benchmark_res_per_rank[0].short_name,
        benchmark_res_per_rank[0].elapsed_time,
        [MemoryStats(rank, 0, 0, 0) for rank in range(world_size)],
        0,
    )

    for res in benchmark_res_per_rank:
        # Each rank's BenchmarkResult contains 1 memory measurement
        total_benchmark_res.mem_stats[res.rank] = res.mem_stats[0]

    return total_benchmark_res


def benchmark_module(
    module: torch.nn.Module,
    sharder: ModuleSharder[T],
    sharding_types: List[ShardingType],
    compile_modes: List[CompileMode],
    tables: Union[List[EmbeddingBagConfig], List[EmbeddingConfig]],
    warmup_iters: int = 20,
    bench_iters: int = 500,
    prof_iters: int = 20,
    batch_size: int = 2048,
    world_size: int = 2,
    num_benchmarks: int = 5,
    output_dir: str = "",
    benchmark_unsharded: bool = False,
    func_to_benchmark: Callable[..., None] = default_func_to_benchmark,
    benchmark_func_kwargs: Optional[Dict[str, Any]] = None,
    pooling_configs: Optional[List[int]] = None,
    variable_batch_embeddings: bool = False,
    device_type: str = "cuda",
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
        pooling_configs: The pooling factor for the tables.
            (Optional; if not set, we'll use 10 as default)
        func_to_benchmark: Custom function to benchmark, check out default_func_to_benchmark for default
        benchmark_func_kwargs: Custom keyword arguments to pass to func_to_benchmark

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

    assert (
        num_benchmarks > 2
    ), "num_benchmarks needs to be greater than 2 for statistical analysis"
    if isinstance(module, QuantEmbeddingBagCollection) or isinstance(
        module, QuantEmbeddingCollection
    ):
        train = False
    else:
        train = True

    benchmark_results: List[BenchmarkResult] = []

    if isinstance(tables[0], EmbeddingBagConfig):
        wrapped_module = EBCWrapper(module)
    else:
        wrapped_module = ECWrapper(module)

    num_inputs_to_gen: int = warmup_iters + bench_iters + prof_iters
    inputs = get_inputs(
        tables,
        batch_size,
        world_size,
        num_inputs_to_gen,
        train,
        pooling_configs,
        variable_batch_embeddings,
    )

    warmup_inputs = [rank_inputs[:warmup_iters] for rank_inputs in inputs]
    bench_inputs = [
        rank_inputs[warmup_iters : (warmup_iters + bench_iters)]
        for rank_inputs in inputs
    ]
    prof_inputs = [rank_inputs[-prof_iters:] for rank_inputs in inputs]

    for sharding_type in sharding_types if not benchmark_unsharded else ["Unsharded"]:
        for compile_mode in compile_modes:
            if not benchmark_unsharded:
                # Test sharders should have a singular sharding_type
                sharder._sharding_type = sharding_type.value
                # pyre-ignore [6]
                benchmark_type = benchmark_type_name(compile_mode, sharding_type)
            else:
                benchmark_type = "unsharded" + compile_mode.name

            logging.info(
                f"\n\n###### Running Benchmark Type: {benchmark_type} ######\n"
            )

            if train:
                res = multi_process_benchmark(
                    # pyre-ignore[6]
                    callable=init_module_and_run_benchmark,
                    module=wrapped_module,
                    sharder=sharder,
                    device=torch.device(device_type),
                    sharding_type=sharding_type,
                    compile_mode=compile_mode,
                    world_size=world_size,
                    batch_size=batch_size,
                    warmup_inputs=warmup_inputs,
                    bench_inputs=bench_inputs,
                    prof_inputs=prof_inputs,
                    tables=tables,
                    num_benchmarks=num_benchmarks,
                    output_dir=output_dir,
                    func_to_benchmark=func_to_benchmark,
                    benchmark_func_kwargs=benchmark_func_kwargs,
                    pooling_configs=pooling_configs,
                )
            else:
                res = init_module_and_run_benchmark(
                    module=wrapped_module,
                    sharder=sharder,
                    device=torch.device(device_type),
                    # pyre-ignore
                    sharding_type=sharding_type,
                    compile_mode=compile_mode,
                    world_size=world_size,
                    batch_size=batch_size,
                    warmup_inputs=warmup_inputs,
                    bench_inputs=bench_inputs,
                    prof_inputs=prof_inputs,
                    tables=tables,
                    num_benchmarks=num_benchmarks,
                    output_dir=output_dir,
                    func_to_benchmark=func_to_benchmark,
                    benchmark_func_kwargs=benchmark_func_kwargs,
                    pooling_configs=pooling_configs,
                    benchmark_unsharded_module=benchmark_unsharded,
                )

            gc.collect()

            benchmark_results.append(res)

    return benchmark_results
