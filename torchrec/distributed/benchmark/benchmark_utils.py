#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

import argparse
import contextlib
import copy
import gc
import logging
import os
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
class BenchmarkResult:
    "Class for holding results of benchmark runs"
    short_name: str
    elapsed_time: torch.Tensor
    max_mem_allocated: List[int]
    rank: int = -1


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
    rank: int = -1,
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

        # If ProcessGroup, place input on correct device. Otherwise, place on cuda:0
        device = torch.device(f"cuda:{rank}") if rank >= 0 else torch.device("cuda:0")
        inputs.append(model_input.idlist_features.to(device))

    return inputs


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
            mem_str += f"Rank {i}: {mem_mb:7}mb  "

        report_str += f"{benchmark_res.short_name:40} Avg QPS:{qps:10} Avg Duration: {int(1000*avg_dur_s):5}"
        report_str += f"ms Standard Dev Duration: {(1000*std_dur_s):.2f}ms\n"
        report_str += f"\tMemory Allocated Per Rank:\n\t{mem_str}\n"

    with open(report_file, "w") as f:
        f.write(report_str)

    logger.info(f"Report written to {report_file}:\n{report_str}")


def init_argparse_and_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--warmup_iters", type=int, default=20)
    parser.add_argument("--bench_iters", type=int, default=500)
    parser.add_argument("--prof_iters", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="/var/tmp/torchrec-bench")
    parser.add_argument("--num_benchmarks", type=int, default=5)

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
    ctx: ContextManager,
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
            # pyre-ignore [6]
            sharders=[sharder],
            device=device,
            plan=plan,
            env=env,
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
    # pyre-ignore[2]
    func_to_benchmark: Any,
    benchmark_func_kwargs: Optional[Dict[str, Any]],
    rank: int,
) -> BenchmarkResult:
    max_mem_allocated: List[int] = []
    logger.info(f" BENCHMARK_MODEL[{name}]:\n{model}")

    for _input in warmup_inputs:
        model(_input)

    if rank == -1:
        # Reset memory for measurement, no process per rank so do all
        for di in range(torch.cuda.device_count()):
            torch.cuda.reset_max_memory_allocated(torch.device(f"cuda:{di}"))
    else:
        torch.cuda.reset_max_memory_allocated(torch.device(f"cuda:{rank}"))

    # Measure time taken for batches in bench_inputs
    start = [torch.cuda.Event(enable_timing=True) for _ in range(num_benchmarks)]
    end = [torch.cuda.Event(enable_timing=True) for _ in range(num_benchmarks)]

    if benchmark_func_kwargs is None:
        # Need this to unwrap
        benchmark_func_kwargs = {}

    for i in range(num_benchmarks):
        start[i].record()
        func_to_benchmark(model, bench_inputs, **benchmark_func_kwargs)
        end[i].record()

    # THis should synchronize all the ranks
    for di in range(torch.cuda.device_count()):
        torch.cuda.synchronize(torch.device(f"cuda:{di}"))

    # TODO: First Benchmark Run for Eager Mode produces outlier
    # Start counting after first as workaround for standard deviation
    elapsed_time = torch.tensor(
        [si.elapsed_time(ei) for si, ei in zip(start[1:], end[1:])]
    )

    if rank == -1:
        # Add up all memory allocated in inference mode
        for di in range(world_size):
            b = torch.cuda.max_memory_allocated(torch.device(f"cuda:{di}"))
            max_mem_allocated.append(b // 1024 // 1024)
    else:
        # Only add up memory allocated for current rank in training mode
        b = torch.cuda.max_memory_allocated(torch.device(f"cuda:{rank}"))
        max_mem_allocated.append(b // 1024 // 1024)

    # pyre-ignore[2]
    def trace_handler(prof) -> None:
        total_average = prof.profiler.total_average()
        logger.info(f" TOTAL_AVERAGE:\n{name}\n{total_average}")
        dir_path: str = output_dir

        # Don't output trace files if dir_path is empty
        # or rank != 0, rank=-1 in no pg case, only 1 rank should output
        # in pg case, so rank=0
        if dir_path == "" or rank > 0:
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
    warmup_iters: int,
    bench_iters: int,
    prof_iters: int,
    tables: Union[List[EmbeddingBagConfig], List[EmbeddingConfig]],
    output_dir: str,
    num_benchmarks: int,
    # pyre-ignore[2]
    func_to_benchmark: Any,
    benchmark_func_kwargs: Optional[Dict[str, Any]],
    rank: int = -1,
    queue: Optional[mp.Queue] = None,
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

    num_inputs_to_gen: int = warmup_iters + bench_iters + prof_iters
    inputs = get_inputs(tables, batch_size, world_size, num_inputs_to_gen, rank)

    warmup_inputs = inputs[:warmup_iters]
    bench_inputs = inputs[warmup_iters : (warmup_iters + bench_iters)]
    prof_inputs = inputs[-prof_iters:]

    with (
        MultiProcessContext(rank, world_size, "nccl", None)
        if rank != -1
        else contextlib.nullcontext()
    ) as ctx:
        module = transform_module(
            module=module,
            device=device,
            inputs=warmup_inputs,
            sharder=sharder,
            sharding_type=sharding_type,
            compile_mode=compile_mode,
            world_size=world_size,
            batch_size=batch_size,
            # pyre-ignore[6]
            ctx=ctx,
        )

        name = benchmark_type_name(compile_mode, sharding_type)

        res = benchmark(
            name,
            module,
            warmup_inputs,
            bench_inputs,
            prof_inputs,
            world_size=world_size,
            output_dir=output_dir,
            num_benchmarks=num_benchmarks,
            func_to_benchmark=func_to_benchmark,
            benchmark_func_kwargs=benchmark_func_kwargs,
            rank=rank,
        )

        if queue is not None:
            queue.put(res)

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
        os.environ["MASTER_ADDR"] = str("localhost")
        os.environ["MASTER_PORT"] = str(get_free_port())
        os.environ["GLOO_DEVICE_TRANSPORT"] = "TCP"
        os.environ["NCCL_SOCKET_IFNAME"] = "lo"

        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    def tearDown() -> None:
        torch.use_deterministic_algorithms(False)
        del os.environ["GLOO_DEVICE_TRANSPORT"]
        del os.environ["NCCL_SOCKET_IFNAME"]
        if torch.cuda.is_available():
            os.unsetenv("CUBLAS_WORKSPACE_CONFIG")

    setUp()
    assert "world_size" in kwargs
    world_size = kwargs["world_size"]

    benchmark_res_per_rank = []
    ctx = mp.get_context("forkserver")
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
        assert len(res.max_mem_allocated) == 1

    total_benchmark_res = BenchmarkResult(
        benchmark_res_per_rank[0].short_name,
        benchmark_res_per_rank[0].elapsed_time,
        [0] * world_size,
        0,
    )

    for res in benchmark_res_per_rank:
        # Each rank's BenchmarkResult contains 1 memory measurement
        total_benchmark_res.max_mem_allocated[res.rank] = res.max_mem_allocated[0]

    for p in processes:
        p.join()
        assert 0 == p.exitcode

    tearDown()
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
    func_to_benchmark: Callable[..., None] = default_func_to_benchmark,
    benchmark_func_kwargs: Optional[Dict[str, Any]] = None,
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

    for sharding_type in sharding_types:
        for compile_mode in compile_modes:
            # Test sharders should have a singular sharding_type
            # pyre-ignore [16]
            sharder._sharding_type = sharding_type.value

            benchmark_type = benchmark_type_name(compile_mode, sharding_type)
            logging.info(
                f"\n\n###### Running Benchmark Type: {benchmark_type} ######\n"
            )

            if train:
                res = multi_process_benchmark(
                    # pyre-ignore[6]
                    callable=init_module_and_run_benchmark,
                    module=wrapped_module,
                    sharder=sharder,
                    # TODO: GPU hardcode for now, expand if needed for heter hardware
                    device=torch.device("cuda:0"),
                    sharding_type=sharding_type,
                    compile_mode=compile_mode,
                    world_size=world_size,
                    batch_size=batch_size,
                    warmup_iters=warmup_iters,
                    bench_iters=bench_iters,
                    prof_iters=prof_iters,
                    tables=tables,
                    num_benchmarks=num_benchmarks,
                    output_dir=output_dir,
                    func_to_benchmark=func_to_benchmark,
                    benchmark_func_kwargs=benchmark_func_kwargs,
                )
            else:
                res = init_module_and_run_benchmark(
                    module=wrapped_module,
                    sharder=sharder,
                    # TODO: GPU hardcode for now, expand if needed for heter hardware
                    device=torch.device("cuda:0"),
                    sharding_type=sharding_type,
                    compile_mode=compile_mode,
                    world_size=world_size,
                    batch_size=batch_size,
                    warmup_iters=warmup_iters,
                    bench_iters=bench_iters,
                    prof_iters=prof_iters,
                    tables=tables,
                    num_benchmarks=num_benchmarks,
                    output_dir=output_dir,
                    func_to_benchmark=func_to_benchmark,
                    benchmark_func_kwargs=benchmark_func_kwargs,
                )

            gc.collect()

            benchmark_results.append(res)

    return benchmark_results
