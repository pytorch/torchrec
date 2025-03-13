#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# pyre-unsafe
import argparse
import contextlib
import logging
import random
import time
from typing import Any, Generator

import torch

logger: logging.Logger = logging.getLogger(__name__)

torch.ops.load_library("//caffe2/torch/fb/retrieval:faster_hash_cpu")
torch.ops.load_library("//caffe2/torch/fb/retrieval:faster_hash_cuda")


@contextlib.contextmanager
def timer(context: str, cpu: bool = False) -> Generator[None, Any, Any]:
    start = time.perf_counter()
    yield
    if not cpu:
        torch.cuda.synchronize()
    logger.info(f"{context} duration {time.perf_counter() - start:.2f}s")


def _run_benchmark_read_only(
    embedding_table_size: int,
    input_nums_size: int,
    warm_up_iteration: int,
    test_iteration: int,
    batch_size: int,
) -> None:
    logger.info(
        "========================= _run_benchmark_read_only START BENCHMARK ========================="
    )

    identities, _ = torch.ops.fb.create_zch_buffer(
        embedding_table_size, device=torch.device("cuda")
    )
    numbers = torch.arange(0, input_nums_size, dtype=torch.int64, device="cuda")
    _, _ = torch.ops.fb.zero_collision_hash(
        input=numbers,
        identities=identities,
        max_probe=512,
        circular_probe=True,
    )

    logger.info(
        f"identities Total slots: {embedding_table_size}; Slot used: {(identities[:, 0] != -1).sum()}"
    )

    identities_uvm = torch.empty_like(identities, device="cpu", pin_memory=True)

    identities_uvm_managed = torch.zeros(
        identities.size(),
        out=torch.ops.fbgemm.new_unified_tensor(
            torch.zeros(
                identities.size(),
                device=identities.device,
                dtype=identities.dtype,
            ),
            [identities.numel()],
            False,
        ),
    )

    identities_uvm.copy_(identities, non_blocking=True)
    identities_uvm_managed.copy_(identities, non_blocking=True)

    to_lookup = torch.randint(
        0, input_nums_size, (test_iteration, batch_size, 2048), device="cuda"
    )
    to_lookup_batch = to_lookup.view(test_iteration, -1)

    logger.info("warm up")
    for i in torch.arange(0, warm_up_iteration):
        for j in torch.arange(0, batch_size):
            output_readonly, evict_slots = torch.ops.fb.zero_collision_hash(
                to_lookup[i][j],
                identities_uvm,
                max_probe=512,
                circular_probe=True,
                exp_hours=-1,
                readonly=True,
                output_on_uvm=True,
            )

    for i in torch.arange(0, warm_up_iteration):
        for j in torch.arange(0, batch_size):
            output_readonly, evict_slots = torch.ops.fb.zero_collision_hash(
                to_lookup[i][j],
                identities_uvm_managed,
                max_probe=512,
                circular_probe=True,
                exp_hours=-1,
                readonly=True,
                output_on_uvm=True,
            )

    for i in torch.arange(0, warm_up_iteration):
        for j in torch.arange(0, batch_size):
            output_readonly, evict_slots = torch.ops.fb.zero_collision_hash(
                to_lookup[i][j],
                identities,
                max_probe=512,
                circular_probe=True,
                exp_hours=-1,
                readonly=True,
                output_on_uvm=True,
            )

    to_lookup_cpu = to_lookup.cpu()
    to_lookup_batch_cpu = to_lookup_cpu.view(test_iteration, -1)
    identities_cpu = identities.cpu()

    for i in torch.arange(0, warm_up_iteration):
        for j in torch.arange(0, batch_size):
            output_readonly, evict_slots = torch.ops.fb.zero_collision_hash(
                to_lookup_cpu[i][j],
                identities_cpu,
                max_probe=512,
                circular_probe=True,
                exp_hours=-1,
                readonly=True,
                output_on_uvm=False,
            )

    logger.info("start benchmark")

    with timer("faster hash benchmark - non batch - uvm"):
        for i in torch.arange(0, test_iteration):
            for j in torch.arange(0, batch_size):
                output_readonly, evict_slots = torch.ops.fb.zero_collision_hash(
                    to_lookup[i][j],
                    identities_uvm,
                    max_probe=512,
                    circular_probe=True,
                    exp_hours=-1,
                    readonly=True,
                    output_on_uvm=True,
                )

    with timer("faster hash benchmark - non batch - uvm managed"):
        for i in torch.arange(0, test_iteration):
            for j in torch.arange(0, batch_size):
                output_readonly, evict_slots = torch.ops.fb.zero_collision_hash(
                    to_lookup[i][j],
                    identities_uvm_managed,
                    max_probe=512,
                    circular_probe=True,
                    exp_hours=-1,
                    readonly=True,
                    output_on_uvm=True,
                )

    with timer("faster hash benchmark - non batch - cuda"):
        for i in torch.arange(0, test_iteration):
            for j in torch.arange(0, batch_size):
                output_readonly, evict_slots = torch.ops.fb.zero_collision_hash(
                    to_lookup[i][j],
                    identities,
                    max_probe=512,
                    circular_probe=True,
                    exp_hours=-1,
                    readonly=True,
                    output_on_uvm=True,
                )

    with timer("faster hash benchmark - non batch - cpu", cpu=True):
        for i in torch.arange(0, test_iteration):
            for j in torch.arange(0, batch_size):
                output_readonly, evict_slots = torch.ops.fb.zero_collision_hash(
                    to_lookup_cpu[i][j],
                    identities_cpu,
                    max_probe=512,
                    circular_probe=True,
                    exp_hours=-1,
                    readonly=True,
                    output_on_uvm=False,
                )

    with timer("faster hash benchmark - batching - uvm"):
        for i in torch.arange(0, test_iteration):
            output_readonly, evict_slots = torch.ops.fb.zero_collision_hash(
                to_lookup_batch[i],
                identities_uvm,
                max_probe=512,
                circular_probe=True,
                exp_hours=-1,
                readonly=True,
                output_on_uvm=True,
            )

    with timer("faster hash benchmark - batching - uvm managed"):
        for i in torch.arange(0, test_iteration):
            output_readonly, evict_slots = torch.ops.fb.zero_collision_hash(
                to_lookup_batch[i],
                identities_uvm_managed,
                max_probe=512,
                circular_probe=True,
                exp_hours=-1,
                readonly=True,
                output_on_uvm=True,
            )

    with timer("faster hash benchmark - batching - cuda"):
        for i in torch.arange(0, test_iteration):
            output_readonly, evict_slots = torch.ops.fb.zero_collision_hash(
                to_lookup_batch[i],
                identities,
                max_probe=512,
                circular_probe=True,
                exp_hours=-1,
                readonly=True,
                output_on_uvm=True,
            )

    with timer("faster hash benchmark - batching - cpu", cpu=True):
        for i in torch.arange(0, test_iteration):
            output_readonly, evict_slots = torch.ops.fb.zero_collision_hash(
                to_lookup_batch_cpu[i],
                identities_cpu,
                max_probe=512,
                circular_probe=True,
                exp_hours=-1,
                readonly=True,
                output_on_uvm=False,
            )
    logger.info("========================= END BENCHMARK =========================")


def _run_benchmark_with_eviction(
    embedding_table_size: int,
    input_nums_size: int,
    eviction_policy_name: str,
    warm_up_iteration: int,
    test_iteration: int,
    batch_size: int,
    zch_exp_hours: int,
) -> None:
    logger.info(
        "========================= _run_benchmark_with_eviction START BENCHMARK ========================="
    )

    eviction_policy = None
    exp_hours = -1
    ts_offset_threshold = 24
    if eviction_policy_name == "lru_eviction":
        eviction_policy = 1
    elif eviction_policy_name == "single_ttl_eviction":
        eviction_policy = 0
        exp_hours = zch_exp_hours
        # make 50% of slots evictable
        ts_offset_threshold = zch_exp_hours * 2

    identities, metadata = torch.ops.fb.create_zch_buffer(
        embedding_table_size, device=torch.device("cuda"), support_evict=True
    )

    numbers = torch.arange(0, input_nums_size, dtype=torch.int64, device="cuda")
    _, _ = torch.ops.fb.zero_collision_hash(
        input=numbers,
        identities=identities,
        max_probe=512,
        circular_probe=True,
        metadata=metadata,
        eviction_policy=eviction_policy,
        exp_hours=exp_hours,
    )

    logger.info(
        f"[After initialization] Total slots : {identities.size()}; Slot used: {(identities[:, 0] != -1).sum()}"
    )

    # assign random timestamps (within range [cur_hour - ts_offset, cur_hour] to slots in metadata
    cur_hour = int(time.time() / 3600)
    ts_offset = random.randint(0, ts_offset_threshold)
    metadata = torch.randint(
        cur_hour - ts_offset,
        cur_hour,
        (embedding_table_size, 1),
        dtype=torch.int32,
        device=metadata.device,
    )

    to_lookup = torch.randint(
        0, input_nums_size, (test_iteration, batch_size, 2048), device="cuda"
    )
    to_lookup_batch = to_lookup.view(test_iteration, -1)

    with timer("faster hash benchmark - batch - eviction - cuda"):
        for i in torch.arange(0, test_iteration):
            output, evict_slots = torch.ops.fb.zero_collision_hash(
                to_lookup_batch[i],
                identities,
                max_probe=512,
                circular_probe=True,
                metadata=metadata,
                eviction_policy=eviction_policy,
                exp_hours=exp_hours,
            )

    logger.info(
        f"[After testing] Total slots: {identities.size()}; Slot used: {(identities[:, 0] != -1).sum()}, evict_slots: {evict_slots.numel()}"
    )

    to_lookup_cpu = to_lookup.cpu()
    to_lookup_batch_cpu = to_lookup_cpu.view(test_iteration, -1)
    identities_cpu = identities.cpu()

    with timer("faster hash benchmark - inference - batch - cpu", cpu=True):
        for i in torch.arange(0, test_iteration):
            output_readonly, _ = torch.ops.fb.zero_collision_hash(
                to_lookup_batch_cpu[i],
                identities_cpu,
                max_probe=512,
                circular_probe=True,
                exp_hours=-1,
                readonly=True,
                output_on_uvm=False,
            )

    with timer("faster hash benchmark - inference - non batch - cpu", cpu=True):
        for i in torch.arange(0, test_iteration):
            for j in torch.arange(0, batch_size):
                output_readonly, _ = torch.ops.fb.zero_collision_hash(
                    to_lookup_cpu[i][j],
                    identities_cpu,
                    max_probe=512,
                    circular_probe=True,
                    exp_hours=-1,
                    readonly=True,
                    output_on_uvm=False,
                )

    logger.info(
        "========================= _run_benchmark_with_eviction END BENCHMARK ========================="
    )


if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    parser = argparse.ArgumentParser(description="Benchmark faster hash operator")

    parser.add_argument(
        "--embedding_table_size",
        type=int,
        default=2000000000,
    )

    parser.add_argument(
        "--input_nums_size",
        type=int,
        default=2000000000,
    )

    parser.add_argument(
        "--eviction_policy",
        type=str,
        default="none",
        help="options include none, lru_eviction, single_ttl_eviction",
    )

    parser.add_argument(
        "--warm_up_iteration",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--test_iteration",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--exp_hours",
        type=int,
        default=-1,
    )

    args = parser.parse_args()

    if args.eviction_policy == "none" and args.exp_hours != -1:
        raise ValueError("'exp_hours' must be -1 when 'eviction_policy' is None")

    if args.eviction_policy == "single_ttl_eviction" and args.exp_hours <= 0:
        raise ValueError(
            "'exp_hours' must be greater than 0 when 'eviction_policy' is single_ttl_eviction"
        )

    if args.eviction_policy == "none":
        _run_benchmark_read_only(
            args.embedding_table_size,
            args.input_nums_size,
            args.warm_up_iteration,
            args.test_iteration,
            args.batch_size,
        )
    else:
        _run_benchmark_with_eviction(
            args.embedding_table_size,
            args.input_nums_size,
            args.eviction_policy,
            args.warm_up_iteration,
            args.test_iteration,
            args.batch_size,
            args.exp_hours,
        )
