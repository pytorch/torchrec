#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

from torchrec.distributed.embedding_types import EmbeddingComputeKernel

MAX_SIZE: int = (1 << 63) - 1

INTRA_NODE_BANDWIDTH: float = 600 * 1024 * 1024 * 1024 / 1000  # bytes/ms
CROSS_NODE_BANDWIDTH: float = 12.5 * 1024 * 1024 * 1024 / 1000  # bytes/ms

MIN_CW_DIM: int = 128
POOLING_FACTOR: float = 1.0

BIGINT_DTYPE: int = 8

HBM_CAP: int = 32 * 1024 * 1024 * 1024  # 32 GB
DDR_CAP: int = 128 * 1024 * 1024 * 1024  # 128 GB
DDR_MEM_BW: float = 51 * 1024 * 1024 * 1024 / 1000  # bytes/ms
HBM_MEM_BW: float = 897 * 1024 * 1024 * 1024 / 1000  # bytes/ms
UVM_CACHING_RATIO: float = 0.2
BATCH_SIZE: int = 512

BATCHED_COPY_PERF_FACTOR: float = 2.455  # empirical studies
FULL_BLOCK_EMB_DIM: int = 128  # FBGEMM Kernel, 32 threads X 4D-Vector
HALF_BLOCK_PENALTY: float = 1.15  # empirical studies
QUARTER_BLOCK_PENALTY: float = 1.75  # empirical studies
BWD_COMPUTE_MULTIPLIER: float = 2  # empirical studies
WEIGHTED_KERNEL_MULTIPLIER: float = 1.1  # empirical studies
DP_ELEMENTWISE_KERNELS_PERF_FACTOR: float = 9.22  # empirical studies


def kernel_bw_lookup(
    compute_device: str,
    compute_kernel: str,
    hbm_mem_bw: float,
    ddr_mem_bw: float,
    caching_ratio: Optional[float] = None,
) -> Optional[float]:
    """
    Calculates the device bandwidth based on given compute device, compute kernel, and
    caching ratio.

    Args:
        compute_kernel (str): compute kernel.
        compute_device (str): compute device.
        hbm_mem_bw (float): the bandwidth of the device HBM.
        ddr_mem_bw (float): the bandwidth of the system DDR memory.
        caching_ratio (Optional[float]): caching ratio used to determine device bandwidth
            if UVM caching is enabled.

    Returns:
        Optional[float]: the device bandwidth.
    """
    caching_ratio = caching_ratio if caching_ratio else UVM_CACHING_RATIO
    lookup = {
        # CPU
        ("cpu", EmbeddingComputeKernel.DENSE.value): 0.5 * ddr_mem_bw,
        ("cpu", EmbeddingComputeKernel.FUSED.value): 1 * ddr_mem_bw,
        ("cpu", EmbeddingComputeKernel.QUANT.value): 1 * ddr_mem_bw,
        # CUDA
        ("cuda", EmbeddingComputeKernel.DENSE.value): 0.5 * hbm_mem_bw,
        ("cuda", EmbeddingComputeKernel.FUSED.value): 1 * hbm_mem_bw,
        ("cuda", EmbeddingComputeKernel.FUSED_UVM.value): ddr_mem_bw / 10,
        ("cuda", EmbeddingComputeKernel.FUSED_UVM_CACHING.value): (
            caching_ratio * hbm_mem_bw + (1 - caching_ratio) * ddr_mem_bw
        )
        / 10,
        ("cuda", EmbeddingComputeKernel.QUANT.value): 1 * hbm_mem_bw,
        ("cuda", EmbeddingComputeKernel.QUANT_UVM.value): ddr_mem_bw / 10,
        ("cuda", EmbeddingComputeKernel.QUANT_UVM_CACHING.value): (
            caching_ratio * hbm_mem_bw + (1 - caching_ratio) * ddr_mem_bw
        )
        / 10,
    }
    return lookup.get((compute_device, compute_kernel))
