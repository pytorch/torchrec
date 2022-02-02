#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

from torchrec.distributed.embedding_types import EmbeddingComputeKernel

MAX_SIZE: int = (1 << 63) - 1

INTRA_NODE_BANDWIDTH: int = 600
CROSS_NODE_BANDWIDTH: int = 12

MIN_CW_DIM: int = 32
POOLING_FACTOR: float = 1.0

BIGINT_DTYPE: int = 8

HBM_CAP: int = 32 * 1024 * 1024 * 1024  # 32 GB
DDR_CAP: int = 128 * 1024 * 1024 * 1024  # 128 GB
DDR_MEM_BW: int = 51
HBM_MEM_BW: int = 897
CACHING_RATIO: float = 0.2
BATCH_SIZE: int = 512


def kernel_bw_lookup(
    compute_device: str,
    compute_kernel: str,
    caching_ratio: Optional[float] = None,
) -> Optional[float]:
    """
    Calculates the device bandwidth based on given compute_device, compute_kernel, and caching_ratio.

    Args:
        compute_kernel (str): compute kernel.
        compute_device (str): compute device.
        caching_ratio (Optional[float]): caching ratio used to determine device bandwidth
            if uvm caching is enabled.

    Returns:
        float: the device bandwidth.

    """
    caching_ratio = caching_ratio if caching_ratio else CACHING_RATIO
    lookup = {
        # CPU
        ("cpu", EmbeddingComputeKernel.DENSE.value): 0.35 * DDR_MEM_BW,
        ("cpu", EmbeddingComputeKernel.SPARSE.value): 0.35 * DDR_MEM_BW,
        ("cpu", EmbeddingComputeKernel.BATCHED_DENSE.value): 0.5 * DDR_MEM_BW,
        ("cpu", EmbeddingComputeKernel.BATCHED_FUSED.value): 1 * DDR_MEM_BW,
        ("cpu", EmbeddingComputeKernel.BATCHED_QUANT.value): 1 * DDR_MEM_BW,
        # CUDA
        ("cuda", EmbeddingComputeKernel.DENSE.value): 0.35 * HBM_MEM_BW,
        ("cuda", EmbeddingComputeKernel.SPARSE.value): 0.35 * HBM_MEM_BW,
        ("cuda", EmbeddingComputeKernel.BATCHED_DENSE.value): 0.5 * HBM_MEM_BW,
        ("cuda", EmbeddingComputeKernel.BATCHED_FUSED.value): 1 * HBM_MEM_BW,
        ("cuda", EmbeddingComputeKernel.BATCHED_QUANT.value): 1 * HBM_MEM_BW,
        ("cuda", EmbeddingComputeKernel.BATCHED_FUSED_UVM.value): DDR_MEM_BW / 100,
        ("cuda", EmbeddingComputeKernel.BATCHED_FUSED_UVM_CACHING.value): (
            caching_ratio * HBM_MEM_BW + (1 - caching_ratio) * DDR_MEM_BW
        )
        / 100,
    }
    return lookup.get((compute_device, compute_kernel))
