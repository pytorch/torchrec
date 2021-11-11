#!/usr/bin/env python3

from typing import Dict, Tuple

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


KERNEL_LOOKUP_BW: Dict[Tuple[str, str], float] = {
    # CPU
    ("cpu", EmbeddingComputeKernel.DENSE.value): 0.35 * DDR_MEM_BW,
    ("cpu", EmbeddingComputeKernel.SPARSE.value): 0.35 * DDR_MEM_BW,
    ("cpu", EmbeddingComputeKernel.BATCHED_DENSE.value): 0.5 * DDR_MEM_BW,
    ("cpu", EmbeddingComputeKernel.BATCHED_FUSED.value): 1 * DDR_MEM_BW,
    # CUDA
    ("cuda", EmbeddingComputeKernel.DENSE.value): 0.35 * HBM_MEM_BW,
    ("cuda", EmbeddingComputeKernel.SPARSE.value): 0.35 * HBM_MEM_BW,
    ("cuda", EmbeddingComputeKernel.BATCHED_DENSE.value): 0.5 * HBM_MEM_BW,
    ("cuda", EmbeddingComputeKernel.BATCHED_FUSED.value): 1 * HBM_MEM_BW,
    ("cuda", EmbeddingComputeKernel.BATCHED_FUSED_UVM.value): DDR_MEM_BW / 100,
    ("cuda", EmbeddingComputeKernel.BATCHED_FUSED_UVM_CACHING.value): (
        CACHING_RATIO * HBM_MEM_BW + (1 - CACHING_RATIO) * DDR_MEM_BW
    )
    / 100,
}
