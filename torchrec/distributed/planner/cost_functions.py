#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict

from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.planner.types import CostInput
from torchrec.distributed.types import ShardingType


# Constants
COMMS_MULTIPLER: Dict[str, int] = {
    ShardingType.TABLE_WISE.value: 2,
    ShardingType.COLUMN_WISE.value: 2,
    ShardingType.ROW_WISE.value: 5,
    ShardingType.TABLE_ROW_WISE.value: 3,
    ShardingType.DATA_PARALLEL.value: 1,
}
KERNEL_MULTIPLER: Dict[str, int] = {
    EmbeddingComputeKernel.DENSE.value: 25,
    EmbeddingComputeKernel.SPARSE.value: 5,
    EmbeddingComputeKernel.BATCHED_DENSE.value: 20,
    EmbeddingComputeKernel.BATCHED_FUSED.value: 1,
    EmbeddingComputeKernel.BATCHED_FUSED_UVM.value: 15,
    EmbeddingComputeKernel.BATCHED_FUSED_UVM_CACHING.value: 10,
    EmbeddingComputeKernel.BATCHED_QUANT.value: 1,
}


def cost_func_compute_based(cost_input: CostInput) -> int:
    sharding_type = cost_input.sharding_type
    compute_kernel = cost_input.compute_kernel
    param = cost_input.param
    input_stats = cost_input.input_stats
    hash_size = param.shape[0]
    emb_dim = param.shape[1]
    pooling_factor = (
        input_stats.mean
        if input_stats is not None
        and input_stats.mean is not None
        and None not in input_stats.mean
        else [1.0]
    )
    cost = math.log(hash_size, 10) * emb_dim * sum(pooling_factor)

    if sharding_type not in COMMS_MULTIPLER:
        raise ValueError(f"cost function does not support {sharding_type}")

    if compute_kernel not in KERNEL_MULTIPLER:
        raise ValueError(f"cost function does not support {compute_kernel}")

    return round(
        cost * COMMS_MULTIPLER[sharding_type] * KERNEL_MULTIPLER[compute_kernel]
    )
