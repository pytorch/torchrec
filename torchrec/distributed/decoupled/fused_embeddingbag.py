#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Iterator, List, Optional, Type

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torchrec.distributed.decoupled.types import (
    ShardedModule,
    Sharding,
    ShardingEnv,
    ShardingType,
)

from torchrec.distributed.embedding_types import (
    BaseEmbeddingSharder,
    EmbeddingComputeKernel,
)
from torchrec.distributed.embeddingbag import ShardedEmbeddingBagCollection
from torchrec.distributed.sharding.dp_sharding import DpPooledEmbeddingSharding
from torchrec.distributed.utils import append_prefix
from torchrec.modules.fused_embedding_modules import (
    convert_optimizer_type_and_kwargs,
    FusedEmbeddingBagCollection,
)


class ShardedFusedEmbeddingBagCollection(ShardedModule):
    def __init__(
        self,
        module: FusedEmbeddingBagCollection,
        shardings: List[Sharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
    ) -> None:
        optimizer_type = module.optimizer_type()
        optimizer_kwargs = module.optimizer_kwargs()


class FusedEmbeddingBagCollectionSharder(
    BaseEmbeddingSharder[FusedEmbeddingBagCollection]
):
    def shard(
        self,
        module: FusedEmbeddingBagCollection,
        shardings: List[Sharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
    ) -> ShardedEmbeddingBagCollection:

        return ShardedFusedEmbeddingBagCollection(
            module,
            shardings,
            env,
            device,
        )

    def construct_shardings(
        module: FusedEmbeddingBagCollection,
        per_parameter_sharding: Dict[str, ShardingType],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
    ) -> List[Sharding]:
        return []
