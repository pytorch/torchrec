#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

from typing import Any, Dict, Optional, Type

import torch

from torchrec.distributed.quant_embedding import (
    QuantEmbeddingCollectionSharder,
    ShardedQuantEmbeddingCollection,
)
from torchrec.distributed.types import ParameterSharding, ShardingEnv
from torchrec.quant.embedding_modules import QuantMappedEmbeddingCollection


class ShardedQuantMappedEmbeddingCollection(ShardedQuantEmbeddingCollection):
    def __init__(
        self,
        module: QuantMappedEmbeddingCollection,
        table_name_to_parameter_sharding: Dict[str, ParameterSharding],
        env: ShardingEnv,
        fused_params: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(
            module,
            table_name_to_parameter_sharding,
            env,
            fused_params,
            device,
        )


class QuantMappedEmbeddingCollectionSharder(QuantEmbeddingCollectionSharder):
    def __init__(
        self,
        fused_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(fused_params)

    @property
    def module_type(self) -> Type[QuantMappedEmbeddingCollection]:
        return QuantMappedEmbeddingCollection

    # pyre-ignore [14]
    def shard(
        self,
        module: QuantMappedEmbeddingCollection,
        table_name_to_parameter_sharding: Dict[str, ParameterSharding],
        env: ShardingEnv,
        fused_params: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> ShardedQuantMappedEmbeddingCollection:
        return ShardedQuantMappedEmbeddingCollection(
            module,
            table_name_to_parameter_sharding,
            env,
            fused_params=fused_params,
            device=device,
        )
