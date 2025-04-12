#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

from typing import Any, Dict, List, Optional, Type

import torch

from torchrec.distributed.embedding import (
    EmbeddingCollectionSharder,
    ShardedEmbeddingCollection,
)
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.types import (
    ParameterSharding,
    QuantizedCommCodecs,
    ShardingEnv,
    ShardingType,
)
from torchrec.modules.mapped_embedding_module import MappedEmbeddingCollection


class ShardedMappedEmbeddingCollection(ShardedEmbeddingCollection):
    def __init__(
        self,
        module: MappedEmbeddingCollection,
        table_name_to_parameter_sharding: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: Optional[torch.device],
        fused_params: Optional[Dict[str, Any]] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
        use_index_dedup: bool = False,
        module_fqn: Optional[str] = None,
    ) -> None:
        super().__init__(
            module=module,
            table_name_to_parameter_sharding=table_name_to_parameter_sharding,
            env=env,
            device=device,
            fused_params=fused_params,
            qcomm_codecs_registry=qcomm_codecs_registry,
            use_index_dedup=use_index_dedup,
            module_fqn=module_fqn,
        )


class MappedEmbeddingCollectionSharder(EmbeddingCollectionSharder):
    def __init__(
        self,
        fused_params: Optional[Dict[str, Any]] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
        use_index_dedup: bool = False,
    ) -> None:
        super().__init__(
            fused_params=fused_params,
            qcomm_codecs_registry=qcomm_codecs_registry,
            use_index_dedup=use_index_dedup,
        )

    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [ShardingType.ROW_WISE.value]

    def compute_kernels(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> List[str]:
        return [EmbeddingComputeKernel.KEY_VALUE.value]

    @property
    def module_type(self) -> Type[MappedEmbeddingCollection]:
        return MappedEmbeddingCollection

    # pyre-ignore: Inconsistent override [14]
    def shard(
        self,
        module: MappedEmbeddingCollection,
        params: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        module_fqn: Optional[str] = None,
    ) -> ShardedMappedEmbeddingCollection:
        return ShardedMappedEmbeddingCollection(
            module=module,
            table_name_to_parameter_sharding=params,
            env=env,
            device=device,
            fused_params=self.fused_params,
            qcomm_codecs_registry=self.qcomm_codecs_registry,
            use_index_dedup=self._use_index_dedup,
            module_fqn=module_fqn,
        )
