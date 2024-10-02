# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Any, cast, Dict, Optional, Type

import torch
from torchrec.distributed.embedding_types import KJTList
from torchrec.distributed.embeddingbag import (
    EmbeddingBagCollectionContext,
    EmbeddingBagCollectionSharder,
    ShardedEmbeddingBagCollection,
)
from torchrec.distributed.mc_embedding_modules import (
    BaseManagedCollisionEmbeddingCollectionSharder,
    BaseShardedManagedCollisionEmbeddingCollection,
)
from torchrec.distributed.mc_modules import ManagedCollisionCollectionSharder
from torchrec.distributed.types import (
    ParameterSharding,
    QuantizedCommCodecs,
    ShardingEnv,
)
from torchrec.modules.mc_embedding_modules import ManagedCollisionEmbeddingBagCollection


@dataclass
class ManagedCollisionEmbeddingBagCollectionContext(EmbeddingBagCollectionContext):
    evictions_per_table: Optional[Dict[str, Optional[torch.Tensor]]] = None
    remapped_kjt: Optional[KJTList] = None

    def record_stream(self, stream: torch.Stream) -> None:
        super().record_stream(stream)
        if self.evictions_per_table:
            #  pyre-ignore
            for value in self.evictions_per_table.values():
                if value is None:
                    continue
                value.record_stream(stream)
        if self.remapped_kjt is not None:
            self.remapped_kjt.record_stream(stream)


class ShardedManagedCollisionEmbeddingBagCollection(
    BaseShardedManagedCollisionEmbeddingCollection[
        ManagedCollisionEmbeddingBagCollectionContext
    ]
):
    def __init__(
        self,
        module: ManagedCollisionEmbeddingBagCollection,
        table_name_to_parameter_sharding: Dict[str, ParameterSharding],
        ebc_sharder: EmbeddingBagCollectionSharder,
        mc_sharder: ManagedCollisionCollectionSharder,
        # TODO - maybe we need this to manage unsharded/sharded consistency/state consistency
        env: ShardingEnv,
        device: torch.device,
    ) -> None:
        super().__init__(
            module,
            table_name_to_parameter_sharding,
            ebc_sharder,
            mc_sharder,
            env,
            device,
        )

    # For backwards compat, some references still to self._embedding_bag_collection
    @property
    def _embedding_bag_collection(self) -> ShardedEmbeddingBagCollection:
        return cast(ShardedEmbeddingBagCollection, self._embedding_module)

    def create_context(
        self,
    ) -> ManagedCollisionEmbeddingBagCollectionContext:
        return ManagedCollisionEmbeddingBagCollectionContext(sharding_contexts=[])


class ManagedCollisionEmbeddingBagCollectionSharder(
    BaseManagedCollisionEmbeddingCollectionSharder[
        ManagedCollisionEmbeddingBagCollection
    ]
):
    def __init__(
        self,
        ebc_sharder: Optional[EmbeddingBagCollectionSharder] = None,
        mc_sharder: Optional[ManagedCollisionCollectionSharder] = None,
        fused_params: Optional[Dict[str, Any]] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__(
            ebc_sharder
            or EmbeddingBagCollectionSharder(
                fused_params=fused_params, qcomm_codecs_registry=qcomm_codecs_registry
            ),
            mc_sharder or ManagedCollisionCollectionSharder(),
            qcomm_codecs_registry=qcomm_codecs_registry,
        )

    def shard(
        self,
        module: ManagedCollisionEmbeddingBagCollection,
        params: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        module_fqn: Optional[str] = None,
    ) -> ShardedManagedCollisionEmbeddingBagCollection:

        if device is None:
            device = torch.device("cuda")

        return ShardedManagedCollisionEmbeddingBagCollection(
            module,
            params,
            # pyre-ignore [6]
            ebc_sharder=self._e_sharder,
            mc_sharder=self._mc_sharder,
            env=env,
            device=device,
        )

    @property
    def module_type(self) -> Type[ManagedCollisionEmbeddingBagCollection]:
        return ManagedCollisionEmbeddingBagCollection
