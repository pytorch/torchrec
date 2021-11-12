#!/usr/bin/env python3

from typing import (
    List,
    Dict,
    Optional,
    Type,
    Any,
    TypeVar,
)

import torch
from torch import nn
from torchrec.distributed.embedding_types import (
    BaseEmbeddingSharder,
    SparseFeaturesList,
)
from torchrec.distributed.types import (
    Awaitable,
    LazyAwaitable,
    ParameterSharding,
    ShardedModule,
    ShardedModuleContext,
    ShardingEnv,
)
from torchrec.modules.embedding_modules import (
    EmbeddingCollection,
)
from torchrec.optim.fused import FusedOptimizerModule
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor


class ShardedEmbeddingCollection(
    ShardedModule[
        SparseFeaturesList,
        List[torch.Tensor],
        KeyedTensor,
    ],
    FusedOptimizerModule,
):
    """
    Sharded implementation of EmbeddingCollection.
    This is part of public API to allow for manual data dist pipelining.
    """

    def __init__(
        self,
        module: EmbeddingCollection,
        table_name_to_parameter_sharding: Dict[str, ParameterSharding],
        env: ShardingEnv,
        fused_params: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

    # pyre-ignore [14]
    def input_dist(
        self, ctx: ShardedModuleContext, features: KeyedJaggedTensor
    ) -> Awaitable[SparseFeaturesList]:
        # pyre-ignore [7]
        pass

    def compute(
        self, ctx: ShardedModuleContext, dist_input: SparseFeaturesList
    ) -> List[torch.Tensor]:
        # pyre-ignore [7]
        pass

    def output_dist(
        self, ctx: ShardedModuleContext, output: List[torch.Tensor]
    ) -> LazyAwaitable[KeyedTensor]:
        # pyre-ignore [7]
        pass


M = TypeVar("M", bound=nn.Module)


class EmbeddingCollectionSharder(BaseEmbeddingSharder[M]):
    """
    This implementation uses non-fused EmbeddingCollection
    """

    def shard(
        self,
        module: EmbeddingCollection,
        params: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
    ) -> ShardedEmbeddingCollection:
        return ShardedEmbeddingCollection(
            module, params, env, self.fused_params, device
        )

    def shardable_parameters(
        self, module: EmbeddingCollection
    ) -> Dict[str, nn.Parameter]:
        return {}

    @property
    def module_type(self) -> Type[EmbeddingCollection]:
        return EmbeddingCollection
