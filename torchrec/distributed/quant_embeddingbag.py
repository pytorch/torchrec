#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Type

import torch
from torch import nn
from torchrec.distributed.embedding_sharding import (
    EmbeddingSharding,
    EmbeddingShardingInfo,
    ListOfKJTListSplitsAwaitable,
)
from torchrec.distributed.embedding_types import (
    BaseQuantEmbeddingSharder,
    FeatureShardingMixIn,
    KJTList,
    ListOfKJTList,
    ShardedEmbeddingModule,
)
from torchrec.distributed.embeddingbag import (
    create_sharding_infos_by_sharding,
    EmbeddingBagCollectionAwaitable,
)
from torchrec.distributed.sharding.tw_sharding import InferTwEmbeddingSharding
from torchrec.distributed.types import (
    Awaitable,
    LazyAwaitable,
    NullShardedModuleContext,
    NullShardingContext,
    ParameterSharding,
    ShardingEnv,
    ShardingType,
)
from torchrec.modules.embedding_configs import (
    data_type_to_sparse_type,
    dtype_to_data_type,
    EmbeddingBagConfig,
)
from torchrec.modules.embedding_modules import EmbeddingBagCollectionInterface
from torchrec.quant.embedding_modules import (
    EmbeddingBagCollection as QuantEmbeddingBagCollection,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor

torch.fx.wrap("len")


def create_infer_embedding_bag_sharding(
    sharding_type: str,
    sharding_infos: List[EmbeddingShardingInfo],
    env: ShardingEnv,
) -> EmbeddingSharding[NullShardingContext, KJTList, List[torch.Tensor], torch.Tensor]:
    if sharding_type == ShardingType.TABLE_WISE.value:
        return InferTwEmbeddingSharding(sharding_infos, env, device=None)
    else:
        raise ValueError(f"Sharding type not supported {sharding_type}")


class ShardedQuantEmbeddingBagCollection(
    ShardedEmbeddingModule[
        ListOfKJTList,
        List[List[torch.Tensor]],
        KeyedTensor,
        NullShardedModuleContext,
    ],
):
    """
    Sharded implementation of `EmbeddingBagCollection`.
    This is part of the public API to allow for manual data dist pipelining.
    """

    def __init__(
        self,
        module: EmbeddingBagCollectionInterface,
        table_name_to_parameter_sharding: Dict[str, ParameterSharding],
        env: ShardingEnv,
        fused_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self._embedding_bag_configs: List[
            EmbeddingBagConfig
        ] = module.embedding_bag_configs()
        sharding_type_to_sharding_infos = create_sharding_infos_by_sharding(
            module, table_name_to_parameter_sharding, "embedding_bags.", fused_params
        )
        self._sharding_type_to_sharding: Dict[
            str,
            EmbeddingSharding[
                NullShardingContext,
                KJTList,
                List[torch.Tensor],
                torch.Tensor,
            ],
        ] = {
            sharding_type: create_infer_embedding_bag_sharding(
                sharding_type, embedding_confings, env
            )
            for sharding_type, embedding_confings in sharding_type_to_sharding_infos.items()
        }

        self._is_weighted: bool = module.is_weighted()
        self._input_dists: List[nn.Module] = []
        self._lookups: List[nn.Module] = []
        self._create_lookups(fused_params)
        self._output_dists: List[nn.Module] = []
        self._embedding_names: List[str] = []
        self._embedding_dims: List[int] = []
        self._feature_splits: List[int] = []
        self._features_order: List[int] = []

        # forward pass flow control
        self._has_uninitialized_input_dist: bool = True
        self._has_uninitialized_output_dist: bool = True
        self._has_features_permute: bool = True

        # This provides consistency between this class and the EmbeddingBagCollection's
        # nn.Module API calls (state_dict, named_modules, etc)
        # Currently, Sharded Quant EBC only uses TW sharding, and returns non-sharded tensors as part of state dict
        # TODO - revisit if we state_dict can be represented as sharded tensor
        self.embedding_bags: nn.ModuleDict = nn.ModuleDict()
        for table in self._embedding_bag_configs:
            self.embedding_bags[table.name] = torch.nn.Module()

        for _sharding_type, lookup in zip(
            self._sharding_type_to_sharding.keys(), self._lookups
        ):
            lookup_state_dict = lookup.state_dict()
            for key in lookup_state_dict:
                if not key.endswith(".weight"):
                    continue
                table_name = key[: -len(".weight")]
                # Register as buffer because this is an inference model, and can potentially use uint8 types.
                self.embedding_bags[table_name].register_buffer(
                    "weight", lookup_state_dict[key]
                )

    def _create_input_dist(
        self,
        input_feature_names: List[str],
        device: torch.device,
    ) -> None:
        feature_names: List[str] = []
        for sharding in self._sharding_type_to_sharding.values():
            self._input_dists.append(sharding.create_input_dist())
            feature_names.extend(sharding.feature_names())
            self._feature_splits.append(len(sharding.feature_names()))

        if feature_names == input_feature_names:
            self._has_features_permute = False
        else:
            for f in feature_names:
                self._features_order.append(input_feature_names.index(f))
            self.register_buffer(
                "_features_order_tensor",
                torch.tensor(self._features_order, device=device, dtype=torch.int32),
                persistent=False,
            )

    def _create_lookups(
        self,
        fused_params: Optional[Dict[str, Any]],
    ) -> None:
        for sharding in self._sharding_type_to_sharding.values():
            self._lookups.append(sharding.create_lookup(fused_params=fused_params))

    def _create_output_dist(self, device: Optional[torch.device] = None) -> None:
        for sharding in self._sharding_type_to_sharding.values():
            self._output_dists.append(sharding.create_output_dist(device))
            self._embedding_names.extend(sharding.embedding_names())
            self._embedding_dims.extend(sharding.embedding_dims())

    # pyre-ignore [14]
    def input_dist(
        self, ctx: NullShardedModuleContext, features: KeyedJaggedTensor
    ) -> Awaitable[Awaitable[ListOfKJTList]]:
        if self._has_uninitialized_input_dist:
            self._create_input_dist(features.keys(), features.device())
            self._has_uninitialized_input_dist = False
        if self._has_uninitialized_output_dist:
            self._create_output_dist(features.device())
            self._has_uninitialized_output_dist = False
        with torch.no_grad():
            if self._has_features_permute:
                features = features.permute(
                    self._features_order,
                    # pyre-ignore [6]
                    self._features_order_tensor,
                )
            features_by_shards = features.split(self._feature_splits)
            awaitables = [
                self._input_dists[i].forward(features_by_shards[i])
                for i in range(len(self._input_dists))
            ]
            return ListOfKJTListSplitsAwaitable(awaitables)

    def compute(
        self,
        ctx: NullShardedModuleContext,
        dist_input: ListOfKJTList,
    ) -> List[List[torch.Tensor]]:
        # syntax for torchscript
        return [lookup.forward(dist_input[i]) for i, lookup in enumerate(self._lookups)]

    def output_dist(
        self,
        ctx: NullShardedModuleContext,
        output: List[List[torch.Tensor]],
    ) -> LazyAwaitable[KeyedTensor]:
        return EmbeddingBagCollectionAwaitable(
            # syntax for torchscript
            awaitables=[
                dist.forward(output[i]) for i, dist in enumerate(self._output_dists)
            ],
            embedding_dims=self._embedding_dims,
            embedding_names=self._embedding_names,
        )

    def compute_and_output_dist(
        self, ctx: NullShardedModuleContext, input: ListOfKJTList
    ) -> LazyAwaitable[KeyedTensor]:
        return self.output_dist(ctx, self.compute(ctx, input))

    def copy(self, device: torch.device) -> nn.Module:
        if self._has_uninitialized_output_dist:
            self._create_output_dist(device)
            self._has_uninitialized_output_dist = False
        return super().copy(device)

    @property
    def shardings(self) -> Dict[str, FeatureShardingMixIn]:
        # pyre-ignore [7]
        return self._sharding_type_to_sharding

    def create_context(self) -> NullShardedModuleContext:
        return NullShardedModuleContext()


class QuantEmbeddingBagCollectionSharder(
    BaseQuantEmbeddingSharder[QuantEmbeddingBagCollection]
):
    def shard(
        self,
        module: QuantEmbeddingBagCollection,
        params: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
    ) -> ShardedQuantEmbeddingBagCollection:
        fused_params = self.fused_params if self.fused_params else {}
        fused_params["output_dtype"] = data_type_to_sparse_type(
            dtype_to_data_type(module.output_dtype())
        )
        return ShardedQuantEmbeddingBagCollection(module, params, env, fused_params)

    @property
    def module_type(self) -> Type[QuantEmbeddingBagCollection]:
        return QuantEmbeddingBagCollection
