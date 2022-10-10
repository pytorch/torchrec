#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Type

import torch
from torch import nn
from torch.nn.modules.module import _IncompatibleKeys
from torchrec.distributed.embedding_sharding import (
    EmbeddingSharding,
    EmbeddingShardingInfo,
    EmptyShardingContext,
    ListOfSparseFeaturesListAwaitable,
)
from torchrec.distributed.embedding_types import (
    BaseQuantEmbeddingSharder,
    ListOfSparseFeaturesList,
    SparseFeatures,
    SparseFeaturesList,
)
from torchrec.distributed.embeddingbag import (
    create_sharding_infos_by_sharding,
    EmbeddingBagCollectionAwaitable,
)
from torchrec.distributed.sharding.tw_sharding import InferTwEmbeddingSharding
from torchrec.distributed.types import (
    Awaitable,
    EmptyShardedModuleContext,
    FeatureShardingMixIn,
    LazyAwaitable,
    ParameterSharding,
    ShardedModule,
    ShardingEnv,
    ShardingType,
)
from torchrec.distributed.utils import filter_state_dict
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


def create_infer_embedding_bag_sharding(
    sharding_type: str,
    sharding_infos: List[EmbeddingShardingInfo],
    env: ShardingEnv,
) -> EmbeddingSharding[
    EmptyShardingContext, SparseFeaturesList, List[torch.Tensor], torch.Tensor
]:
    if sharding_type == ShardingType.TABLE_WISE.value:
        return InferTwEmbeddingSharding(sharding_infos, env, device=None)
    else:
        raise ValueError(f"Sharding type not supported {sharding_type}")


class ShardedQuantEmbeddingBagCollection(
    ShardedModule[
        ListOfSparseFeaturesList,
        List[List[torch.Tensor]],
        KeyedTensor,
        EmptyShardedModuleContext,
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
                EmptyShardingContext,
                SparseFeaturesList,
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
        self._lookups: nn.ModuleList = nn.ModuleList()
        self._create_lookups(fused_params)
        self._output_dists: nn.ModuleList = nn.ModuleList()
        self._embedding_names: List[str] = []
        self._embedding_dims: List[int] = []
        self._feature_splits: List[int] = []
        self._features_order: List[int] = []

        # forward pass flow control
        self._has_uninitialized_input_dist: bool = True
        self._has_uninitialized_output_dist: bool = True
        self._has_features_permute: bool = True

    def _create_input_dist(
        self,
        input_feature_names: List[str],
        device: torch.device,
    ) -> None:
        feature_names: List[str] = []
        for sharding in self._sharding_type_to_sharding.values():
            self._input_dists.append(sharding.create_input_dist())
            feature_names.extend(
                sharding.id_score_list_feature_names()
                if self._is_weighted
                else sharding.id_list_feature_names()
            )
            self._feature_splits.append(
                len(
                    sharding.id_score_list_feature_names()
                    if self._is_weighted
                    else sharding.id_list_feature_names()
                )
            )

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
        self, ctx: EmptyShardedModuleContext, features: KeyedJaggedTensor
    ) -> Awaitable[ListOfSparseFeaturesList]:
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
            features_by_shards = features.split(
                self._feature_splits,
            )
            awaitables = [
                module(
                    SparseFeatures(
                        id_list_features=None
                        if self._is_weighted
                        else features_by_shard,
                        id_score_list_features=features_by_shard
                        if self._is_weighted
                        else None,
                    )
                ).wait()  # a dummy wait since now length indices comm is splited
                for module, features_by_shard in zip(
                    self._input_dists, features_by_shards
                )
            ]
            return ListOfSparseFeaturesListAwaitable(awaitables)

    def compute(
        self,
        ctx: EmptyShardedModuleContext,
        dist_input: ListOfSparseFeaturesList,
    ) -> List[List[torch.Tensor]]:
        return [lookup(features) for lookup, features in zip(self._lookups, dist_input)]

    def output_dist(
        self,
        ctx: EmptyShardedModuleContext,
        output: List[List[torch.Tensor]],
    ) -> LazyAwaitable[KeyedTensor]:
        return EmbeddingBagCollectionAwaitable(
            awaitables=[
                dist(embeddings) for dist, embeddings in zip(self._output_dists, output)
            ],
            embedding_dims=self._embedding_dims,
            embedding_names=self._embedding_names,
        )

    def compute_and_output_dist(
        self, ctx: EmptyShardedModuleContext, input: ListOfSparseFeaturesList
    ) -> LazyAwaitable[KeyedTensor]:
        return self.output_dist(ctx, self.compute(ctx, input))

    # pyre-fixme[14]: `state_dict` overrides method defined in `Module` inconsistently.
    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        if destination is None:
            destination = OrderedDict()
            # pyre-ignore [16]
            destination._metadata = OrderedDict()
        for lookup in self._lookups:
            lookup.state_dict(destination, prefix + "embedding_bags.", keep_vars)
        return destination

    # pyre-fixme[14]: `load_state_dict` overrides method defined in `Module`
    #  inconsistently.
    def load_state_dict(
        self,
        state_dict: "OrderedDict[str, torch.Tensor]",
        strict: bool = True,
    ) -> _IncompatibleKeys:
        missing_keys = []
        unexpected_keys = []
        for lookup in self._lookups:
            missing, unexpected = lookup.load_state_dict(
                filter_state_dict(state_dict, "embedding_bags"),
                strict,
            )
            missing_keys.extend(missing)
            unexpected_keys.extend(unexpected)
        return _IncompatibleKeys(
            missing_keys=missing_keys, unexpected_keys=unexpected_keys
        )

    def copy(self, device: torch.device) -> nn.Module:
        if self._has_uninitialized_output_dist:
            self._create_output_dist(device)
            self._has_uninitialized_output_dist = False
        return self

    @property
    def shardings(self) -> Dict[str, FeatureShardingMixIn]:
        # pyre-ignore [7]
        return self._sharding_type_to_sharding

    def create_context(self) -> EmptyShardedModuleContext:
        return EmptyShardedModuleContext()


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

    def shardable_parameters(
        self, module: QuantEmbeddingBagCollection
    ) -> Dict[str, nn.Parameter]:
        return {
            name.split(".")[-2]: param
            for name, param in module.state_dict().items()
            if name.endswith(".weight")
        }

    @property
    def module_type(self) -> Type[QuantEmbeddingBagCollection]:
        return QuantEmbeddingBagCollection
