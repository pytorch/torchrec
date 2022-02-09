#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Dict, Optional, Type, Any

import torch
from torch import nn
from torchrec.distributed.embedding_sharding import (
    ListOfSparseFeaturesListAwaitable,
)
from torchrec.distributed.embedding_types import (
    SparseFeatures,
    EmbeddingComputeKernel,
    ListOfSparseFeaturesList,
)
from torchrec.distributed.embeddingbag import ShardedEmbeddingBagCollectionBase
from torchrec.distributed.types import (
    Awaitable,
    ParameterSharding,
    ParameterStorage,
    ShardingType,
    ShardedModuleContext,
    ModuleSharder,
    ShardingEnv,
)
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollectionInterface,
)
from torchrec.quant.embedding_modules import (
    EmbeddingBagCollection as QuantEmbeddingBagCollection,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class ShardedQuantEmbeddingBagCollection(
    ShardedEmbeddingBagCollectionBase[
        ListOfSparseFeaturesList,
        List[List[torch.Tensor]],
    ],
):
    """
    Sharded implementation of EmbeddingBagCollection.
    This is part of public API to allow for manual data dist pipelining.
    """

    def __init__(
        self,
        module: EmbeddingBagCollectionInterface,
        table_name_to_parameter_sharding: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(module, table_name_to_parameter_sharding, env, None, device)

    # pyre-ignore [3]
    def input_dist(
        self, ctx: ShardedModuleContext, features: KeyedJaggedTensor
    ) -> Awaitable[Any]:
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

    def _create_input_dist(
        self,
        input_feature_names: List[str],
        device: torch.device,
    ) -> None:
        feature_names: List[str] = []
        for sharding in self._sharding_type_to_sharding.values():
            self._input_dists.append(sharding.create_infer_input_dist())
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
            )

    def _create_lookups(
        self,
        fused_params: Optional[Dict[str, Any]],
    ) -> None:
        self._lookups = nn.ModuleList()
        for sharding in self._sharding_type_to_sharding.values():
            self._lookups.append(sharding.create_infer_lookup(fused_params))

    def _create_output_dist(self, device: Optional[torch.device] = None) -> None:
        for sharding in self._sharding_type_to_sharding.values():
            self._output_dists.append(sharding.create_infer_pooled_output_dist(device))
            self._embedding_names.extend(sharding.embedding_names())
            self._embedding_dims.extend(sharding.embedding_dims())


class QuantEmbeddingBagCollectionSharder(ModuleSharder[QuantEmbeddingBagCollection]):
    def shard(
        self,
        module: QuantEmbeddingBagCollection,
        params: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
    ) -> ShardedQuantEmbeddingBagCollection:
        return ShardedQuantEmbeddingBagCollection(module, params, env, device)

    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [ShardingType.DATA_PARALLEL.value, ShardingType.TABLE_WISE.value]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [
            EmbeddingComputeKernel.BATCHED_QUANT.value,
        ]

    def storage_usage(
        self, tensor: torch.Tensor, compute_device_type: str, compute_kernel: str
    ) -> Dict[str, int]:
        tensor_bytes = tensor.numel() * tensor.element_size() + tensor.shape[0] * 4
        assert compute_device_type in {"cuda", "cpu"}
        storage_map = {"cuda": ParameterStorage.HBM, "cpu": ParameterStorage.DDR}
        return {storage_map[compute_device_type].value: tensor_bytes}

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
