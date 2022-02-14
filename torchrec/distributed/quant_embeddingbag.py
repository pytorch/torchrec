#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
from typing import List, Dict, Optional, Type, Any, Tuple

import torch
from torch import nn
from torch.nn.modules.module import _IncompatibleKeys
from torchrec.distributed.embedding_sharding import (
    ListOfSparseFeaturesListAwaitable,
    EmbeddingSharding,
)
from torchrec.distributed.embedding_types import (
    SparseFeatures,
    EmbeddingComputeKernel,
    ListOfSparseFeaturesList,
    SparseFeaturesList,
)
from torchrec.distributed.embeddingbag import (
    create_embedding_configs_by_sharding,
    replace_placement_with_meta_device,
    EmbeddingCollectionAwaitable,
    filter_state_dict,
)
from torchrec.distributed.sharding.tw_sharding import InferTwEmbeddingSharding
from torchrec.distributed.types import (
    Awaitable,
    ParameterSharding,
    ParameterStorage,
    ShardingType,
    ShardedModuleContext,
    ModuleSharder,
    ShardingEnv,
    ShardedModule,
    LazyAwaitable,
)
from torchrec.modules.embedding_configs import EmbeddingTableConfig
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollectionInterface,
)
from torchrec.quant.embedding_modules import (
    EmbeddingBagCollection as QuantEmbeddingBagCollection,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor


def create_infer_embedding_bag_sharding(
    sharding_type: str,
    embedding_configs: List[
        Tuple[EmbeddingTableConfig, ParameterSharding, torch.Tensor]
    ],
    env: ShardingEnv,
    device: Optional[torch.device] = None,
    permute_embeddings: bool = False,
) -> EmbeddingSharding[SparseFeaturesList, List[torch.Tensor]]:
    if device is not None and device.type == "meta":
        replace_placement_with_meta_device(embedding_configs)
    if sharding_type == ShardingType.TABLE_WISE.value:
        return InferTwEmbeddingSharding(embedding_configs, env, device)
    else:
        raise ValueError(f"Sharding type not supported {sharding_type}")


class ShardedQuantEmbeddingBagCollection(
    ShardedModule[ListOfSparseFeaturesList, List[List[torch.Tensor]], KeyedTensor],
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
        super().__init__()
        sharding_type_to_embedding_configs = create_embedding_configs_by_sharding(
            module, table_name_to_parameter_sharding, "embedding_bags."
        )
        self._sharding_type_to_sharding: Dict[
            str, EmbeddingSharding[SparseFeaturesList, List[torch.Tensor]]
        ] = {
            sharding_type: create_infer_embedding_bag_sharding(
                sharding_type, embedding_confings, env, device, permute_embeddings=True
            )
            for sharding_type, embedding_confings in sharding_type_to_embedding_configs.items()
        }

        self._is_weighted: bool = module.is_weighted
        self._device = device
        self._input_dists: nn.ModuleList = nn.ModuleList()
        self._lookups: nn.ModuleList = nn.ModuleList()
        self._create_lookups()
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
            )

    def _create_lookups(
        self,
    ) -> None:
        for sharding in self._sharding_type_to_sharding.values():
            self._lookups.append(sharding.create_lookup())

    def _create_output_dist(self, device: Optional[torch.device] = None) -> None:
        for sharding in self._sharding_type_to_sharding.values():
            self._output_dists.append(sharding.create_output_dist(device))
            self._embedding_names.extend(sharding.embedding_names())
            self._embedding_dims.extend(sharding.embedding_dims())

    # pyre-ignore [14]
    def input_dist(
        self, ctx: ShardedModuleContext, features: KeyedJaggedTensor
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
        ctx: ShardedModuleContext,
        dist_input: ListOfSparseFeaturesList,
    ) -> List[List[torch.Tensor]]:
        return [lookup(features) for lookup, features in zip(self._lookups, dist_input)]

    def output_dist(
        self,
        ctx: ShardedModuleContext,
        output: List[List[torch.Tensor]],
    ) -> LazyAwaitable[KeyedTensor]:
        if self._has_uninitialized_output_dist:
            self._create_output_dist(self._device)
            self._has_uninitialized_output_dist = False
        return EmbeddingCollectionAwaitable(
            awaitables=[
                dist(embeddings) for dist, embeddings in zip(self._output_dists, output)
            ],
            embedding_dims=self._embedding_dims,
            embedding_names=self._embedding_names,
        )

    def compute_and_output_dist(
        self, ctx: ShardedModuleContext, input: ListOfSparseFeaturesList
    ) -> LazyAwaitable[KeyedTensor]:
        if self._has_uninitialized_output_dist:
            self._create_output_dist(self._device)
            self._has_uninitialized_output_dist = False
        return EmbeddingCollectionAwaitable(
            awaitables=[
                dist(lookup(features))
                for lookup, dist, features in zip(
                    self._lookups,
                    self._output_dists,
                    input,
                )
            ],
            embedding_dims=self._embedding_dims,
            embedding_names=self._embedding_names,
        )

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
