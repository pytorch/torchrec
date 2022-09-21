#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
from typing import Dict, List, Optional, Type

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torchrec.distributed.embedding_sharding import (
    EmbeddingSharding,
    EmptyShardingContext,
    SparseFeaturesListAwaitable,
)
from torchrec.distributed.embedding_types import (
    BaseEmbeddingSharder,
    EmbeddingComputeKernel,
    SparseFeatures,
    SparseFeaturesList,
)

from torchrec.distributed.embeddingbag import (
    _check_need_pos,
    create_embedding_bag_sharding,
    create_sharding_infos_by_sharding,
    EmbeddingBagCollectionAwaitable,
)
from torchrec.distributed.sharding.dp_sharding import DpPooledEmbeddingSharding
from torchrec.distributed.types import (
    Awaitable,
    EmptyShardedModuleContext,
    LazyAwaitable,
    ParameterSharding,
    QuantizedCommCodecs,
    ShardedModule,
    ShardedTensor,
    ShardingEnv,
    ShardingType,
)
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingBagCollectionInterface,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor


class ShardedEmbeddingBagCollection(
    ShardedModule[
        SparseFeaturesList, List[torch.Tensor], KeyedTensor, EmptyShardedModuleContext
    ],
):
    """
    Sharded implementation of EmbeddingBagCollection. This version decouples compute kernel from sharding.
    This is part of the public API to allow for manual data dist pipelining.
    """

    def __init__(
        self,
        module: EmbeddingBagCollectionInterface,
        table_name_to_parameter_sharding: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__()

        self._embedding_bag_configs: List[
            EmbeddingBagConfig
        ] = module.embedding_bag_configs()

        self._table_name_to_parameter_sharding = table_name_to_parameter_sharding
        self._env = env

        fqn_to_parameter = {}
        state_dict = module.state_dict()
        for fqn, parameter in module.named_parameters():
            fqn_to_parameter[fqn] = parameter

        # Users are no longer exposed to Compute Kernel type, instead we infer it per parameter
        # If a parameter has _optimizer applied onto it, and it is supported, and not data_parallel", use FUSED kernel,
        # otherwise, (if data_parallel)
        for table_name, parameter_sharding in table_name_to_parameter_sharding.items():
            param_name = f"embedding_bags.{table_name}.weight"
            param = fqn_to_parameter.get(param_name, state_dict[param_name])
            if hasattr(param, "_optimizer_class") and (
                parameter_sharding.sharding_type != ShardingType.DATA_PARALLEL.value
            ):
                # TODO check that this optimizer is supported
                parameter_sharding.compute_kernel = EmbeddingComputeKernel.FUSED.value
            else:
                parameter_sharding.compute_kernel = EmbeddingComputeKernel.DENSE.value

        sharding_type_to_sharding_infos = create_sharding_infos_by_sharding(
            module, table_name_to_parameter_sharding, "embedding_bags.", None
        )
        need_pos = _check_need_pos(module)
        self._sharding_type_to_sharding: Dict[
            str,
            EmbeddingSharding[
                EmptyShardingContext, SparseFeatures, torch.Tensor, torch.Tensor
            ],
        ] = {
            sharding_type: create_embedding_bag_sharding(
                sharding_type,
                embedding_configs,
                env,
                device,
                permute_embeddings=True,
                need_pos=need_pos,
                qcomm_codecs_registry=self.qcomm_codecs_registry,
            )
            for sharding_type, embedding_configs in sharding_type_to_sharding_infos.items()
        }

        self._is_weighted: bool = module.is_weighted()
        self._device = device
        self._input_dists: List[nn.Module] = []
        self._lookups: List[nn.Module] = []
        self._create_lookups()
        self._output_dists: List[nn.Module] = []
        self._embedding_names: List[str] = []
        self._embedding_dims: List[int] = []
        self._feature_splits: List[int] = []
        self._features_order: List[int] = []
        # to support the FP16 hook
        self._create_output_dist()

        # forward pass flow control
        self._has_uninitialized_input_dist: bool = True
        self._has_features_permute: bool = True

        for index, (sharding, lookup, sharding_infos) in enumerate(
            zip(
                self._sharding_type_to_sharding.values(),
                self._lookups,
                sharding_type_to_sharding_infos.values(),
            )
        ):
            if isinstance(sharding, DpPooledEmbeddingSharding):
                # pyre-fixme[28]: Unexpected keyword argument `gradient_as_bucket_view`.
                self._lookups[index] = DistributedDataParallel(
                    module=lookup,
                    device_ids=[device],
                    process_group=env.process_group,
                    gradient_as_bucket_view=True,
                    broadcast_buffers=False,
                    static_graph=True,
                )

                optimizer_class = None
                optimizer_kwargs = {}
                for sharding_info in sharding_infos:
                    unsharded_param = sharding_info.param
                    # TODO have to checks to ensure that everything in this group has the same _optimizer_class and kwargs
                    optimizer_class = getattr(unsharded_param, "_optimizer_class", None)
                    optimizer_kwargs = getattr(
                        unsharded_param, "_optimizer_kwargs", None
                    )
                    break

                # TODO get rid of below once supported, we should apply these directly on top of lookup as apply_overlapped_optimizer
                # DDP will handle the rest
                if optimizer_class is not None:
                    # pyre-ignore
                    self._lookups[index]._register_fused_optim(
                        optimizer_class, **optimizer_kwargs
                    )

        self._initialize_torch_state()

    def _initialize_torch_state(self) -> None:
        model_parallel_name_to_local_shards = OrderedDict()
        for (
            table_name,
            parameter_sharding,
        ) in self._table_name_to_parameter_sharding.items():
            if parameter_sharding.sharding_type == ShardingType.DATA_PARALLEL.value:
                continue
            model_parallel_name_to_local_shards[table_name] = []

        data_parallel_table_name_to_state = {}
        for sharding_type, lookup in zip(
            self._sharding_type_to_sharding.keys(), self._lookups
        ):
            lookup_state_dict = lookup.state_dict()
            for key in lookup_state_dict:
                assert key.endswith(".weight")
                key_without_weight = key[: -len(".weight")]
                if sharding_type == ShardingType.DATA_PARALLEL.value:
                    data_parallel_table_name_to_state[
                        key_without_weight
                    ] = lookup_state_dict[key]
                elif key_without_weight in model_parallel_name_to_local_shards:
                    lookup_sharded_tensor = lookup_state_dict[key]
                    model_parallel_name_to_local_shards[key_without_weight].extend(
                        lookup_sharded_tensor.local_shards()
                    )

        name_to_table_size = {}
        # This provides consistency between this class and the EmbeddingBagCollection's
        # nn.Module API calls (state_dict, named_modules, etc)
        self.embedding_bags: nn.ModuleDict = nn.ModuleDict()
        for table in self._embedding_bag_configs:
            name_to_table_size[table.name] = (table.num_embeddings, table.embedding_dim)
            self.embedding_bags[table.name] = torch.nn.Module()

        for table_name, local_shards in model_parallel_name_to_local_shards.items():
            if (
                self._table_name_to_parameter_sharding[table_name].sharding_type
                != ShardingType.DATA_PARALLEL.value
            ):
                weight = ShardedTensor._init_from_local_shards(
                    local_shards,
                    name_to_table_size[table_name],
                    process_group=self._env.process_group,
                )
                # TODO - ensure that optimizers over named_parameters works
                self.embedding_bags[table_name].register_parameter(
                    "weight", torch.nn.Parameter(weight)
                )

        # Register data_parallel state as regular tensors
        for table_name, table_state in data_parallel_table_name_to_state.items():
            # Because DDP isn't composable, its named_parameters have a module. prefix, so we need to get rid of it here
            table_name = table_name.lstrip("module.")
            self.embedding_bags[table_name].register_parameter(
                "weight", torch.nn.Parameter(table_state)
            )

    def _create_input_dist(
        self,
        input_feature_names: List[str],
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
                torch.tensor(
                    self._features_order, device=self._device, dtype=torch.int32
                ),
                persistent=False,
            )

    def _create_lookups(
        self,
    ) -> None:
        for sharding in self._sharding_type_to_sharding.values():
            self._lookups.append(sharding.create_lookup())

    def _create_output_dist(self) -> None:
        for sharding in self._sharding_type_to_sharding.values():
            self._output_dists.append(sharding.create_output_dist(device=self._device))
            self._embedding_names.extend(sharding.embedding_names())
            self._embedding_dims.extend(sharding.embedding_dims())

    # pyre-ignore [14]
    def input_dist(
        self, ctx: EmptyShardedModuleContext, features: KeyedJaggedTensor
    ) -> Awaitable[SparseFeaturesList]:
        if self._has_uninitialized_input_dist:
            self._create_input_dist(features.keys())
            self._has_uninitialized_input_dist = False
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
            awaitables = []
            for module, features_by_shard in zip(self._input_dists, features_by_shards):
                all2all_lengths = module(
                    SparseFeatures(
                        id_list_features=None
                        if self._is_weighted
                        else features_by_shard,
                        id_score_list_features=features_by_shard
                        if self._is_weighted
                        else None,
                    )
                )
                awaitables.append(all2all_lengths.wait())
            return SparseFeaturesListAwaitable(awaitables)

    def compute(
        self,
        ctx: EmptyShardedModuleContext,
        dist_input: SparseFeaturesList,
    ) -> List[torch.Tensor]:
        return [lookup(features) for lookup, features in zip(self._lookups, dist_input)]

    def output_dist(
        self,
        ctx: EmptyShardedModuleContext,
        output: List[torch.Tensor],
    ) -> LazyAwaitable[KeyedTensor]:
        return EmbeddingBagCollectionAwaitable(
            awaitables=[
                dist(embeddings) for dist, embeddings in zip(self._output_dists, output)
            ],
            embedding_dims=self._embedding_dims,
            embedding_names=self._embedding_names,
        )

    def compute_and_output_dist(
        self, ctx: EmptyShardedModuleContext, input: SparseFeaturesList
    ) -> LazyAwaitable[KeyedTensor]:
        return EmbeddingBagCollectionAwaitable(
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

    def create_context(self) -> EmptyShardedModuleContext:
        return EmptyShardedModuleContext()


class EmbeddingBagCollectionSharder(BaseEmbeddingSharder[EmbeddingBagCollection]):
    """
    Experimental, composable version of ShardedEmbeddingBagCollection and EmbeddingBagCollectionSharder
    with new APIs.

    This implementation uses EmbeddingBagCollection
    """

    def __init__(
        self,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__(qcomm_codecs_registry=qcomm_codecs_registry)

    def shard(
        self,
        module: EmbeddingBagCollection,
        params: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
    ) -> ShardedEmbeddingBagCollection:

        return ShardedEmbeddingBagCollection(
            module=module,
            table_name_to_parameter_sharding=params,
            env=env,
            device=device,
            qcomm_codecs_registry=self.qcomm_codecs_registry,
        )

    def shardable_parameters(
        self, module: EmbeddingBagCollection
    ) -> Dict[str, nn.Parameter]:
        return {
            name.split(".")[0]: param
            for name, param in module.embedding_bags.named_parameters()
        }

    @property
    def module_type(self) -> Type[EmbeddingBagCollection]:
        return EmbeddingBagCollection
