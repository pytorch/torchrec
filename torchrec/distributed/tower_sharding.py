#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
from typing import Dict, TypeVar, Optional, Any, Type, List, Union, Iterator, Tuple, Set

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torchrec.distributed.comm import intra_and_cross_node_pg
from torchrec.distributed.dist_data import (
    PooledEmbeddingsAllToAll,
    PooledEmbeddingsAwaitable,
)
from torchrec.distributed.embedding import (
    ShardedEmbeddingCollection,
    EmbeddingCollectionSharder,
)
from torchrec.distributed.embedding_sharding import (
    SparseFeaturesAllToAll,
    SparseFeaturesListAwaitable,
)
from torchrec.distributed.embedding_types import (
    BaseEmbeddingSharder,
    SparseFeatures,
    SparseFeaturesList,
)
from torchrec.distributed.embeddingbag import (
    EmbeddingBagCollectionSharder,
    ShardedEmbeddingBagCollection,
)
from torchrec.distributed.types import (
    ParameterSharding,
    ShardingEnv,
    ShardedModule,
    Awaitable,
    ShardedModuleContext,
    ShardingType,
    LazyAwaitable,
)
from torchrec.distributed.utils import append_prefix
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)
from torchrec.modules.tower import EmbeddingTower
from torchrec.optim.fused import FusedOptimizerModule
from torchrec.optim.keyed import KeyedOptimizer, CombinedOptimizer
from torchrec.sparse.jagged_tensor import (
    KeyedJaggedTensor,
)

M = TypeVar("M", bound=nn.Module)


def _get_feature_names(
    module: Union[EmbeddingBagCollection, EmbeddingCollection]
) -> List[str]:
    if not (
        isinstance(module, EmbeddingBagCollection)
        or isinstance(module, EmbeddingCollection)
    ):
        raise RuntimeError(f"unsupported embedding type: {type(module)}")
    ret: List[str] = []
    configs = (
        module.embedding_bag_configs
        if isinstance(module, EmbeddingBagCollection)
        else module.embedding_configs
    )
    for config in configs:
        if not config.feature_names:
            ret.append(config.name)
        else:
            ret.extend(config.feature_names)
    return ret


class DenseOutputLazyAwaitable(LazyAwaitable[torch.Tensor]):
    def __init__(
        self,
        awaitable: PooledEmbeddingsAwaitable,
    ) -> None:
        super().__init__()
        self._awaitable = awaitable

    def _wait_impl(self) -> torch.Tensor:
        return self._awaitable.wait()


def _replace_sharding_with_intra_node(
    table_name_to_parameter_sharding: Dict[str, ParameterSharding], local_size: int
) -> None:
    for _, value in table_name_to_parameter_sharding.items():
        if value.sharding_type == ShardingType.TABLE_ROW_WISE.value:
            value.sharding_type = ShardingType.ROW_WISE.value
        elif value.sharding_type == ShardingType.TABLE_COLUMN_WISE.value:
            value.sharding_type = ShardingType.COLUMN_WISE.value
        else:
            raise ValueError(f"Sharding type not supported {value.sharding_type}")
        if value.ranks:
            value.ranks = [rank % local_size for rank in value.ranks]
        if value.sharding_spec:
            # pyre-ignore [6, 16]
            for (shard, rank) in zip(value.sharding_spec.shards, value.ranks):
                shard.placement._rank = rank


class ShardedEmbeddingTower(
    ShardedModule[
        SparseFeaturesList,
        torch.Tensor,
        torch.Tensor,
    ],
    FusedOptimizerModule,
):
    def __init__(
        self,
        module: EmbeddingTower,
        table_name_to_parameter_sharding: Dict[str, ParameterSharding],
        env: ShardingEnv,
        fused_params: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        intra_pg, cross_pg = intra_and_cross_node_pg(device)
        # pyre-ignore [11]
        self._intra_pg: Optional[dist.ProcessGroup] = intra_pg
        self._cross_pg: Optional[dist.ProcessGroup] = cross_pg
        self._device = device
        self._output_dist: Optional[PooledEmbeddingsAllToAll] = None
        self._cross_pg_global_batch_size: int = 0
        self._cross_pg_world_size: int = dist.get_world_size(self._cross_pg)

        self._has_uninitialized_output_dist = True

        # make sure all sharding on single physical node
        devices_per_node = dist.get_world_size(intra_pg)
        tower_devices = set()
        for sharding in table_name_to_parameter_sharding.values():
            # pyre-ignore [6]
            tower_devices.update(sharding.ranks)
        node = {tower_device // devices_per_node for tower_device in tower_devices}
        assert len(node) == 1
        self._tower_node: int = next(iter(node))
        self._active_device: bool = {dist.get_rank() // devices_per_node} == node

        # input_dist
        self._feature_names: List[str] = _get_feature_names(module.embedding)
        # pyre-ignore [8]
        self._is_weighted: bool = (
            False
            if isinstance(module.embedding, EmbeddingCollection)
            else module.embedding.is_weighted
        )
        self._has_uninitialized_input_dist: bool = True
        self._cross_dist: nn.Module = nn.Module()
        self._features_order: List[int] = []
        self._has_features_permute: bool = True

        self.embedding: Union[
            None, ShardedEmbeddingBagCollection, ShardedEmbeddingCollection
        ] = None
        self.interaction: Optional[nn.Module] = None

        if self._active_device:
            _replace_sharding_with_intra_node(
                table_name_to_parameter_sharding, dist.get_world_size(self._intra_pg)
            )
            intra_env: ShardingEnv = ShardingEnv(
                world_size=dist.get_world_size(self._intra_pg),
                rank=dist.get_rank(self._intra_pg),
                pg=self._intra_pg,
            )
            # shard embedding module
            if isinstance(module.embedding, EmbeddingBagCollection):
                self.embedding = ShardedEmbeddingBagCollection(
                    module.embedding,
                    table_name_to_parameter_sharding,
                    intra_env,
                    fused_params,
                    device,
                )
            elif isinstance(module.embedding, EmbeddingCollection):
                self.embedding = ShardedEmbeddingCollection(
                    module.embedding,
                    table_name_to_parameter_sharding,
                    intra_env,
                    fused_params,
                    device,
                )
            else:
                raise RuntimeError("The embedding module should be EBC or EC")

            # Hiearcherial DDP
            self.interaction = DistributedDataParallel(
                module=module.interaction.to(self._device),
                device_ids=[self._device],
                process_group=self._intra_pg,
                gradient_as_bucket_view=True,
                broadcast_buffers=False,
            )

    def _create_input_dist(
        self,
        input_feature_names: List[str],
        local_batch_size: int,
    ) -> None:
        self._cross_pg_global_batch_size = local_batch_size * self._cross_pg_world_size
        if self._feature_names == input_feature_names:
            self._has_features_permute = False
        else:
            for f in self._feature_names:
                self._features_order.append(input_feature_names.index(f))
            self.register_buffer(
                "_features_order_tensor",
                torch.tensor(
                    self._features_order, device=self._device, dtype=torch.int32
                ),
            )

        node_count = dist.get_world_size(self._cross_pg)
        features_per_node = [0 for _ in range(node_count)]
        features_per_node[self._tower_node] = len(self._feature_names)
        self._cross_dist = SparseFeaturesAllToAll(
            self._cross_pg,
            [0] * node_count if self._is_weighted else features_per_node,
            features_per_node if self._is_weighted else [0] * node_count,
            self._device,
        )

    # pyre-ignore [14]
    def input_dist(
        self, ctx: ShardedModuleContext, features: KeyedJaggedTensor
    ) -> Awaitable[SparseFeaturesList]:
        # return self.embedding.input_dist(ctx, features)
        if self._has_uninitialized_input_dist:
            self._create_input_dist(features.keys(), features.stride())
            self._has_uninitialized_input_dist = False
        with torch.no_grad():
            if self._has_features_permute:
                features = features.permute(
                    self._features_order,
                    # pyre-ignore [6]
                    self._features_order_tensor,
                )
                tensor_awaitable = self._cross_dist(
                    SparseFeatures(
                        id_list_features=None if self._is_weighted else features,
                        id_score_list_features=features if self._is_weighted else None,
                    )
                )
            return SparseFeaturesListAwaitable([tensor_awaitable.wait()])

    def compute(
        self, ctx: ShardedModuleContext, dist_input: SparseFeaturesList
    ) -> torch.Tensor:
        if self._active_device:
            features = (
                dist_input[0].id_score_list_features
                if self._is_weighted
                else dist_input[0].id_list_features
            )
            # pyre-ignore [29]
            embeddings = self.embedding(features)
            # pyre-ignore [29]
            output = self.interaction(embeddings)
        else:
            output = torch.empty(
                [self._cross_pg_global_batch_size, 0],
                device=self._device,
                requires_grad=True,
            )
        return output

    def _create_output_dist(
        self, ctx: ShardedModuleContext, output: torch.Tensor
    ) -> None:
        # Determine the output_dist splits and the all_to_all output size
        assert len(output.shape) == 2
        local_dim_sum = torch.tensor(
            [
                output.shape[1],
            ],
            dtype=torch.int64,
            device=self._device,
        )
        dim_sum_per_rank = [
            torch.zeros(
                1,
                dtype=torch.int64,
                device=self._device,
            )
            for i in range(dist.get_world_size(self._cross_pg))
        ]
        dist.all_gather(
            dim_sum_per_rank,
            local_dim_sum,
            group=self._cross_pg,
        )
        dim_sum_per_rank = [x.item() for x in dim_sum_per_rank]
        self._output_dist = PooledEmbeddingsAllToAll(
            pg=self._cross_pg, dim_sum_per_rank=dim_sum_per_rank, device=self._device
        )

    def output_dist(
        self, ctx: ShardedModuleContext, output: torch.Tensor
    ) -> LazyAwaitable[torch.Tensor]:
        if self._has_uninitialized_output_dist:
            self._create_output_dist(ctx, output)
            self._has_uninitialized_output_dist = False
        # pyre-ignore [29]
        return DenseOutputLazyAwaitable(self._output_dist(output))

    # pyre-ignore [14]
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
        if self._active_device:
            # pyre-ignore [16]
            self.embedding.state_dict(destination, prefix + "embedding.", keep_vars)
            # pyre-ignore [16]
            self.interaction.module.state_dict(
                destination, prefix + "interaction.", keep_vars
            )
        return destination

    @property
    def fused_optimizer(self) -> KeyedOptimizer:
        if self.embedding:
            return self.embedding.fused_optimizer
        else:
            return CombinedOptimizer([])

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        if self._active_device:
            # pyre-ignore[16]
            yield from self.embedding.named_parameters(
                append_prefix(prefix, "embedding"), recurse
            )
            # pyre-ignore[16]
            yield from self.interaction.module.named_parameters(
                append_prefix(prefix, "interaction"), recurse
            )
        else:
            yield from ()

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        if self._active_device:
            # pyre-ignore[16]
            yield from self.embedding.named_buffers(
                append_prefix(prefix, "embedding"), recurse
            )
            # pyre-ignore[16]
            yield from self.interaction.module.named_buffers(
                append_prefix(prefix, "interaction"), recurse
            )
        yield from ()

    def sparse_grad_parameter_names(
        self,
        destination: Optional[List[str]] = None,
        prefix: str = "",
    ) -> List[str]:
        destination = [] if destination is None else destination
        if self._active_device:
            # pyre-ignore[16]
            self.embedding.sparse_grad_parameter_names(
                destination, append_prefix(prefix, "embedding")
            )
        return destination

    def sharded_parameter_names(self, prefix: str = "") -> Iterator[str]:
        if self._active_device:
            # pyre-ignore[16]
            yield from self.embedding.sharded_parameter_names(
                append_prefix(prefix, "embedding")
            )
            # pyre-ignore[16]
            for name, _ in self.interaction.module.named_parameters(
                append_prefix(prefix, "interaction")
            ):
                yield name
        else:
            yield from ()

    def named_modules(
        self,
        memo: Optional[Set[nn.Module]] = None,
        prefix: str = "",
        remove_duplicate: bool = True,
    ) -> Iterator[Tuple[str, nn.Module]]:
        yield from [(prefix, self)]


class EmbeddingTowerSharder(BaseEmbeddingSharder[EmbeddingTower]):
    def __init__(self, fused_params: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(fused_params)

    def shard(
        self,
        module: EmbeddingTower,
        params: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
    ) -> ShardedEmbeddingTower:
        return ShardedEmbeddingTower(module, params, env, self.fused_params, device)

    def sharding_types(self, compute_device_type: str) -> List[str]:
        """
        List of supported sharding types. See ShardingType for well-known examples.
        """
        return [
            ShardingType.TABLE_ROW_WISE.value,
            # TABLE_COLUNN_WISE only works for pooled embedding, not sequence embedding
            ShardingType.TABLE_COLUMN_WISE.value,
        ]

    def shardable_parameters(self, module: EmbeddingTower) -> Dict[str, nn.Parameter]:
        """
        List of parameters, which can be sharded.
        """
        embedding_sharder = None
        if isinstance(module.embedding, EmbeddingBagCollection):
            embedding_sharder = EmbeddingBagCollectionSharder(self.fused_params)
        elif isinstance(module.embedding, EmbeddingCollection):
            embedding_sharder = EmbeddingCollectionSharder(self.fused_params)
        else:
            raise RuntimeError("unsupported embedding type")
        # pyre-ignore [6]
        return embedding_sharder.shardable_parameters(module.embedding)

    @property
    def module_type(self) -> Type[EmbeddingTower]:
        return EmbeddingTower
