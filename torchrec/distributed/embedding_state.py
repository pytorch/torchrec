#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch

import torch.distributed as dist
from torch import nn
from torch.distributed._shard.sharded_tensor import Shard
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.types import (
    EmbeddingModuleShardingPlan,
    ShardedTensor,
    ShardingType,
)
from torchrec.modules.embedding_configs import BaseEmbeddingConfig
from torchrec.optim.fused import EmptyFusedOptimizer


class ShardedEmbeddingModuleState(abc.ABC):
    _model_parallel_name_to_sharded_tensor: "OrderedDict[str, ShardedTensor]"
    _model_parallel_name_to_local_shards: "OrderedDict[str, List[Shard]]"

    @abc.abstractmethod
    def __init__(self) -> None:
        super().__init__()
        self._model_parallel_name_to_sharded_tensor = OrderedDict()
        self._model_parallel_name_to_local_shards = OrderedDict()

    @abc.abstractproperty
    def module_weight_key(self) -> str:
        ...

    def init_embedding_modules(
        self,
        module_sharding_plan: EmbeddingModuleShardingPlan,
        embedding_configs: Sequence[BaseEmbeddingConfig],
        sharding_types: Iterable[str],
        lookups: List[nn.Module],
        pg: Optional[dist.ProcessGroup],
    ) -> nn.ModuleDict:
        model_parallel_name_to_compute_kernel: Dict[str, str] = {}
        for (
            table_name,
            parameter_sharding,
        ) in module_sharding_plan.items():
            if parameter_sharding.sharding_type == ShardingType.DATA_PARALLEL.value:
                continue
            self._model_parallel_name_to_local_shards[table_name] = []
            model_parallel_name_to_compute_kernel[
                table_name
            ] = parameter_sharding.compute_kernel

        name_to_table_size = {}
        embeddings = nn.ModuleDict()

        for table in embedding_configs:
            embeddings[table.name] = nn.Module()
            name_to_table_size[table.name] = (
                table.num_embeddings,
                table.embedding_dim,
            )

        for sharding_type, lookup in zip(sharding_types, lookups):
            if sharding_type == ShardingType.DATA_PARALLEL.value:
                # unwrap DDP
                lookup = lookup.module
            else:
                # save local_shards for transforming MP params to shardedTensor
                for key, v in lookup.state_dict().items():
                    table_name = key[: -len(".weight")]
                    self._model_parallel_name_to_local_shards[table_name].extend(
                        v.local_shards()
                    )
            for (
                table_name,
                tbe_slice,
            ) in lookup.named_parameters_by_table():
                embeddings[table_name].register_parameter("weight", tbe_slice)

        for (
            table_name,
            local_shards,
        ) in self._model_parallel_name_to_local_shards.items():
            # for shards that don't exist on this rank, register with empty tensor
            if not hasattr(embeddings[table_name], "weight"):
                embeddings[table_name].register_parameter(
                    "weight", nn.Parameter(torch.empty(0))
                )
                if (
                    model_parallel_name_to_compute_kernel[table_name]
                    != EmbeddingComputeKernel.DENSE.value
                ):
                    embeddings[table_name].weight._in_backward_optimizers = [
                        EmptyFusedOptimizer()
                    ]
            # created ShardedTensors once in init, use in post_state_dict_hook
            self._model_parallel_name_to_sharded_tensor[
                table_name
            ] = ShardedTensor._init_from_local_shards(
                local_shards,
                name_to_table_size[table_name],
                process_group=pg,
            )

        return embeddings

    def construct_state_dict_key(
        self: "ShardedEmbeddingModuleState",
        prefix: str,
        table_name: str,
    ) -> str:
        return f"{prefix}{self.module_weight_key}.{table_name}.weight"

    @staticmethod
    def post_state_dict_hook(
        self: "ShardedEmbeddingModuleState",
        destination: Dict[str, torch.Tensor],
        prefix: str,
        _local_metadata: Dict[str, Any],
    ) -> None:
        # Adjust dense MP
        for (
            table_name,
            sharded_t,
        ) in self._model_parallel_name_to_sharded_tensor.items():
            destination_key = self.construct_state_dict_key(prefix, table_name)
            destination[destination_key] = sharded_t

    @staticmethod
    def pre_load_state_dict_hook(
        self: "ShardedEmbeddingModuleState",
        state_dict: Dict[str, Any],
        prefix: str,
        *args: Any,
    ) -> None:
        """
        Modify the destination state_dict for model parallel
        to transform from ShardedTensors into tensors
        """
        for (
            table_name,
            model_shards,
        ) in self._model_parallel_name_to_local_shards.items():
            key = self.construct_state_dict_key(prefix, table_name)
            # If state_dict[key] is already a ShardedTensor, use its local shards
            if isinstance(state_dict[key], ShardedTensor):
                local_shards = state_dict[key].local_shards()
                if len(local_shards) == 0:
                    state_dict[key] = torch.empty(0)
                else:
                    dim = state_dict[key].metadata().shards_metadata[0].shard_sizes[1]
                    # CW multiple shards are merged
                    if len(local_shards) > 1:
                        state_dict[key] = torch.cat(
                            [s.tensor.view(-1) for s in local_shards], dim=0
                        ).view(-1, dim)
                    else:
                        state_dict[key] = local_shards[0].tensor.view(-1, dim)
            elif isinstance(state_dict[key], torch.Tensor):
                local_shards = []
                for shard in model_shards:
                    # Extract shard size and offsets for splicing
                    shard_sizes = shard.metadata.shard_sizes
                    shard_offsets = shard.metadata.shard_offsets

                    # Prepare tensor by splicing and placing on appropriate device
                    spliced_tensor = state_dict[key][
                        shard_offsets[0] : shard_offsets[0] + shard_sizes[0],
                        shard_offsets[1] : shard_offsets[1] + shard_sizes[1],
                    ]

                    # Append spliced tensor into local shards
                    local_shards.append(spliced_tensor)

                state_dict[key] = (
                    torch.empty(0)
                    if not local_shards
                    else torch.cat(local_shards, dim=0)
                )
            else:
                raise RuntimeError(
                    f"Unexpected state_dict key type {type(state_dict[key])} found for {key}"
                )
