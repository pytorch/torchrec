#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Type, Union

import torch
from torch import nn

from torchrec.distributed.embedding_types import (
    BaseEmbeddingSharder,
    EmbeddingComputeKernel,
)
from torchrec.distributed.modular_embeddingbag import ShardedEmbeddingBagCollection
from torchrec.distributed.sharding.dp_sharding import DpPooledEmbeddingSharding
from torchrec.distributed.types import (
    ParameterSharding,
    QuantizedCommCodecs,
    ShardedTensor,
    ShardingEnv,
    ShardingType,
)
from torchrec.modules.fused_embedding_modules import (
    convert_optimizer_type_and_kwargs,
    FusedEmbeddingBagCollection,
)
from torchrec.optim.fused import FusedOptimizerModule
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizer


class ShardedFusedEmbeddingBagCollection(
    ShardedEmbeddingBagCollection,
    FusedOptimizerModule,
):
    def __init__(
        self,
        module: FusedEmbeddingBagCollection,
        table_name_to_parameter_sharding: Dict[str, ParameterSharding],
        env: ShardingEnv,
        fused_params: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:

        optimizer_type = module.optimizer_type()
        optimizer_kwargs = module.optimizer_kwargs()

        fused_params = {}
        emb_opt_type_and_kwargs = convert_optimizer_type_and_kwargs(
            optimizer_type, optimizer_kwargs
        )
        assert emb_opt_type_and_kwargs is not None
        (emb_optim_type, emb_opt_kwargs) = emb_opt_type_and_kwargs

        fused_params["optimizer"] = emb_optim_type
        fused_params.update(emb_opt_kwargs)

        super().__init__(
            module=module,
            table_name_to_parameter_sharding=table_name_to_parameter_sharding,
            env=env,
            fused_params=fused_params,
            device=device,
            qcomm_codecs_registry=qcomm_codecs_registry,
        )

        for index, (sharding, _) in enumerate(
            zip(self._sharding_type_to_sharding.values(), self._lookups)
        ):
            if isinstance(sharding, DpPooledEmbeddingSharding):
                # pyre-ignore
                self._lookups[index]._register_fused_optim(
                    optimizer_type, **optimizer_kwargs
                )
                # TODO - We need a way to get this optimizer back (and add to optims) so it
                # can be checkpointed.
                # We need to ensure that a checkpoint from DDP and a checkpoint from a
                # model parallel version are compatible.

        # Get all fused optimizers and combine them.
        optims = []
        for lookup in self._lookups:
            for _, module in lookup.named_modules():
                if isinstance(module, FusedOptimizerModule):
                    # modify param keys to match EmbeddingBagCollection
                    params: Dict[str, Union[torch.Tensor, ShardedTensor]] = {}
                    for param_key, weight in module.fused_optimizer.params.items():
                        params["embedding_bags." + param_key] = weight
                    module.fused_optimizer.params = params
                    optims.append(("", module.fused_optimizer))
        self._optim: CombinedOptimizer = CombinedOptimizer(optims)

    def fused_optimizer(self) -> KeyedOptimizer:
        return self._optim


class FusedEmbeddingBagCollectionSharder(
    BaseEmbeddingSharder[FusedEmbeddingBagCollection]
):
    def shard(
        self,
        module: FusedEmbeddingBagCollection,
        params: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
    ) -> ShardedEmbeddingBagCollection:

        return ShardedFusedEmbeddingBagCollection(
            module,
            params,
            env,
            device=device,
            qcomm_codecs_registry=self.qcomm_codecs_registry,
        )

    def shardable_parameters(
        self, module: FusedEmbeddingBagCollection
    ) -> Dict[str, nn.Parameter]:
        return {
            name.split(".")[0]: param
            for name, param in module.embedding_bags.named_parameters()
        }

    @property
    def module_type(self) -> Type[FusedEmbeddingBagCollection]:
        return FusedEmbeddingBagCollection

    def sharding_types(self, compute_device_type: str) -> List[str]:
        types = [
            ShardingType.DATA_PARALLEL.value,
            ShardingType.TABLE_WISE.value,
            ShardingType.COLUMN_WISE.value,
            ShardingType.TABLE_COLUMN_WISE.value,
        ]
        if compute_device_type in {"cuda"}:
            types += [
                ShardingType.ROW_WISE.value,
                ShardingType.TABLE_ROW_WISE.value,
            ]

        return types

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        # Need to get rid of this, this should be placement only
        ret = []
        if sharding_type != ShardingType.DATA_PARALLEL.value:
            ret += [
                EmbeddingComputeKernel.FUSED.value,
            ]
            if compute_device_type in {"cuda"}:
                ret += [
                    EmbeddingComputeKernel.FUSED_UVM.value,
                    EmbeddingComputeKernel.FUSED_UVM_CACHING.value,
                ]
        else:
            ret.append(EmbeddingComputeKernel.DENSE.value)
        return ret
