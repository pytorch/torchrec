#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torchrec
from torch import quantization as quant
from torchrec import EmbeddingCollection, EmbeddingConfig, KeyedJaggedTensor
from torchrec.distributed.embedding_types import ModuleSharder
from torchrec.distributed.fused_params import FUSED_PARAM_REGISTER_TBE_BOOL
from torchrec.distributed.planner import Topology
from torchrec.distributed.quant_embedding import (
    QuantEmbeddingCollectionSharder,
    ShardedQuantEmbeddingCollection,
)
from torchrec.distributed.quant_embeddingbag import (
    QuantEmbeddingBagCollection,
    QuantEmbeddingBagCollectionSharder,
    ShardedQuantEmbeddingBagCollection,
)

from torchrec.distributed.test_utils.test_model import ModelInput
from torchrec.distributed.types import ParameterSharding, ShardingEnv
from torchrec.distributed.utils import CopyableMixin
from torchrec.modules.embedding_configs import (
    data_type_to_sparse_type,
    dtype_to_data_type,
    EmbeddingBagConfig,
)
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.quant.embedding_modules import (
    EmbeddingCollection as QuantEmbeddingCollection,
    MODULE_ATTR_REGISTER_TBES_BOOL,
)


@dataclass
class TestModelInfo:
    sparse_device: torch.device
    dense_device: torch.device
    num_features: int
    num_float_features: int
    num_weighted_features: int
    tables: Union[List[EmbeddingBagConfig], List[EmbeddingConfig]] = field(
        default_factory=list
    )
    weighted_tables: List[EmbeddingBagConfig] = field(default_factory=list)
    model: torch.nn.Module = torch.nn.Module()
    quant_model: torch.nn.Module = torch.nn.Module()
    sharders: List[ModuleSharder] = field(default_factory=list)
    topology: Optional[Topology] = None


def prep_inputs(
    model_info: TestModelInfo, world_size: int, batch_size: int = 1, count: int = 5
) -> List[ModelInput]:
    inputs = []
    for _ in range(count):
        inputs.append(
            ModelInput.generate(
                batch_size=batch_size,
                world_size=world_size,
                num_float_features=model_info.num_float_features,
                tables=model_info.tables,
                weighted_tables=model_info.weighted_tables,
            )[1][0],
        )
    return inputs


def prep_inputs_multiprocess(
    model_info: TestModelInfo, world_size: int, batch_size: int = 1, count: int = 5
) -> List[Tuple[ModelInput, List[ModelInput]]]:
    inputs = []
    for _ in range(count):
        inputs.append(
            ModelInput.generate(
                batch_size=batch_size,
                world_size=world_size,
                num_float_features=model_info.num_float_features,
                tables=model_info.tables,
                weighted_tables=model_info.weighted_tables,
            )
        )
    return inputs


def model_input_to_forward_args_kjt(
    mi: ModelInput,
) -> Tuple[List[str], torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    kjt = mi.idlist_features
    return (
        kjt._keys,
        kjt._values,
        kjt._lengths,
        kjt._offsets,
    )


# We want to be torch types bound, args for TorchTypesModelInputWrapper
def model_input_to_forward_args(
    mi: ModelInput,
) -> Tuple[
    torch.Tensor,
    List[str],
    torch.Tensor,
    List[str],
    torch.Tensor,
    Optional[torch.Tensor],
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    idlist_kjt = mi.idlist_features
    idscore_kjt = mi.idscore_features
    assert idscore_kjt is not None
    return (
        mi.float_features,
        idlist_kjt._keys,
        idlist_kjt._values,
        idscore_kjt._keys,
        idscore_kjt._values,
        idscore_kjt._weights,
        mi.label,
        idlist_kjt._lengths,
        idlist_kjt._offsets,
        idscore_kjt._lengths,
        idscore_kjt._offsets,
    )


def quantize(
    module: torch.nn.Module,
    inplace: bool,
    output_type: torch.dtype = torch.float,
    register_tbes: bool = False,
) -> torch.nn.Module:
    if register_tbes:
        for m in module.modules():
            if isinstance(
                m, torchrec.modules.embedding_modules.EmbeddingBagCollection
            ) or isinstance(m, torchrec.modules.embedding_modules.EmbeddingCollection):
                setattr(m, MODULE_ATTR_REGISTER_TBES_BOOL, True)

    qconfig = quant.QConfig(
        activation=quant.PlaceholderObserver.with_args(dtype=output_type),
        weight=quant.PlaceholderObserver.with_args(dtype=torch.qint8),
    )
    return quant.quantize_dynamic(
        module,
        qconfig_spec={
            EmbeddingBagCollection: qconfig,
            EmbeddingCollection: qconfig,
        },
        mapping={
            EmbeddingBagCollection: QuantEmbeddingBagCollection,
            EmbeddingCollection: QuantEmbeddingCollection,
        },
        inplace=inplace,
    )


class TestQuantEBCSharder(QuantEmbeddingBagCollectionSharder):
    def __init__(
        self,
        sharding_type: str,
        kernel_type: str,
        fused_params: Optional[Dict[str, Any]] = None,
        shardable_params: Optional[List[str]] = None,
    ) -> None:
        super().__init__(fused_params=fused_params, shardable_params=shardable_params)
        self._sharding_type = sharding_type
        self._kernel_type = kernel_type

    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [self._sharding_type]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [self._kernel_type]

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
        fused_params[FUSED_PARAM_REGISTER_TBE_BOOL] = True
        return ShardedQuantEmbeddingBagCollection(
            module=module,
            table_name_to_parameter_sharding=params,
            env=env,
            fused_params=fused_params,
        )


class TestQuantECSharder(QuantEmbeddingCollectionSharder):
    def __init__(self, sharding_type: str, kernel_type: str) -> None:
        super().__init__()
        self._sharding_type = sharding_type
        self._kernel_type = kernel_type

    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [self._sharding_type]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [self._kernel_type]

    def shard(
        self,
        module: QuantEmbeddingCollection,
        params: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
    ) -> ShardedQuantEmbeddingCollection:
        fused_params = self.fused_params if self.fused_params else {}
        fused_params["output_dtype"] = data_type_to_sparse_type(
            dtype_to_data_type(module.output_dtype())
        )
        fused_params[FUSED_PARAM_REGISTER_TBE_BOOL] = True
        return ShardedQuantEmbeddingCollection(module, params, env, fused_params)


class KJTInputWrapper(torch.nn.Module):
    def __init__(
        self,
        module_kjt_input: torch.nn.Module,
    ) -> None:
        super().__init__()
        self._module_kjt_input = module_kjt_input
        self.add_module("_module_kjt_input", self._module_kjt_input)

    # pyre-ignore
    def forward(
        self,
        keys: List[str],
        values: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
    ):
        kjt = KeyedJaggedTensor(
            keys=keys,
            values=values,
            lengths=lengths,
            offsets=offsets,
        )
        return self._module_kjt_input(kjt)


# Wrapper for module that accepts ModelInput to avoid jit scripting of ModelInput (dataclass) and be fully torch types bound.
class TorchTypesModelInputWrapper(CopyableMixin):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self._module = module

    def forward(
        self,
        float_features: torch.Tensor,
        idlist_features_keys: List[str],
        idlist_features_values: torch.Tensor,
        idscore_features_keys: List[str],
        idscore_features_values: torch.Tensor,
        idscore_features_weights: torch.Tensor,
        label: torch.Tensor,
        idlist_features_lengths: Optional[torch.Tensor] = None,
        idlist_features_offsets: Optional[torch.Tensor] = None,
        idscore_features_lengths: Optional[torch.Tensor] = None,
        idscore_features_offsets: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        idlist_kjt = KeyedJaggedTensor(
            keys=idlist_features_keys,
            values=idlist_features_values,
            lengths=idlist_features_lengths,
            offsets=idlist_features_offsets,
        )
        idscore_kjt = KeyedJaggedTensor(
            keys=idscore_features_keys,
            values=idscore_features_values,
            weights=idscore_features_weights,
            lengths=idscore_features_lengths,
            offsets=idscore_features_offsets,
        )
        mi = ModelInput(
            float_features=float_features,
            idlist_features=idlist_kjt,
            idscore_features=idscore_kjt,
            label=label,
        )
        return self._module(mi)
