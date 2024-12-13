#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
)
from torch import nn
from torchrec.distributed.embedding_lookup import EmbeddingComputeKernel
from torchrec.distributed.embedding_sharding import (
    EmbeddingSharding,
    EmbeddingShardingInfo,
)
from torchrec.distributed.embedding_types import (
    BaseQuantEmbeddingSharder,
    FeatureShardingMixIn,
    GroupedEmbeddingConfig,
    InputDistOutputs,
    KJTList,
    ListOfKJTList,
)
from torchrec.distributed.embeddingbag import (
    construct_output_kt,
    create_sharding_infos_by_sharding_device_group,
)
from torchrec.distributed.fused_params import (
    FUSED_PARAM_QUANT_STATE_DICT_SPLIT_SCALE_BIAS,
    FUSED_PARAM_REGISTER_TBE_BOOL,
    get_tbes_to_register_from_iterable,
    is_fused_param_quant_state_dict_split_scale_bias,
    is_fused_param_register_tbe,
)
from torchrec.distributed.global_settings import get_propogate_device
from torchrec.distributed.quant_state import ShardedQuantEmbeddingModuleState
from torchrec.distributed.sharding.cw_sharding import InferCwPooledEmbeddingSharding
from torchrec.distributed.sharding.rw_sharding import InferRwPooledEmbeddingSharding
from torchrec.distributed.sharding.tw_sharding import InferTwEmbeddingSharding
from torchrec.distributed.types import (
    NullShardedModuleContext,
    NullShardingContext,
    ParameterSharding,
    ShardingEnv,
    ShardingType,
)
from torchrec.distributed.utils import copy_to_device
from torchrec.modules.embedding_configs import (
    data_type_to_sparse_type,
    dtype_to_data_type,
    EmbeddingBagConfig,
)
from torchrec.modules.embedding_modules import EmbeddingBagCollectionInterface
from torchrec.modules.feature_processor_ import FeatureProcessorsCollection
from torchrec.pt2.checks import is_torchdynamo_compiling
from torchrec.quant.embedding_modules import (
    EmbeddingBagCollection as QuantEmbeddingBagCollection,
    FeatureProcessedEmbeddingBagCollection as QuantFeatureProcessedEmbeddingBagCollection,
    MODULE_ATTR_QUANT_STATE_DICT_SPLIT_SCALE_BIAS,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor


def get_device_from_parameter_sharding(ps: ParameterSharding) -> str:
    # pyre-ignore
    return ps.sharding_spec.shards[0].placement.device().type


def get_device_from_sharding_infos(
    emb_shard_infos: List[EmbeddingShardingInfo],
) -> str:
    res = list(
        {
            get_device_from_parameter_sharding(ps.param_sharding)
            for ps in emb_shard_infos
        }
    )
    assert len(res) == 1, "All shards should be on the same type of device"
    return res[0]


torch.fx.wrap("len")


@torch.fx.wrap
def flatten_feature_lengths(features: KeyedJaggedTensor) -> KeyedJaggedTensor:
    return features.flatten_lengths() if features.lengths().dim() > 1 else features


def create_infer_embedding_bag_sharding(
    sharding_type: str,
    sharding_infos: List[EmbeddingShardingInfo],
    env: ShardingEnv,
    device: Optional[torch.device] = None,
) -> EmbeddingSharding[
    NullShardingContext, InputDistOutputs, List[torch.Tensor], torch.Tensor
]:
    propogate_device: bool = get_propogate_device()
    if sharding_type == ShardingType.TABLE_WISE.value:
        return InferTwEmbeddingSharding(
            sharding_infos, env, device=device if propogate_device else None
        )
    elif sharding_type == ShardingType.ROW_WISE.value:
        return InferRwPooledEmbeddingSharding(
            sharding_infos, env, device=device if propogate_device else None
        )
    elif sharding_type == ShardingType.COLUMN_WISE.value:
        return InferCwPooledEmbeddingSharding(
            sharding_infos,
            env,
            device=device if propogate_device else None,
            permute_embeddings=True,
        )
    else:
        raise ValueError(f"Sharding type not supported {sharding_type}")


class ShardedQuantEmbeddingBagCollection(
    ShardedQuantEmbeddingModuleState[
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
        env: Union[ShardingEnv, Dict[str, ShardingEnv]],  # support for Hybrid Sharding
        fused_params: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self._embedding_bag_configs: List[EmbeddingBagConfig] = (
            module.embedding_bag_configs()
        )
        self._sharding_type_device_group_to_sharding_infos: Dict[
            Tuple[str, str], List[EmbeddingShardingInfo]
        ] = create_sharding_infos_by_sharding_device_group(
            module, table_name_to_parameter_sharding, "embedding_bags.", fused_params
        )
        self._sharding_type_device_group_to_sharding: Dict[
            Tuple[str, str],
            EmbeddingSharding[
                NullShardingContext,
                InputDistOutputs,
                List[torch.Tensor],
                torch.Tensor,
            ],
        ] = {
            (sharding_type, device_group): create_infer_embedding_bag_sharding(
                sharding_type,
                embedding_configs,
                (
                    env
                    if not isinstance(env, Dict)
                    else env[get_device_from_sharding_infos(embedding_configs)]
                ),
                device if get_propogate_device() else None,
            )
            for (
                sharding_type,
                device_group,
            ), embedding_configs in self._sharding_type_device_group_to_sharding_infos.items()
        }
        self._device = device
        self._is_weighted: bool = module.is_weighted()
        self._lookups: List[nn.Module] = []
        self._create_lookups(fused_params, device)

        # Ensure output dist is set for post processing from an inference runtime (ie. setting device from runtime).
        self._output_dists: torch.nn.ModuleList = torch.nn.ModuleList()

        self._embedding_names: List[str] = []
        self._embedding_dims: List[int] = []

        # forward pass flow control
        self._has_uninitialized_output_dist: bool = True

        tbes: Dict[IntNBitTableBatchedEmbeddingBagsCodegen, GroupedEmbeddingConfig] = (
            get_tbes_to_register_from_iterable(self._lookups)
        )

        self._tbes_configs: Dict[
            IntNBitTableBatchedEmbeddingBagsCodegen, GroupedEmbeddingConfig
        ] = tbes

        # Optional registration of TBEs for model post processing utilities
        if is_fused_param_register_tbe(fused_params):
            self.tbes: torch.nn.ModuleList = torch.nn.ModuleList(tbes.keys())

        quant_state_dict_split_scale_bias = (
            is_fused_param_quant_state_dict_split_scale_bias(fused_params)
        )

        if quant_state_dict_split_scale_bias:
            self._initialize_torch_state(
                tbes=tbes,
                table_name_to_parameter_sharding=table_name_to_parameter_sharding,
                tables_weights_prefix="embedding_bags",
            )
        else:
            table_wise_sharded_only: bool = all(
                sharding_type == ShardingType.TABLE_WISE.value
                for (
                    sharding_type,
                    _,
                ) in self._sharding_type_device_group_to_sharding.keys()
            )
            assert (
                table_wise_sharded_only
            ), "ROW_WISE,COLUMN_WISE shardings can be used only in 'quant_state_dict_split_scale_bias' mode, specify fused_params[FUSED_PARAM_QUANT_STATE_DICT_SPLIT_SCALE_BIAS]=True to __init__ argument"

            self.embedding_bags: nn.ModuleDict = nn.ModuleDict()
            for table in self._embedding_bag_configs:
                self.embedding_bags[table.name] = torch.nn.Module()

            for _, lookup in zip(
                self._sharding_type_device_group_to_sharding.keys(), self._lookups
            ):
                lookup_state_dict = lookup.state_dict()
                for key in lookup_state_dict:
                    if key.endswith(".weight"):
                        table_name = key[: -len(".weight")]
                        self.embedding_bags[table_name].register_buffer(
                            "weight", lookup_state_dict[key]
                        )

        self._input_dist_module: ShardedQuantEbcInputDist = ShardedQuantEbcInputDist(
            self._sharding_type_device_group_to_sharding, self._device
        )

    def tbes_configs(
        self,
    ) -> Dict[IntNBitTableBatchedEmbeddingBagsCodegen, GroupedEmbeddingConfig]:
        return self._tbes_configs

    def sharding_type_device_group_to_sharding_infos(
        self,
    ) -> Dict[Tuple[str, str], List[EmbeddingShardingInfo]]:
        return self._sharding_type_device_group_to_sharding_infos

    def embedding_bag_configs(self) -> List[EmbeddingBagConfig]:
        return self._embedding_bag_configs

    def _create_lookups(
        self,
        fused_params: Optional[Dict[str, Any]],
        device: Optional[torch.device] = None,
    ) -> None:
        for sharding in self._sharding_type_device_group_to_sharding.values():
            self._lookups.append(
                sharding.create_lookup(
                    device=device,
                    fused_params=fused_params,
                )
            )

    def _create_output_dist(self, device: Optional[torch.device] = None) -> None:
        for sharding in self._sharding_type_device_group_to_sharding.values():
            self._output_dists.append(sharding.create_output_dist(device))
            self._embedding_names.extend(sharding.embedding_names())
            self._embedding_dims.extend(sharding.embedding_dims())

    # pyre-ignore [14]
    # pyre-ignore
    def input_dist(
        self, ctx: NullShardedModuleContext, features: KeyedJaggedTensor
    ) -> ListOfKJTList:
        input_dist_outputs = self._input_dist_module(features)

        if self._has_uninitialized_output_dist:
            self._create_output_dist(features.device())
            self._has_uninitialized_output_dist = False

        return input_dist_outputs

    def compute(
        self,
        ctx: NullShardedModuleContext,
        dist_input: ListOfKJTList,
    ) -> List[List[torch.Tensor]]:
        # syntax for torchscript
        return [lookup.forward(dist_input[i]) for i, lookup in enumerate(self._lookups)]

    # pyre-ignore
    def output_dist(
        self,
        ctx: NullShardedModuleContext,
        output: List[List[torch.Tensor]],
    ) -> KeyedTensor:
        return construct_output_kt(
            embeddings=[
                dist.forward(output[i]) for i, dist in enumerate(self._output_dists)
            ],
            embedding_dims=self._embedding_dims,
            embedding_names=self._embedding_names,
        )

    # pyre-ignore
    def compute_and_output_dist(
        self, ctx: NullShardedModuleContext, input: ListOfKJTList
    ) -> KeyedTensor:
        return self.output_dist(ctx, self.compute(ctx, input))

    # pyre-ignore
    def forward(self, *input, **kwargs) -> KeyedTensor:
        ctx = self.create_context()
        dist_input = self.input_dist(ctx, *input, **kwargs)
        return self.compute_and_output_dist(ctx, dist_input)

    def copy(self, device: torch.device) -> nn.Module:
        if self._has_uninitialized_output_dist:
            self._create_output_dist(device)
            self._has_uninitialized_output_dist = False
        return super().copy(device)

    @property
    def shardings(self) -> Dict[Tuple[str, str], FeatureShardingMixIn]:
        # pyre-ignore [7]
        return self._sharding_type_device_group_to_sharding

    def create_context(self) -> NullShardedModuleContext:
        if is_torchdynamo_compiling():
            # Context creation is not supported by dynamo yet.
            # Context is not needed for TW sharding =>
            # Unblocking dynamo TW with None.
            # pyre-ignore
            return None

        return NullShardedModuleContext()


class QuantEmbeddingBagCollectionSharder(
    BaseQuantEmbeddingSharder[QuantEmbeddingBagCollection]
):
    def shard(
        self,
        module: QuantEmbeddingBagCollection,
        params: Dict[str, ParameterSharding],
        env: Union[ShardingEnv, Dict[str, ShardingEnv]],
        device: Optional[torch.device] = None,
        module_fqn: Optional[str] = None,
    ) -> ShardedQuantEmbeddingBagCollection:
        fused_params = self.fused_params if self.fused_params else {}
        fused_params["output_dtype"] = data_type_to_sparse_type(
            dtype_to_data_type(module.output_dtype())
        )
        if FUSED_PARAM_QUANT_STATE_DICT_SPLIT_SCALE_BIAS not in fused_params:
            fused_params[FUSED_PARAM_QUANT_STATE_DICT_SPLIT_SCALE_BIAS] = getattr(
                module, MODULE_ATTR_QUANT_STATE_DICT_SPLIT_SCALE_BIAS, False
            )
        if FUSED_PARAM_REGISTER_TBE_BOOL not in fused_params:
            fused_params[FUSED_PARAM_REGISTER_TBE_BOOL] = getattr(
                module, FUSED_PARAM_REGISTER_TBE_BOOL, False
            )

        return ShardedQuantEmbeddingBagCollection(
            module, params, env, fused_params, device=device
        )

    @property
    def module_type(self) -> Type[QuantEmbeddingBagCollection]:
        return QuantEmbeddingBagCollection


class ShardedQuantFeatureProcessedEmbeddingBagCollection(
    ShardedQuantEmbeddingBagCollection,
):
    def __init__(
        self,
        module: EmbeddingBagCollectionInterface,
        table_name_to_parameter_sharding: Dict[str, ParameterSharding],
        env: ShardingEnv,
        fused_params: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        feature_processor: Optional[FeatureProcessorsCollection] = None,
    ) -> None:
        super().__init__(
            module,
            table_name_to_parameter_sharding,
            env,
            fused_params,
            device,
        )
        assert feature_processor is not None
        device_type: str = self._device.type if self._device is not None else "cuda"
        self.feature_processors_per_rank: nn.ModuleList = torch.nn.ModuleList()
        feature_processor_device = None
        for _, param in feature_processor.named_parameters():
            if feature_processor_device is None:
                feature_processor_device = param.device
            elif feature_processor_device != param.device:
                raise RuntimeError(
                    f"Feature processor has inconsistent devices. Expected {feature_processor_device}, got {param.device}"
                )

        for _, buffer in feature_processor.named_buffers():
            if feature_processor_device is None:
                feature_processor_device = buffer.device
            elif feature_processor_device != buffer.device:
                raise RuntimeError(
                    f"Feature processor has inconsistent devices. Expected {feature_processor_device}, got {param.device}"
                )

        if feature_processor_device is None:
            for _ in range(env.world_size):
                self.feature_processors_per_rank.append(feature_processor)
        else:
            for i in range(env.world_size):
                # Generic copy, for example initailized on cpu but -> sharding as meta
                self.feature_processors_per_rank.append(
                    copy.deepcopy(feature_processor)
                    if device_type == "meta"
                    else copy_to_device(
                        feature_processor,
                        feature_processor_device,
                        (
                            torch.device(f"{device_type}:{i}")
                            if device_type == "cuda"
                            else torch.device(f"{device_type}")
                        ),
                    )
                )

    def apply_feature_processor(
        self,
        kjt_list: KJTList,
    ) -> KJTList:
        l: List[KeyedJaggedTensor] = []
        for i in range(len(self.feature_processors_per_rank)):
            l.append(self.feature_processors_per_rank[i](kjt_list[i]))
        return KJTList(l)

    def compute(
        self,
        ctx: NullShardedModuleContext,
        dist_input: ListOfKJTList,  # List_per_sharding[List_per_rank[KJT]]
    ) -> List[List[torch.Tensor]]:
        return [
            lookup.forward(self.apply_feature_processor(dist_input[i]))
            for i, lookup in enumerate(self._lookups)
        ]


class QuantFeatureProcessedEmbeddingBagCollectionSharder(
    BaseQuantEmbeddingSharder[QuantFeatureProcessedEmbeddingBagCollection]
):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [
            ShardingType.TABLE_WISE.value,
            ShardingType.COLUMN_WISE.value,
        ]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [EmbeddingComputeKernel.QUANT.value]

    def shard(
        self,
        module: QuantFeatureProcessedEmbeddingBagCollection,
        params: Dict[str, ParameterSharding],
        env: ShardingEnv,
        device: Optional[torch.device] = None,
        module_fqn: Optional[str] = None,
    ) -> ShardedQuantEmbeddingBagCollection:
        qebc = module
        assert isinstance(qebc, QuantEmbeddingBagCollection)
        fused_params = self.fused_params if self.fused_params else {}
        fused_params["output_dtype"] = data_type_to_sparse_type(
            dtype_to_data_type(qebc.output_dtype())
        )
        if FUSED_PARAM_QUANT_STATE_DICT_SPLIT_SCALE_BIAS not in fused_params:
            fused_params[FUSED_PARAM_QUANT_STATE_DICT_SPLIT_SCALE_BIAS] = getattr(
                qebc, MODULE_ATTR_QUANT_STATE_DICT_SPLIT_SCALE_BIAS, False
            )
        if FUSED_PARAM_REGISTER_TBE_BOOL not in fused_params:
            fused_params[FUSED_PARAM_REGISTER_TBE_BOOL] = getattr(
                qebc, FUSED_PARAM_REGISTER_TBE_BOOL, False
            )

        return ShardedQuantFeatureProcessedEmbeddingBagCollection(
            qebc,
            params,
            env,
            fused_params,
            device=device,
            feature_processor=module.feature_processor,
        )

    @property
    def module_type(self) -> Type[QuantFeatureProcessedEmbeddingBagCollection]:
        return QuantFeatureProcessedEmbeddingBagCollection


class ShardedQuantEbcInputDist(torch.nn.Module):
    """
    This module implements distributed inputs of a ShardedQuantEmbeddingBagCollection.

    Args:
        sharding_type_to_sharding (Dict[
            str,
            EmbeddingSharding[
                NullShardingContext,
                KJTList,
                List[torch.Tensor],
                torch.Tensor,
            ],
        ]): map from sharding type to EmbeddingSharding.
        device (Optional[torch.device]): default compute device.

    Example::

        sqebc_input_dist = ShardedQuantEbcInputDist(
            sharding_type_to_sharding={
                ShardingType.TABLE_WISE: InferTwSequenceEmbeddingSharding(
                    [],
                    ShardingEnv(
                        world_size=2,
                        rank=0,
                        pg=0,
                    ),
                    torch.device("cpu")
                )
            },
            device=torch.device("cpu"),
        )

        features = KeyedJaggedTensor(
            keys=["f1", "f2"],
            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
            offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8]),
        )

        sqebc_input_dist(features)
    """

    def __init__(
        self,
        sharding_type_device_group_to_sharding: Dict[
            Tuple[str, str],
            EmbeddingSharding[
                NullShardingContext,
                InputDistOutputs,
                List[torch.Tensor],
                torch.Tensor,
            ],
        ],
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self._sharding_type_device_group_to_sharding = (
            sharding_type_device_group_to_sharding
        )
        self._device = device

        self._input_dists: List[nn.Module] = []

        self._feature_splits: List[int] = []
        self._features_order: List[int] = []

        # forward pass flow control
        self._has_uninitialized_input_dist: bool = True
        self._has_features_permute: bool = True

    def _create_input_dist(
        self,
        input_feature_names: List[str],
        features_device: torch.device,
        input_dist_device: Optional[torch.device] = None,
    ) -> None:
        feature_names: List[str] = []
        for sharding in self._sharding_type_device_group_to_sharding.values():
            self._input_dists.append(
                sharding.create_input_dist(device=input_dist_device)
            )
            feature_names.extend(sharding.feature_names())
            self._feature_splits.append(len(sharding.feature_names()))

        if feature_names == input_feature_names:
            self._has_features_permute = False
        else:
            for f in feature_names:
                self._features_order.append(input_feature_names.index(f))
            self.register_buffer(
                "_features_order_tensor",
                torch.tensor(
                    self._features_order, device=features_device, dtype=torch.int32
                ),
                persistent=False,
            )

    def forward(self, features: KeyedJaggedTensor) -> ListOfKJTList:
        """
        Args:
            features (KeyedJaggedTensor): KJT of form [F X B X L].

        Returns:
            ListOfKJTList
        """
        if self._has_uninitialized_input_dist:
            self._create_input_dist(
                features.keys(),
                features.device(),
                self._device,
            )
            self._has_uninitialized_input_dist = False
        with torch.no_grad():
            if self._has_features_permute:
                features = features.permute(
                    self._features_order,
                    # pyre-fixme[6]: For 2nd argument expected `Optional[Tensor]`
                    #  but got `Union[Module, Tensor]`.
                    self._features_order_tensor,
                )
            else:
                features = flatten_feature_lengths(features)
            features_by_shards = (
                [features]
                if len(self._feature_splits) == 1
                else features.split(self._feature_splits)
            )
            return ListOfKJTList(
                [
                    self._input_dists[i].forward(features_by_shards[i]).features
                    for i in range(len(self._input_dists))
                ]
            )
