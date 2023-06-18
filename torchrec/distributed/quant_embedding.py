#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

import torch
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
)
from torch import nn
from torchrec.distributed.embedding import (
    create_sharding_infos_by_sharding,
    EmbeddingShardingInfo,
)
from torchrec.distributed.embedding_sharding import EmbeddingSharding
from torchrec.distributed.embedding_types import (
    BaseQuantEmbeddingSharder,
    FeatureShardingMixIn,
    GroupedEmbeddingConfig,
    KJTList,
    ListOfKJTList,
    ShardingType,
)
from torchrec.distributed.fused_params import (
    FUSED_PARAM_QUANT_STATE_DICT_SPLIT_SCALE_BIAS,
    get_tbes_to_register_from_iterable,
    is_fused_param_quant_state_dict_split_scale_bias,
    is_fused_param_register_tbe,
)
from torchrec.distributed.quant_state import ShardedQuantEmbeddingModuleState
from torchrec.distributed.sharding.sequence_sharding import InferSequenceShardingContext
from torchrec.distributed.sharding.tw_sequence_sharding import (
    InferTwSequenceEmbeddingSharding,
)
from torchrec.distributed.types import ParameterSharding, ShardingEnv
from torchrec.modules.embedding_configs import (
    data_type_to_sparse_type,
    dtype_to_data_type,
    EmbeddingConfig,
)
from torchrec.quant.embedding_modules import (
    EmbeddingCollection as QuantEmbeddingCollection,
    MODULE_ATTR_QUANT_STATE_DICT_SPLIT_SCALE_BIAS,
)
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor
from torchrec.streamable import Multistreamable

torch.fx.wrap("len")

try:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
except OSError:
    pass


@dataclass
class EmbeddingCollectionContext(Multistreamable):
    sharding_contexts: List[InferSequenceShardingContext]

    def record_stream(self, stream: torch.cuda.streams.Stream) -> None:
        for ctx in self.sharding_contexts:
            ctx.record_stream(stream)


def create_infer_embedding_sharding(
    sharding_type: str,
    sharding_infos: List[EmbeddingShardingInfo],
    env: ShardingEnv,
    device: Optional[torch.device] = None,
) -> EmbeddingSharding[
    InferSequenceShardingContext,
    KJTList,
    List[torch.Tensor],
    List[torch.Tensor],
]:
    if sharding_type == ShardingType.TABLE_WISE.value:
        return InferTwSequenceEmbeddingSharding(sharding_infos, env, device)
    else:
        raise ValueError(f"Sharding type not supported {sharding_type}")


def _construct_jagged_tensors(
    embeddings: torch.Tensor,
    features: KeyedJaggedTensor,
    need_indices: bool = False,
) -> Dict[str, JaggedTensor]:
    # ignore cw consideration for inference now.
    ret: Dict[str, JaggedTensor] = {}
    lengths = features.lengths().view(-1, features.stride())
    values = features.values()
    length_per_key = features.length_per_key()

    embeddings_list = torch.split(embeddings, length_per_key, dim=0)
    stride = features.stride()
    lengths_tuple = torch.unbind(lengths.view(-1, stride), dim=0)
    if need_indices:
        values_list = torch.split(values, length_per_key)
        for i, key in enumerate(features.keys()):
            ret[key] = JaggedTensor(
                lengths=lengths_tuple[i],
                values=embeddings_list[i],
                weights=values_list[i],
            )
    else:
        for i, key in enumerate(features.keys()):
            ret[key] = JaggedTensor(
                lengths=lengths_tuple[i],
                values=embeddings_list[i],
                weights=None,
            )
    return ret


@torch.fx.wrap
def output_jt_dict(
    emb_per_sharding: List[List[torch.Tensor]],
    features_per_sharding: List[KJTList],
    need_indices: bool,
) -> Dict[str, JaggedTensor]:
    jt_dict: Dict[str, JaggedTensor] = {}
    for emb_sharding, f_sharding in zip(
        emb_per_sharding,
        features_per_sharding,
    ):
        # Can not use zip here as Iterator of KJTList is not supported by jit
        for i in range(len(emb_sharding)):
            jt_dict.update(
                _construct_jagged_tensors(
                    embeddings=emb_sharding[i],
                    features=f_sharding[i],
                    need_indices=need_indices,
                )
            )
    return jt_dict


class ShardedQuantEmbeddingCollection(
    ShardedQuantEmbeddingModuleState[
        ListOfKJTList,
        List[List[torch.Tensor]],
        Dict[str, JaggedTensor],
        EmbeddingCollectionContext,
    ],
):
    """
    Sharded implementation of `QuantEmbeddingCollection`.
    """

    def __init__(
        self,
        module: QuantEmbeddingCollection,
        table_name_to_parameter_sharding: Dict[str, ParameterSharding],
        env: ShardingEnv,
        fused_params: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        self._embedding_configs: List[EmbeddingConfig] = module.embedding_configs()

        sharding_type_to_sharding_infos = create_sharding_infos_by_sharding(
            module, table_name_to_parameter_sharding, fused_params
        )
        self._sharding_type_to_sharding: Dict[
            str,
            EmbeddingSharding[
                InferSequenceShardingContext,
                KJTList,
                List[torch.Tensor],
                List[torch.Tensor],
            ],
        ] = {
            sharding_type: create_infer_embedding_sharding(
                sharding_type, embedding_confings, env
            )
            for sharding_type, embedding_confings in sharding_type_to_sharding_infos.items()
        }

        self._device = device
        self._input_dists: List[nn.Module] = []
        self._lookups: List[nn.Module] = []
        self._create_lookups(fused_params, device)
        self._output_dists: List[nn.Module] = []

        self._feature_splits: List[int] = []
        self._features_order: List[int] = []

        self._has_uninitialized_input_dist: bool = True
        self._has_uninitialized_output_dist: bool = True

        self._embedding_dim: int = module.embedding_dim()
        self._need_indices: bool = module.need_indices()

        self._fused_params = fused_params

        tbes: Dict[
            IntNBitTableBatchedEmbeddingBagsCodegen, GroupedEmbeddingConfig
        ] = get_tbes_to_register_from_iterable(self._lookups)

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
                tables_weights_prefix="embeddings",
            )
        else:
            table_wise_sharded_only: bool = all(
                [
                    sharding_type == ShardingType.TABLE_WISE.value
                    for sharding_type in self._sharding_type_to_sharding.keys()
                ]
            )
            assert (
                table_wise_sharded_only
            ), "ROW_WISE,COLUMN_WISE shardings can be used only in 'quant_state_dict_split_scale_bias' mode, specify fused_params[FUSED_PARAM_QUANT_STATE_DICT_SPLIT_SCALE_BIAS]=True to __init__ argument"

            self.embeddings: nn.ModuleDict = nn.ModuleDict()
            for table in self._embedding_configs:
                self.embeddings[table.name] = torch.nn.Module()

            for _sharding_type, lookup in zip(
                self._sharding_type_to_sharding.keys(), self._lookups
            ):
                lookup_state_dict = lookup.state_dict()
                for key in lookup_state_dict:
                    if key.endswith(".weight"):
                        table_name = key[: -len(".weight")]
                        self.embeddings[table_name].register_buffer(
                            "weight", lookup_state_dict[key]
                        )

    def _create_input_dist(
        self,
        input_feature_names: List[str],
        device: torch.device,
        input_dist_device: Optional[torch.device] = None,
    ) -> None:
        feature_names: List[str] = []
        self._feature_splits: List[int] = []
        for sharding in self._sharding_type_to_sharding.values():
            self._input_dists.append(
                sharding.create_input_dist(device=input_dist_device)
            )
            feature_names.extend(sharding.feature_names())
            self._feature_splits.append(len(sharding.feature_names()))
        self._features_order: List[int] = []
        for f in feature_names:
            self._features_order.append(input_feature_names.index(f))
        self._features_order = (
            []
            if self._features_order == list(range(len(self._features_order)))
            else self._features_order
        )
        self.register_buffer(
            "_features_order_tensor",
            torch.tensor(self._features_order, device=device, dtype=torch.int32),
            persistent=False,
        )

    def _create_lookups(
        self,
        fused_params: Optional[Dict[str, Any]],
        device: Optional[torch.device] = None,
    ) -> None:
        for sharding in self._sharding_type_to_sharding.values():
            self._lookups.append(
                sharding.create_lookup(fused_params=fused_params, device=device)
            )

    def _create_output_dist(
        self,
        device: Optional[torch.device] = None,
    ) -> None:
        for sharding in self._sharding_type_to_sharding.values():
            self._output_dists.append(sharding.create_output_dist(device))

    # pyre-ignore [14]
    # pyre-ignore
    def input_dist(
        self,
        ctx: EmbeddingCollectionContext,
        features: KeyedJaggedTensor,
    ) -> ListOfKJTList:
        if self._has_uninitialized_input_dist:
            self._create_input_dist(
                input_feature_names=features.keys() if features is not None else [],
                device=features.device(),
                input_dist_device=self._device,
            )
            self._has_uninitialized_input_dist = False
        if self._has_uninitialized_output_dist:
            self._create_output_dist(features.device())
            self._has_uninitialized_output_dist = False
        with torch.no_grad():
            features_by_sharding = []
            if self._features_order:
                features = features.permute(
                    self._features_order,
                    # pyre-ignore [6]
                    self._features_order_tensor,
                )
            features_by_sharding = features.split(
                self._feature_splits,
            )

            return ListOfKJTList(
                [
                    self._input_dists[i].forward(features_by_sharding[i])
                    for i in range(len(self._input_dists))
                ]
            )

    def compute(
        self, ctx: EmbeddingCollectionContext, dist_input: ListOfKJTList
    ) -> List[List[torch.Tensor]]:
        ret: List[List[torch.Tensor]] = []
        for lookup, features in zip(
            self._lookups,
            dist_input,
        ):
            ctx.sharding_contexts.append(InferSequenceShardingContext(features))
            ret.append(
                [o.view(-1, self._embedding_dim) for o in lookup.forward(features)]
            )
        return ret

    # pyre-ignore
    def output_dist(
        self, ctx: EmbeddingCollectionContext, output: List[List[torch.Tensor]]
    ) -> Dict[str, JaggedTensor]:
        emb_per_sharding: List[List[torch.Tensor]] = []
        features_per_sharding: List[List[KeyedJaggedTensor]] = []
        for odist, embeddings, sharding_ctx in zip(
            self._output_dists,
            output,
            ctx.sharding_contexts,
        ):
            emb_per_sharding.append(odist.forward(embeddings, sharding_ctx))
            # pyre-fixme[6]: For 1st argument expected `List[KeyedJaggedTensor]` but
            #  got `Optional[KJTList]`.
            features_per_sharding.append(sharding_ctx.features)

        return output_jt_dict(
            emb_per_sharding, features_per_sharding, self._need_indices
        )

    # pyre-ignore
    def compute_and_output_dist(
        self, ctx: EmbeddingCollectionContext, input: ListOfKJTList
    ) -> Dict[str, JaggedTensor]:
        return self.output_dist(ctx, self.compute(ctx, input))

    # pyre-ignore
    def forward(self, *input, **kwargs) -> Dict[str, JaggedTensor]:
        ctx = self.create_context()
        dist_input = self.input_dist(ctx, *input, **kwargs)
        return self.compute_and_output_dist(ctx, dist_input)

    def copy(self, device: torch.device) -> nn.Module:
        if self._has_uninitialized_output_dist:
            self._create_output_dist(device)
            self._has_uninitialized_output_dist = False
        return super().copy(device)

    def create_context(self) -> EmbeddingCollectionContext:
        return EmbeddingCollectionContext(sharding_contexts=[])

    @property
    def shardings(self) -> Dict[str, FeatureShardingMixIn]:
        # pyre-ignore [7]
        return self._sharding_type_to_sharding


class QuantEmbeddingCollectionSharder(
    BaseQuantEmbeddingSharder[QuantEmbeddingCollection]
):
    """
    This implementation uses non-fused EmbeddingCollection
    """

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
        fused_params[FUSED_PARAM_QUANT_STATE_DICT_SPLIT_SCALE_BIAS] = getattr(
            module, MODULE_ATTR_QUANT_STATE_DICT_SPLIT_SCALE_BIAS, False
        )
        return ShardedQuantEmbeddingCollection(module, params, env, fused_params)

    @property
    def module_type(self) -> Type[QuantEmbeddingCollection]:
        return QuantEmbeddingCollection
