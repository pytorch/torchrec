#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Mapping, Optional, Tuple, TypeVar, Union

import torch
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
)
from torch.distributed import _remote_device
from torch.distributed._shard.sharded_tensor import (
    Shard,
    ShardedTensorBase,
    ShardedTensorMetadata,
    ShardMetadata,
)
from torchrec.distributed.embedding_sharding import EmbeddingShardingInfo
from torchrec.distributed.embedding_types import (
    GroupedEmbeddingConfig,
    ShardedEmbeddingModule,
)
from torchrec.distributed.types import ParameterSharding, ShardingType
from torchrec.modules.embedding_configs import DataType
from torchrec.streamable import Multistreamable
from torchrec.tensor_types import UInt2Tensor, UInt4Tensor

Out = TypeVar("Out")
CompIn = TypeVar("CompIn")
DistOut = TypeVar("DistOut")
ShrdCtx = TypeVar("ShrdCtx", bound=Multistreamable)


def _append_table_shard(
    d: Dict[str, List[Shard]], table_name: str, shard: Shard
) -> None:
    if table_name not in d:
        d[table_name] = []
    d[table_name].append(shard)


def post_state_dict_hook(
    # Union["ShardedQuantEmbeddingBagCollection", "ShardedQuantEmbeddingCollection"]
    # pyre-ignore [24]
    module: ShardedEmbeddingModule,
    destination: Dict[str, torch.Tensor],
    prefix: str,
    _local_metadata: Dict[str, Any],
    tables_weights_prefix: str,  # "embedding_bags" or "embeddings"
) -> None:
    for (
        table_name,
        sharded_t,
        # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute `items`.
    ) in module._table_name_to_sharded_tensor.items():
        destination[f"{prefix}{tables_weights_prefix}.{table_name}.weight"] = sharded_t

    for sfx, dict_sharded_t, dict_t_list in [
        (
            "weight_qscale",
            module._table_name_to_sharded_tensor_qscale,
            module._table_name_to_tensors_list_qscale,
        ),
        (
            "weight_qbias",
            module._table_name_to_sharded_tensor_qbias,
            module._table_name_to_tensors_list_qbias,
        ),
    ]:
        for (
            table_name,
            sharded_t,
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute `items`.
        ) in dict_sharded_t.items():
            destination[f"{prefix}{tables_weights_prefix}.{table_name}.{sfx}"] = (
                sharded_t
            )
        for (
            table_name,
            t_list,
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute `items`.
        ) in dict_t_list.items():
            destination[f"{prefix}{tables_weights_prefix}.{table_name}.{sfx}"] = t_list


class ShardedQuantEmbeddingModuleState(
    ShardedEmbeddingModule[CompIn, DistOut, Out, ShrdCtx]
):
    def _initialize_torch_state(  # noqa: C901
        # Union[ShardedQuantEmbeddingBagCollection, ShardedQuantEmbeddingCollection]
        self,
        tbes: Dict[IntNBitTableBatchedEmbeddingBagsCodegen, GroupedEmbeddingConfig],
        table_name_to_parameter_sharding: Dict[str, ParameterSharding],
        tables_weights_prefix: str,  # "embedding_bags" or "embeddings"
    ) -> None:  # noqa
        # State is prepared only in "quant_state_dict_split_scale_bias" mode
        assert (
            tables_weights_prefix == "embedding_bags"
            or tables_weights_prefix == "embeddings"
        )

        # weight
        # pyre-fixme[16]: `ShardedQuantEmbeddingModuleState` has no attribute
        #  `_table_name_to_local_shards`.
        self._table_name_to_local_shards: Dict[str, List[Shard]] = {}
        # pyre-fixme[16]: `ShardedQuantEmbeddingModuleState` has no attribute
        #  `_table_name_to_sharded_tensor`.
        self._table_name_to_sharded_tensor: Dict[
            str, Union[torch.Tensor, ShardedTensorBase]
        ] = {}

        # weight_qscale
        # pyre-fixme[16]: `ShardedQuantEmbeddingModuleState` has no attribute
        #  `_table_name_to_local_shards_qscale`.
        self._table_name_to_local_shards_qscale: Dict[str, List[Shard]] = {}
        # pyre-fixme[16]: `ShardedQuantEmbeddingModuleState` has no attribute
        #  `_table_name_to_sharded_tensor_qscale`.
        self._table_name_to_sharded_tensor_qscale: Dict[
            str, Union[torch.Tensor, ShardedTensorBase]
        ] = {}
        # pyre-fixme[16]: `ShardedQuantEmbeddingModuleState` has no attribute
        #  `_table_name_to_tensors_list_qscale`.
        self._table_name_to_tensors_list_qscale: Dict[str, List[torch.Tensor]] = {}

        # weight_qbias
        # pyre-fixme[16]: `ShardedQuantEmbeddingModuleState` has no attribute
        #  `_table_name_to_local_shards_qbias`.
        self._table_name_to_local_shards_qbias: Dict[str, List[Shard]] = {}
        # pyre-fixme[16]: `ShardedQuantEmbeddingModuleState` has no attribute
        #  `_table_name_to_sharded_tensor_qbias`.
        self._table_name_to_sharded_tensor_qbias: Dict[
            str, Union[torch.Tensor, ShardedTensorBase]
        ] = {}
        # pyre-fixme[16]: `ShardedQuantEmbeddingModuleState` has no attribute
        #  `_table_name_to_tensors_list_qbias`.
        self._table_name_to_tensors_list_qbias: Dict[str, List[torch.Tensor]] = {}

        for tbe, config in tbes.items():
            for (tbe_split_w, tbe_split_qscale, tbe_split_qbias), table in zip(
                tbe.split_embedding_weights_with_scale_bias(split_scale_bias_mode=2),
                config.embedding_tables,
            ):
                if table.data_type == DataType.INT4:
                    tbe_split_w = UInt4Tensor(tbe_split_w)
                elif table.data_type == DataType.INT2:
                    tbe_split_w = UInt2Tensor(tbe_split_w)

                # weight shards section:
                assert table.local_metadata
                metadata: ShardMetadata = copy.deepcopy(table.local_metadata)
                metadata.shard_sizes = [tbe_split_w.size(0), tbe_split_w.size(1)]

                # TODO(ivankobzarev): "meta" sharding support: cleanup when copy to "meta" moves all tensors to "meta"
                # pyre-ignore
                if metadata.placement.device != tbe_split_w.device:
                    metadata.placement = _remote_device(tbe_split_w.device)
                _append_table_shard(
                    # pyre-fixme[6]: For 1st argument expected `Dict[str,
                    #  List[Shard]]` but got `Union[Tensor, Module]`.
                    self._table_name_to_local_shards,
                    table.name,
                    Shard(tensor=tbe_split_w, metadata=metadata),
                )
                # end of weight shards section

                # weight_qscale & weight_qbias section:
                # For RW - ShardedTensorBase
                # For CW - List[Tensor] that logically corresponds to the same unsharded Tensor, but present on each sharded rank
                for (
                    tbe_split_qparam,
                    table_name_to_local_shards,
                    table_name_to_tensors_list,
                ) in [
                    (
                        tbe_split_qscale,
                        self._table_name_to_local_shards_qscale,
                        self._table_name_to_tensors_list_qscale,
                    ),
                    (
                        tbe_split_qbias,
                        self._table_name_to_local_shards_qbias,
                        self._table_name_to_tensors_list_qbias,
                    ),
                ]:
                    assert table.local_metadata
                    metadata: ShardMetadata = copy.deepcopy(table.local_metadata)
                    shard_sizes = metadata.shard_sizes
                    shard_offsets = metadata.shard_offsets

                    shard_sizes_cols = shard_sizes[1]
                    shard_offsets_cols = shard_offsets[1]

                    parameter_sharding: ParameterSharding = (
                        table_name_to_parameter_sharding[table.name]
                    )
                    sharding_type: str = parameter_sharding.sharding_type

                    if sharding_type == ShardingType.COLUMN_WISE.value:
                        # pyre-fixme[58]: `not in` is not supported for right
                        #  operand type `Union[Tensor, Module]`.
                        if table.name not in table_name_to_tensors_list:
                            assert parameter_sharding.ranks
                            num_shards: int = len(parameter_sharding.ranks)
                            # pyre-fixme[29]: `Union[(self: TensorBase, indices: Unio...
                            table_name_to_tensors_list[table.name] = [
                                torch.empty([])
                            ] * num_shards

                        column_idx = int(shard_offsets_cols / shard_sizes_cols)
                        # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[No...
                        table_name_to_tensors_list[table.name][
                            column_idx
                        ] = tbe_split_qparam
                    else:
                        qmetadata = ShardMetadata(
                            shard_offsets=metadata.shard_offsets,
                            shard_sizes=[
                                tbe_split_qparam.shape[0],
                                tbe_split_qparam.shape[1],
                            ],
                            # pyre-ignore
                            placement=table.local_metadata.placement,
                        )
                        # TODO(ivankobzarev): "meta" sharding support: cleanup when copy to "meta" moves all tensors to "meta"
                        if qmetadata.placement.device != tbe_split_qparam.device:
                            qmetadata.placement = _remote_device(
                                tbe_split_qparam.device
                            )
                        _append_table_shard(
                            # pyre-fixme[6]: For 1st argument expected `Dict[str,
                            #  List[Shard]]` but got `Union[Tensor, Module]`.
                            table_name_to_local_shards,
                            table.name,
                            Shard(tensor=tbe_split_qparam, metadata=qmetadata),
                        )
                    # end of weight_qscale & weight_qbias section

        for table_name_to_local_shards, table_name_to_sharded_tensor in [
            (self._table_name_to_local_shards, self._table_name_to_sharded_tensor),
            (
                self._table_name_to_local_shards_qscale,
                self._table_name_to_sharded_tensor_qscale,
            ),
            (
                self._table_name_to_local_shards_qbias,
                self._table_name_to_sharded_tensor_qbias,
            ),
        ]:
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `items`.
            for table_name, local_shards in table_name_to_local_shards.items():
                if len(local_shards) == 1:
                    # Single Tensor per table (TW sharding)
                    # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, ...
                    table_name_to_sharded_tensor[table_name] = local_shards[0].tensor
                    continue

                # ShardedTensor per table
                global_rows = max(
                    [
                        ls.metadata.shard_offsets[0] + ls.metadata.shard_sizes[0]
                        for ls in local_shards
                    ]
                )
                global_cols = max(
                    [
                        ls.metadata.shard_offsets[1] + ls.metadata.shard_sizes[1]
                        for ls in local_shards
                    ]
                )
                global_metadata: ShardedTensorMetadata = ShardedTensorMetadata(
                    shards_metadata=[ls.metadata for ls in local_shards],
                    size=torch.Size([global_rows, global_cols]),
                )
                # pyre-fixme[29]: `Union[(self: TensorBase, indices: Union[None, _Nes...
                table_name_to_sharded_tensor[table_name] = (
                    ShardedTensorBase._init_from_local_shards_and_global_metadata(
                        local_shards=local_shards,
                        sharded_tensor_metadata=global_metadata,
                    )
                )

        self._register_state_dict_hook(
            partial(post_state_dict_hook, tables_weights_prefix=tables_weights_prefix)
        )

    def _load_from_state_dict(
        # Union["ShardedQuantEmbeddingBagCollection", "ShardedQuantEmbeddingCollection"]
        self,
        state_dict: Mapping[str, Any],
        prefix: str,
        # pyre-ignore
        local_metadata,
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        dst_state_dict = self.state_dict()
        _missing_keys: List[str] = []
        _unexpected_keys: List[str] = list(state_dict.keys())
        for name, dst_tensor in dst_state_dict.items():
            src_state_dict_name = prefix + name
            if src_state_dict_name not in state_dict:
                _missing_keys.append(src_state_dict_name)
                continue

            src_tensor = state_dict[src_state_dict_name]
            if isinstance(dst_tensor, ShardedTensorBase) and isinstance(
                src_tensor, ShardedTensorBase
            ):
                # sharded to sharded model, only identically sharded
                for dst_local_shard in dst_tensor.local_shards():
                    copied: bool = False
                    for src_local_shard in src_tensor.local_shards():
                        if (
                            dst_local_shard.metadata.shard_offsets
                            == src_local_shard.metadata.shard_offsets
                            and dst_local_shard.metadata.shard_sizes
                            == src_local_shard.metadata.shard_sizes
                        ):
                            dst_local_shard.tensor.copy_(src_local_shard.tensor)
                            copied = True
                            break
                    assert copied, "Incompatible state_dict"
            elif isinstance(dst_tensor, ShardedTensorBase) and isinstance(
                src_tensor, torch.Tensor
            ):
                # non_sharded to sharded model
                for dst_local_shard in dst_tensor.local_shards():
                    dst_tensor = dst_local_shard.tensor
                    assert src_tensor.ndim == dst_tensor.ndim
                    meta = dst_local_shard.metadata
                    t = src_tensor.detach()
                    rows_from = meta.shard_offsets[0]
                    rows_to = rows_from + meta.shard_sizes[0]
                    if t.ndim == 1:
                        dst_tensor.copy_(t[rows_from:rows_to])
                    elif t.ndim == 2:
                        cols_from = meta.shard_offsets[1]
                        cols_to = cols_from + meta.shard_sizes[1]
                        dst_tensor.copy_(
                            t[
                                rows_from:rows_to,
                                cols_from:cols_to,
                            ]
                        )
                    else:
                        raise RuntimeError("Tensors with ndim > 2 are not supported")
            elif isinstance(dst_tensor, list) and isinstance(src_tensor, torch.Tensor):
                # non_sharded to CW columns qscale, qbias (one to many)
                for t in dst_tensor:
                    assert isinstance(t, torch.Tensor)
                    t.copy_(src_tensor)
            else:
                dst_tensor.copy_(src_tensor)

            _unexpected_keys.remove(src_state_dict_name)
        missing_keys.extend(_missing_keys)
        unexpected_keys.extend(_unexpected_keys)


@dataclass
class WeightSpec:
    fqn: str  # "ebc.embedding_bags.table_0.weight"
    shard_offsets: List[int]  # shard offsets
    shard_sizes: List[int]  # shard sizes
    sharding_type: Optional[str]  # e.g. ShardingType.ROW_WISE.value=="row_wise"


def sharded_tbes_weights_spec(
    sharded_model: torch.nn.Module,
) -> Dict[str, WeightSpec]:
    # OUTPUT:
    # Example:
    # {
    # tbes.0
    # table_0 in tbes.0
    # 	"ebc.tbes.0.0.table_0.weight": WeightSpec("ebc.embedding_bags.table_0.weight", [0, 0], [500, 192])
    # 	"ebc.tbes.0.0.table_0.weight_qscale":WeightSpec("ebc.embedding_bags.table_0.weight_qscale", [0, 0], [500, 2])
    # 	"ebc.tbes.0.0.table_0.weight_qbias":WeightSpec("ebc.embedding_bags.table_0.weight_qbias", [0, 0], [500, 2])
    # table_1 in tbes.0
    # 	"ebc.tbes.0.1.table_1.weight": WeightSpec("ebc.embedding_bags.table_1.weight", [0, 0], [500, 192])
    # 	"ebc.tbes.0.1.table_1.weight_qscale":WeightSpec("ebc.embedding_bags.table_1.weight_qscale", [0, 0], [500, 2])
    # 	"ebc.tbes.0.1.table_1.weight_qbias":WeightSpec("ebc.embedding_bags.table_1.weight_qbias", [0, 0], [500, 2])
    # tbes.1
    # table_0 in tbes.1
    # 	"ebc.tbes.1.0.table_0.weight": WeightSpec("ebc.embedding_bags.table_0.weight", [500, 0], [500, 192])
    # 	"ebc.tbes.1.0.table_0.weight_qscale":WeightSpec("ebc.embedding_bags.table_0.weight_qscale", [500, 0], [500, 2])
    # 	"ebc.tbes.1.0.table_0.weight_qbias":WeightSpec("ebc.embedding_bags.table_0.weight_qbias", [500, 0], [500, 2])
    # table_1 in tbes.1
    # 	"ebc.tbes.1.1.table_1.weight": WeightSpec("ebc.embedding_bags.table_1.weight", [500, 0], [500, 192])
    # 	"ebc.tbes.1.1.table_1.weight_qscale":WeightSpec("ebc.embedding_bags.table_1.weight_qscale", [500, 0], [500, 2])
    # 	"ebc.tbes.1.1.table_1.weight_qbias":WeightSpec("ebc.embedding_bags.table_1.weight_qbias", [500, 0], [500, 2])
    # }
    # In the format of ebc.tbes.i.j.table_k.weight, where i is the index of the TBE, j is the index of the embedding bag within TBE i, k is the index of the original table set in the ebc embedding_configs
    # e.g. ebc.tbes.1.1.table_1.weight, it represents second embedding bag within the second TBE. This part of weight is from a shard of table_1

    ret: Dict[str, WeightSpec] = {}
    for module_fqn, module in sharded_model.named_modules():
        type_name: str = type(module).__name__
        is_sqebc: bool = "ShardedQuantEmbeddingBagCollection" in type_name
        is_sqec: bool = "ShardedQuantEmbeddingCollection" in type_name
        is_sqmcec: bool = "ShardedQuantManagedCollisionEmbeddingCollection" in type_name

        if is_sqebc or is_sqec or is_sqmcec:
            assert (
                is_sqec + is_sqebc + is_sqmcec == 1
            ), "Cannot have any two of ShardedQuantEmbeddingBagCollection, ShardedQuantEmbeddingCollection and ShardedQuantManagedCollisionEmbeddingCollection are true"
            tbes_configs: Dict[
                IntNBitTableBatchedEmbeddingBagsCodegen, GroupedEmbeddingConfig
            ] = module.tbes_configs()
            table_shardings: Dict[str, str] = {}

            sharding_type_device_group_to_sharding_infos: Dict[
                Tuple[str, str], List[EmbeddingShardingInfo]
            ] = module.sharding_type_device_group_to_sharding_infos()

            for (
                (sharding_type, _),
                sharding_infos,
            ) in sharding_type_device_group_to_sharding_infos.items():
                for info in sharding_infos:
                    table_shardings[info.embedding_config.name] = sharding_type

            for tbe_idx, (_tbe, config) in enumerate(tbes_configs.items()):
                tables = config.embedding_tables
                for table_idx, table in enumerate(tables):
                    table_name: str = table.name
                    # pyre-ignore
                    table_metadata: ShardMetadata = table.local_metadata
                    # TODO(ivankobzarev) Switch to use table_metadata.shard_sizes when it works correctly with int4 quantized modules
                    shard_sizes: List[int] = [table.local_rows, table.local_cols]
                    shard_offsets: List[int] = table_metadata.shard_offsets
                    s: str = "embedding_bags" if is_sqebc else "embeddings"
                    unsharded_fqn_weight: str = f"{module_fqn}.{s}.{table_name}.weight"

                    sharded_fqn_weight: str = (
                        f"{module_fqn}.tbes.{tbe_idx}.{table_idx}.{table_name}.weight"
                    )
                    sharding_type: str = table_shardings[table_name]
                    ret[sharded_fqn_weight] = WeightSpec(
                        fqn=unsharded_fqn_weight,
                        shard_offsets=shard_offsets,
                        shard_sizes=shard_sizes,
                        sharding_type=sharding_type,
                    )

                    for qcomponent in ["qscale", "qbias"]:
                        qcomp_shard_offsets: List[int] = copy.deepcopy(shard_offsets)
                        # handling CW - no columns shift for qscale/qbias
                        qcomp_shard_offsets[1] = 0
                        qcomp_shard_sizes: List[int] = copy.deepcopy(shard_sizes)
                        # Assuming qscale and qbias are always torch.half (float16), represented as tensor of byte type => sizeof(float16) == 2 (bytes)
                        qcomp_shard_sizes[1] = 2

                        ret[f"{sharded_fqn_weight}_{qcomponent}"] = WeightSpec(
                            fqn=f"{unsharded_fqn_weight}_{qcomponent}",
                            shard_offsets=qcomp_shard_offsets,
                            shard_sizes=qcomp_shard_sizes,
                            sharding_type=sharding_type,
                        )
    return ret
