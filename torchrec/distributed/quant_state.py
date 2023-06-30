#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Any, Dict, List, Mapping, TypeVar, Union

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
from torchrec.distributed.embedding_types import (
    GroupedEmbeddingConfig,
    ShardedEmbeddingModule,
)
from torchrec.distributed.types import ParameterSharding, ShardingType
from torchrec.streamable import Multistreamable

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
        self._table_name_to_local_shards: Dict[str, List[Shard]] = {}
        self._table_name_to_sharded_tensor: Dict[
            str, Union[torch.Tensor, ShardedTensorBase]
        ] = {}

        # weight_qscale
        self._table_name_to_local_shards_qscale: Dict[str, List[Shard]] = {}
        self._table_name_to_sharded_tensor_qscale: Dict[
            str, Union[torch.Tensor, ShardedTensorBase]
        ] = {}
        self._table_name_to_tensors_list_qscale: Dict[str, List[torch.Tensor]] = {}

        # weight_qbias
        self._table_name_to_local_shards_qbias: Dict[str, List[Shard]] = {}
        self._table_name_to_sharded_tensor_qbias: Dict[
            str, Union[torch.Tensor, ShardedTensorBase]
        ] = {}
        self._table_name_to_tensors_list_qbias: Dict[str, List[torch.Tensor]] = {}

        for tbe, config in tbes.items():
            for (tbe_split_w, tbe_split_qscale, tbe_split_qbias), table in zip(
                tbe.split_embedding_weights_with_scale_bias(split_scale_bias_mode=2),
                config.embedding_tables,
            ):
                # weight shards section:
                assert table.local_metadata
                metadata: ShardMetadata = copy.deepcopy(table.local_metadata)
                metadata.shard_sizes = [tbe_split_w.size(0), tbe_split_w.size(1)]

                # TODO(ivankobzarev): "meta" sharding support: cleanup when copy to "meta" moves all tensors to "meta"
                # pyre-ignore
                if metadata.placement.device != tbe_split_w.device:
                    metadata.placement = _remote_device(tbe_split_w.device)
                _append_table_shard(
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
                        if table.name not in table_name_to_tensors_list:
                            assert parameter_sharding.ranks
                            num_shards: int = len(parameter_sharding.ranks)
                            table_name_to_tensors_list[table.name] = [
                                torch.empty([])
                            ] * num_shards

                        column_idx = int(shard_offsets_cols / shard_sizes_cols)
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
            for table_name, local_shards in table_name_to_local_shards.items():
                if len(local_shards) == 1:
                    # Single Tensor per table (TW sharding)
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
                table_name_to_sharded_tensor[
                    table_name
                ] = ShardedTensorBase._init_from_local_shards_and_global_metadata(
                    local_shards=local_shards,
                    sharded_tensor_metadata=global_metadata,
                )

        def post_state_dict_hook(
            # Union["ShardedQuantEmbeddingBagCollection", "ShardedQuantEmbeddingCollection"]
            module: ShardedQuantEmbeddingModuleState[CompIn, DistOut, Out, ShrdCtx],
            destination: Dict[str, torch.Tensor],
            prefix: str,
            _local_metadata: Dict[str, Any],
        ) -> None:
            for (
                table_name,
                sharded_t,
            ) in module._table_name_to_sharded_tensor.items():
                destination[
                    f"{prefix}{tables_weights_prefix}.{table_name}.weight"
                ] = sharded_t

            for sfx, dict_sharded_t, dict_t_list in [
                (
                    "qscale",
                    module._table_name_to_sharded_tensor_qscale,
                    module._table_name_to_tensors_list_qscale,
                ),
                (
                    "qbias",
                    module._table_name_to_sharded_tensor_qbias,
                    module._table_name_to_tensors_list_qbias,
                ),
            ]:
                for (
                    table_name,
                    sharded_t,
                ) in dict_sharded_t.items():
                    destination[
                        f"{prefix}{tables_weights_prefix}.{table_name}.weight_{sfx}"
                    ] = sharded_t
                for (
                    table_name,
                    t_list,
                ) in dict_t_list.items():
                    destination[
                        f"{prefix}{tables_weights_prefix}.{table_name}.weight_{sfx}"
                    ] = t_list

        self._register_state_dict_hook(post_state_dict_hook)

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
