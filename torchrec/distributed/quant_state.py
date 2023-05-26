#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Any, Dict, List, Mapping, TypeVar, Union

import torch
from fbgemm_gpu.split_table_batched_embeddings_ops import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
)
from torch.distributed import _remote_device
from torch.distributed._shard.sharded_tensor import (
    Shard,
    ShardedTensorBase,
    ShardedTensorMetadata,
    ShardMetadata,
    TensorProperties,
)
from torchrec.distributed.embedding_types import (
    GroupedEmbeddingConfig,
    ShardedEmbeddingModule,
)
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
        tables_weights_prefix: str,  # "embedding_bags" or "embeddings"
    ) -> None:  # noqa
        assert (
            tables_weights_prefix == "embedding_bags"
            or tables_weights_prefix == "embeddings"
        )
        # pyre-ignore[16]
        self._table_name_to_local_shards: Dict[str, List[Shard]] = {}
        # pyre-ignore[16]
        self._table_name_to_sharded_tensor: Dict[
            str, Union[torch.Tensor, ShardedTensorBase]
        ] = {}
        # pyre-ignore[16]
        self._table_name_to_local_shards_qss: Dict[str, List[Shard]] = {}
        # pyre-ignore[16]
        self._table_name_to_sharded_tensor_qss: Dict[
            str, Union[torch.Tensor, ShardedTensorBase]
        ] = {}

        for tbe, config in tbes.items():
            for (tbe_split_w, tbe_split_wq), table in zip(
                tbe.split_embedding_weights(split_scale_shifts=True),
                config.embedding_tables,
            ):
                metadata = copy.deepcopy(table.local_metadata)
                metadata.shard_sizes = [tbe_split_w.size(0), tbe_split_w.size(1)]

                # TODO(ivankobzarev): only for "meta" sharding support: cleanup when copy to  "meta" moves all tensors to "meta"
                if metadata.placement.device != tbe_split_w.device:
                    metadata.placement = _remote_device(tbe_split_w.device)
                _append_table_shard(
                    # pyre-ignore
                    self._table_name_to_local_shards,
                    table.name,
                    Shard(tensor=tbe_split_w, metadata=metadata),
                )

                metadata = copy.deepcopy(table.local_metadata)
                shard_sizes = metadata.shard_sizes
                shard_sizes_cols = shard_sizes[1]
                shard_offsets = table.local_metadata.shard_offsets
                shard_offsets_cols = shard_offsets[1]
                col_idx = int(shard_offsets_cols / shard_sizes_cols)
                qss_cols_size = tbe_split_wq.shape[1]

                qss_metadata = ShardMetadata(
                    shard_offsets=[
                        metadata.shard_offsets[0],
                        col_idx * qss_cols_size,
                    ],
                    shard_sizes=[tbe_split_wq.shape[0], tbe_split_wq.shape[1]],
                    placement=table.local_metadata.placement,
                )
                # TODO(ivankobzarev): only for "meta" sharding support: cleanup when copy to  "meta" moves all tensors to "meta"
                # pyre-ignore[16]
                if qss_metadata.placement.device != tbe_split_wq.device:
                    qss_metadata.placement = _remote_device(tbe_split_wq.device)
                _append_table_shard(
                    # pyre-ignore
                    self._table_name_to_local_shards_qss,
                    table.name,
                    Shard(tensor=tbe_split_wq, metadata=qss_metadata),
                )

        for table_name_to_local_shards, table_name_to_sharded_tensor in [
            (self._table_name_to_local_shards, self._table_name_to_sharded_tensor),
            (
                self._table_name_to_local_shards_qss,
                self._table_name_to_sharded_tensor_qss,
            ),
        ]:
            # pyre-ignore
            for table_name, local_shards in table_name_to_local_shards.items():
                if len(local_shards) == 1:
                    # Single Tensor per table (TW sharding)
                    # pyre-ignore
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
                    tensor_properties=TensorProperties(
                        dtype=torch.uint8,
                    ),
                )
                # pyre-ignore
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
            ) in module._table_name_to_sharded_tensor.items():  # pyre-ignore
                destination[
                    f"{prefix}{tables_weights_prefix}.{table_name}.weight"
                ] = sharded_t
            for (
                table_name,
                sharded_t,
            ) in module._table_name_to_sharded_tensor_qss.items():  # pyre-ignore
                destination[
                    f"{prefix}{tables_weights_prefix}.{table_name}.weight_qscaleshift"
                ] = sharded_t

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
                        if cols_from >= t.shape[1]:
                            # CW sharding qscaleshift handle:
                            cols_from = 0
                        cols_to = cols_from + meta.shard_sizes[1]
                        dst_tensor.copy_(
                            t[
                                rows_from:rows_to,
                                cols_from:cols_to,
                            ]
                        )
                    else:
                        raise RuntimeError("Tensors with ndim > 2 are not supported")
            else:
                dst_tensor.copy_(src_tensor)

            _unexpected_keys.remove(src_state_dict_name)
        missing_keys.extend(_missing_keys)
        unexpected_keys.extend(_unexpected_keys)
