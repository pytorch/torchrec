#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops import (
    EmbeddingLocation,
    IntNBitTableBatchedEmbeddingBagsCodegen,
    PoolingMode,
    rounded_row_size_in_bytes,
)
from torchrec.distributed.batched_embedding_kernel import (
    BaseBatchedEmbedding,
    BaseBatchedEmbeddingBag,
    BatchedDenseEmbedding,
    BatchedDenseEmbeddingBag,
)
from torchrec.distributed.embedding_kernel import BaseEmbedding
from torchrec.distributed.embedding_types import (
    compute_kernel_to_embedding_location,
    GroupedEmbeddingConfig,
)
from torchrec.distributed.utils import append_prefix
from torchrec.modules.embedding_configs import (
    DATA_TYPE_NUM_BITS,
    data_type_to_sparse_type,
    DataType,
    dtype_to_data_type,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


logger: logging.Logger = logging.getLogger(__name__)


def _copy_config(
    original: GroupedEmbeddingConfig,
    data_type: DataType,
    sparse_type: SparseType,
    device: torch.device,
) -> GroupedEmbeddingConfig:
    # Adjust config to quantized version.
    # This obviously doesn't work for column-wise sharding.
    config = copy.deepcopy(original)
    config.data_type = data_type
    for table in config.embedding_tables:
        row_alignment = 16 if device.type == "cuda" else 1
        table.local_cols = rounded_row_size_in_bytes(
            table.local_cols, sparse_type, row_alignment
        )
        if table.local_metadata is not None:
            table.local_metadata.shard_sizes = [
                table.local_rows,
                table.local_cols,
            ]

        global_metadata = table.global_metadata
        if global_metadata is not None:
            for shard_meta in global_metadata.shards_metadata:
                if shard_meta != table.local_metadata:
                    shard_meta.shard_sizes = [
                        shard_meta.shard_sizes[0],
                        rounded_row_size_in_bytes(
                            shard_meta.shard_sizes[1], sparse_type, row_alignment
                        ),
                    ]
            global_metadata.size = torch.Size(
                [
                    global_metadata.size[0],
                    sum(
                        shard_meta.shard_sizes[1]
                        for shard_meta in global_metadata.shards_metadata
                    ),
                ]
            )
    return config


def _quantize_weight(
    state_dict: Dict[str, torch.Tensor],
    data_type: DataType,
) -> List[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
    quant_weight_list = []
    for weight in state_dict.values():
        if weight.dtype == torch.float or weight.dtype == torch.float16:
            quantized_weights = (
                torch.ops.fbgemm.FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf(
                    weight, DATA_TYPE_NUM_BITS[data_type]
                )
            )
        else:
            raise Exception("Unsupported dtype: {weight.dtype}")

        # weight and 4 byte scale shift (2xfp16)
        quant_weight = quantized_weights[:, :-4]
        scale_shift = quantized_weights[:, -4:]

        quant_weight_list.append((quant_weight, scale_shift))
    return quant_weight_list


class QuantBatchedEmbeddingBag(BaseBatchedEmbeddingBag):
    def __init__(
        self,
        config: GroupedEmbeddingConfig,
        pg: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
        fused_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(config, pg, device)

        managed: List[EmbeddingLocation] = []
        for table in config.embedding_tables:
            if device is not None and device.type == "cuda":
                managed.append(
                    compute_kernel_to_embedding_location(table.compute_kernel)
                )
            else:
                managed.append(EmbeddingLocation.HOST)
        self._emb_module: IntNBitTableBatchedEmbeddingBagsCodegen = IntNBitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (
                    "",
                    local_rows,
                    table.embedding_dim,
                    data_type_to_sparse_type(config.data_type),
                    location,
                )
                for local_rows, table, location in zip(
                    self._local_rows, config.embedding_tables, managed
                )
            ],
            device=device,
            pooling_mode=self._pooling,
            feature_table_map=self._feature_table_map,
            row_alignment=16,
            uvm_host_mapped=True,  # Use cudaHostAlloc for UVM CACHING to fix imbalance numa memory issue
            **(fused_params or {}),
        )
        if device is not None and device.type != "meta":
            self._emb_module.initialize_weights()

    def init_parameters(self) -> None:
        pass

    @property
    def emb_module(
        self,
    ) -> IntNBitTableBatchedEmbeddingBagsCodegen:
        return self._emb_module

    def forward(self, features: KeyedJaggedTensor) -> torch.Tensor:
        return self.emb_module.forward(
            indices=features.values().int(),
            offsets=features.offsets().int(),
            per_sample_weights=features.weights_or_none(),
        )

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        assert (
            remove_duplicate
        ), "remove_duplicate=False not supported in QuantBatchedEmbeddingBag.named_split_embedding_weights"
        for config, weight in zip(
            self._config.embedding_tables,
            self.emb_module.split_embedding_weights(),
        ):
            yield append_prefix(prefix, f"{config.name}.weight"), weight[0]

    def split_embedding_weights(self) -> List[torch.Tensor]:
        return [
            weight
            for weight, _ in self.emb_module.split_embedding_weights(
                split_scale_shifts=False
            )
        ]

    @classmethod
    def from_float(cls, module: BaseEmbedding) -> "QuantBatchedEmbeddingBag":
        assert hasattr(
            module, "qconfig"
        ), "BaseEmbedding input float module must have qconfig defined"

        # pyre-ignore [16]
        data_type = dtype_to_data_type(module.qconfig.weight().dtype)
        sparse_type = data_type_to_sparse_type(data_type)

        # TODO Can we simplify this with state_dict = module.state_dict()?
        state_dict = (
            dict(module.named_split_embedding_weights())
            if isinstance(module, BatchedDenseEmbeddingBag)
            else dict(module.named_parameters())
        )
        device = next(iter(state_dict.values())).device

        config = _copy_config(module.config, data_type, sparse_type, device)
        ret = QuantBatchedEmbeddingBag(config=config, device=device)

        # pyre-ignore
        quant_weight_list = _quantize_weight(state_dict, data_type)
        ret.emb_module.assign_embedding_weights(quant_weight_list)

        return ret


class QuantBatchedEmbedding(BaseBatchedEmbedding):
    def __init__(
        self,
        config: GroupedEmbeddingConfig,
        pg: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
        fused_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(config, pg, device)

        managed: List[EmbeddingLocation] = []
        for table in config.embedding_tables:
            if device is not None and device.type == "cuda":
                managed.append(
                    compute_kernel_to_embedding_location(table.compute_kernel)
                )
            else:
                managed.append(EmbeddingLocation.HOST)
        self._emb_module: IntNBitTableBatchedEmbeddingBagsCodegen = IntNBitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (
                    "",
                    local_rows,
                    table.embedding_dim,
                    data_type_to_sparse_type(config.data_type),
                    location,
                )
                for local_rows, table, location in zip(
                    self._local_rows, config.embedding_tables, managed
                )
            ],
            device=device,
            pooling_mode=PoolingMode.NONE,
            feature_table_map=self._feature_table_map,
            row_alignment=16,
            uvm_host_mapped=True,  # Use cudaHostAlloc for UVM CACHING to fix imbalance numa memory issue
            **(fused_params or {}),
        )
        if device is not None and device.type != "meta":
            self._emb_module.initialize_weights()

    @property
    def emb_module(
        self,
    ) -> IntNBitTableBatchedEmbeddingBagsCodegen:
        return self._emb_module

    def split_embedding_weights(self) -> List[torch.Tensor]:
        return [
            weight
            for weight, _ in self.emb_module.split_embedding_weights(
                split_scale_shifts=False
            )
        ]

    def forward(self, features: KeyedJaggedTensor) -> torch.Tensor:
        return self.emb_module(
            indices=features.values().int(),
            offsets=features.offsets().int(),
        )

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        for config, weight in zip(
            self._config.embedding_tables,
            self.emb_module.split_embedding_weights(),
        ):
            yield append_prefix(prefix, f"{config.name}.weight"), weight[0]

    @classmethod
    def from_float(cls, module: BaseEmbedding) -> "QuantBatchedEmbedding":
        assert hasattr(
            module, "qconfig"
        ), "BaseEmbedding input float module must have qconfig defined"

        # pyre-ignore [16]
        data_type = dtype_to_data_type(module.qconfig.weight().dtype)
        sparse_type = data_type_to_sparse_type(data_type)

        # TODO Can we simplify this with state_dict = module.state_dict()?
        state_dict = (
            dict(module.named_split_embedding_weights())
            if isinstance(module, BatchedDenseEmbedding)
            else dict(module.named_parameters())
        )
        device = next(iter(state_dict.values())).device

        config = _copy_config(module.config, data_type, sparse_type, device)
        ret = QuantBatchedEmbedding(config=config, device=device)

        # pyre-ignore
        quant_weight_list = _quantize_weight(state_dict, data_type)
        ret.emb_module.assign_embedding_weights(quant_weight_list)

        return ret
