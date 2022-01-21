#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import itertools
import logging
from typing import List, Optional, Tuple, Iterator

import torch
import torch.distributed as dist
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops import (
    EmbeddingLocation,
    IntNBitTableBatchedEmbeddingBagsCodegen,
    rounded_row_size_in_bytes,
)
from torchrec.distributed.batched_embedding_kernel import BaseBatchedEmbeddingBag
from torchrec.distributed.embedding_kernel import BaseEmbeddingBag
from torchrec.distributed.embedding_types import GroupedEmbeddingConfig
from torchrec.distributed.utils import append_prefix
from torchrec.modules.embedding_configs import (
    DataType,
    DATA_TYPE_NUM_BITS,
)
from torchrec.sparse.jagged_tensor import (
    KeyedJaggedTensor,
    KeyedTensor,
)

logger: logging.Logger = logging.getLogger(__name__)


class QuantBatchedEmbeddingBag(BaseBatchedEmbeddingBag):
    def __init__(
        self,
        config: GroupedEmbeddingConfig,
        # pyre-fixme[11]
        pg: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(config, pg, device)

        self._emb_module: IntNBitTableBatchedEmbeddingBagsCodegen = (
            IntNBitTableBatchedEmbeddingBagsCodegen(
                embedding_specs=[
                    (
                        "",
                        local_rows,
                        table.embedding_dim,
                        QuantBatchedEmbeddingBag.to_sparse_type(config.data_type),
                        EmbeddingLocation.DEVICE
                        if (device is not None and device.type == "cuda")
                        else EmbeddingLocation.HOST,
                    )
                    for local_rows, table in zip(
                        self._local_rows, config.embedding_tables
                    )
                ],
                device=device,
                pooling_mode=self._pooling,
                feature_table_map=self._feature_table_map,
            )
        )
        if device is not None and device.type != "meta":
            self._emb_module.initialize_weights()

    @staticmethod
    def to_sparse_type(data_type: DataType) -> SparseType:
        if data_type == DataType.FP16:
            return SparseType.FP16
        elif data_type == DataType.INT8:
            return SparseType.INT8
        elif data_type == DataType.INT4:
            return SparseType.INT4
        elif data_type == DataType.INT2:
            return SparseType.INT2
        else:
            raise ValueError(f"Invalid DataType {data_type}")

    def init_parameters(self) -> None:
        pass

    @property
    def emb_module(
        self,
    ) -> IntNBitTableBatchedEmbeddingBagsCodegen:
        return self._emb_module

    def forward(self, features: KeyedJaggedTensor) -> KeyedTensor:
        values = self.emb_module(
            indices=features.values().int(),
            offsets=features.offsets().int(),
            per_sample_weights=features.weights_or_none(),
        ).float()
        return KeyedTensor(
            keys=self._emb_names,
            values=values,
            length_per_key=self._lengths_per_emb,
        )

    def named_buffers(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
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
    def from_float(cls, module: BaseEmbeddingBag) -> "QuantBatchedEmbeddingBag":
        assert hasattr(
            module, "qconfig"
        ), "EmbeddingBagCollectionInterface input float module must have qconfig defined"

        def _to_data_type(dtype: torch.dtype) -> DataType:
            if dtype == torch.quint8 or dtype == torch.qint8:
                return DataType.INT8
            elif dtype == torch.quint4 or dtype == torch.qint4:
                return DataType.INT4
            elif dtype == torch.quint2 or dtype == torch.qint2:
                return DataType.INT2
            else:
                raise Exception(f"Invalid data type {dtype}")

        # pyre-ignore [16]
        data_type = _to_data_type(module.qconfig.weight().dtype)
        sparse_type = QuantBatchedEmbeddingBag.to_sparse_type(data_type)

        state_dict = dict(
            itertools.chain(module.named_buffers(), module.named_parameters())
        )
        device = next(iter(state_dict.values())).device

        # Adjust config to quantized version.
        # This obviously doesn't work for column-wise sharding.
        # pyre-ignore [29]
        config = copy.deepcopy(module.config())
        config.data_type = data_type
        for table in config.embedding_tables:
            table.local_cols = rounded_row_size_in_bytes(table.local_cols, sparse_type)
            if table.local_metadata is not None:
                table.local_metadata.shard_sizes = [
                    table.local_rows,
                    table.local_cols,
                ]

            if table.global_metadata is not None:
                for shard_meta in table.global_metadata.shards_metadata:
                    if shard_meta != table.local_metadata:
                        shard_meta.shard_sizes = [
                            shard_meta.shard_sizes[0],
                            rounded_row_size_in_bytes(
                                shard_meta.shard_sizes[1], sparse_type
                            ),
                        ]
                table.global_metadata.size = torch.Size(
                    [
                        table.global_metadata.size[0],
                        sum(
                            shard_meta.shard_sizes[1]
                            for shard_meta in table.global_metadata.shards_metadata
                        ),
                    ]
                )

        ret = QuantBatchedEmbeddingBag(config=config, device=device)

        # Quantize weights.
        quant_weight_list = []
        for _, weight in state_dict.items():
            quantized_weights = torch.ops.fbgemm.FloatToFusedNBitRowwiseQuantizedSBHalf(
                weight, DATA_TYPE_NUM_BITS[data_type]
            )
            # weight and 4 byte scale shift (2xfp16)
            quant_weight = quantized_weights[:, :-4]
            scale_shift = quantized_weights[:, -4:]

            quant_weight_list.append((quant_weight, scale_shift))
        ret.emb_module.assign_embedding_weights(quant_weight_list)

        return ret
