#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import List, Optional, Tuple, Iterator

import torch
import torch.distributed as dist
from fbgemm_gpu.split_table_batched_embeddings_ops import (
    EmbeddingLocation,
    IntNBitTableBatchedEmbeddingBagsCodegen,
)
from torchrec.distributed.batched_embedding_kernel import BaseBatchedEmbeddingBag
from torchrec.distributed.embedding_types import GroupedEmbeddingConfig
from torchrec.distributed.utils import append_prefix
from torchrec.modules.embedding_configs import data_type_to_sparse_type
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

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
                        data_type_to_sparse_type(config.data_type),
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

    def init_parameters(self) -> None:
        pass

    @property
    def emb_module(
        self,
    ) -> IntNBitTableBatchedEmbeddingBagsCodegen:
        return self._emb_module

    def forward(self, features: KeyedJaggedTensor) -> torch.Tensor:
        return self.emb_module(
            indices=features.values().int(),
            offsets=features.offsets().int(),
            per_sample_weights=features.weights_or_none(),
        ).float()

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
