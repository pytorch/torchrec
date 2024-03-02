#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import abc
import logging
from collections import defaultdict, OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
from torchrec.distributed.embedding_types import (
    EmbeddingComputeKernel,
    GroupedEmbeddingConfig,
    ShardedEmbeddingTable,
)
from torchrec.distributed.types import Shard, ShardedTensor, ShardedTensorMetadata
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

logger: logging.Logger = logging.getLogger(__name__)


class BaseEmbedding(abc.ABC, nn.Module):
    """
    Abstract base class for grouped `nn.Embedding` and `nn.EmbeddingBag`
    """

    @abc.abstractmethod
    def forward(
        self,
        features: KeyedJaggedTensor,
    ) -> torch.Tensor:
        """
        Args:
            features (KeyedJaggedTensor):
        Returns:
            torch.Tensor: sparse gradient parameter names.
        """
        pass

    @property
    @abc.abstractmethod
    def config(self) -> GroupedEmbeddingConfig:
        pass


def get_state_dict(
    embedding_tables: List[ShardedEmbeddingTable],
    params: Union[
        nn.ModuleList,
        List[Union[nn.Module, torch.Tensor]],
        List[torch.Tensor],
        List[Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]],
    ],
    pg: Optional[dist.ProcessGroup] = None,
    destination: Optional[Dict[str, Any]] = None,
    prefix: str = "",
) -> Dict[str, Any]:
    if destination is None:
        destination = OrderedDict()
        # pyre-ignore [16]
        destination._metadata = OrderedDict()
    """
    It is possible for there to be multiple shards from a table on a single rank.
    We accumulate them in key_to_local_shards. Repeat shards should have identical
    global ShardedTensorMetadata.
    """
    key_to_local_shards: Dict[str, List[Shard]] = defaultdict(list)
    key_to_global_metadata: Dict[str, ShardedTensorMetadata] = {}

    def get_key_from_embedding_table(embedding_table: ShardedEmbeddingTable) -> str:
        return prefix + f"{embedding_table.name}.weight"

    for embedding_table, param in zip(embedding_tables, params):
        key = get_key_from_embedding_table(embedding_table)
        is_quant = embedding_table.compute_kernel in [
            EmbeddingComputeKernel.QUANT,
            EmbeddingComputeKernel.QUANT_UVM,
            EmbeddingComputeKernel.QUANT_UVM_CACHING,
        ]
        qscale = None
        qbias = None
        if is_quant:
            # For QUANT* param is Tuple[torch.Tensor, Optional[torch.Tensor]] where first argument is the weight table, the second is optional quantization extra information, depending on quantization type. e.g. for fbgemm rowwise quantization this is scale and shift for each row.
            assert isinstance(param, tuple)
            qscale = param[1]
            qbias = param[2]
            param = param[0]

        assert embedding_table.local_rows == param.size(0)  # pyre-ignore[16]

        if qscale is not None:
            assert embedding_table.local_cols == param.size(1)  # pyre-ignore[16]

        if embedding_table.global_metadata is not None and pg is not None:
            # set additional field of sharded tensor based on local tensor properties
            embedding_table.global_metadata.tensor_properties.dtype = (
                param.dtype  # pyre-ignore[16]
            )
            embedding_table.global_metadata.tensor_properties.requires_grad = (
                param.requires_grad  # pyre-ignore[16]
            )
            key_to_global_metadata[key] = embedding_table.global_metadata

            key_to_local_shards[key].append(
                # pyre-fixme[6]: For 1st argument expected `Tensor` but got
                #  `Union[Module, Tensor]`.
                # pyre-fixme[6]: For 2nd argument expected `ShardMetadata` but got
                #  `Optional[ShardMetadata]`.
                Shard(param, embedding_table.local_metadata)
            )
        else:
            destination[key] = param
            if qscale is not None:
                destination[f"{key}_qscale"] = qscale
            if qbias is not None:
                destination[f"{key}_qbias"] = qbias

    if pg is not None:
        # Populate the remaining destinations that have a global metadata
        for key in key_to_local_shards:
            global_metadata = key_to_global_metadata[key]
            destination[
                key
            ] = ShardedTensor._init_from_local_shards_and_global_metadata(
                local_shards=key_to_local_shards[key],
                sharded_tensor_metadata=global_metadata,
                process_group=pg,
            )

    return destination
