#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
import logging
from collections import defaultdict, OrderedDict
from typing import List, Optional, Dict, Any, Union, Tuple, Iterator

import torch
import torch.distributed as dist
from torch import nn
from torchrec.distributed.embedding_types import (
    ShardedEmbeddingTable,
    GroupedEmbeddingConfig,
)
from torchrec.distributed.types import (
    Shard,
    ShardedTensorMetadata,
    ShardedTensor,
)
from torchrec.distributed.utils import append_prefix
from torchrec.modules.embedding_configs import PoolingType
from torchrec.sparse.jagged_tensor import (
    KeyedJaggedTensor,
    KeyedTensor,
)

logger: logging.Logger = logging.getLogger(__name__)


class BaseEmbedding(abc.ABC, nn.Module):
    """
    abstract base class for grouped nn.Embedding
    """

    @abc.abstractmethod
    def forward(
        self,
        features: KeyedJaggedTensor,
    ) -> torch.Tensor:
        pass

    """
    return sparse gradient parameter names
    """

    def sparse_grad_parameter_names(
        self, destination: Optional[List[str]] = None, prefix: str = ""
    ) -> List[str]:
        destination = [] if destination is None else destination
        return destination


def get_state_dict(
    embedding_tables: List[ShardedEmbeddingTable],
    params: Union[
        nn.ModuleList,
        List[Union[nn.Module, torch.Tensor]],
        List[torch.Tensor],
    ],
    # pyre-fixme[11]
    pg: Optional[dist.ProcessGroup] = None,
    destination: Optional[Dict[str, Any]] = None,
    prefix: str = "",
) -> Dict[str, Any]:
    if destination is None:
        destination = OrderedDict()
        # pyre-ignore [16]
        destination._metadata = OrderedDict()

    """ It is possible for there to be multiple shards from a table on a single rank. """
    """ We accumulate them in key_to_local_shards. Repeat shards should have identical """
    """ global ShardedTensorMetadata"""
    key_to_local_shards: Dict[str, List[Shard]] = defaultdict(list)
    key_to_global_metadata: Dict[str, ShardedTensorMetadata] = {}

    def get_key_from_embedding_table(embedding_table: ShardedEmbeddingTable) -> str:
        return prefix + f"{embedding_table.name}.weight"

    for embedding_table, param in zip(embedding_tables, params):
        key = get_key_from_embedding_table(embedding_table)
        assert embedding_table.local_rows == param.size(0)
        assert embedding_table.local_cols == param.size(1)
        if embedding_table.global_metadata is not None:
            # set additional field of sharded tensor based on local tensor properties
            embedding_table.global_metadata.tensor_properties.dtype = param.dtype
            embedding_table.global_metadata.tensor_properties.requires_grad = (
                param.requires_grad
            )
            key_to_global_metadata[key] = embedding_table.global_metadata

            key_to_local_shards[key].append(
                Shard(param, embedding_table.local_metadata)
            )
        else:
            destination[key] = param

    # Populate the remaining destinations that have a global metadata
    for key in key_to_local_shards:
        global_metadata = key_to_global_metadata[key]
        destination[key] = ShardedTensor._init_from_local_shards_and_global_metadata(
            local_shards=key_to_local_shards[key],
            sharded_tensor_metadata=global_metadata,
            process_group=pg,
        )

    return destination


class GroupedEmbedding(BaseEmbedding):
    def __init__(
        self,
        config: GroupedEmbeddingConfig,
        sparse: bool,
        pg: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torchrec.distributed.{self.__class__.__name__}")
        self._config = config
        # pyre-fixme[4]: Attribute must be annotated.
        self._pg = pg
        # pyre-fixme[24]: Non-generic type `nn.modules.container.ModuleList` cannot
        #  take parameters.
        self._emb_modules: nn.ModuleList[nn.Module] = nn.ModuleList()
        self._sparse = sparse
        for embedding_config in self._config.embedding_tables:
            self._emb_modules.append(
                nn.Embedding(
                    num_embeddings=embedding_config.local_rows,
                    embedding_dim=embedding_config.local_cols,
                    device=device,
                    sparse=self._sparse,
                    _weight=torch.empty(
                        embedding_config.local_rows,
                        embedding_config.local_cols,
                        device=device,
                    ).uniform_(
                        embedding_config.get_weight_init_min(),
                        embedding_config.get_weight_init_max(),
                    ),
                )
            )

    def forward(self, features: KeyedJaggedTensor) -> torch.Tensor:
        indices_dict: Dict[str, torch.Tensor] = {}
        indices_list = torch.split(features.values(), features.length_per_key())
        for key, indices in zip(features.keys(), indices_list):
            indices_dict[key] = indices
        unpooled_embeddings: List[torch.Tensor] = []
        for embedding_config, emb_module in zip(
            self._config.embedding_tables, self._emb_modules
        ):
            for feature_name in embedding_config.feature_names:
                unpooled_embeddings.append(emb_module(input=indices_dict[feature_name]))
        return torch.cat(unpooled_embeddings, dim=0)

    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        params = [
            emb_module.weight if keep_vars else emb_module.weight.data
            for emb_module in self._emb_modules
        ]
        return get_state_dict(
            self._config.embedding_tables, params, self._pg, destination, prefix
        )

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        for config, emb_module in zip(
            self._config.embedding_tables,
            self._emb_modules,
        ):
            param = emb_module.weight
            assert config.local_rows == param.size(0)
            assert config.local_cols == param.size(1)
            yield append_prefix(prefix, f"{config.name}.weight"), param

    def sparse_grad_parameter_names(
        self, destination: Optional[List[str]] = None, prefix: str = ""
    ) -> List[str]:
        destination = [] if destination is None else destination
        if self._sparse:
            for config in self._config.embedding_tables:
                destination.append(append_prefix(prefix, f"{config.name}.weight"))
        return destination

    def config(self) -> GroupedEmbeddingConfig:
        return self._config


class BaseEmbeddingBag(nn.Module):
    """
    abstract base class for grouped nn.EmbeddingBag
    """

    """
    return sparse gradient parameter names
    """

    def sparse_grad_parameter_names(
        self, destination: Optional[List[str]] = None, prefix: str = ""
    ) -> List[str]:
        destination = [] if destination is None else destination
        return destination

    @property
    @abc.abstractmethod
    def config(self) -> GroupedEmbeddingConfig:
        pass


class GroupedEmbeddingBag(BaseEmbeddingBag):
    def __init__(
        self,
        config: GroupedEmbeddingConfig,
        sparse: bool,
        pg: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        def _to_mode(pooling: PoolingType) -> str:
            if pooling == PoolingType.SUM:
                return "sum"
            elif pooling == PoolingType.MEAN:
                return "mean"
            else:
                raise ValueError(f"Unsupported pooling {pooling}")

        super().__init__()
        torch._C._log_api_usage_once(f"torchrec.distributed.{self.__class__.__name__}")
        self._config = config
        # pyre-fixme[4]: Attribute must be annotated.
        self._pg = pg
        # pyre-fixme[24]: Non-generic type `nn.modules.container.ModuleList` cannot
        #  take parameters.
        self._emb_modules: nn.ModuleList[nn.Module] = nn.ModuleList()
        self._sparse = sparse
        self._emb_names: List[str] = []
        self._lengths_per_emb: List[int] = []

        shared_feature: Dict[str, bool] = {}
        for embedding_config in self._config.embedding_tables:
            self._emb_modules.append(
                nn.EmbeddingBag(
                    num_embeddings=embedding_config.local_rows,
                    embedding_dim=embedding_config.local_cols,
                    mode=_to_mode(embedding_config.pooling),
                    device=device,
                    include_last_offset=True,
                    sparse=self._sparse,
                    _weight=torch.empty(
                        embedding_config.local_rows,
                        embedding_config.local_cols,
                        device=device,
                    ).uniform_(
                        embedding_config.get_weight_init_min(),
                        embedding_config.get_weight_init_max(),
                    ),
                )
            )
            for feature_name in embedding_config.feature_names:
                if feature_name not in shared_feature:
                    shared_feature[feature_name] = False
                else:
                    shared_feature[feature_name] = True
                self._lengths_per_emb.append(embedding_config.embedding_dim)

        for embedding_config in self._config.embedding_tables:
            for feature_name in embedding_config.feature_names:
                if shared_feature[feature_name]:
                    self._emb_names.append(feature_name + "@" + embedding_config.name)
                else:
                    self._emb_names.append(feature_name)

    def forward(self, features: KeyedJaggedTensor) -> KeyedTensor:
        pooled_embeddings: List[torch.Tensor] = []
        for embedding_config, emb_module in zip(
            self._config.embedding_tables, self._emb_modules
        ):
            for feature_name in embedding_config.feature_names:
                values = features[feature_name].values()
                offsets = features[feature_name].offsets()
                weights = features[feature_name].weights_or_none()
                if weights is not None and not torch.is_floating_point(weights):
                    weights = None
                pooled_embeddings.append(
                    emb_module(
                        input=values,
                        offsets=offsets,
                        per_sample_weights=weights,
                    )
                )
        return KeyedTensor(
            keys=self._emb_names,
            values=torch.cat(pooled_embeddings, dim=1),
            length_per_key=self._lengths_per_emb,
        )

    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        params = [
            emb_module.weight if keep_vars else emb_module.weight.data
            for emb_module in self._emb_modules
        ]
        return get_state_dict(
            self._config.embedding_tables, params, self._pg, destination, prefix
        )

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        for config, emb_module in zip(
            self._config.embedding_tables,
            self._emb_modules,
        ):
            param = emb_module.weight
            assert config.local_rows == param.size(0)
            assert config.local_cols == param.size(1)
            yield append_prefix(prefix, f"{config.name}.weight"), param

    def sparse_grad_parameter_names(
        self, destination: Optional[List[str]] = None, prefix: str = ""
    ) -> List[str]:
        destination = [] if destination is None else destination
        if self._sparse:
            for config in self._config.embedding_tables:
                destination.append(append_prefix(prefix, f"{config.name}.weight"))
        return destination

    def config(self) -> GroupedEmbeddingConfig:
        return self._config
