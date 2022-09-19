#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import abc
import logging
from collections import defaultdict, OrderedDict
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, cast
import torch
import torch.distributed as dist
from torch import nn
from torchrec.distributed.embedding_types import (
    EmbeddingComputeKernel,
    GroupedEmbeddingConfig,
    ShardedEmbeddingTable,
)
from torchrec.distributed.types import Shard, ShardedTensor, ShardedTensorMetadata
from torchrec.distributed.utils import append_prefix
from torchrec.modules.embedding_configs import pooling_type_to_str
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
logger: logging.Logger = logging.getLogger(__name__)
from torchrec.distributed.embedding_kernel import BaseEmbedding, get_state_dict
import numpy as np

try:
    from colossalai.nn.parallel.layers.cache_embedding import FreqAwareEmbeddingBag
except ImportError: 
    print('install colossalai')

from .batched_embedding_kernel import BaseBatchedEmbeddingBag
from torchrec.modules.embedding_configs import pooling_mode_to_str
                
class CAIGroupedEmbeddingBag(BaseEmbedding):
    def __init__(
        self,
        config: GroupedEmbeddingConfig,
        sparse: bool,
        pg: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
        use_cache : bool = True,
        cache_ratio : float = 1.0,
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torchrec.distributed.{self.__class__.__name__}")
        self._config = config
        # pyre-fixme[4]: Attribute must be annotated.
        self._pg = pg
        self._emb_modules: nn.ModuleList = nn.ModuleList()
        self._sparse = sparse
        self._emb_names: List[str] = []
        self._lengths_per_emb: List[int] = []
        
        for embedding_config in self._config.embedding_tables:
            if use_cache:
                emb = FreqAwareEmbeddingBag(
                        num_embeddings=embedding_config.local_rows,
                        embedding_dim=embedding_config.local_cols,
                        mode=pooling_type_to_str(embedding_config.pooling),
                        include_last_offset=True,
                        sparse=self._sparse,
                        _weight=torch.empty(
                            embedding_config.local_rows,
                            embedding_config.local_cols,
                            device='cpu',
                        ).uniform_(
                            embedding_config.get_weight_init_min(),
                            embedding_config.get_weight_init_max(),
                        ),
                        cuda_row_num = int(embedding_config.local_rows * cache_ratio),
                    )
                self._emb_modules.append(
                    emb
                )
            else:   
                self._emb_modules.append(
                    nn.EmbeddingBag(
                        num_embeddings=embedding_config.local_rows,
                        embedding_dim=embedding_config.local_cols,
                        mode=pooling_type_to_str(embedding_config.pooling),
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

    def forward(self, features: KeyedJaggedTensor) -> torch.Tensor:
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
        return torch.cat(pooled_embeddings, dim=1)
    # pyre-fixme[14]: `state_dict` overrides method defined in `Module` inconsistently.
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
    def named_buffers(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
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
    @property
    def config(self) -> GroupedEmbeddingConfig:
        return self._config
s
class CAIBatchedDenseEmbeddingBag(BaseBatchedEmbeddingBag):
    def __init__(
        self,
        config: GroupedEmbeddingConfig,
        pg: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
        cache_ratio : float = 0.01, 
    ) -> None:
        super().__init__(config, pg, device)

        #  fused multiple embedding bags into a single one as self._emb_module
        # replace fbgemm implementation with colossalai FAW
        # Table-batched version of nn.EmbeddingBag(sparse=False)
        # https://github.com/pytorch/FBGEMM/blob/1a61102ad65af645cdd9d4a78b6dfd6388dc7735/fbgemm_gpu/fbgemm_gpu/split_table_batched_embeddings_ops.py
        # self._emb_module: DenseTableBatchedEmbeddingBagsCodegen = (
        #     DenseTableBatchedEmbeddingBagsCodegen(
        #         list(zip(self._local_rows, self._local_cols)),
        #         feature_table_map=self._feature_table_map,
        #         pooling_mode=self._pooling,
        #         use_cpu=device is None
        #         or device.type == "cpu"
        #         or not torch.cuda.is_available(),
        #     )
        # )

        num_embeddings = sum(self._num_embeddings)
        assert all(x == self._local_cols[0] for x in self._local_cols), "local col should be consistent in all embeddings"
        embedding_dim = self._local_cols[0]
        pool_str = pooling_mode_to_str(self._pooling)
        
        self._weight_list: List[torch.Tensor] = []
        for embedding_config in self._config.embedding_tables:
            self._weight_list.append(torch.empty(
                        embedding_config.local_rows,
                        embedding_config.local_cols,
                        device='cpu',
                    ).uniform_(
                        embedding_config.get_weight_init_min(),
                        embedding_config.get_weight_init_max(),
                    )
            )
        
        self._emb_module = FreqAwareEmbeddingBag(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            mode=pool_str,
            include_last_offset=True,
            _weight=torch.cat(self._weight_list, 0),
            cuda_row_num = int(num_embeddings * cache_ratio),
        )
        # prepare for features concatenation
        self._table_idx_offset_list = np.cumsum([0] + self._num_embeddings[:-1])
        
        # TODO() not support split_embedding_weights currently
        # init parameter by uniformly init the _weight
        # self.init_parameters()

    @property
    def emb_module(
        self,
    ):
        return self._emb_module

    def named_buffers(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        yield from ()

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        combined_key = "/".join(
            [config.name for config in self._config.embedding_tables]
        )
        yield append_prefix(prefix, f"{combined_key}.weight"), cast(
            nn.Parameter, self._emb_module.weight
        )

    def concatenate_offsets(self, offsets_list: List[torch.Tensor]) -> torch.Tensor:
        # assert include last offset
        offset_hold = 0
        output: List[torch.Tensor] = []
        for offsets in offsets_list:
            offsets = torch.add(offsets, offset_hold)
            offset_hold = offsets[-1]
            output.append(offsets[:-1])
        offset_hold = torch.tensor([offset_hold], dtype= offsets_list[0].dtype,
                                   device=offsets_list[0].device)
        return torch.cat(output+[offset_hold], 0)
    
    def forward(self, features: KeyedJaggedTensor) -> torch.Tensor:
        batch_size = len(features._lengths)//len(features._keys)
        values_list: List[torch.Tensor] = []
        offsets_list: List[torch.Tensor] = []
        weights_list: List[torch.Tensor] = []
        for i, embedding_config in enumerate(self._config.embedding_tables):
            values_in_table = torch.empty(0,device=torch.cuda.current_device())
            offsets_in_table = torch.empty(0,device=torch.cuda.current_device())
            weights_in_table = torch.empty(0,device=torch.cuda.current_device())
            for feature_name in embedding_config.feature_names:
                values_in_table = torch.cat((values_in_table,features[feature_name].values()))
                offsets_in_table = torch.cat((offsets_in_table,features[feature_name].offsets()))
                weights = features[feature_name].weights_or_none()
                if weights is not None and not torch.is_floating_point(weights):
                    weights = None
                if weights is not None:
                    weights_in_table = torch.cat((weights_in_table, weights))
            values_list.append(values_in_table.add(self._table_idx_offset_list[i]))
            offsets_list.append(offsets_in_table)
            weights_list.append(weights_in_table)
            
        values_full = torch.cat(values_list)
        offsets_full = self.concatenate_offsets(offsets_list)
        weights_full = torch.cat(weights_list)
        # TODO: pad weights
        assert len(weights_full) == 0 or len(weights_full) == len(values_full)
        if len(weights_full) == 0:
            weights_full = None
        output = self.emb_module(values_full.type(torch.int), offsets_full.type(torch.int), weights_full)
        output = output.reshape(batch_size,-1)
        return output