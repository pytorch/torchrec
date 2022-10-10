#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from torchrec.modules.embedding_configs import pooling_mode_to_str
from .batched_embedding_kernel import BaseBatchedEmbeddingBag
from torch.profiler import record_function
from torchrec.distributed.embedding_kernel import BaseEmbedding, get_state_dict
import logging
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, cast
import torch
import torch.distributed as dist
from torch import nn
from torchrec.distributed.embedding_types import (
    EmbeddingComputeKernel,
    GroupedEmbeddingConfig,
    ShardedEmbeddingTable,
)
from torchrec.distributed.utils import append_prefix
from torchrec.modules.embedding_configs import pooling_type_to_str
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
logger: logging.Logger = logging.getLogger(__name__)

try:
    from colossalai.nn.parallel.layers.cache_embedding import FreqAwareEmbeddingBag
except ImportError:
    print('please pip install colossalai')

from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType, SparseType

class CAIGroupedEmbeddingBag(BaseEmbedding):
    def __init__(
        self,
        config: GroupedEmbeddingConfig,
        pg: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
        use_cache: bool = True,
        cache_ratio: float = 1.0,
        sparse: bool = False
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(
            f"torchrec.distributed.{self.__class__.__name__}")
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
                    cache_ratio=cache_ratio,
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
                destination.append(append_prefix(
                    prefix, f"{config.name}.weight"))
        return destination

    @property
    def config(self) -> GroupedEmbeddingConfig:
        return self._config


class CAIBatchedDenseEmbeddingBag(BaseBatchedEmbeddingBag):
    def __init__(
        self,
        config: GroupedEmbeddingConfig,
        pg: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
        cache_ratio: float = 0.05,
    ) -> None:
        super().__init__(config, pg, device)

        num_embeddings = sum(self._num_embeddings)
        assert all(x == self._local_cols[0]
                   for x in self._local_cols), "local col should be consistent in all embeddings"
        embedding_dim = self._local_cols[0]
        self.pool_str = pooling_mode_to_str(self._pooling)

        weight_list = []
        for embedding_config in self._config.embedding_tables:
            weight_list.append(torch.empty(
                embedding_config.local_rows,
                embedding_config.local_cols,
                device='cpu',
            ).uniform_(
                embedding_config.get_weight_init_min(),
                embedding_config.get_weight_init_max(),
            ))
        self._emb_module = FreqAwareEmbeddingBag(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            mode=self.pool_str,
            include_last_offset=True,
            sparse=True,
            # _weight=torch.empty(num_embeddings, embedding_dim, device='cpu',).uniform_(
            #     min(self._weight_init_mins), max(self._weight_init_maxs)),
            _weight=torch.cat(weight_list,0).pin_memory(),
            warmup_ratio=0.7,
            cache_ratio = cache_ratio,

        )
        self._table_idx_offsets = torch.cumsum(torch.tensor(
            [0] + self._num_embeddings, device="cuda"), 0, dtype=torch.long)
        self._already_linearized = False
        # TODO count different idx num
        # self._idx_input_record = torch.zeros(num_embeddings)
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
            nn.Parameter, self._emb_module.cache_weight_mgr.cuda_cached_weight
        )
        
    def forward(self, features: KeyedJaggedTensor) -> torch.Tensor:
        offsets = features.offsets().long()
        weights = features.weights_or_none()
        batch_size = len(features.offsets())//len(features.keys())
        if not self._already_linearized:
            values = self.linearize_features(features)
        else:
            values = features.values().long()
        output = self._emb_module(values, offsets, weights)
        ret =  torch.cat(output.split(batch_size), 1)
        return ret

    
    def linearize_features(self, features: KeyedJaggedTensor):
        # apply table offset to values 
        
        with torch.no_grad():
            with record_function("add id offsets"):
                values = features.values().long()
                split_view = torch.tensor_split(
                    values, features.offset_per_key()[1:-1])
                for i, chunk in enumerate(split_view):
                    torch.add(chunk, self._table_idx_offsets[i], out=chunk)
        return values
        
        # # alternative approach
        # return torch.ops.fbgemm.linearize_cache_indices(
        #     self._table_idx_offsets,
        #     features.values(),
        #     torch.tensor(features.offset_per_key(), dtype=torch.int, device="cuda")
        # )
    def set_already_linearized(self, linearized = False):
        self._already_linearized = linearized