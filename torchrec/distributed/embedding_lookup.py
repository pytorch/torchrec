#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from abc import ABC
from collections import OrderedDict
from typing import Any, cast, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.modules.module import _IncompatibleKeys
from torchrec.distributed.batched_embedding_kernel import (
    BaseBatchedEmbedding,
    BaseBatchedEmbeddingBag,
    BatchedDenseEmbedding,
    BatchedDenseEmbeddingBag,
    BatchedFusedEmbedding,
    BatchedFusedEmbeddingBag,
)
from torchrec.distributed.embedding_kernel import BaseEmbedding
from torchrec.distributed.embedding_types import (
    BaseEmbeddingLookup,
    BaseGroupedFeatureProcessor,
    EmbeddingComputeKernel,
    GroupedEmbeddingConfig,
    SparseFeatures,
    SparseFeaturesList,
)
from torchrec.distributed.quant_embedding_kernel import (
    QuantBatchedEmbedding,
    QuantBatchedEmbeddingBag,
)
from torchrec.distributed.types import ShardedTensor

logger: logging.Logger = logging.getLogger(__name__)


def _load_state_dict(
    emb_modules: "nn.ModuleList",
    state_dict: "OrderedDict[str, Union[torch.Tensor, ShardedTensor]]",
) -> Tuple[List[str], List[str]]:
    missing_keys = []
    unexpected_keys = list(state_dict.keys())
    for emb_module in emb_modules:
        for key, dst_param in emb_module.state_dict().items():
            if key in state_dict:
                src_param = state_dict[key]
                if isinstance(dst_param, ShardedTensor):
                    assert isinstance(src_param, ShardedTensor)
                    assert len(dst_param.local_shards()) == len(
                        src_param.local_shards()
                    )
                    for dst_local_shard, src_local_shard in zip(
                        dst_param.local_shards(), src_param.local_shards()
                    ):
                        assert (
                            dst_local_shard.metadata.shard_offsets
                            == src_local_shard.metadata.shard_offsets
                        )
                        assert (
                            dst_local_shard.metadata.shard_sizes
                            == src_local_shard.metadata.shard_sizes
                        )

                        dst_local_shard.tensor.detach().copy_(src_local_shard.tensor)
                else:
                    assert isinstance(src_param, torch.Tensor) and isinstance(
                        dst_param, torch.Tensor
                    )
                    dst_param.detach().copy_(src_param)
                unexpected_keys.remove(key)
            else:
                missing_keys.append(cast(str, key))
    return missing_keys, unexpected_keys


class GroupedEmbeddingsLookup(BaseEmbeddingLookup[SparseFeatures, torch.Tensor]):
    def __init__(
        self,
        grouped_configs: List[GroupedEmbeddingConfig],
        pg: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        def _create_lookup(
            config: GroupedEmbeddingConfig,
        ) -> BaseEmbedding:
            if config.compute_kernel == EmbeddingComputeKernel.DENSE:
                return BatchedDenseEmbedding(
                    config=config,
                    pg=pg,
                    device=device,
                )
            elif config.compute_kernel == EmbeddingComputeKernel.FUSED:
                return BatchedFusedEmbedding(
                    config=config,
                    pg=pg,
                    device=device,
                )
            else:
                raise ValueError(
                    f"Compute kernel not supported {config.compute_kernel}"
                )

        super().__init__()
        self._emb_modules: nn.ModuleList = nn.ModuleList()
        for config in grouped_configs:
            self._emb_modules.append(_create_lookup(config))

        self._id_list_feature_splits: List[int] = []
        for config in grouped_configs:
            self._id_list_feature_splits.append(config.num_features())

        # return a dummy empty tensor when grouped_configs is empty
        self.register_buffer(
            "_dummy_embs_tensor",
            torch.empty(
                [0],
                dtype=torch.float32,
                device=device,
                requires_grad=True,
            ),
        )

        self.grouped_configs = grouped_configs

    def forward(
        self,
        sparse_features: SparseFeatures,
    ) -> torch.Tensor:
        assert sparse_features.id_list_features is not None
        embeddings: List[torch.Tensor] = []
        id_list_features_by_group = sparse_features.id_list_features.split(
            self._id_list_feature_splits,
        )
        for emb_op, features in zip(self._emb_modules, id_list_features_by_group):
            embeddings.append(emb_op(features).view(-1))

        if len(embeddings) == 0:
            # a hack for empty ranks
            return self._dummy_embs_tensor
        elif len(embeddings) == 1:
            return embeddings[0]
        else:
            return torch.cat(embeddings)

    # pyre-fixme[14]: `state_dict` overrides method defined in `Module` inconsistently.
    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        if destination is None:
            destination = OrderedDict()
            # pyre-ignore [16]
            destination._metadata = OrderedDict()

        for emb_module in self._emb_modules:
            emb_module.state_dict(destination, prefix, keep_vars)

        return destination

    # pyre-fixme[14]: `load_state_dict` overrides method defined in `Module`
    #  inconsistently.
    def load_state_dict(
        self,
        state_dict: "OrderedDict[str, Union[torch.Tensor, ShardedTensor]]",
        strict: bool = True,
    ) -> _IncompatibleKeys:
        m, u = _load_state_dict(self._emb_modules, state_dict)
        return _IncompatibleKeys(missing_keys=m, unexpected_keys=u)

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        assert remove_duplicate, (
            "remove_duplicate=False in named_parameters for"
            "GroupedEmbeddingsLookup is not supported"
        )
        for emb_module in self._emb_modules:
            yield from emb_module.named_parameters(prefix, recurse)

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        assert remove_duplicate, (
            "remove_duplicate=False in named_buffers for"
            "GroupedEmbeddingsLookup is not supported"
        )
        for emb_module in self._emb_modules:
            yield from emb_module.named_buffers(prefix, recurse)


class GroupedPooledEmbeddingsLookup(BaseEmbeddingLookup[SparseFeatures, torch.Tensor]):
    def __init__(
        self,
        grouped_configs: List[GroupedEmbeddingConfig],
        grouped_score_configs: List[GroupedEmbeddingConfig],
        device: Optional[torch.device] = None,
        pg: Optional[dist.ProcessGroup] = None,
        feature_processor: Optional[BaseGroupedFeatureProcessor] = None,
    ) -> None:
        def _create_lookup(
            config: GroupedEmbeddingConfig,
            device: Optional[torch.device] = None,
        ) -> BaseEmbedding:
            if config.compute_kernel == EmbeddingComputeKernel.DENSE:
                return BatchedDenseEmbeddingBag(
                    config=config,
                    pg=pg,
                    device=device,
                )
            elif config.compute_kernel == EmbeddingComputeKernel.FUSED:
                return BatchedFusedEmbeddingBag(
                    config=config,
                    pg=pg,
                    device=device,
                )
            else:
                raise ValueError(
                    f"Compute kernel not supported {config.compute_kernel}"
                )

        super().__init__()
        self._emb_modules: nn.ModuleList = nn.ModuleList()
        for config in grouped_configs:
            self._emb_modules.append(_create_lookup(config, device))

        self._score_emb_modules: nn.ModuleList = nn.ModuleList()
        for config in grouped_score_configs:
            self._score_emb_modules.append(_create_lookup(config, device))

        self._id_list_feature_splits: List[int] = []
        for config in grouped_configs:
            self._id_list_feature_splits.append(config.num_features())
        self._id_score_list_feature_splits: List[int] = []
        for config in grouped_score_configs:
            self._id_score_list_feature_splits.append(config.num_features())

        # return a dummy empty tensor
        # when grouped_configs and grouped_score_configs are empty
        self.register_buffer(
            "_dummy_embs_tensor",
            torch.empty(
                [0],
                dtype=torch.float32,
                device=device,
                requires_grad=True,
            ),
        )

        self.grouped_configs = grouped_configs
        self.grouped_score_configs = grouped_score_configs
        self._feature_processor = feature_processor

    def forward(
        self,
        sparse_features: SparseFeatures,
    ) -> torch.Tensor:
        assert (
            sparse_features.id_list_features is not None
            or sparse_features.id_score_list_features is not None
        )
        embeddings: List[torch.Tensor] = []
        if len(self._emb_modules) > 0:
            assert sparse_features.id_list_features is not None
            id_list_features_by_group = sparse_features.id_list_features.split(
                self._id_list_feature_splits,
            )
            for config, emb_op, features in zip(
                self.grouped_configs, self._emb_modules, id_list_features_by_group
            ):
                # keep this to avoid break ads code using feature_processor, for ebc
                # the has_feature_processor will always be false. Remove this block when
                # finishing the migration
                if (
                    config.has_feature_processor
                    and self._feature_processor is not None
                    and isinstance(self._feature_processor, BaseGroupedFeatureProcessor)
                ):
                    features = self._feature_processor(features)
                embeddings.append(emb_op(features))
        if len(self._score_emb_modules) > 0:
            assert sparse_features.id_score_list_features is not None
            id_score_list_features_by_group = (
                sparse_features.id_score_list_features.split(
                    self._id_score_list_feature_splits,
                )
            )
            for config, emb_op, features in zip(
                self.grouped_score_configs,
                self._score_emb_modules,
                id_score_list_features_by_group,
            ):
                if (
                    config.has_feature_processor
                    and self._feature_processor is not None
                    and isinstance(self._feature_processor, BaseGroupedFeatureProcessor)
                ):
                    features = self._feature_processor(features)
                embeddings.append(emb_op(features))

        if len(embeddings) == 0:
            # a hack for empty ranks
            batch_size: int = (
                sparse_features.id_list_features.stride()
                if sparse_features.id_list_features is not None
                # pyre-fixme[16]: `Optional` has no attribute `stride`.
                else sparse_features.id_score_list_features.stride()
            )
            return self._dummy_embs_tensor.view(batch_size, 0)
        elif len(embeddings) == 1:
            return embeddings[0]
        else:
            return torch.cat(embeddings, dim=1)

    # pyre-fixme[14]: `state_dict` overrides method defined in `Module` inconsistently.
    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        if destination is None:
            destination = OrderedDict()
            # pyre-ignore [16]
            destination._metadata = OrderedDict()

        for emb_module in self._emb_modules:
            emb_module.state_dict(destination, prefix, keep_vars)
        for emb_module in self._score_emb_modules:
            emb_module.state_dict(destination, prefix, keep_vars)

        return destination

    # pyre-fixme[14]: `load_state_dict` overrides method defined in `Module`
    #  inconsistently.
    def load_state_dict(
        self,
        state_dict: "OrderedDict[str, Union[ShardedTensor, torch.Tensor]]",
        strict: bool = True,
    ) -> _IncompatibleKeys:
        m1, u1 = _load_state_dict(self._emb_modules, state_dict)
        m2, u2 = _load_state_dict(self._score_emb_modules, state_dict)
        return _IncompatibleKeys(missing_keys=m1 + m2, unexpected_keys=u1 + u2)

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        assert remove_duplicate, (
            "remove_duplicate=False in named_parameters for"
            "GroupedPooledEmbeddingsLookup is not supported"
        )
        for emb_module in self._emb_modules:
            yield from emb_module.named_parameters(prefix, recurse)
        for emb_module in self._score_emb_modules:
            yield from emb_module.named_parameters(prefix, recurse)

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        assert remove_duplicate, (
            "remove_duplicate=False in named_buffers for"
            "GroupedPooledEmbeddingsLookup is not supported"
        )
        for emb_module in self._emb_modules:
            yield from emb_module.named_buffers(prefix, recurse)
        for emb_module in self._score_emb_modules:
            yield from emb_module.named_buffers(prefix, recurse)


class MetaInferGroupedEmbeddingsLookup(
    BaseEmbeddingLookup[SparseFeatures, torch.Tensor]
):
    """
    meta embedding lookup module for inference since inference lookup has references
    for multiple TBE ops over all gpu workers.
    inference grouped embedding lookup module contains meta modules allocated over gpu workers.
    """

    def __init__(
        self,
        grouped_configs: List[GroupedEmbeddingConfig],
        device: Optional[torch.device] = None,
        fused_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        def _create_lookup(
            config: GroupedEmbeddingConfig,
            device: Optional[torch.device] = None,
            fused_params: Optional[Dict[str, Any]] = None,
        ) -> BaseBatchedEmbedding:
            return QuantBatchedEmbedding(
                config=config,
                device=device,
                fused_params=fused_params,
            )

        super().__init__()
        self._emb_modules: nn.ModuleList = nn.ModuleList()
        for config in grouped_configs:
            self._emb_modules.append(_create_lookup(config, device, fused_params))

        self._id_list_feature_splits: List[int] = [
            config.num_features() for config in grouped_configs
        ]

        # return a dummy empty tensor when grouped_configs is empty
        self.register_buffer(
            "_dummy_embs_tensor",
            torch.empty(
                [0],
                dtype=torch.float32,
                device=device,
            ),
        )

        self.grouped_configs = grouped_configs

    def forward(
        self,
        sparse_features: SparseFeatures,
    ) -> torch.Tensor:
        assert sparse_features.id_list_features is not None
        embeddings: List[torch.Tensor] = []
        id_list_features_by_group = sparse_features.id_list_features.split(
            self._id_list_feature_splits,
        )
        for emb_op, features in zip(self._emb_modules, id_list_features_by_group):
            embeddings.append(emb_op(features).view(-1))

        if len(embeddings) == 0:
            # a hack for empty ranks
            return self._dummy_embs_tensor
        elif len(embeddings) == 1:
            return embeddings[0]
        else:
            return torch.cat(embeddings)

    # pyre-ignore [14]
    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        if destination is None:
            destination = OrderedDict()
            # pyre-ignore [16]
            destination._metadata = OrderedDict()

        for emb_module in self._emb_modules:
            emb_module.state_dict(destination, prefix, keep_vars)

        return destination

    # pyre-fixme[14]: `load_state_dict` overrides method defined in `Module`
    #  inconsistently.
    def load_state_dict(
        self,
        state_dict: "OrderedDict[str, Union[ShardedTensor, torch.Tensor]]",
        strict: bool = True,
    ) -> _IncompatibleKeys:
        m, u = _load_state_dict(self._emb_modules, state_dict)
        return _IncompatibleKeys(missing_keys=m, unexpected_keys=u)

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        assert remove_duplicate, (
            "remove_duplicate=False in named_buffers for"
            "MetaInferGroupedEmbeddingsLookup is not supported"
        )
        for emb_module in self._emb_modules:
            yield from emb_module.named_parameters(prefix, recurse)

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        assert remove_duplicate, (
            "remove_duplicate=False in named_buffers for"
            "MetaInferGroupedEmbeddingsLookup is not supported"
        )
        for emb_module in self._emb_modules:
            yield from emb_module.named_buffers(prefix, recurse)


class MetaInferGroupedPooledEmbeddingsLookup(
    BaseEmbeddingLookup[SparseFeatures, torch.Tensor]
):
    """
    meta embedding bag lookup module for inference since inference lookup has references
    for multiple TBE ops over all gpu workers.
    inference grouped embedding bag lookup module contains meta modules allocated over gpu workers.
    """

    def __init__(
        self,
        grouped_configs: List[GroupedEmbeddingConfig],
        grouped_score_configs: List[GroupedEmbeddingConfig],
        device: Optional[torch.device] = None,
        feature_processor: Optional[BaseGroupedFeatureProcessor] = None,
        fused_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        def _create_lookup(
            config: GroupedEmbeddingConfig,
            device: Optional[torch.device] = None,
            fused_params: Optional[Dict[str, Any]] = None,
        ) -> BaseBatchedEmbeddingBag:
            return QuantBatchedEmbeddingBag(
                config=config,
                device=device,
                fused_params=fused_params,
            )

        super().__init__()
        self._emb_modules: nn.ModuleList = nn.ModuleList()
        for config in grouped_configs:
            self._emb_modules.append(_create_lookup(config, device, fused_params))

        self._score_emb_modules: nn.ModuleList = nn.ModuleList()
        for config in grouped_score_configs:
            self._score_emb_modules.append(_create_lookup(config, device))

        self._id_list_feature_splits: List[int] = [
            config.num_features() for config in grouped_configs
        ]
        self._id_score_list_feature_splits: List[int] = [
            config.num_features() for config in grouped_score_configs
        ]

        # return a dummy empty tensor
        # when grouped_configs and grouped_score_configs are empty
        self.register_buffer(
            "_dummy_embs_tensor",
            torch.empty(
                [0],
                dtype=torch.float32,
                device=device,
            ),
        )

        self.grouped_configs = grouped_configs
        self.grouped_score_configs = grouped_score_configs
        self._feature_processor = feature_processor

    def forward(
        self,
        sparse_features: SparseFeatures,
    ) -> torch.Tensor:
        assert (
            sparse_features.id_list_features is not None
            or sparse_features.id_score_list_features is not None
        )
        embeddings: List[torch.Tensor] = []
        if len(self._emb_modules) > 0:
            assert sparse_features.id_list_features is not None
            id_list_features_by_group = sparse_features.id_list_features.split(
                self._id_list_feature_splits,
            )
            for config, emb_op, features in zip(
                self.grouped_configs, self._emb_modules, id_list_features_by_group
            ):
                if (
                    config.has_feature_processor
                    and self._feature_processor is not None
                    and isinstance(self._feature_processor, BaseGroupedFeatureProcessor)
                ):
                    features = self._feature_processor(features)
                embeddings.append(emb_op(features))
        if len(self._score_emb_modules) > 0:
            assert sparse_features.id_score_list_features is not None
            id_score_list_features_by_group = (
                sparse_features.id_score_list_features.split(
                    self._id_score_list_feature_splits,
                )
            )
            for emb_op, features in zip(
                self._score_emb_modules, id_score_list_features_by_group
            ):
                embeddings.append(emb_op(features))

        if len(embeddings) == 0:
            # a hack for empty ranks
            batch_size: int = (
                sparse_features.id_list_features.stride()
                if sparse_features.id_list_features is not None
                # pyre-fixme[16]: `Optional` has no attribute `stride`.
                else sparse_features.id_score_list_features.stride()
            )
            return self._dummy_embs_tensor.view(batch_size, 0)
        elif len(embeddings) == 1:
            return embeddings[0]
        else:
            return torch.cat(embeddings, dim=1)

    # pyre-ignore [14]
    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        if destination is None:
            destination = OrderedDict()
            # pyre-ignore [16]
            destination._metadata = OrderedDict()

        for emb_module in self._emb_modules:
            emb_module.state_dict(destination, prefix, keep_vars)
        for emb_module in self._score_emb_modules:
            emb_module.state_dict(destination, prefix, keep_vars)

        return destination

    # pyre-fixme[14]: `load_state_dict` overrides method defined in `Module`
    #  inconsistently.
    def load_state_dict(
        self,
        state_dict: "OrderedDict[str, Union[ShardedTensor, torch.Tensor]]",
        strict: bool = True,
    ) -> _IncompatibleKeys:
        m1, u1 = _load_state_dict(self._emb_modules, state_dict)
        m2, u2 = _load_state_dict(self._score_emb_modules, state_dict)
        return _IncompatibleKeys(missing_keys=m1 + m2, unexpected_keys=u1 + u2)

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        assert remove_duplicate, (
            "remove_duplicate=False in named_parameters for"
            "MetaInferGroupedPooledEmbeddingsLookup is not supported"
        )
        for emb_module in self._emb_modules:
            yield from emb_module.named_parameters(prefix, recurse)
        for emb_module in self._score_emb_modules:
            yield from emb_module.named_parameters(prefix, recurse)

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        assert remove_duplicate, (
            "remove_duplicate=False in named_buffers for"
            "MetaInferGroupedPooledEmbeddingsLookup is not supported"
        )
        for emb_module in self._emb_modules:
            yield from emb_module.named_buffers(prefix, recurse)
        for emb_module in self._score_emb_modules:
            yield from emb_module.named_buffers(prefix, recurse)


class InferGroupedLookupMixin(ABC):
    def forward(
        self,
        sparse_features: SparseFeaturesList,
    ) -> List[torch.Tensor]:
        embeddings: List[torch.Tensor] = []
        for sparse_features_rank, embedding_lookup in zip(
            sparse_features,
            # pyre-fixme[16]
            self._embedding_lookups_per_rank,
        ):
            assert (
                sparse_features_rank.id_list_features is not None
                or sparse_features_rank.id_score_list_features is not None
            )
            embeddings.append(embedding_lookup(sparse_features_rank))
        return embeddings

    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        if destination is None:
            destination = OrderedDict()
            # pyre-ignore [16]
            destination._metadata = OrderedDict()

        # pyre-fixme[16]
        for rank_modules in self._embedding_lookups_per_rank:
            rank_modules.state_dict(destination, prefix, keep_vars)

        return destination

    def load_state_dict(
        self,
        state_dict: "OrderedDict[str, torch.Tensor]",
        strict: bool = True,
    ) -> _IncompatibleKeys:
        missing_keys = []
        unexpected_keys = []
        # pyre-fixme[16]
        for rank_modules in self._embedding_lookups_per_rank:
            incompatible_keys = rank_modules.load_state_dict(state_dict)
            missing_keys.extend(incompatible_keys.missing_keys)
            unexpected_keys.extend(incompatible_keys.unexpected_keys)
        return _IncompatibleKeys(
            missing_keys=missing_keys, unexpected_keys=unexpected_keys
        )

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        # pyre-fixme[16]
        for rank_modules in self._embedding_lookups_per_rank:
            yield from rank_modules.named_parameters(prefix, recurse)

    def named_buffers(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        # pyre-fixme[16]
        for rank_modules in self._embedding_lookups_per_rank:
            yield from rank_modules.named_buffers(prefix, recurse)


class InferGroupedPooledEmbeddingsLookup(
    InferGroupedLookupMixin,
    BaseEmbeddingLookup[SparseFeaturesList, List[torch.Tensor]],
):
    def __init__(
        self,
        grouped_configs_per_rank: List[List[GroupedEmbeddingConfig]],
        grouped_score_configs_per_rank: List[List[GroupedEmbeddingConfig]],
        world_size: int,
        fused_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self._embedding_lookups_per_rank: List[
            MetaInferGroupedPooledEmbeddingsLookup
        ] = []
        for rank in range(world_size):
            self._embedding_lookups_per_rank.append(
                # TODO add position weighted module support
                MetaInferGroupedPooledEmbeddingsLookup(
                    grouped_configs=grouped_configs_per_rank[rank],
                    grouped_score_configs=grouped_score_configs_per_rank[rank],
                    device=torch.device("cuda", rank),
                    fused_params=fused_params,
                )
            )


class InferGroupedEmbeddingsLookup(
    InferGroupedLookupMixin,
    BaseEmbeddingLookup[SparseFeaturesList, List[torch.Tensor]],
):
    def __init__(
        self,
        grouped_configs_per_rank: List[List[GroupedEmbeddingConfig]],
        world_size: int,
        fused_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self._embedding_lookups_per_rank: List[MetaInferGroupedEmbeddingsLookup] = []
        for rank in range(world_size):
            self._embedding_lookups_per_rank.append(
                MetaInferGroupedEmbeddingsLookup(
                    grouped_configs=grouped_configs_per_rank[rank],
                    device=torch.device("cuda", rank),
                    fused_params=fused_params,
                )
            )
