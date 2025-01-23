#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import random
from dataclasses import dataclass
from typing import Any, cast, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from torchrec.distributed.embedding_tower_sharding import (
    EmbeddingTowerCollectionSharder,
    EmbeddingTowerSharder,
)
from torchrec.distributed.embedding_types import EmbeddingTableConfig
from torchrec.distributed.embeddingbag import (
    EmbeddingBagCollectionSharder,
    EmbeddingBagSharder,
)
from torchrec.distributed.fused_embedding import FusedEmbeddingCollectionSharder
from torchrec.distributed.fused_embeddingbag import FusedEmbeddingBagCollectionSharder
from torchrec.distributed.types import QuantizedCommCodecs
from torchrec.distributed.utils import CopyableMixin
from torchrec.modules.activation import SwishLayerNorm
from torchrec.modules.embedding_configs import (
    BaseEmbeddingConfig,
    EmbeddingBagConfig,
    EmbeddingConfig,
)
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.modules.embedding_tower import EmbeddingTower, EmbeddingTowerCollection
from torchrec.modules.feature_processor import PositionWeightedProcessor
from torchrec.modules.feature_processor_ import PositionWeightedModuleCollection
from torchrec.modules.fp_embedding_modules import FeatureProcessedEmbeddingBagCollection
from torchrec.modules.regroup import KTRegroupAsDict
from torchrec.sparse.jagged_tensor import _to_offsets, KeyedJaggedTensor, KeyedTensor
from torchrec.streamable import Pipelineable


@dataclass
class ModelInput(Pipelineable):
    float_features: torch.Tensor
    idlist_features: KeyedJaggedTensor
    idscore_features: Optional[KeyedJaggedTensor]
    label: torch.Tensor

    @staticmethod
    def generate(
        batch_size: int,
        world_size: int,
        num_float_features: int,
        tables: Union[
            List[EmbeddingTableConfig], List[EmbeddingBagConfig], List[EmbeddingConfig]
        ],
        weighted_tables: Union[
            List[EmbeddingTableConfig], List[EmbeddingBagConfig], List[EmbeddingConfig]
        ],
        pooling_avg: int = 10,
        dedup_tables: Optional[
            Union[
                List[EmbeddingTableConfig],
                List[EmbeddingBagConfig],
                List[EmbeddingConfig],
            ]
        ] = None,
        variable_batch_size: bool = False,
        long_indices: bool = True,
        tables_pooling: Optional[List[int]] = None,
        weighted_tables_pooling: Optional[List[int]] = None,
        randomize_indices: bool = True,
        device: Optional[torch.device] = None,
        max_feature_lengths: Optional[List[int]] = None,
    ) -> Tuple["ModelInput", List["ModelInput"]]:
        """
        Returns a global (single-rank training) batch
        and a list of local (multi-rank training) batches of world_size.
        """
        batch_size_by_rank = [batch_size] * world_size
        if variable_batch_size:
            batch_size_by_rank = [
                batch_size_by_rank[r] - r if batch_size_by_rank[r] - r > 0 else 1
                for r in range(world_size)
            ]

        def _validate_pooling_factor(
            tables: Union[
                List[EmbeddingTableConfig],
                List[EmbeddingBagConfig],
                List[EmbeddingConfig],
            ],
            pooling_factor: Optional[List[int]],
        ) -> None:
            if pooling_factor and len(pooling_factor) != len(tables):
                raise ValueError(
                    "tables_pooling and tables must have the same length. "
                    f"Got {len(pooling_factor)} and {len(tables)}."
                )

        _validate_pooling_factor(tables, tables_pooling)
        _validate_pooling_factor(weighted_tables, weighted_tables_pooling)

        idlist_features_to_num_embeddings = {}
        idlist_features_to_pooling_factor = {}
        idlist_features_to_max_length = {}
        feature_idx = 0
        for idx in range(len(tables)):
            for feature in tables[idx].feature_names:
                idlist_features_to_num_embeddings[feature] = (
                    tables[idx].num_embeddings_post_pruning
                    if tables[idx].num_embeddings_post_pruning is not None
                    else tables[idx].num_embeddings
                )

                idlist_features_to_max_length[feature] = (
                    max_feature_lengths[feature_idx] if max_feature_lengths else None
                )
                if tables_pooling is not None:
                    idlist_features_to_pooling_factor[feature] = tables_pooling[idx]
                feature_idx += 1

        idlist_features = list(idlist_features_to_num_embeddings.keys())
        idscore_features = [
            feature for table in weighted_tables for feature in table.feature_names
        ]

        idlist_ind_ranges = list(idlist_features_to_num_embeddings.values())
        idscore_ind_ranges = [
            (
                table.num_embeddings_post_pruning
                if table.num_embeddings_post_pruning is not None
                else table.num_embeddings
            )
            for table in weighted_tables
        ]

        idlist_pooling_factor = list(idlist_features_to_pooling_factor.values())
        idscore_pooling_factor = weighted_tables_pooling

        idlist_max_lengths = list(idlist_features_to_max_length.values())

        # Generate global batch.
        global_idlist_lengths = []
        global_idlist_indices = []
        global_idscore_lengths = []
        global_idscore_indices = []
        global_idscore_weights = []

        for idx in range(len(idlist_ind_ranges)):
            ind_range = idlist_ind_ranges[idx]
            if idlist_pooling_factor:
                lengths_ = torch.max(
                    torch.normal(
                        idlist_pooling_factor[idx],
                        idlist_pooling_factor[idx] / 10,
                        [batch_size * world_size],
                        device=device,
                    ),
                    torch.tensor(1.0, device=device),
                ).int()
            else:
                lengths_ = torch.abs(
                    torch.randn(batch_size * world_size, device=device) + pooling_avg,
                ).int()

            if idlist_max_lengths[idx]:
                lengths_ = torch.clamp(lengths_, max=idlist_max_lengths[idx])

            if variable_batch_size:
                lengths = torch.zeros(batch_size * world_size, device=device).int()
                for r in range(world_size):
                    lengths[r * batch_size : r * batch_size + batch_size_by_rank[r]] = (
                        lengths_[
                            r * batch_size : r * batch_size + batch_size_by_rank[r]
                        ]
                    )
            else:
                lengths = lengths_

            num_indices = cast(int, torch.sum(lengths).item())
            if randomize_indices:
                indices = torch.randint(
                    0,
                    ind_range,
                    (num_indices,),
                    dtype=torch.long if long_indices else torch.int32,
                    device=device,
                )
            else:
                indices = torch.zeros(
                    (num_indices),
                    dtype=torch.long if long_indices else torch.int32,
                    device=device,
                )
            global_idlist_lengths.append(lengths)
            global_idlist_indices.append(indices)
        global_idlist_kjt = KeyedJaggedTensor(
            keys=idlist_features,
            values=torch.cat(global_idlist_indices),
            lengths=torch.cat(global_idlist_lengths),
        )

        for idx in range(len(idscore_ind_ranges)):
            ind_range = idscore_ind_ranges[idx]
            lengths_ = torch.abs(
                torch.randn(batch_size * world_size, device=device)
                + (
                    idscore_pooling_factor[idx]
                    if idscore_pooling_factor
                    else pooling_avg
                )
            ).int()
            if variable_batch_size:
                lengths = torch.zeros(batch_size * world_size, device=device).int()
                for r in range(world_size):
                    lengths[r * batch_size : r * batch_size + batch_size_by_rank[r]] = (
                        lengths_[
                            r * batch_size : r * batch_size + batch_size_by_rank[r]
                        ]
                    )
            else:
                lengths = lengths_
            num_indices = cast(int, torch.sum(lengths).item())
            if randomize_indices:
                indices = torch.randint(
                    0,
                    # pyre-ignore [6]
                    ind_range,
                    (num_indices,),
                    dtype=torch.long if long_indices else torch.int32,
                    device=device,
                )
            else:
                indices = torch.zeros(
                    (num_indices),
                    dtype=torch.long if long_indices else torch.int32,
                    device=device,
                )
            weights = torch.rand((num_indices,), device=device)
            global_idscore_lengths.append(lengths)
            global_idscore_indices.append(indices)
            global_idscore_weights.append(weights)
        global_idscore_kjt = (
            KeyedJaggedTensor(
                keys=idscore_features,
                values=torch.cat(global_idscore_indices),
                lengths=torch.cat(global_idscore_lengths),
                weights=torch.cat(global_idscore_weights),
            )
            if global_idscore_indices
            else None
        )

        if randomize_indices:
            global_float = torch.rand(
                (batch_size * world_size, num_float_features), device=device
            )
            global_label = torch.rand(batch_size * world_size, device=device)
        else:
            global_float = torch.zeros(
                (batch_size * world_size, num_float_features), device=device
            )
            global_label = torch.zeros(batch_size * world_size, device=device)

        # Split global batch into local batches.
        local_inputs = []
        for r in range(world_size):
            local_idlist_lengths = []
            local_idlist_indices = []
            local_idscore_lengths = []
            local_idscore_indices = []
            local_idscore_weights = []

            for lengths, indices in zip(global_idlist_lengths, global_idlist_indices):
                local_idlist_lengths.append(
                    lengths[r * batch_size : r * batch_size + batch_size_by_rank[r]]
                )
                lengths_cumsum = [0] + lengths.view(world_size, -1).sum(dim=1).cumsum(
                    dim=0
                ).tolist()
                local_idlist_indices.append(
                    indices[lengths_cumsum[r] : lengths_cumsum[r + 1]]
                )

            for lengths, indices, weights in zip(
                global_idscore_lengths, global_idscore_indices, global_idscore_weights
            ):
                local_idscore_lengths.append(
                    lengths[r * batch_size : r * batch_size + batch_size_by_rank[r]]
                )
                lengths_cumsum = [0] + lengths.view(world_size, -1).sum(dim=1).cumsum(
                    dim=0
                ).tolist()
                local_idscore_indices.append(
                    indices[lengths_cumsum[r] : lengths_cumsum[r + 1]]
                )
                local_idscore_weights.append(
                    weights[lengths_cumsum[r] : lengths_cumsum[r + 1]]
                )

            local_idlist_kjt = KeyedJaggedTensor(
                keys=idlist_features,
                values=torch.cat(local_idlist_indices),
                lengths=torch.cat(local_idlist_lengths),
            )

            local_idscore_kjt = (
                KeyedJaggedTensor(
                    keys=idscore_features,
                    values=torch.cat(local_idscore_indices),
                    lengths=torch.cat(local_idscore_lengths),
                    weights=torch.cat(local_idscore_weights),
                )
                if local_idscore_indices
                else None
            )

            local_input = ModelInput(
                float_features=global_float[r * batch_size : (r + 1) * batch_size],
                idlist_features=local_idlist_kjt,
                idscore_features=local_idscore_kjt,
                label=global_label[r * batch_size : (r + 1) * batch_size],
            )
            local_inputs.append(local_input)

        return (
            ModelInput(
                float_features=global_float,
                idlist_features=global_idlist_kjt,
                idscore_features=global_idscore_kjt,
                label=global_label,
            ),
            local_inputs,
        )

    @staticmethod
    def _generate_variable_batch_local_features(
        feature_num_embeddings: Dict[str, int],
        average_batch_size: int,
        world_size: int,
        dedup_factor: int,
        values_per_rank_per_feature: Dict[int, Dict[str, torch.Tensor]],
        lengths_per_rank_per_feature: Dict[int, Dict[str, torch.Tensor]],
        strides_per_rank_per_feature: Dict[int, Dict[str, int]],
        inverse_indices_per_rank_per_feature: Dict[int, Dict[str, torch.Tensor]],
        weights_per_rank_per_feature: Optional[Dict[int, Dict[str, torch.Tensor]]],
    ) -> List[KeyedJaggedTensor]:
        local_kjts = []
        keys = list(feature_num_embeddings.keys())
        for rank in range(world_size):
            lengths_per_rank_per_feature[rank] = {}
            values_per_rank_per_feature[rank] = {}
            strides_per_rank_per_feature[rank] = {}
            inverse_indices_per_rank_per_feature[rank] = {}
            if weights_per_rank_per_feature is not None:
                weights_per_rank_per_feature[rank] = {}

            for key, num_embeddings in feature_num_embeddings.items():
                batch_size = random.randint(1, average_batch_size * dedup_factor - 1)
                lengths = torch.randint(low=0, high=5, size=(batch_size,))
                lengths_per_rank_per_feature[rank][key] = lengths
                lengths_sum = sum(lengths.tolist())
                values = torch.randint(0, num_embeddings, (lengths_sum,))
                values_per_rank_per_feature[rank][key] = values
                if weights_per_rank_per_feature is not None:
                    weights_per_rank_per_feature[rank][key] = torch.rand(lengths_sum)
                strides_per_rank_per_feature[rank][key] = batch_size
                inverse_indices_per_rank_per_feature[rank][key] = torch.randint(
                    0, batch_size, (dedup_factor * average_batch_size,)
                )

            values = torch.cat(list(values_per_rank_per_feature[rank].values()))
            lengths = torch.cat(list(lengths_per_rank_per_feature[rank].values()))
            weights = (
                torch.cat(list(weights_per_rank_per_feature[rank].values()))
                if weights_per_rank_per_feature is not None
                else None
            )
            stride_per_key_per_rank = [
                [stride] for stride in strides_per_rank_per_feature[rank].values()
            ]
            inverse_indices = (
                keys,
                torch.stack(list(inverse_indices_per_rank_per_feature[rank].values())),
            )
            local_kjts.append(
                KeyedJaggedTensor(
                    keys=keys,
                    values=values,
                    lengths=lengths,
                    weights=weights,
                    stride_per_key_per_rank=stride_per_key_per_rank,
                    inverse_indices=inverse_indices,
                )
            )
        return local_kjts

    @staticmethod
    def _generate_variable_batch_global_features(
        keys: List[str],
        world_size: int,
        global_constant_batch: bool,
        values_per_rank_per_feature: Dict[int, Dict[str, torch.Tensor]],
        lengths_per_rank_per_feature: Dict[int, Dict[str, torch.Tensor]],
        strides_per_rank_per_feature: Dict[int, Dict[str, int]],
        inverse_indices_per_rank_per_feature: Dict[int, Dict[str, torch.Tensor]],
        weights_per_rank_per_feature: Optional[Dict[int, Dict[str, torch.Tensor]]],
    ) -> KeyedJaggedTensor:
        global_values = []
        global_lengths = []
        global_stride_per_key_per_rank = []
        inverse_indices_per_feature_per_rank = []
        global_weights = [] if weights_per_rank_per_feature is not None else None

        for key in keys:
            sum_stride = 0
            for rank in range(world_size):
                global_values.append(values_per_rank_per_feature[rank][key])
                global_lengths.append(lengths_per_rank_per_feature[rank][key])
                if weights_per_rank_per_feature is not None:
                    assert global_weights is not None
                    global_weights.append(weights_per_rank_per_feature[rank][key])
                sum_stride += strides_per_rank_per_feature[rank][key]
                inverse_indices_per_feature_per_rank.append(
                    inverse_indices_per_rank_per_feature[rank][key]
                )
            global_stride_per_key_per_rank.append([sum_stride])

        inverse_indices_list: List[torch.Tensor] = []
        for key in keys:
            accum_batch_size = 0
            inverse_indices = []
            for rank in range(world_size):
                inverse_indices.append(
                    inverse_indices_per_rank_per_feature[rank][key] + accum_batch_size
                )
                accum_batch_size += strides_per_rank_per_feature[rank][key]
            inverse_indices_list.append(torch.cat(inverse_indices))
        global_inverse_indices = (keys, torch.stack(inverse_indices_list))

        if global_constant_batch:
            global_offsets = []
            for length in global_lengths:
                global_offsets.append(_to_offsets(length))
            reindexed_lengths = []
            for length, indices in zip(
                global_lengths, inverse_indices_per_feature_per_rank
            ):
                reindexed_lengths.append(torch.index_select(length, 0, indices))
            lengths = torch.cat(reindexed_lengths)
            reindexed_values, reindexed_weights = [], []
            for i, (values, offsets, indices) in enumerate(
                zip(global_values, global_offsets, inverse_indices_per_feature_per_rank)
            ):
                for idx in indices:
                    reindexed_values.append(values[offsets[idx] : offsets[idx + 1]])
                    if global_weights is not None:
                        reindexed_weights.append(
                            global_weights[i][offsets[idx] : offsets[idx + 1]]
                        )
            values = torch.cat(reindexed_values)
            weights = (
                torch.cat(reindexed_weights) if global_weights is not None else None
            )
            global_stride_per_key_per_rank = None
            global_inverse_indices = None
        else:
            values = torch.cat(global_values)
            lengths = torch.cat(global_lengths)
            weights = torch.cat(global_weights) if global_weights is not None else None

        return KeyedJaggedTensor(
            keys=keys,
            values=values,
            lengths=lengths,
            weights=weights,
            stride_per_key_per_rank=global_stride_per_key_per_rank,
            inverse_indices=global_inverse_indices,
        )

    @staticmethod
    def _generate_variable_batch_features(
        tables: Union[
            List[EmbeddingTableConfig], List[EmbeddingBagConfig], List[EmbeddingConfig]
        ],
        average_batch_size: int,
        world_size: int,
        dedup_factor: int,
        global_constant_batch: bool,
    ) -> Tuple[KeyedJaggedTensor, List[KeyedJaggedTensor]]:
        is_weighted = (
            True if tables and getattr(tables[0], "is_weighted", False) else False
        )
        feature_num_embeddings = {}
        for table in tables:
            for feature_name in table.feature_names:
                feature_num_embeddings[feature_name] = (
                    table.num_embeddings_post_pruning
                    if table.num_embeddings_post_pruning
                    else table.num_embeddings
                )

        local_kjts = []
        values_per_rank_per_feature = {}
        lengths_per_rank_per_feature = {}
        strides_per_rank_per_feature = {}
        inverse_indices_per_rank_per_feature = {}
        weights_per_rank_per_feature = {} if is_weighted else None

        local_kjts = ModelInput._generate_variable_batch_local_features(
            feature_num_embeddings,
            average_batch_size,
            world_size,
            dedup_factor,
            values_per_rank_per_feature,
            lengths_per_rank_per_feature,
            strides_per_rank_per_feature,
            inverse_indices_per_rank_per_feature,
            weights_per_rank_per_feature,
        )

        global_kjt = ModelInput._generate_variable_batch_global_features(
            list(feature_num_embeddings.keys()),
            world_size,
            global_constant_batch,
            values_per_rank_per_feature,
            lengths_per_rank_per_feature,
            strides_per_rank_per_feature,
            inverse_indices_per_rank_per_feature,
            weights_per_rank_per_feature,
        )

        return (global_kjt, local_kjts)

    @staticmethod
    def generate_variable_batch_input(
        average_batch_size: int,
        world_size: int,
        num_float_features: int,
        tables: Union[
            List[EmbeddingTableConfig], List[EmbeddingBagConfig], List[EmbeddingConfig]
        ],
        weighted_tables: Optional[
            Union[
                List[EmbeddingTableConfig],
                List[EmbeddingBagConfig],
                List[EmbeddingConfig],
            ]
        ] = None,
        pooling_avg: int = 10,
        global_constant_batch: bool = False,
    ) -> Tuple["ModelInput", List["ModelInput"]]:
        torch.manual_seed(100)
        random.seed(100)
        dedup_factor = 2
        global_kjt, local_kjts = ModelInput._generate_variable_batch_features(
            tables, average_batch_size, world_size, dedup_factor, global_constant_batch
        )
        if weighted_tables:
            global_score_kjt, local_score_kjts = (
                ModelInput._generate_variable_batch_features(
                    weighted_tables,
                    average_batch_size,
                    world_size,
                    dedup_factor,
                    global_constant_batch,
                )
            )
        else:
            global_score_kjt, local_score_kjts = None, []
        global_float = torch.rand(
            (dedup_factor * average_batch_size * world_size, num_float_features)
        )
        local_model_input = []
        label_per_rank = []
        for rank in range(world_size):
            label_per_rank.append(torch.rand(dedup_factor * average_batch_size))
            local_float = global_float[
                rank
                * dedup_factor
                * average_batch_size : (rank + 1)
                * dedup_factor
                * average_batch_size
            ]
            local_model_input.append(
                ModelInput(
                    idlist_features=local_kjts[rank],
                    idscore_features=(
                        local_score_kjts[rank] if local_score_kjts else None
                    ),
                    label=label_per_rank[rank],
                    float_features=local_float,
                ),
            )
        global_model_input = ModelInput(
            idlist_features=global_kjt,
            idscore_features=global_score_kjt,
            label=torch.cat(label_per_rank),
            float_features=global_float,
        )
        return (global_model_input, local_model_input)

    def to(self, device: torch.device, non_blocking: bool = False) -> "ModelInput":
        return ModelInput(
            float_features=self.float_features.to(
                device=device, non_blocking=non_blocking
            ),
            idlist_features=self.idlist_features.to(
                device=device, non_blocking=non_blocking
            ),
            idscore_features=(
                self.idscore_features.to(device=device, non_blocking=non_blocking)
                if self.idscore_features is not None
                else None
            ),
            label=self.label.to(device=device, non_blocking=non_blocking),
        )

    def record_stream(self, stream: torch.Stream) -> None:
        self.float_features.record_stream(stream)
        self.idlist_features.record_stream(stream)
        if self.idscore_features is not None:
            self.idscore_features.record_stream(stream)
        self.label.record_stream(stream)


class TestDenseArch(nn.Module):
    """
    Basic nn.Module for testing

    Args:
        device

    Call Args:
        dense_input: torch.Tensor

    Returns:
        KeyedTensor

    Example::

        TestDenseArch()
    """

    def __init__(
        self,
        num_float_features: int = 10,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if device is None:
            device = torch.device("cpu")
        self.linear: nn.modules.Linear = nn.Linear(
            in_features=num_float_features, out_features=8, device=device
        )

        self.dummy_param = torch.nn.Parameter(torch.zeros(2, device=device))
        self.register_buffer(
            "dummy_buffer",
            torch.nn.Parameter(torch.zeros(1, device=device)),
        )

    def forward(self, dense_input: torch.Tensor) -> torch.Tensor:
        return self.linear(dense_input)


class TestDHNArch(nn.Module):
    """
    Simple version of a model with two linear layers.
    We use this to test out recursively wrapped FSDP

    Args:
        in_feature: the size of input dim
        device: the device on which this module will be placed.

    Call Args:
        input: input tensor,

    Returns:
        torch.Tensor

    Example::

        TestDHNArch()
    """

    def __init__(
        self,
        in_features: int,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if device is None:
            device = torch.device("cpu")

        self.device = device
        self.linear0 = nn.Linear(
            in_features=in_features, out_features=16, device=device
        )
        self.linear1 = nn.Linear(in_features=16, out_features=16, device=device)

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        return self.linear1(self.linear0(input))


@torch.fx.wrap
def _concat(
    dense: torch.Tensor,
    sparse_embeddings: List[torch.Tensor],
) -> torch.Tensor:
    return torch.cat([dense] + sparse_embeddings, dim=1)


class TestOverArchRegroupModule(nn.Module):
    """
    Basic nn.Module for testing

    Args:
        device

    Call Args:
        dense: torch.Tensor,
        sparse: KeyedTensor,

    Returns:
        torch.Tensor

    Example::

        TestOverArch()
    """

    def __init__(
        self,
        tables: List[EmbeddingBagConfig],
        weighted_tables: List[EmbeddingBagConfig],
        embedding_names: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if device is None:
            device = torch.device("cpu")
        self._embedding_names: List[str] = (
            embedding_names
            if embedding_names
            else [feature for table in tables for feature in table.feature_names]
        )
        self._weighted_features: List[str] = [
            feature for table in weighted_tables for feature in table.feature_names
        ]
        in_features = (
            8
            + sum([table.embedding_dim * len(table.feature_names) for table in tables])
            + sum(
                [
                    table.embedding_dim * len(table.feature_names)
                    for table in weighted_tables
                ]
            )
        )
        self.dhn_arch: nn.Module = TestDHNArch(in_features, device)
        self.regroup_module = KTRegroupAsDict(
            [self._embedding_names, self._weighted_features],
            ["unweighted", "weighted"],
        )

    def forward(
        self,
        dense: torch.Tensor,
        sparse: KeyedTensor,
    ) -> torch.Tensor:
        pooled_emb = self.regroup_module([sparse])
        values = list(pooled_emb.values())
        return self.dhn_arch(_concat(dense, values))


class TestOverArch(nn.Module):
    """
    Basic nn.Module for testing

    Args:
        device

    Call Args:
        dense: torch.Tensor,
        sparse: KeyedTensor,

    Returns:
        torch.Tensor

    Example::

        TestOverArch()
    """

    def __init__(
        self,
        tables: List[EmbeddingBagConfig],
        weighted_tables: List[EmbeddingBagConfig],
        embedding_names: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if device is None:
            device = torch.device("cpu")
        self._embedding_names: List[str] = (
            embedding_names
            if embedding_names
            else [feature for table in tables for feature in table.feature_names]
        )
        self._weighted_features: List[str] = [
            feature for table in weighted_tables for feature in table.feature_names
        ]
        in_features = (
            8
            + sum([table.embedding_dim * len(table.feature_names) for table in tables])
            + sum(
                [
                    table.embedding_dim * len(table.feature_names)
                    for table in weighted_tables
                ]
            )
        )
        self.dhn_arch: nn.Module = TestDHNArch(in_features, device)

    def forward(
        self,
        dense: torch.Tensor,
        sparse: KeyedTensor,
    ) -> torch.Tensor:
        sparse_regrouped: List[torch.Tensor] = KeyedTensor.regroup(
            [sparse], [self._embedding_names, self._weighted_features]
        )

        return self.dhn_arch(_concat(dense, sparse_regrouped))


class TestOverArchLarge(nn.Module):
    """
    Basic nn.Module for testing, w 5/ layers.
    """

    def __init__(
        self,
        tables: List[EmbeddingBagConfig],
        weighted_tables: List[EmbeddingBagConfig],
        embedding_names: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if device is None:
            device = torch.device("cpu")
        self._embedding_names: List[str] = (
            embedding_names
            if embedding_names
            else [feature for table in tables for feature in table.feature_names]
        )
        self._weighted_features: List[str] = [
            feature for table in weighted_tables for feature in table.feature_names
        ]
        in_features = (
            8
            + sum([table.embedding_dim * len(table.feature_names) for table in tables])
            + sum(
                [
                    table.embedding_dim * len(table.feature_names)
                    for table in weighted_tables
                ]
            )
        )
        out_features = 1000
        layers = [
            torch.nn.Linear(
                in_features=in_features,
                out_features=out_features,
            ),
            SwishLayerNorm([out_features]),
        ]

        for _ in range(5):
            layers += [
                torch.nn.Linear(
                    in_features=out_features,
                    out_features=out_features,
                ),
                SwishLayerNorm([out_features]),
            ]

        self.overarch = torch.nn.Sequential(*layers)

    def forward(
        self,
        dense: torch.Tensor,
        sparse: KeyedTensor,
    ) -> torch.Tensor:
        ret_list = [dense]
        ret_list.extend(
            KeyedTensor.regroup(
                [sparse], [self._embedding_names, self._weighted_features]
            )
        )
        return self.overarch(torch.cat(ret_list, dim=1))


@torch.fx.wrap
def _post_sparsenn_forward(
    ebc: KeyedTensor,
    fp_ebc: Optional[KeyedTensor],
    w_ebc: Optional[KeyedTensor],
    batch_size: Optional[int] = None,
) -> KeyedTensor:
    if batch_size is None or ebc.values().size(0) == batch_size:
        ebc_values = ebc.values()
        fp_ebc_values = fp_ebc.values() if fp_ebc is not None else None
        w_ebc_values = w_ebc.values() if w_ebc is not None else None
    else:
        ebc_values = torch.zeros(
            batch_size,
            ebc.values().size(1),
            dtype=ebc.values().dtype,
            device=ebc.values().device,
        )
        ebc_values[: ebc.values().size(0), :] = ebc.values()
        if fp_ebc is not None:
            fp_ebc_values = torch.zeros(
                batch_size,
                fp_ebc.values().size(1),
                dtype=fp_ebc.values().dtype,
                device=fp_ebc.values().device,
            )
            fp_ebc_values[: fp_ebc.values().size(0), :] = fp_ebc.values()
        else:
            fp_ebc_values = None
        if w_ebc is not None:
            w_ebc_values = torch.zeros(
                batch_size,
                w_ebc.values().size(1),
                dtype=w_ebc.values().dtype,
                device=w_ebc.values().device,
            )
            w_ebc_values[: w_ebc.values().size(0), :] = w_ebc.values()
        else:
            w_ebc_values = None

    if fp_ebc is None and w_ebc is None:
        return KeyedTensor(
            keys=ebc.keys(),
            length_per_key=ebc.length_per_key(),
            values=ebc_values,
        )
    elif fp_ebc is None and w_ebc is not None:
        return KeyedTensor(
            keys=ebc.keys() + w_ebc.keys(),
            length_per_key=ebc.length_per_key() + w_ebc.length_per_key(),
            values=torch.cat(
                [ebc_values, torch.jit._unwrap_optional(w_ebc_values)], dim=1
            ),
        )
    elif fp_ebc is not None and w_ebc is None:
        return KeyedTensor(
            keys=ebc.keys() + fp_ebc.keys(),
            length_per_key=ebc.length_per_key() + fp_ebc.length_per_key(),
            values=torch.cat(
                [ebc_values, torch.jit._unwrap_optional(fp_ebc_values)], dim=1
            ),
        )
    else:
        assert fp_ebc is not None and w_ebc is not None
        return KeyedTensor(
            keys=ebc.keys() + fp_ebc.keys() + w_ebc.keys(),
            length_per_key=ebc.length_per_key()
            + fp_ebc.length_per_key()
            + w_ebc.length_per_key(),
            # Comment to torch.jit._unwrap_optional fp_ebc_values is inferred as Optional[Tensor] as it can be None when fp_ebc is None. But at this point we now that it has a value and doing jit._unwrap_optional will tell jit to treat it as Tensor type.
            values=torch.cat(
                [
                    ebc_values,
                    torch.jit._unwrap_optional(fp_ebc_values),
                    torch.jit._unwrap_optional(w_ebc_values),
                ],
                dim=1,
            ),
        )


class TestSparseArch(nn.Module):
    """
    Basic nn.Module for testing

    Args:
        tables
        device

    Call Args:
        features

    Returns:
        KeyedTensor
    """

    def __init__(
        self,
        tables: List[EmbeddingBagConfig],
        weighted_tables: List[EmbeddingBagConfig],
        device: Optional[torch.device] = None,
        max_feature_lengths: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__()
        if device is None:
            device = torch.device("cpu")
        self.fps: Optional[nn.ModuleList] = None
        self.fp_ebc: Optional[FeatureProcessedEmbeddingBagCollection] = None

        if max_feature_lengths is not None:
            fp_tables_names = set(max_feature_lengths.keys())
            normal_tables_names = {table.name for table in tables} - fp_tables_names

            self.ebc: EmbeddingBagCollection = EmbeddingBagCollection(
                tables=[table for table in tables if table.name in normal_tables_names],
                device=device,
            )

            fp = PositionWeightedModuleCollection(
                max_feature_lengths=max_feature_lengths,
                device=(
                    device if device != torch.device("meta") else torch.device("cpu")
                ),
            )
            self.fp_ebc = FeatureProcessedEmbeddingBagCollection(
                embedding_bag_collection=EmbeddingBagCollection(
                    tables=[table for table in tables if table.name in fp_tables_names],
                    device=device,
                    is_weighted=True,
                ),
                feature_processors=fp,
            )
        else:
            self.ebc: EmbeddingBagCollection = EmbeddingBagCollection(
                tables=tables,
                device=device,
            )

        self.weighted_ebc: Optional[EmbeddingBagCollection] = (
            EmbeddingBagCollection(
                tables=weighted_tables,
                is_weighted=True,
                device=device,
            )
            if weighted_tables
            else None
        )

    def forward(
        self,
        features: KeyedJaggedTensor,
        weighted_features: Optional[KeyedJaggedTensor] = None,
        batch_size: Optional[int] = None,
    ) -> KeyedTensor:
        fp_features = features
        if self.fps:
            # pyre-ignore[16]: Undefined attribute [16]: `Optional` has no attribute `__iter__`.
            for fp in self.fps:
                fp_features = fp(fp_features)
        ebc = self.ebc(features)
        ebc = _post_ebc_test_wrap_function(ebc)
        fp_ebc: Optional[KeyedTensor] = (
            self.fp_ebc(fp_features) if self.fp_ebc is not None else None
        )
        w_ebc = (
            self.weighted_ebc(weighted_features)
            if self.weighted_ebc is not None and weighted_features is not None
            else None
        )
        result = _post_sparsenn_forward(ebc, fp_ebc, w_ebc, batch_size)
        return result


class TestSparseNNBase(nn.Module):
    """
    Base class for a SparseNN model.

    Args:
        tables: List[BaseEmbeddingConfig],
        weighted_tables: Optional[List[BaseEmbeddingConfig]],
        embedding_groups: Optional[Dict[str, List[str]]],
        dense_device: Optional[torch.device],
        sparse_device: Optional[torch.device],
    """

    def __init__(
        self,
        tables: List[BaseEmbeddingConfig],
        weighted_tables: Optional[List[BaseEmbeddingConfig]] = None,
        embedding_groups: Optional[Dict[str, List[str]]] = None,
        dense_device: Optional[torch.device] = None,
        sparse_device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if dense_device is None:
            dense_device = torch.device("cpu")
        if sparse_device is None:
            sparse_device = torch.device("cpu")


class TestSparseNN(TestSparseNNBase, CopyableMixin):
    """
    Simple version of a SparseNN model.

    Args:
        tables: List[EmbeddingBagConfig],
        weighted_tables: Optional[List[EmbeddingBagConfig]],
        embedding_groups: Optional[Dict[str, List[str]]],
        dense_device: Optional[torch.device],
        sparse_device: Optional[torch.device],

    Call Args:
        input: ModelInput,

    Returns:
        torch.Tensor

    Example::

        TestSparseNN()
    """

    def __init__(
        self,
        tables: List[EmbeddingBagConfig],
        num_float_features: int = 10,
        weighted_tables: Optional[List[EmbeddingBagConfig]] = None,
        embedding_groups: Optional[Dict[str, List[str]]] = None,
        dense_device: Optional[torch.device] = None,
        sparse_device: Optional[torch.device] = None,
        max_feature_lengths: Optional[Dict[str, int]] = None,
        feature_processor_modules: Optional[Dict[str, torch.nn.Module]] = None,
        over_arch_clazz: Type[nn.Module] = TestOverArch,
        postproc_module: Optional[nn.Module] = None,
    ) -> None:
        super().__init__(
            tables=cast(List[BaseEmbeddingConfig], tables),
            weighted_tables=cast(Optional[List[BaseEmbeddingConfig]], weighted_tables),
            embedding_groups=embedding_groups,
            dense_device=dense_device,
            sparse_device=sparse_device,
        )
        if weighted_tables is None:
            weighted_tables = []
        self.dense = TestDenseArch(num_float_features, dense_device)
        self.sparse = TestSparseArch(
            tables,
            weighted_tables,
            sparse_device,
            max_feature_lengths,
        )

        embedding_names = (
            list(embedding_groups.values())[0] if embedding_groups else None
        )
        self._embedding_names: List[str] = (
            embedding_names
            if embedding_names
            else [feature for table in tables for feature in table.feature_names]
        )
        self._weighted_features: List[str] = [
            feature for table in weighted_tables for feature in table.feature_names
        ]
        self.over: nn.Module = over_arch_clazz(
            tables, weighted_tables, embedding_names, dense_device
        )
        self.register_buffer(
            "dummy_ones",
            torch.ones(1, device=dense_device),
        )
        self.postproc_module = postproc_module

    def sparse_forward(self, input: ModelInput) -> KeyedTensor:
        return self.sparse(
            features=input.idlist_features,
            weighted_features=input.idscore_features,
            batch_size=input.float_features.size(0),
        )

    def dense_forward(
        self, input: ModelInput, sparse_output: KeyedTensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        dense_r = self.dense(input.float_features)
        over_r = self.over(dense_r, sparse_output)
        pred = torch.sigmoid(torch.mean(over_r, dim=1)) + self.dummy_ones
        if self.training:
            return (
                torch.nn.functional.binary_cross_entropy_with_logits(pred, input.label),
                pred,
            )
        else:
            return pred

    def forward(
        self,
        input: ModelInput,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.postproc_module:
            input = self.postproc_module(input)
        return self.dense_forward(input, self.sparse_forward(input))


class TestTowerInteraction(nn.Module):
    """
    Basic nn.Module for testing

    Args:
        tables: List[EmbeddingBagConfig],
        device: Optional[torch.device],

    Call Args:
        sparse: KeyedTensor,

    Returns:
        torch.Tensor

    Example:
        >>> TestOverArch()
    """

    def __init__(
        self,
        tables: List[EmbeddingBagConfig],
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if device is None:
            device = torch.device("cpu")
        self._features: List[str] = [
            feature for table in tables for feature in table.feature_names
        ]
        in_features = sum(
            [table.embedding_dim * len(table.feature_names) for table in tables]
        )
        self.linear: nn.modules.Linear = nn.Linear(
            in_features=in_features,
            out_features=in_features,
            device=device,
        )

    def forward(
        self,
        sparse: KeyedTensor,
    ) -> torch.Tensor:
        ret_list = []
        for feature_name in self._features:
            ret_list.append(sparse[feature_name])
        return self.linear(torch.cat(ret_list, dim=1))


class TestTowerSparseNN(TestSparseNNBase):
    """
    Simple version of a SparseNN model.

    Args:
        tables: List[EmbeddingBagConfig],
        embedding_groups: Optional[Dict[str, List[str]]],
        dense_device: Optional[torch.device],
        sparse_device: Optional[torch.device],

    Call Args:
        input: ModelInput,

    Returns:
        torch.Tensor

    Example:
        >>> TestSparseNN()
    """

    def __init__(
        self,
        tables: List[EmbeddingBagConfig],
        num_float_features: int = 10,
        weighted_tables: Optional[List[EmbeddingBagConfig]] = None,
        embedding_groups: Optional[Dict[str, List[str]]] = None,
        dense_device: Optional[torch.device] = None,
        sparse_device: Optional[torch.device] = None,
        feature_processor_modules: Optional[Dict[str, torch.nn.Module]] = None,
    ) -> None:
        super().__init__(
            tables=cast(List[BaseEmbeddingConfig], tables),
            weighted_tables=cast(Optional[List[BaseEmbeddingConfig]], weighted_tables),
            embedding_groups=embedding_groups,
            dense_device=dense_device,
            sparse_device=sparse_device,
        )

        self.dense = TestDenseArch(num_float_features, dense_device)

        # TODO: after adding planner support for tower_module, we can random assign
        # tables to towers, but for now the match planner default layout
        self.tower_0 = EmbeddingTower(
            embedding_module=EmbeddingBagCollection(tables=[tables[2], tables[3]]),
            interaction_module=TestTowerInteraction(tables=[tables[2], tables[3]]),
        )
        self.tower_1 = EmbeddingTower(
            embedding_module=EmbeddingBagCollection(tables=[tables[0]]),
            interaction_module=TestTowerInteraction(tables=[tables[0]]),
        )
        self.sparse_arch = TestSparseArch(
            [tables[1]],
            # pyre-ignore [16]
            [weighted_tables[0]],
            sparse_device,
        )
        self.sparse_arch_feature_names: List[str] = (
            tables[1].feature_names + weighted_tables[0].feature_names
        )

        self.over = nn.Linear(
            in_features=8
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `out_features`.
            + self.tower_0.interaction.linear.out_features
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `out_features`.
            + self.tower_1.interaction.linear.out_features
            + tables[1].embedding_dim * len(tables[1].feature_names)
            + weighted_tables[0].embedding_dim * len(weighted_tables[0].feature_names),
            out_features=16,
            device=dense_device,
        )

    def forward(
        self,
        input: ModelInput,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        dense_r = self.dense(input.float_features)
        tower_0_r = self.tower_0(input.idlist_features)
        tower_1_r = self.tower_1(input.idlist_features)
        sparse_arch_r = self.sparse_arch(input.idlist_features, input.idscore_features)
        sparse_arch_r = torch.cat(
            [sparse_arch_r[f] for f in self.sparse_arch_feature_names], dim=1
        )

        sparse_r = torch.cat([tower_0_r, tower_1_r, sparse_arch_r], dim=1)
        over_r = self.over(torch.cat([dense_r, sparse_r], dim=1))
        pred = torch.sigmoid(torch.mean(over_r, dim=1))
        if self.training:
            return (
                torch.nn.functional.binary_cross_entropy_with_logits(pred, input.label),
                pred,
            )
        else:
            return pred


class TestTowerCollectionSparseNN(TestSparseNNBase):
    """
    Simple version of a SparseNN model.

    Constructor Args:
        tables: List[EmbeddingBagConfig],
        embedding_groups: Optional[Dict[str, List[str]]],
        dense_device: Optional[torch.device],
        sparse_device: Optional[torch.device],

    Call Args:
        input: ModelInput,

    Returns:
        torch.Tensor

    Example:
        >>> TestSparseNN()
    """

    def __init__(
        self,
        tables: List[EmbeddingBagConfig],
        num_float_features: int = 10,
        weighted_tables: Optional[List[EmbeddingBagConfig]] = None,
        embedding_groups: Optional[Dict[str, List[str]]] = None,
        dense_device: Optional[torch.device] = None,
        sparse_device: Optional[torch.device] = None,
        feature_processor_modules: Optional[Dict[str, torch.nn.Module]] = None,
    ) -> None:
        super().__init__(
            tables=cast(List[BaseEmbeddingConfig], tables),
            weighted_tables=cast(Optional[List[BaseEmbeddingConfig]], weighted_tables),
            embedding_groups=embedding_groups,
            dense_device=dense_device,
            sparse_device=sparse_device,
        )

        self.dense = TestDenseArch(num_float_features, dense_device)
        # TODO: after adding planner support for tower_module, we can random assign
        # tables to towers, but for now the match planner default layout
        tower_0 = EmbeddingTower(
            embedding_module=EmbeddingBagCollection(tables=[tables[0], tables[2]]),
            interaction_module=TestTowerInteraction(tables=[tables[0], tables[2]]),
        )
        tower_1 = EmbeddingTower(
            embedding_module=EmbeddingBagCollection(tables=[tables[1]]),
            interaction_module=TestTowerInteraction(tables=[tables[1]]),
        )
        tower_2 = EmbeddingTower(
            embedding_module=EmbeddingBagCollection(
                # pyre-ignore [16]
                tables=[weighted_tables[0]],
                is_weighted=True,
            ),
            interaction_module=TestTowerInteraction(tables=[weighted_tables[0]]),
        )
        self.tower_arch = EmbeddingTowerCollection(towers=[tower_0, tower_1, tower_2])
        self.over = nn.Linear(
            in_features=8
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `out_features`.
            + tower_0.interaction.linear.out_features
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `out_features`.
            + tower_1.interaction.linear.out_features
            # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute
            #  `out_features`.
            + tower_2.interaction.linear.out_features,
            out_features=16,
            device=dense_device,
        )

    def forward(
        self,
        input: ModelInput,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        dense_r = self.dense(input.float_features)
        sparse_r = self.tower_arch(input.idlist_features, input.idscore_features)
        over_r = self.over(torch.cat([dense_r, sparse_r], dim=1))
        pred = torch.sigmoid(torch.mean(over_r, dim=1))
        if self.training:
            return (
                torch.nn.functional.binary_cross_entropy_with_logits(pred, input.label),
                pred,
            )
        else:
            return pred


class TestEBCSharder(EmbeddingBagCollectionSharder):
    def __init__(
        self,
        sharding_type: str,
        kernel_type: str,
        fused_params: Optional[Dict[str, Any]] = None,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        if fused_params is None:
            fused_params = {}

        self._sharding_type = sharding_type
        self._kernel_type = kernel_type
        super().__init__(fused_params, qcomm_codecs_registry)

    """
    Restricts sharding to single type only.
    """

    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [self._sharding_type]

    """
    Restricts to single impl.
    """

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [self._kernel_type]


class TestFusedEBCSharder(FusedEmbeddingBagCollectionSharder):
    def __init__(
        self,
        sharding_type: str,
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__(fused_params={}, qcomm_codecs_registry=qcomm_codecs_registry)
        self._sharding_type = sharding_type

    """
    Restricts sharding to single type only.
    """

    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [self._sharding_type]


class TestFusedECSharder(FusedEmbeddingCollectionSharder):
    def __init__(
        self,
        sharding_type: str,
    ) -> None:
        super().__init__()
        self._sharding_type = sharding_type

    """
    Restricts sharding to single type only.
    """

    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [self._sharding_type]


class TestEBSharder(EmbeddingBagSharder):
    def __init__(
        self,
        sharding_type: str,
        kernel_type: str,
        fused_params: Dict[str, Any],
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__(fused_params, qcomm_codecs_registry)
        self._sharding_type = sharding_type
        self._kernel_type = kernel_type

    """
    Restricts sharding to single type only.
    """

    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [self._sharding_type]

    """
    Restricts to single impl.
    """

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [self._kernel_type]

    @property
    def fused_params(self) -> Optional[Dict[str, Any]]:
        return self._fused_params


class TestETSharder(EmbeddingTowerSharder):
    def __init__(
        self,
        sharding_type: str,
        kernel_type: str,
        fused_params: Dict[str, Any],
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__(fused_params, qcomm_codecs_registry=qcomm_codecs_registry)
        self._sharding_type = sharding_type
        self._kernel_type = kernel_type

    """
    Restricts sharding to single type only.
    """

    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [self._sharding_type]

    """
    Restricts to single impl.
    """

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [self._kernel_type]

    @property
    def fused_params(self) -> Optional[Dict[str, Any]]:
        return self._fused_params


class TestETCSharder(EmbeddingTowerCollectionSharder):
    def __init__(
        self,
        sharding_type: str,
        kernel_type: str,
        fused_params: Dict[str, Any],
        qcomm_codecs_registry: Optional[Dict[str, QuantizedCommCodecs]] = None,
    ) -> None:
        super().__init__(fused_params, qcomm_codecs_registry=qcomm_codecs_registry)
        self._sharding_type = sharding_type
        self._kernel_type = kernel_type

    """
    Restricts sharding to single type only.
    """

    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [self._sharding_type]

    """
    Restricts to single impl.
    """

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [self._kernel_type]

    @property
    def fused_params(self) -> Optional[Dict[str, Any]]:
        return self._fused_params


def _get_default_rtol_and_atol(
    actual: torch.Tensor, expected: torch.Tensor
) -> Tuple[float, float]:
    """
    default tolerance values for torch.testing.assert_close,
    consistent with the values of torch.testing.assert_close
    """
    _DTYPE_PRECISIONS = {
        torch.float16: (1e-3, 1e-3),
        torch.float32: (1e-4, 1e-5),
        torch.float64: (1e-5, 1e-8),
    }
    actual_rtol, actual_atol = _DTYPE_PRECISIONS.get(actual.dtype, (0.0, 0.0))
    expected_rtol, expected_atol = _DTYPE_PRECISIONS.get(expected.dtype, (0.0, 0.0))
    return max(actual_rtol, expected_rtol), max(actual_atol, expected_atol)


@torch.fx.wrap
def _post_ebc_test_wrap_function(kt: KeyedTensor) -> KeyedTensor:
    for _ in kt.values():
        continue

    return kt


class TestPreprocNonWeighted(nn.Module):
    """
    Basic module for testing

    Args: None
    Examples:
        >>> TestPreprocNonWeighted()
    Returns:
        List[KeyedJaggedTensor]
    """

    def forward(self, kjt: KeyedJaggedTensor) -> List[KeyedJaggedTensor]:
        """
        Selects 3 features from a specific KJT
        """
        # split
        jt_0 = kjt["feature_0"]
        jt_1 = kjt["feature_1"]
        jt_2 = kjt["feature_2"]

        # merge only features 0,1,2, removing feature 3
        return [
            KeyedJaggedTensor.from_jt_dict(
                {
                    "feature_0": jt_0,
                    "feature_1": jt_1,
                    "feature_2": jt_2,
                }
            )
        ]


class TestPreprocWeighted(nn.Module):
    """
    Basic module for testing

    Args: None
    Examples:
        >>> TestPreprocWeighted()
    Returns:
        List[KeyedJaggedTensor]
    """

    def forward(self, kjt: KeyedJaggedTensor) -> List[KeyedJaggedTensor]:
        """
        Selects 1 feature from specific weighted KJT
        """

        # split
        jt_0 = kjt["weighted_feature_0"]

        # keep only weighted_feature_0
        return [
            KeyedJaggedTensor.from_jt_dict(
                {
                    "weighted_feature_0": jt_0,
                }
            )
        ]


class TestModelWithPreproc(nn.Module):
    """
    Basic module with up to 3 postproc modules:
    - postproc on idlist_features for non-weighted EBC
    - postproc on idscore_features for weighted EBC
    - optional postproc on model input shared by both EBCs

    Args:
        tables,
        weighted_tables,
        device,
        postproc_module,
        num_float_features,
        run_postproc_inline,

    Example:
        >>> TestModelWithPreproc(tables, weighted_tables, device)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]
    """

    def __init__(
        self,
        tables: List[EmbeddingBagConfig],
        weighted_tables: List[EmbeddingBagConfig],
        device: torch.device,
        postproc_module: Optional[nn.Module] = None,
        num_float_features: int = 10,
        run_postproc_inline: bool = False,
    ) -> None:
        super().__init__()
        self.dense = TestDenseArch(num_float_features, device)

        self.ebc: EmbeddingBagCollection = EmbeddingBagCollection(
            tables=tables,
            device=device,
        )
        self.weighted_ebc = EmbeddingBagCollection(
            tables=weighted_tables,
            is_weighted=True,
            device=device,
        )
        self.postproc_nonweighted = TestPreprocNonWeighted()
        self.postproc_weighted = TestPreprocWeighted()
        self._postproc_module = postproc_module
        self._run_postproc_inline = run_postproc_inline

    def forward(
        self,
        input: ModelInput,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs preprco for EBC and weighted EBC, optionally runs postproc for input

        Args:
            input
        Returns:
            Tuple[torch.Tensor, torch.Tensor]
        """
        modified_input = input

        if self._postproc_module is not None:
            modified_input = self._postproc_module(modified_input)
        elif self._run_postproc_inline:
            idlist_features = modified_input.idlist_features
            modified_input.idlist_features = KeyedJaggedTensor.from_lengths_sync(
                idlist_features.keys(),  # pyre-ignore [6]
                idlist_features.values(),  # pyre-ignore [6]
                idlist_features.lengths(),  # pyre-ignore [16]
            )

        modified_idlist_features = self.postproc_nonweighted(
            modified_input.idlist_features
        )
        modified_idscore_features = self.postproc_weighted(
            modified_input.idscore_features
        )
        ebc_out = self.ebc(modified_idlist_features[0])
        weighted_ebc_out = self.weighted_ebc(modified_idscore_features[0])

        pred = torch.cat([ebc_out.values(), weighted_ebc_out.values()], dim=1)
        return pred.sum(), pred


class TestNegSamplingModule(torch.nn.Module):
    """
    Basic module to simulate feature augmentation postproc (e.g. neg sampling) for testing

    Args:
        extra_input
        has_params

    Example:
        >>> postproc = TestNegSamplingModule(extra_input)
        >>> out = postproc(in)

    Returns:
        ModelInput
    """

    TEST_BUFFER_NAME = "test_buffer"

    def __init__(
        self,
        extra_input: ModelInput,
        has_params: bool = False,
    ) -> None:
        super().__init__()
        self._extra_input = extra_input
        self.register_buffer(self.TEST_BUFFER_NAME, torch.zeros(1))
        if has_params:
            self._linear: nn.Module = nn.Linear(30, 30)

    def forward(self, input: ModelInput) -> ModelInput:
        """
        Appends extra features to model input

        Args:
            input
        Returns:
            ModelInput
        """

        # merge extra input
        modified_input = copy.deepcopy(input)

        # dim=0 (batch dimensions) increases by self._extra_input.float_features.shape[0]
        modified_input.float_features = torch.concat(
            (modified_input.float_features, self._extra_input.float_features), dim=0
        )

        # stride will be same but features will be joined
        modified_input.idlist_features = KeyedJaggedTensor.concat(
            [modified_input.idlist_features, self._extra_input.idlist_features]
        )
        if self._extra_input.idscore_features is not None:
            # stride will be smae but features will be joined
            modified_input.idscore_features = KeyedJaggedTensor.concat(
                # pyre-ignore
                [modified_input.idscore_features, self._extra_input.idscore_features]
            )

        # dim=0 (batch dimensions) increases by self._extra_input.input_label.shape[0]
        modified_input.label = torch.concat(
            (modified_input.label, self._extra_input.label), dim=0
        )

        return modified_input


class TestPositionWeightedPreprocModule(torch.nn.Module):
    """
    Basic module for testing

    Args: None
    Example:
        >>> postproc = TestPositionWeightedPreprocModule(max_feature_lengths, device)
        >>> out = postproc(in)
    Returns:
        ModelInput
    """

    def __init__(
        self, max_feature_lengths: Dict[str, int], device: torch.device
    ) -> None:
        super().__init__()
        self.fp_proc = PositionWeightedProcessor(
            max_feature_lengths=max_feature_lengths,
            device=device,
        )

    def forward(self, input: ModelInput) -> ModelInput:
        """
        Runs PositionWeightedProcessor

        Args:
            input
        Returns:
            ModelInput
        """
        modified_input = copy.deepcopy(input)
        modified_input.idlist_features = self.fp_proc(modified_input.idlist_features)
        return modified_input
