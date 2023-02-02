#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, cast, Dict, List, Optional, Tuple, Union

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
from torchrec.modules.embedding_configs import (
    BaseEmbeddingConfig,
    EmbeddingBagConfig,
    EmbeddingConfig,
)
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.modules.embedding_tower import EmbeddingTower, EmbeddingTowerCollection
from torchrec.modules.feature_processor import PositionWeightedProcessor
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor
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

        idlist_features_to_num_embeddings = {}
        for table in tables:
            for feature in table.feature_names:
                idlist_features_to_num_embeddings[feature] = table.num_embeddings

        idlist_features = list(idlist_features_to_num_embeddings.keys())
        idscore_features = [
            feature for table in weighted_tables for feature in table.feature_names
        ]

        idlist_ind_ranges = list(idlist_features_to_num_embeddings.values())
        idscore_ind_ranges = [table.num_embeddings for table in weighted_tables]

        # Generate global batch.
        global_idlist_lengths = []
        global_idlist_indices = []
        global_idscore_lengths = []
        global_idscore_indices = []
        global_idscore_weights = []

        for ind_range in idlist_ind_ranges:
            lengths_ = torch.abs(
                torch.randn(batch_size * world_size) + pooling_avg
            ).int()
            if variable_batch_size:
                lengths = torch.zeros(batch_size * world_size).int()
                for r in range(world_size):
                    lengths[
                        r * batch_size : r * batch_size + batch_size_by_rank[r]
                    ] = lengths_[
                        r * batch_size : r * batch_size + batch_size_by_rank[r]
                    ]
            else:
                lengths = lengths_
            num_indices = cast(int, torch.sum(lengths).item())
            indices = torch.randint(0, ind_range, (num_indices,))
            global_idlist_lengths.append(lengths)
            global_idlist_indices.append(indices)
        global_idlist_kjt = KeyedJaggedTensor(
            keys=idlist_features,
            values=torch.cat(global_idlist_indices),
            lengths=torch.cat(global_idlist_lengths),
        )

        for ind_range in idscore_ind_ranges:
            lengths_ = torch.abs(
                torch.randn(batch_size * world_size) + pooling_avg
            ).int()
            if variable_batch_size:
                lengths = torch.zeros(batch_size * world_size).int()
                for r in range(world_size):
                    lengths[
                        r * batch_size : r * batch_size + batch_size_by_rank[r]
                    ] = lengths_[
                        r * batch_size : r * batch_size + batch_size_by_rank[r]
                    ]
            else:
                lengths = lengths_
            num_indices = cast(int, torch.sum(lengths).item())
            indices = torch.randint(0, ind_range, (num_indices,))
            weights = torch.rand((num_indices,))
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

        global_float = torch.rand((batch_size * world_size, num_float_features))
        global_label = torch.rand(batch_size * world_size)

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

    def to(self, device: torch.device, non_blocking: bool = False) -> "ModelInput":
        return ModelInput(
            float_features=self.float_features.to(
                device=device, non_blocking=non_blocking
            ),
            idlist_features=self.idlist_features.to(
                device=device, non_blocking=non_blocking
            ),
            idscore_features=self.idscore_features.to(
                device=device, non_blocking=non_blocking
            )
            if self.idscore_features is not None
            else None,
            label=self.label.to(device=device, non_blocking=non_blocking),
        )

    def record_stream(self, stream: torch.cuda.streams.Stream) -> None:
        # pyre-fixme[6]: For 1st param expected `Stream` but got `Stream`.
        self.float_features.record_stream(stream)
        self.idlist_features.record_stream(stream)
        if self.idscore_features is not None:
            self.idscore_features.record_stream(stream)
        # pyre-fixme[6]: For 1st param expected `Stream` but got `Stream`.
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

        self.dummy_param = torch.nn.Parameter(torch.empty(2, device=device))
        self.register_buffer(
            "dummy_buffer",
            torch.nn.Parameter(torch.empty(1, device=device)),
        )

    def forward(self, dense_input: torch.Tensor) -> torch.Tensor:
        return self.linear(dense_input)


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
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if device is None:
            device = torch.device("cpu")
        self._features: List[str] = [
            feature for table in tables for feature in table.feature_names
        ]
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
        self.linear: nn.modules.Linear = nn.Linear(
            in_features=in_features, out_features=16, device=device
        )

    def forward(
        self,
        dense: torch.Tensor,
        sparse: KeyedTensor,
    ) -> torch.Tensor:
        ret_list = []
        ret_list.append(dense)
        for feature_name in self._features:
            ret_list.append(sparse[feature_name])
        for feature_name in self._weighted_features:
            ret_list.append(sparse[feature_name])
        return self.linear(torch.cat(ret_list, dim=1))


@torch.fx.wrap
def _post_sparsenn_forward(
    ebc: KeyedTensor,
    fp_ebc: Optional[KeyedTensor],
    w_ebc: KeyedTensor,
    batch_size: Optional[int] = None,
) -> KeyedTensor:
    if batch_size is None or ebc.values().size(0) == batch_size:
        ebc_values = ebc.values()
        fp_ebc_values = fp_ebc.values() if fp_ebc is not None else None
        w_ebc_values = w_ebc.values()
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
        w_ebc_values = torch.zeros(
            batch_size,
            w_ebc.values().size(1),
            dtype=w_ebc.values().dtype,
            device=w_ebc.values().device,
        )
        w_ebc_values[: w_ebc.values().size(0), :] = w_ebc.values()
    result = (
        KeyedTensor(
            keys=ebc.keys() + w_ebc.keys(),
            length_per_key=ebc.length_per_key() + w_ebc.length_per_key(),
            values=torch.cat([ebc_values, w_ebc_values], dim=1),
        )
        if fp_ebc is None
        else KeyedTensor(
            keys=ebc.keys() + fp_ebc.keys() + w_ebc.keys(),
            length_per_key=ebc.length_per_key()
            + fp_ebc.length_per_key()
            + w_ebc.length_per_key(),
            # Comment to torch.jit._unwrap_optional fp_ebc_values is inferred as Optional[Tensor] as it can be None when fp_ebc is None. But at this point we now that it has a value and doing jit._unwrap_optional will tell jit to treat it as Tensor type.
            values=torch.cat(
                [ebc_values, torch.jit._unwrap_optional(fp_ebc_values), w_ebc_values],
                dim=1,
            ),
        )
    )
    return result


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
        max_feature_lengths_list: Optional[List[Dict[str, int]]] = None,
    ) -> None:
        super().__init__()
        if device is None:
            device = torch.device("cpu")
        self.fps: Optional[nn.ModuleList] = None
        self.fp_ebc: Optional[EmbeddingBagCollection] = None
        if max_feature_lengths_list is not None:
            self.fps = nn.ModuleList(
                [
                    PositionWeightedProcessor(
                        max_feature_lengths=max_feature_lengths,
                        device=device
                        if device != torch.device("meta")
                        else torch.device("cpu"),
                    )
                    for max_feature_lengths in max_feature_lengths_list
                ]
            )
            normal_id_list_tables = []
            fp_id_list_tables = []
            for table in tables:
                # the key set of feature_processor is either subset or none in the feature_names
                if set(table.feature_names).issubset(
                    set(max_feature_lengths_list[0].keys())
                ):
                    fp_id_list_tables.append(table)
                else:
                    normal_id_list_tables.append(table)

            self.ebc: EmbeddingBagCollection = EmbeddingBagCollection(
                tables=normal_id_list_tables,
                device=device,
            )
            self.fp_ebc: EmbeddingBagCollection = EmbeddingBagCollection(
                tables=fp_id_list_tables,
                device=device,
                is_weighted=True,
            )
        else:
            self.ebc: EmbeddingBagCollection = EmbeddingBagCollection(
                tables=tables,
                device=device,
            )
        self.weighted_ebc: EmbeddingBagCollection = EmbeddingBagCollection(
            tables=weighted_tables,
            is_weighted=True,
            device=device,
        )

    def forward(
        self,
        features: KeyedJaggedTensor,
        weighted_features: KeyedJaggedTensor,
        batch_size: Optional[int] = None,
    ) -> KeyedTensor:
        fp_features = features
        if self.fps:
            # pyre-ignore[16]: Undefined attribute [16]: `Optional` has no attribute `__iter__`.
            for fp in self.fps:
                fp_features = fp(fp_features)
        ebc = self.ebc(features)
        fp_ebc: Optional[KeyedTensor] = (
            self.fp_ebc(fp_features) if self.fp_ebc is not None else None
        )
        w_ebc = self.weighted_ebc(weighted_features)
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
        feature_processor_modules: Optional[Dict[str, torch.nn.Module]] = None,
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
        max_feature_lengths_list: Optional[List[Dict[str, int]]] = None,
        feature_processor_modules: Optional[Dict[str, torch.nn.Module]] = None,
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
            max_feature_lengths_list if max_feature_lengths_list is not None else None,
        )
        self.over = TestOverArch(tables, weighted_tables, dense_device)

    def forward(
        self,
        input: ModelInput,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        dense_r = self.dense(input.float_features)
        sparse_r = self.sparse(
            input.idlist_features, input.idscore_features, input.float_features.size(0)
        )
        over_r = self.over(dense_r, sparse_r)
        pred = torch.sigmoid(torch.mean(over_r, dim=1))
        if self.training:
            return (
                torch.nn.functional.binary_cross_entropy_with_logits(pred, input.label),
                pred,
            )
        else:
            return pred


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
            # pyre-ignore [16]
            + self.tower_0.interaction.linear.out_features
            # pyre-ignore [16]
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
            # pyre-ignore [16]
            + tower_0.interaction.linear.out_features
            # pyre-ignore [16]
            + tower_1.interaction.linear.out_features
            # pyre-ignore [16]
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
