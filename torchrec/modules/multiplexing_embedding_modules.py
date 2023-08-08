#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, List

import torch
import torch.nn as nn

from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollectionInterface,
    get_embedding_names_by_table,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor


class MultiplexingEmbeddingBagCollection(EmbeddingBagCollectionInterface):
    """
    MultiplexingEmbeddingBagCollection is a special EBC that allows splitting
    the collection of embedding bags into multiple groups, each of which has a
    different variant of EBC implementation.

    It has identical semantics to `EmbeddingBagCollection` (See its documentation
    for input/output details).

    Args:
        tables (List[EmbeddingBagConfig]): list of embedding tables.
        regroup_functor (Callable[[EmbeddingBagConfig], str]):
            a function accepting an embedding table config and returning a string
            representing the subgroup name the table belongs to.
        ebc_init_functor (str ->  Callable[[List[EmbeddingBagConfig]], EmbeddingBagCollectionInterface]):
            a dictionary of constructor functions, which constructs the EBC for
            that group given the list of embedding tables. Functions are keyed by
            group name, or "*" for catch-call constructor.
        is_weighted (bool): whether input `KeyedJaggedTensor` is weighted.

    Examples:

        tables=[
            EmbeddingBagConfig(
                name="t1", embedding_dim=4, num_embeddings=10, feature_names=["f1"]
            ),
            EmbeddingBagConfig(
                name="t2", embedding_dim=8, num_embeddings=10, feature_names=["f1", "f2"]
            ),
        ]
        ebc = MultiplexingEmbeddingBagCollection(
            tables=tables,
            regroup_functor=lambda c: "multifeature" if len(c.feature_names) > 1 else "singlefeature",
            ebc_init_functor={
                "multifeature": lambda l : EmbeddingBagCollection(l),
                "*": lambda l : FusedEmbeddingBagCollection(l, optimizer_type=torch.optim.SGD, optimizer_kwargs={"lr": .01}),
            },
        )
        features = KeyedJaggedTensor(
            keys=["f1", "f2"],
            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
            offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8]),
        )
        pooled_embeddings = ebc(features)
        print(pooled_embeddings.values())

        tensor([[-0.5943, -0.0921,  0.0480, -0.2055, -0.2443,  0.0262,  0.2740, -0.1015,
              0.5262, -0.1523, -0.4258, -0.0306, -0.2695, -0.1459,  0.1162,  0.2864,
              0.1066, -0.0724,  0.2839,  0.0389],
            [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
              0.0000,  0.0000,  0.0000,  0.0000, -0.2597, -0.3038, -0.3083, -0.1560,
              0.2962,  0.2461, -0.1721,  0.2606],
            [-0.1516,  0.1559,  0.0599, -0.1350,  0.1958, -0.2178, -0.2824,  0.1925,
            -0.0526,  0.2030,  0.2736,  0.1965, -0.2912, -0.1038,  0.6177, -0.1836,
            -0.3240,  0.4003, -0.0822, -0.1145]],
          grad_fn=<SplitWithSizesBackward0>)

        print(pooled_embeddings.keys())

        ['f1@t1', 'f1@t2', 'f2']

        print(pooled_embeddings.offset_per_key())

        [0, 4, 12, 20]
    """

    def __init__(
        self,
        tables: List[EmbeddingBagConfig],
        regroup_functor: Callable[[EmbeddingBagConfig], str],
        ebc_init_functor: Dict[
            str, Callable[[List[EmbeddingBagConfig]], EmbeddingBagCollectionInterface]
        ],
        is_weighted: bool = False,
    ) -> None:
        super().__init__()
        self.tables: List[EmbeddingBagConfig] = tables
        self._is_weighted: bool = is_weighted
        torch._C._log_api_usage_once(f"torchrec.modules.{self.__class__.__name__}")

        assert len(tables) > 0, "Need at least one table."
        # Sanity check and pre-processing
        table_names = set()
        for table in tables:
            if table.name in table_names:
                raise ValueError(f"Duplicate table name {table.name}")
            table_names.add(table.name)
            if not table.feature_names:
                table.feature_names = [table.name]

        ebcs: Dict[str, EmbeddingBagCollectionInterface] = {}
        grouped_tables: Dict[str, List[EmbeddingBagConfig]] = {}
        final_key_list_by_table: List[List[str]] = get_embedding_names_by_table(tables)
        assert len(final_key_list_by_table) == len(tables)

        self.grouped_embedding_namelist: Dict[str, List[str]] = {}
        self.regroup_target_embedding_namelist_flattened: List[str] = [
            embedding_name
            for embedding_names in final_key_list_by_table
            for embedding_name in embedding_names
        ]
        self._post_regroup_length_per_key: List[int] = []
        for table, final_key_list in zip(tables, final_key_list_by_table):
            self._post_regroup_length_per_key.extend(
                len(table.feature_names) * [table.embedding_dim]
            )
            group: str = regroup_functor(table)
            grouped_tables.setdefault(group, []).append(table)
            self.grouped_embedding_namelist.setdefault(group, []).extend(final_key_list)

        assert len(self._post_regroup_length_per_key) == len(
            self.regroup_target_embedding_namelist_flattened
        )
        for group, tables_in_group in grouped_tables.items():
            ebc_init_key = group if group in ebc_init_functor else "*"
            ebcs[group] = ebc_init_functor[ebc_init_key](tables_in_group)
            if ebcs[group].is_weighted() != self._is_weighted:
                raise ValueError(
                    f"Inconsistent is_weighted for group {group}, "
                    f"expected {self._is_weighted}, got {ebcs[group].is_weighted()}"
                )

        self.ebcs = nn.ModuleDict(modules=ebcs)

    def forward(
        self,
        features: KeyedJaggedTensor,
    ) -> KeyedTensor:
        results: List[KeyedTensor] = []
        for group, ebc in self.ebcs.items():
            result: KeyedTensor = ebc(features)
            # We need to replace the returned KeyedTensor key list so duplicate
            # feature across tables can be distinguished.
            # assert len(result.keys()) == len(self.grouped_embedding_namelist[group])
            result = KeyedTensor(
                keys=self.grouped_embedding_namelist[group],
                values=result.values(),
                length_per_key=result.length_per_key(),
            )
            results.append(result)
        flattened_results = KeyedTensor.regroup(
            results, [self.regroup_target_embedding_namelist_flattened]
        )

        return KeyedTensor(
            keys=self.regroup_target_embedding_namelist_flattened,
            length_per_key=self._post_regroup_length_per_key,
            values=flattened_results[0],
        )

    def embedding_bag_configs(
        self,
    ) -> List[EmbeddingBagConfig]:
        return self.tables

    def is_weighted(self) -> bool:
        return self._is_weighted
