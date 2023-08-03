#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingBagCollectionInterface,
)
from torchrec.modules.managed_collision_modules import ManagedCollisionModule
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor


@torch.fx.wrap
def apply_managed_collision_modules_to_kjt(
    features: KeyedJaggedTensor,
    managed_collisions: nn.ModuleDict,
    feature_to_table: Dict[str, str],
) -> KeyedJaggedTensor:

    if len(features.keys()) == 0:
        return features

    features_dict = features.to_dict()
    managed_ids = []

    for key in features.keys():
        jt = features_dict[key]
        table_name = feature_to_table[key]
        if table_name in managed_collisions:
            mc_jt = managed_collisions[table_name](jt)
        else:
            mc_jt = jt
        managed_ids.append(mc_jt.values())

    return KeyedJaggedTensor(
        keys=features.keys(),
        values=torch.cat(managed_ids),
        weights=features.weights_or_none(),
        lengths=features.lengths(),
        offsets=features._offsets,
        stride=features._stride,
        length_per_key=features._length_per_key,
        offset_per_key=features._offset_per_key,
        index_per_key=features._index_per_key,
    )


def evict(
    evictions: Dict[str, Optional[torch.Tensor]], ebc: EmbeddingBagCollection
) -> None:
    # evicts ids corresponding with table name from ebcs
    # If none, do a no-op
    return


class ManagedCollisionEmbeddingBagCollection(EmbeddingBagCollectionInterface):
    """
    ManagedCollisionEmbeddingBagCollection represents a EmbeddingBagCollection module and a set of managed collision modules.
    The inputs into the MC-EBC will first be modified by the managed collision module before being passed into the embedding bag collection.

    For details of input and output types, see EmbeddingBagCollection

    Args:
        embedding_bag_collection: EmbeddingBagCollection to lookup embeddings
        managed_collision_modules: Dict of managed collision modules

    Example:
        ebc = EmbeddingBagCollection(
                tables=[
                    EmbeddingBagConfig(
                        name="t1", embedding_dim=8, num_embeddings=16, feature_names=["f1"]
                    ),
                    EmbeddingBagConfig(
                        name="t2", embedding_dim=8, num_embeddings=16, feature_names=["f2"]
                    ),
                ],
                device=device,
            )
        mc_modules = {
            "t1": ManagedCollisionModule(
                    max_output_id=16, max_input_id=32, device=device
                ),
            "t2":
                ManagedCollisionModule(
                    max_output_id=16, max_input_id=32, device=device
                ),
            ),
        }

        mc_ebc = ManagedCollisionEmbeddingBagCollection(ebc, mc_modules)
        kt = mc_ebc(kjt)
    """

    def __init__(
        self,
        embedding_bag_collection: EmbeddingBagCollection,
        managed_collision_modules: Dict[str, ManagedCollisionModule],
    ) -> None:
        super().__init__()
        self._embedding_bag_collection = embedding_bag_collection
        self._managed_collision_modules = nn.ModuleDict(managed_collision_modules)

        self._features_to_tables: Dict[str, str] = {}
        self._table_to_features: Dict[str, List[str]] = defaultdict(list)
        for table in self._embedding_bag_collection.embedding_bag_configs():
            if table.name not in managed_collision_modules:
                raise ValueError(
                    f"{table.name} is not present in managed_collision_modules"
                )

            assert (
                self._managed_collision_modules[table.name]._max_output_id
                == table.num_embeddings
            ), f"max_output_id in managed collision module for {table.name} must match {table.num_embeddings=}"

            for feature in table.feature_names:
                self._features_to_tables[feature] = table.name
                self._table_to_features[table.name].append(feature)

    def forward(
        self,
        features: KeyedJaggedTensor,
    ) -> KeyedTensor:
        """
        Args:
            features (KeyedJaggedTensor): KJT of form [F X B X L].

        Returns:
            KeyedTensor
        """

        mc_features = apply_managed_collision_modules_to_kjt(
            features, self._managed_collision_modules, self._features_to_tables
        )
        ret = self._embedding_bag_collection(mc_features)

        evictions: Dict[str, Optional[torch.Tensor]] = {}
        for table, managed_collision_module in self._managed_collision_modules.items():
            evictions[table] = managed_collision_module.evict()
        evict(evictions, self._embedding_bag_collection)

        return ret

    def is_weighted(self) -> bool:
        return self._embedding_bag_collection.is_weighted()

    def embedding_bag_configs(self) -> List[EmbeddingBagConfig]:
        return self._embedding_bag_collection.embedding_bag_configs()

    @property
    def device(self) -> torch.device:
        return self._embedding_bag_collection.device
