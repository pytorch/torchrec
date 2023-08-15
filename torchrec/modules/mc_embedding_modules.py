#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.modules.managed_collision_modules import ManagedCollisionModule
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor, KeyedTensor


@torch.fx.wrap
def apply_managed_collision_modules_to_kjt(
    features: KeyedJaggedTensor,
    managed_collisions: nn.ModuleDict,
    feature_to_table: Dict[str, str],
    mode: Optional[str] = None,
    mc_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> KeyedJaggedTensor:
    if len(features.keys()) == 0:
        return features

    features_dict = features.to_dict()
    managed_ids = []
    for key in features.keys():
        jt = features_dict[key]
        if key in feature_to_table:
            table_name = feature_to_table[key]
            mc_module = managed_collisions[table_name]
            if mode == "preprocess":
                mc_jt = mc_module.preprocess(jt)
                mc_jt_values = mc_jt.values()
            elif mode == "local_to_global":
                local_to_global_offset = mc_module.local_map_global_offset()
                mc_jt_values = jt.values() + local_to_global_offset
            else:
                mc_jt = mc_module(
                    jt,
                    mc_kwargs=mc_kwargs.get(table_name, None)
                    if mc_kwargs is not None
                    else None,
                )
                mc_jt_values = mc_jt.values()
        else:
            mc_jt = jt
            mc_jt_values = mc_jt.values()

        managed_ids.append(mc_jt_values)

    mc_kjt = KeyedJaggedTensor(
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

    return mc_kjt


def evict(
    evictions: Dict[str, Optional[torch.Tensor]], ebc: EmbeddingBagCollection
) -> None:
    # evicts ids corresponding with table name from ebcs
    # If none, do a no-op
    return


class ManagedCollisionCollection(nn.Module):
    """
    ManagedCollisionCollection represents a collection of managed collision modules.
    The inputs passed to the MCC will be remapped by the managed collision modules
        and returned.
    Args:
        feature_to_mc: Dict of feature name to its managed collision module to use.
        managed_collision_modules: Dict of managed collision modules

    Example:
        feature_to_mc = {
            "f1": "t1",
            "f2": "t2",
        }
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
        Dict[str, JaggedTensor] = mc_ebc(kjt)
    """

    def __init__(
        self,
        feature_to_mc: Dict[str, str],
        managed_collision_modules: Dict[str, ManagedCollisionModule],
    ) -> None:
        super().__init__()
        self._device: torch.device = list(managed_collision_modules.values())[0]._device
        self._features_to_mc = feature_to_mc
        self._managed_collision_modules = nn.ModuleDict(managed_collision_modules)
        self._mc_to_features: Dict[str, List[str]] = defaultdict(list)
        for feature_name, mc_name in feature_to_mc.items():
            self._mc_to_features[mc_name].append(feature_name)
        for mc_name in self._mc_to_features.keys():
            if mc_name not in self._managed_collision_modules:
                raise ValueError(
                    f"{mc_name} is not present in managed_collision_modules"
                )

    def compute(
        self,
        features: KeyedJaggedTensor,
        mc_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> KeyedJaggedTensor:
        features = apply_managed_collision_modules_to_kjt(
            features,
            self._managed_collision_modules,
            self._features_to_mc,
            mode="preprocess",
        )
        mc_features = apply_managed_collision_modules_to_kjt(
            features,
            self._managed_collision_modules,
            self._features_to_mc,
            mc_kwargs=mc_kwargs,
        )

        return mc_features

    def forward(
        self,
        features: KeyedJaggedTensor,
        mc_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, JaggedTensor]:
        """
        Args:
            features (KeyedJaggedTensor): KJT of form [F X B X L].
            mc_kwargs (Optional[Dict[str, Dict[str, Any]]]): optional args dict to
                pass to MC modules
        Returns:
            Dict[str, JaggedTensor] of remapped features
        """
        mc_features = self.compute(features, mc_kwargs)

        return mc_features.to_dict()


class ManagedCollisionEmbeddingBagCollection(nn.Module):
    """
    ManagedCollisionEmbeddingBagCollection represents a EmbeddingBagCollection module and a set of managed collision modules.
    The inputs into the MC-EBC will first be modified by the managed collision module before being passed into the embedding bag collection.

    For details of input and output types, see EmbeddingBagCollection

    Args:
        embedding_bag_collection: EmbeddingBagCollection to lookup embeddings
        managed_collision_modules: Dict of managed collision modules
        return_remapped_features (bool): whether to return remapped input features
            in addition to embeddings

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
        managed_collision_modules: Union[
            ManagedCollisionCollection, Dict[str, ManagedCollisionModule]
        ],
        return_remapped_features: bool = False,
    ) -> None:
        super().__init__()
        self._embedding_bag_collection = embedding_bag_collection
        self._return_remapped_features = return_remapped_features

        self._features_to_tables: Dict[str, str] = {}
        self._table_to_features: Dict[str, List[str]] = defaultdict(list)
        for table in self._embedding_bag_collection.embedding_bag_configs():
            for feature in table.feature_names:
                assert feature not in self._features_to_tables, (
                    "shared (same input feature to multiple tables) is "
                    "not currently supported."
                )
                self._features_to_tables[feature] = table.name
                self._table_to_features[table.name].append(feature)

        if isinstance(managed_collision_modules, ManagedCollisionCollection):
            self._managed_collision_collection: ManagedCollisionCollection = (
                managed_collision_modules
            )
            managed_collision_modules = (
                self._managed_collision_collection._managed_collision_modules
            )
        else:
            self._managed_collision_collection = ManagedCollisionCollection(
                self._features_to_tables,
                managed_collision_modules,
            )

        for table in self._embedding_bag_collection.embedding_bag_configs():
            if table.name not in managed_collision_modules:
                raise ValueError(
                    f"{table.name} is not present in managed_collision_modules"
                )
            assert (
                managed_collision_modules[table.name]._max_output_id
                == table.num_embeddings
            ), (
                f"max_output_id in managed collision module for {table.name} "
                f"must match {table.num_embeddings=}"
            )

    def forward(
        self,
        features: KeyedJaggedTensor,
        mc_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Union[KeyedTensor, Tuple[KeyedTensor, Dict[str, JaggedTensor]]]:
        """
        Args:
            features (KeyedJaggedTensor): KJT of form [F X B X L].
            mc_kwargs (Optional[Dict[str, Dict[str, Any]]]): optional args dict to
                pass to MC modules
        Returns:
            KeyedTensor or Tuple[KeyedTensor, Dict[str, JaggedTensor]]
        """

        mc_features = self._managed_collision_collection.compute(features, mc_kwargs)

        ret = self._embedding_bag_collection(mc_features)

        evictions: Dict[str, Optional[torch.Tensor]] = {}
        for (
            table,
            managed_collision_module,
        ) in self._managed_collision_collection._managed_collision_modules.items():
            evictions[table] = managed_collision_module.evict()
        evict(evictions, self._embedding_bag_collection)

        if not self._return_remapped_features:
            return ret
        else:
            return ret, mc_features.to_dict()
