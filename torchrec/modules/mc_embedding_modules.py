#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.modules.mc_modules import ManagedCollisionCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor


def evict(
    evictions: Dict[str, Optional[torch.Tensor]], ebc: EmbeddingBagCollection
) -> None:
    # TODO: write function
    return


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

    """

    def __init__(
        self,
        embedding_bag_collection: EmbeddingBagCollection,
        managed_collision_collection: ManagedCollisionCollection,
        return_remapped_features: bool = False,
    ) -> None:
        super().__init__()
        self._embedding_bag_collection = embedding_bag_collection
        self._managed_collision_collection = managed_collision_collection
        self._return_remapped_features = return_remapped_features

        assert (
            self._embedding_bag_collection.embedding_bag_configs()
            == self._managed_collision_collection.embedding_configs()
        ), "Embedding Collection and Managed Collision Collection must contain the Embedding Configs"

    def forward(
        self,
        features: KeyedJaggedTensor,
    ) -> Tuple[KeyedTensor, Optional[KeyedJaggedTensor]]:

        features = self._managed_collision_collection(features)

        pooled_embeddings = self._embedding_bag_collection(features)

        evict(
            self._managed_collision_collection.evict(), self._embedding_bag_collection
        )

        if not self._return_remapped_features:
            return pooled_embeddings, None
        return pooled_embeddings, features
