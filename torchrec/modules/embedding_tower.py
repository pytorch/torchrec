#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


def tower_input_params(module: nn.Module) -> Tuple[bool, bool]:
    """
    Utilty to compute mapping of tower kjt args to pass to embedding modules

    Args:
        module: nn.Module
    Returns:
        Tuple[bool, bool]: representing kjt and wkjt required, respectively
    """
    if isinstance(module, EmbeddingCollection):
        return True, False
    elif isinstance(module, EmbeddingBagCollection):
        return not module.is_weighted, module.is_weighted
    # default to assuming both kjt and weight_kjt required
    return True, True


class EmbeddingTower(nn.Module):
    """
    Logical "Tower" of embeddings directly passed to custom interaction

    Args:
        embedding_module: nn.Module,
        interaction_module: nn.Module,
        device: Optional[torch.device],

    Example:

        >>> ebc, interaction = EmbeddingBagCollection(), MyInteractionModule()
        >>> tower = EmbeddingTower(ebc, interaction, device)
        >>> kjt = KeyedJaggedTensor()
        >>> output = tower(kjt)
    """

    def __init__(
        self,
        embedding_module: nn.Module,
        interaction_module: nn.Module,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.embedding = embedding_module
        self.interaction = interaction_module

    def forward(
        self,
        # pyre-ignore [2]
        *args,
        # pyre-ignore [2]
        **kwargs,
    ) -> torch.Tensor:
        """
        Run the embedding module and interaction module, support all torchrec shardable embedding modules

        Args:
            *args: user provided
            **kwargs: user provided

        Returns:
            torch.Tensor, 2-D shape of N X B, where B is local batch size
        """
        embeddings = self.embedding(*args, **kwargs)
        return self.interaction(embeddings)


class EmbeddingTowerCollection(nn.Module):
    """
    Collection of EmbeddingTowers

    Args:
        towers: List[EmbeddingTower]
        device: Optional[torch.device],

    Example:

        >>> ebc, ebc_interaction = EmbeddingBagCollection(), MyEBCInteractionModule()
        >>> eb, eb_interaction = EmbeddingCollection(), MyECInteractionModule()
        >>> tower_0 = EmbeddingTower(ebc, ebc_interaction, device)
        >>> tower_1 = EmbeddingTower(eb, eb_interaction, device)
        >>> tower_collection = EmbeddingTowerCollection([tower_0, tower_1])
        >>> kjt = KeyedJaggedTensor()
        >>> output = tower_collection(kjt)
    """

    def __init__(
        self,
        towers: List[EmbeddingTower],
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.towers = nn.ModuleList(towers)
        self._input_params: List[Tuple[bool, bool]] = []
        for tower in towers:
            self._input_params.append(tower_input_params(tower.embedding))

    def forward(
        self,
        features: Optional[KeyedJaggedTensor] = None,
        weighted_features: Optional[KeyedJaggedTensor] = None,
    ) -> torch.Tensor:
        """
        Run the collection of towers.  User is requires to pass features and/or weighted features as
        required by underlying embedding modules

        Args:
            features: Optional[KeyedJaggedTensor]
            weighted_features: Optional[KeyedJaggedTensor]

        Returns:
            torch.Tensor, 2-D shape of M X B, where M sum(N_i) for tower output i, and B is local batch size
        """

        tower_outputs = []
        for tower, input_params in zip(self.towers, self._input_params):
            has_kjt_param, has_wkjt_param = input_params
            if has_kjt_param and has_wkjt_param:
                assert features is not None
                assert weighted_features is not None
                tower_outputs.append(tower(features, weighted_features))
            elif has_wkjt_param:
                assert weighted_features is not None
                tower_outputs.append(tower(weighted_features))
            else:
                assert features is not None
                tower_outputs.append(tower(features))

        return torch.cat(tower_outputs, dim=1)
