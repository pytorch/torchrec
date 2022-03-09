#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union

import torch
import torch.nn as nn
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)
from torchrec.sparse.jagged_tensor import (
    KeyedJaggedTensor,
)


class EmbeddingTower(nn.Module):
    """
    Logical "Tower" of embeddings directly passed to custom interaction

    Args:
        embedding_module: Union[EmbeddingBagCollection, EmbeddingCollection],
        interaction_module: nn.Module,
        device: Optional[torch.device],

    Example:

        >>> ebc = EmbeddingBagCollection()
        >>> interaction = MyInteractionModule()
        >>> embedding_tower = EmbeddingTower(ebc, interaction, device)
        >>> kjt = KeyedJaggedTensor()
        >>> output = embedding_tower(kjt)
    """

    def __init__(
        self,
        embedding_module: Union[EmbeddingBagCollection, EmbeddingCollection],
        interaction_module: nn.Module,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.embedding = embedding_module
        self.interaction = interaction_module

    def forward(
        self,
        features: KeyedJaggedTensor,
    ) -> torch.Tensor:
        """
        Run the embedding module and interaction module

        Args:
            features: KeyedJaggedTensor,

        Returns:
            torch.Tensor: 2D tensor
        """
        embeddings = self.embedding(features)
        return self.interaction(embeddings)
