#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
from typing import Dict

import torch
import torch.nn as nn
from torchrec.sparse.jagged_tensor import JaggedTensor


class BaseFeatureProcessor(nn.Module):
    """
    Abstract base class for feature processor.
    """

    @abc.abstractmethod
    def forward(
        self,
        features: Dict[str, JaggedTensor],
    ) -> Dict[str, JaggedTensor]:
        pass


class PositionWeightedModule(BaseFeatureProcessor):
    """
    Adds position weights to id list features.

    Args:
        max_feature_lengths (Dict[str, int]): feature name to `max_length` mapping.
            `max_length`, a.k.a truncation size, specifies the maximum number of ids
            each sample has. For each feature, its position weight parameter size is
            `max_length`.
    """

    def __init__(
        self,
        max_feature_lengths: Dict[str, int],
    ) -> None:
        super().__init__()
        self.max_feature_lengths = max_feature_lengths
        self.position_weights: nn.ParameterDict = nn.ParameterDict()
        for key, length in max_feature_lengths.items():
            self.position_weights[key] = nn.Parameter(torch.empty([length]).fill_(1.0))

    def forward(
        self,
        features: Dict[str, JaggedTensor],
    ) -> Dict[str, JaggedTensor]:
        """
        Args:
            features (Dict[str, JaggedTensor]): dictionary of keys to `JaggedTensor`,
                representing the features.

        Returns:
            Dict[str, JaggedTensor]: same as input features with `weights` field being populated.
        """

        weighted_features: Dict[str, JaggedTensor] = {}
        for key, pos_weight in self.position_weights.items():
            seq = torch.ops.fbgemm.offsets_range(
                features[key].offsets().long(), torch.numel(features[key].values())
            )
            weighted_features[key] = JaggedTensor(
                values=features[key].values(),
                lengths=features[key].lengths(),
                offsets=features[key].offsets(),
                weights=torch.gather(pos_weight, dim=0, index=seq),
            )
        return weighted_features
