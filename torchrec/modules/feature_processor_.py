#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

import abc

import torch

from torch import nn
from torchrec.sparse.jagged_tensor import JaggedTensor


class FeatureProcessor(nn.Module):
    """
    Abstract base class for feature processor.

    Args:
        features (JaggedTensor]): feature representation

    Returns:
        JaggedTensor: modified JT


    Example::
        jt = JaggedTensor(...)
        fp = FeatureProcessor(...)
        fp_jt = FeatureProcessor(fp)
    """

    @abc.abstractmethod
    def forward(
        self,
        features: JaggedTensor,
    ) -> JaggedTensor:
        """
        Args:
        features (JaggedTensor]): feature representation

        Returns:
            JaggedTensor: modified JT
        """
        pass


class PositionWeightedModule(FeatureProcessor):
    """
    Adds position weights to id list features.

    Args:
        `max_length`, a.k.a truncation size, specifies the maximum number of ids
        each sample has. For each feature, its position weight parameter size is
        `max_length`.
    """

    def __init__(self, max_feature_length: int) -> None:
        super().__init__()
        self.position_weight = nn.Parameter(
            torch.empty([max_feature_length]).fill_(1.0)
        )

    def forward(
        self,
        features: JaggedTensor,
    ) -> JaggedTensor:

        """
        Args:
            features (JaggedTensor]): feature representation

        Returns:
            JaggedTensor: same as input features with `weights` field being populated.
        """

        seq = torch.ops.fbgemm.offsets_range(
            features.offsets().long(), torch.numel(features.values())
        )
        weighted_features = JaggedTensor(
            values=features.values(),
            lengths=features.lengths(),
            offsets=features.offsets(),
            weights=torch.gather(self.position_weight, dim=0, index=seq),
        )
        return weighted_features
