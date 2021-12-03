#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torch
import torch.nn as nn
from torchrec.sparse.jagged_tensor import (
    KeyedJaggedTensor,
)


# pyre-fixme[56]: Pyre was not able to infer the type of the decorator `torch.fx.wrap`.
@torch.fx.wrap
def lengths_range_fill(lengths: torch.Tensor) -> torch.Tensor:
    """
    Generate arange list for each length element
    Example:
    lengths = torch.Tensor([3, 1, 2])
    return torch.Tensor([0, 1, 2, 0, 0, 1])
    """
    seq_list = [torch.arange(start=0, end=i, dtype=torch.int64) for i in lengths]
    return torch.cat(seq_list)


class PositionWeightsAttacher(nn.Module):
    """
    Map id list features to id score list features using each id's
    position in the sample.

    Constructor Args:
        features_max_length (Dict[str, int]): feature name to max_length mapping.
            max_length, a.k.a truncation size, specifies the maximum number of ids
            each sample has. For each feature, its position weight parameter size
            is max_length.

    Call Args:
        features: KeyedJaggedTensor

    Returns:
        weighted_features (KeyedJaggedTensor): same as input features with weights
            field being populated

    Example:
        >>> features_max_length = {"f1": 10, "f2": 3}
        pw = PositionWeightsAttacher(features_max_lengths)

        #     0       1        2  <-- batch
        # 0   [0,1] None    [2]
        # 1   [3]    [4]    [5,6,7]
        # ^
        # feature
        features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f2"],
            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
            offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8]),
        )
        weighted_features = pw(features)
    """

    def __init__(
        self,
        features_max_length: Dict[str, int],
    ) -> None:
        super().__init__()
        self.features_max_length = features_max_length
        self.position_weights = nn.ParameterDict()
        for feature_name, max_length in features_max_length.items():
            # pyre-fixme[29]: `Union[nn.Module, torch.Tensor]` is not a function.
            self.position_weights[feature_name] = nn.Parameter(torch.ones(max_length))

    def forward(
        self,
        features: KeyedJaggedTensor,
    ) -> KeyedJaggedTensor:
        features_weights = []
        for feature_name, _ in self.features_max_length.items():
            lengths = features[feature_name].lengths()
            # TODO(T92151660): replace pt ops with fbgemm's lengths_range_w_truncation_size
            # and fast_gather
            seq = lengths_range_fill(lengths)
            weights = torch.gather(self.position_weights[feature_name], 0, seq)
            features_weights.append(weights)
        weights = torch.cat(features_weights)
        return KeyedJaggedTensor.from_lengths_sync(
            keys=features.keys(),
            values=features.values(),
            lengths=features.lengths(),
            weights=weights,
        )
