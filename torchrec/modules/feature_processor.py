#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
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
    abstract base class for feature processor
    """

    @abc.abstractmethod
    def forward(
        self,
        features: Dict[str, JaggedTensor],
    ) -> Dict[str, JaggedTensor]:
        pass


class PositionWeightedModule(BaseFeatureProcessor):
    def __init__(
        self,
        max_feature_lengths: Dict[str, int],
    ) -> None:
        super().__init__()
        self.max_feature_lengths = max_feature_lengths
        self.position_weights: nn.ParameterDict = nn.ParameterDict()
        for key, length in max_feature_lengths.items():
            # pyre-fixme[29]
            self.position_weights[key] = nn.Parameter(torch.empty([length]).fill_(1.0))

    def forward(
        self,
        features: Dict[str, JaggedTensor],
    ) -> Dict[str, JaggedTensor]:
        ret: Dict[str, JaggedTensor] = {}
        # pyre-fixme[29]
        for key, pos_weight in self.position_weights.items():
            seq = torch.ops.fbgemm.offsets_range(
                features[key].lengths().long(), features[key].values().long()
            )
            ret[key] = JaggedTensor(
                values=features[key].values(),
                lengths=features[key].lengths(),
                offsets=features[key].offsets(),
                weights=torch.gather(pos_weight, dim=0, index=seq),
            )
        return ret
