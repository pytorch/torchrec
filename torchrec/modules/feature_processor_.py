#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

#!/usr/bin/env python3

import abc
from typing import Dict, Optional

import torch

from torch import nn
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor


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

    def __init__(
        self, max_feature_length: int, device: Optional[torch.device] = None
    ) -> None:
        super().__init__()
        self.position_weight = nn.Parameter(
            torch.empty([max_feature_length], device=device),
            requires_grad=True,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.position_weight.fill_(1.0)

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


class FeatureProcessorsCollection(nn.Module):
    """
    Abstract base class for feature processor.

    Args:
        features (KeyedJaggedTensor]): feature representation

    Returns:
        KeyedJaggedTensor: modified KJT


    Example::
        kjt = JaggedTensor(...)
        grouped_fp = FeatureProcessorsCollection(...)
        fp_kjt = grouped_fp(kjt)
    """

    @abc.abstractmethod
    def forward(
        self,
        features: KeyedJaggedTensor,
    ) -> KeyedJaggedTensor:
        """
        Args:
        features (JaggedTensor]): feature representation

        Returns:
            JaggedTensor: modified JT
        """
        pass


@torch.fx.wrap
def get_weights_list(
    cat_seq: torch.Tensor,
    features: KeyedJaggedTensor,
    position_weights: Dict[str, nn.Parameter],
) -> Optional[torch.Tensor]:
    weights_list = []
    seqs = torch.split(cat_seq, features.length_per_key())
    for key, seq in zip(features.keys(), seqs):
        if key in position_weights.keys():
            weights_list.append(torch.gather(position_weights[key], dim=0, index=seq))
        else:
            weights_list.append(
                torch.ones(seq.shape[0], device=features.values().device)
            )
    return torch.cat(weights_list) if weights_list else features.weights_or_none()


class PositionWeightedModuleCollection(FeatureProcessorsCollection):
    def __init__(
        self, max_feature_lengths: Dict[str, int], device: Optional[torch.device] = None
    ) -> None:
        super().__init__()
        self.max_feature_lengths = max_feature_lengths
        for length in self.max_feature_lengths.values():
            if length <= 0:
                raise

        self.position_weights: nn.ParameterDict = nn.ParameterDict()
        # needed since nn.ParameterDict isn't torchscriptable (get_items)
        self.position_weights_dict: Dict[str, nn.Parameter] = {}

        for key, length in max_feature_lengths.items():
            self.position_weights[key] = nn.Parameter(
                torch.empty([length], device=device)
            )

            self.position_weights_dict[key] = self.position_weights[key]

        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            for key, _length in self.max_feature_lengths.items():
                self.position_weights[key].fill_(1.0)
                # Re-assign python dict to param dict in case of re-materialization
                self.position_weights_dict[key] = self.position_weights[key]

    def forward(self, features: KeyedJaggedTensor) -> KeyedJaggedTensor:
        cat_seq = torch.ops.fbgemm.offsets_range(
            features.offsets().long(), torch.numel(features.values())
        )

        return KeyedJaggedTensor(
            keys=features.keys(),
            values=features.values(),
            weights=get_weights_list(cat_seq, features, self.position_weights_dict),
            lengths=features.lengths(),
            offsets=features.offsets(),
            stride=features.stride(),
            length_per_key=features.length_per_key(),
        )
