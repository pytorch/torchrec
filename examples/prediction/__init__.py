#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
TorchRec recommendation model examples.

This package contains examples of different recommendation models implemented using TorchRec:
- DLRM (Deep Learning Recommendation Model)
- Two-Tower Model

Each model is organized in its own subdirectory with complete implementation, tests, and documentation.
"""

# Import main components from DLRM
from torchrec.github.examples.prediction.dlrm.predict_using_dlrm import (
    create_kjt_from_batch,
    DLRMRatingWrapper,
    RecommendationDataset as DLRMDataset,
    TorchRecDLRM,
)

# Import main components from Two-Tower
from torchrec.github.examples.prediction.twoTower.predict_using_twotower import (
    create_kjt_from_ids,
    RecommendationDataset as TwoTowerDataset,
    TwoTowerModel,
    TwoTowerRatingWrapper,
)

__all__ = [
    # DLRM components
    "DLRMRatingWrapper",
    "DLRMDataset",
    "TorchRecDLRM",
    "create_kjt_from_batch",
    # Two-Tower components
    "TwoTowerModel",
    "TwoTowerRatingWrapper",
    "TwoTowerDataset",
    "create_kjt_from_ids",
]
