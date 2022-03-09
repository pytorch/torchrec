#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Torchrec Common Modules

The torchrec modules contain a collection of various modules.

These modules include:
    - extensions of `nn.Embedding` and `nn.EmbeddingBag`, called `EmbeddingBagCollection`
      and `EmbeddingCollection` respectively.
    - established modules such as `DeepFM <https://arxiv.org/pdf/1703.04247.pdf>`_ and
      `CrossNet <https://arxiv.org/abs/1708.05123>`_.
    - common module patterns such as `MLP` and `SwishLayerNorm`.
    - custom modules for torchrec such as `PositionWeightedModule` and
      `LazyModuleExtensionMixin`.
"""

from . import activation  # noqa
from . import crossnet  # noqa
from . import deepfm  # noqa
from . import embedding_configs  # noqa
from . import embedding_modules  # noqa
from . import feature_processor  # noqa
from . import lazy_extension  # noqa
from . import mlp  # noqa
from . import tower  # noqa
