#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Torchrec Models

Torchrec provides the architecture for two popular recsys models;
`DeepFM <https://arxiv.org/pdf/1703.04247.pdf>`_ and `DLRM (Deep Learning Recommendation Model)
<https://arxiv.org/abs/1906.00091>`_.

Along with the overall model, the individual architectures of each layer are also
provided (e.g. `SparseArch`, `DenseArch`, `InteractionArch`, and `OverArch`).

Examples can be found within each model.
"""

from . import deepfm  # noqa
from . import dlrm  # noqa
