#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchrec.distributed.test_utils.test_model_parallel import ModelParallelBase
from torchrec.distributed.test_utils.test_model_parallel_base import (
    ModelParallelSparseOnlyBase,
    ModelParallelStateDictBase,
)

# CPU tests for Gloo.


class ModelParallelTestGloo(ModelParallelBase):
    def setUp(self, backend: str = "gloo") -> None:
        super().setUp(backend=backend)


class ModelParallelStateDictTestGloo(ModelParallelStateDictBase):
    def setUp(self, backend: str = "gloo") -> None:
        super().setUp(backend=backend)


class ModelParallelSparseOnlyTestGloo(ModelParallelSparseOnlyBase):
    def setUp(self, backend: str = "gloo") -> None:
        super().setUp(backend=backend)
