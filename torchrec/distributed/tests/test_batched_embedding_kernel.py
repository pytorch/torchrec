#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from fbgemm_gpu.split_embedding_configs import SparseType
from torchrec.distributed.batched_embedding_kernel import configure_fused_params


class BatchedEmbeddingKernelTest(unittest.TestCase):
    def test_configure_fused_params(self) -> None:
        # Case fused_params is None, change it to be an empty dict
        # and set cache_precision to be the same as weights_precision
        fused_params = None
        weights_precision = SparseType.FP16
        configured_fused_params = configure_fused_params(
            fused_params=fused_params, weights_precision=weights_precision
        )
        self.assertFalse(configured_fused_params is None)
        self.assertEqual(configured_fused_params["cache_precision"], weights_precision)

        # Case fused_params does not preset cache_precision
        # and set cache_precision to be the same as weights_precision
        fused_params = {}
        weights_precision = SparseType.FP16
        configured_fused_params = configure_fused_params(
            fused_params=fused_params, weights_precision=weights_precision
        )
        self.assertEqual(configured_fused_params["cache_precision"], weights_precision)

        # Case fused_params presets cache_precision, return the same fused_params as it is
        fused_params = {}
        configured_weights_precision = SparseType.INT8
        fused_params = {"cache_precision": configured_weights_precision}
        configured_fused_params = configure_fused_params(
            fused_params=fused_params, weights_precision=SparseType.FP16
        )
        self.assertEqual(configured_fused_params, fused_params)
