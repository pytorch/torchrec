#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import List, Optional

from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.planner.constants import (
    DDR_MEM_BW,
    HBM_MEM_BW,
    kernel_bw_lookup,
)


class TestKernelBWLookup(unittest.TestCase):
    def test_uvm_caching_bw(self) -> None:
        compute_device: str = "cuda"
        computer_kernel: str = EmbeddingComputeKernel.FUSED_UVM_CACHING.value

        caching_ratios: List[float] = [0, 0.25, 0.5, 0.75, 1]

        uvm_caching_bw: list[Optional[float]] = [
            kernel_bw_lookup(
                compute_device, computer_kernel, HBM_MEM_BW, DDR_MEM_BW, caching_ratio
            )
            for caching_ratio in caching_ratios
        ]
        expected_uvm_caching_bw: List[float] = [
            23643794.96448,
            28185722.880000003,
            50895362.457600005,
            73605002.0352,
            96314641.6128,
        ]

        self.assertEqual(expected_uvm_caching_bw, uvm_caching_bw)

    def test_uvm_caching_bw_with_prefetch_pipeline(self) -> None:
        compute_device: str = "cuda"
        computer_kernel: str = EmbeddingComputeKernel.FUSED_UVM_CACHING.value
        prefetch_pipeline: bool = True

        caching_ratios: List[float] = [0, 0.25, 0.5, 0.75, 1]

        uvm_caching_bw: list[Optional[float]] = [
            kernel_bw_lookup(
                compute_device,
                computer_kernel,
                HBM_MEM_BW,
                DDR_MEM_BW,
                caching_ratio,
                prefetch_pipeline,
            )
            for caching_ratio in caching_ratios
        ]
        expected_uvm_caching_bw: List[float] = [
            963146416.128,
            963146416.128,
            963146416.128,
            963146416.128,
            963146416.128,
        ]

        self.assertEqual(expected_uvm_caching_bw, uvm_caching_bw)
