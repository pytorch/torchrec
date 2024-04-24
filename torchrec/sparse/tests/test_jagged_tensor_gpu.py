#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import torch
from torchrec.sparse.jagged_tensor import _regroup_keyed_tensors, KeyedTensor
from torchrec.sparse.tests.utils import build_groups, build_kts
from torchrec.test_utils import skip_if_asan_class


@skip_if_asan_class
class TestKeyedTensorGPU(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.device = torch.cuda.current_device()

    # pyre-ignore
    @unittest.skipIf(
        torch.cuda.device_count() <= 0,
        "Not enough GPUs, this test requires at least one GPUs",
    )
    def test_regroup_backward_skips_and_duplicates(self) -> None:
        kts = build_kts(
            dense_features=20,
            sparse_features=20,
            dim_dense=64,
            dim_sparse=128,
            batch_size=128,
            device=self.device,
            run_backward=True,
        )
        groups = build_groups(kts=kts, num_groups=2, skips=True, duplicates=True)
        labels = torch.randint(0, 1, (128,), device=self.device).float()

        tensor_groups = KeyedTensor.regroup(kts, groups)
        pred0 = tensor_groups[0].sum(dim=1).mul(tensor_groups[1].sum(dim=1))
        loss = torch.nn.functional.l1_loss(pred0, labels).sum()
        actual_kt_0_grad = torch.autograd.grad(
            loss, kts[0].values(), retain_graph=True
        )[0]
        actual_kt_1_grad = torch.autograd.grad(
            loss, kts[1].values(), retain_graph=True
        )[0]

        # clear grads are return
        kts[0].values().grad = None
        kts[1].values().grad = None

        tensor_groups = _regroup_keyed_tensors(kts, groups)
        pred1 = tensor_groups[0].sum(dim=1).mul(tensor_groups[1].sum(dim=1))
        loss = torch.nn.functional.l1_loss(pred1, labels).sum()
        expected_kt_0_grad = torch.autograd.grad(
            loss, kts[0].values(), retain_graph=True
        )[0]
        expected_kt_1_grad = torch.autograd.grad(
            loss, kts[1].values(), retain_graph=True
        )[0]

        torch.allclose(actual_kt_0_grad, expected_kt_0_grad)
        torch.allclose(actual_kt_1_grad, expected_kt_1_grad)

    # pyre-ignore
    @unittest.skipIf(
        torch.cuda.device_count() <= 0,
        "Not enough GPUs, this test requires at least one GPUs",
    )
    def test_regroup_backward(self) -> None:
        kts = build_kts(
            dense_features=20,
            sparse_features=20,
            dim_dense=64,
            dim_sparse=128,
            batch_size=128,
            device=self.device,
            run_backward=True,
        )
        groups = build_groups(kts=kts, num_groups=2, skips=False, duplicates=False)
        labels = torch.randint(0, 1, (128,), device=self.device).float()

        tensor_groups = KeyedTensor.regroup(kts, groups)
        pred0 = tensor_groups[0].sum(dim=1).mul(tensor_groups[1].sum(dim=1))
        loss = torch.nn.functional.l1_loss(pred0, labels).sum()
        actual_kt_0_grad = torch.autograd.grad(
            loss, kts[0].values(), retain_graph=True
        )[0]
        actual_kt_1_grad = torch.autograd.grad(
            loss, kts[1].values(), retain_graph=True
        )[0]

        # clear grads are return
        kts[0].values().grad = None
        kts[1].values().grad = None

        tensor_groups = _regroup_keyed_tensors(kts, groups)
        pred1 = tensor_groups[0].sum(dim=1).mul(tensor_groups[1].sum(dim=1))
        loss = torch.nn.functional.l1_loss(pred1, labels).sum()
        expected_kt_0_grad = torch.autograd.grad(
            loss, kts[0].values(), retain_graph=True
        )[0]
        expected_kt_1_grad = torch.autograd.grad(
            loss, kts[1].values(), retain_graph=True
        )[0]

        torch.allclose(actual_kt_0_grad, expected_kt_0_grad)
        torch.allclose(actual_kt_1_grad, expected_kt_1_grad)
