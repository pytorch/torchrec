#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import unittest

import torch
from torchrec.sparse.jagged_tensor import (
    _regroup_keyed_tensors,
    KeyedJaggedTensor,
    KeyedTensor,
)
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


@skip_if_asan_class
class TestKeyedJaggedTensorGPU(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.device = torch.cuda.current_device()

    # pyre-ignore
    @unittest.skipIf(
        torch.cuda.device_count() <= 0,
        "Not enough GPUs, this test requires at least one GPUs",
    )
    def test_permute(self) -> None:
        values = torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], device=self.device
        )
        lengths = torch.tensor([0, 2, 0, 1, 1, 1, 0, 3, 0], device=self.device)
        keys = ["index_0", "index_1", "index_2"]

        jag_tensor = KeyedJaggedTensor.from_lengths_sync(
            values=values,
            keys=keys,
            lengths=lengths,
        )
        indices = [1, 0, 2]
        permuted_jag_tensor = jag_tensor.permute(indices)

        self.assertEqual(permuted_jag_tensor.keys(), ["index_1", "index_0", "index_2"])
        self.assertEqual(
            permuted_jag_tensor.offset_per_key(),
            [0, 3, 5, 8],
        )
        self.assertEqual(
            permuted_jag_tensor.values().tolist(),
            [3.0, 4.0, 5.0, 1.0, 2.0, 6.0, 7.0, 8.0],
        )
        self.assertEqual(
            permuted_jag_tensor.lengths().tolist(), [1, 1, 1, 0, 2, 0, 0, 3, 0]
        )
        self.assertEqual(permuted_jag_tensor.weights_or_none(), None)

    # pyre-ignore
    @unittest.skipIf(
        torch.cuda.device_count() <= 0,
        "Not enough GPUs, this test requires at least one GPUs",
    )
    def test_permute_vb(self) -> None:
        values = torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], device=self.device
        )
        lengths = torch.tensor([1, 0, 1, 3, 0, 1, 0, 2, 0], device=self.device)
        keys = ["index_0", "index_1", "index_2"]
        stride_per_key_per_rank = [[2], [4], [3]]

        jag_tensor = KeyedJaggedTensor.from_lengths_sync(
            values=values,
            keys=keys,
            lengths=lengths,
            stride_per_key_per_rank=stride_per_key_per_rank,
        )

        indices = [1, 0, 2]
        permuted_jag_tensor = jag_tensor.permute(indices)

        self.assertEqual(permuted_jag_tensor.keys(), ["index_1", "index_0", "index_2"])
        self.assertEqual(
            permuted_jag_tensor.offset_per_key(),
            [0, 5, 6, 8],
        )
        self.assertEqual(
            permuted_jag_tensor.values().tolist(),
            [2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 7.0, 8.0],
        )
        self.assertEqual(
            permuted_jag_tensor.lengths().tolist(), [1, 3, 0, 1, 1, 0, 0, 2, 0]
        )
        self.assertEqual(permuted_jag_tensor.weights_or_none(), None)

    # pyre-ignore
    @unittest.skipIf(
        torch.cuda.device_count() <= 0,
        "Not enough GPUs, this test requires at least one GPUs",
    )
    def test_permute_vb_duplicate(self) -> None:
        values = torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], device=self.device
        )
        lengths = torch.tensor([1, 0, 1, 3, 0, 1, 0, 2, 0], device=self.device)
        keys = ["index_0", "index_1", "index_2"]
        stride_per_key_per_rank = [[2], [4], [3]]

        jag_tensor = KeyedJaggedTensor.from_lengths_sync(
            values=values,
            keys=keys,
            lengths=lengths,
            stride_per_key_per_rank=stride_per_key_per_rank,
        )

        indices = [1, 1, 0, 0, 2, 2]
        permuted_jag_tensor = jag_tensor.permute(indices)

        self.assertEqual(
            permuted_jag_tensor.keys(),
            ["index_1", "index_1", "index_0", "index_0", "index_2", "index_2"],
        )
        self.assertTrue(
            torch.equal(
                permuted_jag_tensor.values().cpu(),
                torch.Tensor(
                    [
                        2.0,
                        3.0,
                        4.0,
                        5.0,
                        6.0,
                        2.0,
                        3.0,
                        4.0,
                        5.0,
                        6.0,
                        1.0,
                        1.0,
                        7.0,
                        8.0,
                        7.0,
                        8.0,
                    ]
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                permuted_jag_tensor.lengths().cpu(),
                torch.IntTensor([1, 3, 0, 1, 1, 3, 0, 1, 1, 0, 1, 0, 0, 2, 0, 0, 2, 0]),
            )
        )
        self.assertEqual(permuted_jag_tensor.weights_or_none(), None)

    # pyre-ignore
    @unittest.skipIf(
        torch.cuda.device_count() <= 0,
        "Not enough GPUs, this test requires at least one GPUs",
    )
    def test_permute_duplicates(self) -> None:
        values = torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], device=self.device
        )
        lengths = torch.tensor([0, 2, 0, 1, 1, 1, 0, 3, 0], device=self.device)
        keys = ["index_0", "index_1", "index_2"]

        jag_tensor = KeyedJaggedTensor.from_lengths_sync(
            values=values,
            keys=keys,
            lengths=lengths,
        )

        indices = [1, 0, 2, 1, 1]
        permuted_jag_tensor = jag_tensor.permute(indices)

        self.assertEqual(
            permuted_jag_tensor.keys(),
            ["index_1", "index_0", "index_2", "index_1", "index_1"],
        )
        self.assertEqual(
            permuted_jag_tensor.offset_per_key(),
            [0, 3, 5, 8, 11, 14],
        )
        self.assertEqual(
            permuted_jag_tensor.values().tolist(),
            [
                3.0,
                4.0,
                5.0,
                1.0,
                2.0,
                6.0,
                7.0,
                8.0,
                3.0,
                4.0,
                5.0,
                3.0,
                4.0,
                5.0,
            ],
        )
        self.assertEqual(
            permuted_jag_tensor.lengths().tolist(),
            [1, 1, 1, 0, 2, 0, 0, 3, 0, 1, 1, 1, 1, 1, 1],
        )
        self.assertEqual(permuted_jag_tensor.weights_or_none(), None)
