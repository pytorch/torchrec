#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import unittest

import hypothesis.strategies as st

import torch
from hypothesis import Verbosity, given, settings
from fbgemm_gpu.test.test_utils import gpu_unavailable

try:
    torch.ops.load_library("fbgemm_gpu_py.so")
except Exception:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")

MAX_EXAMPLES = 20

class LayoutTransformOpsTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    # pyre-fixme[56]
    @given(
        B=st.integers(min_value=1, max_value=20),
        T=st.integers(min_value=1, max_value=20),
        D=st.integers(min_value=2, max_value=20),
        W=st.integers(min_value=1, max_value=20),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_recat_embedding_grad_output(self, B: int, T: int, D: int, W: int) -> None:
        num_features_per_rank = np.random.randint(low=1, high=20, size=(W,)).tolist()
        grad_output = torch.randn(B, sum(num_features_per_rank), D).float().cuda()
        grad_outputs_by_rank = grad_output.split(num_features_per_rank, dim=1)
        sharded_grad_output = torch.cat(
            [
                grad_output_by_rank.contiguous().view(-1)
                for grad_output_by_rank in grad_outputs_by_rank
            ],
            dim=0,
        )
        sharded_grad_output_impl = torch.ops.fbgemm.recat_embedding_grad_output(
            grad_output, num_features_per_rank
        )
        torch.testing.assert_allclose(sharded_grad_output_impl.cpu(), sharded_grad_output.cpu())

    @unittest.skipIf(*gpu_unavailable)
    # pyre-fixme[56]
    @given(
        B=st.integers(min_value=1, max_value=20),
        W=st.integers(min_value=1, max_value=20),
        cuda=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_recat_embedding_grad_output_mixed_D(self, B: int, W: int, cuda: bool) -> None:
        num_features_per_rank = np.random.randint(low=1, high=20, size=(W,)).tolist()
        global_T = sum(num_features_per_rank)
        mixed_D_list = np.random.randint(low=1, high=10, size=(global_T,))
        grad_output = torch.randn(B, sum(mixed_D_list)).float().cuda()
        if cuda:
            grad_output = grad_output.cuda()
        num_feature_offsets_list = torch.tensor(
            [0] + np.cumsum(num_features_per_rank).tolist()
        )
        dim_sum_per_rank = [
            sum(
                mixed_D_list[
                    num_feature_offsets_list[i] : num_feature_offsets_list[i + 1]
                ]
            )
            for i in range(W)
        ]
        grad_outputs_by_rank = grad_output.split(dim_sum_per_rank, dim=1)
        sharded_grad_output = torch.cat(
            [
                grad_output_by_rank.contiguous().view(-1)
                for grad_output_by_rank in grad_outputs_by_rank
            ],
            dim=0,
        )
        sharded_grad_output_impl = torch.ops.fbgemm.recat_embedding_grad_output_mixed_D(
            grad_output, dim_sum_per_rank
        )
        torch.testing.assert_allclose(sharded_grad_output_impl.cpu(), sharded_grad_output.cpu())

    @unittest.skipIf(*gpu_unavailable)
    # pyre-fixme[56]
    @given(
        B=st.integers(min_value=1, max_value=20),
        W=st.integers(min_value=1, max_value=20),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_recat_embedding_grad_output_mixed_D_batch(self, B: int, W: int) -> None:
        num_features_per_rank = np.random.randint(low=1, high=20, size=(W,)).tolist()
        global_T = sum(num_features_per_rank)
        mixed_D_list = np.random.randint(low=1, high=10, size=(global_T,))
        grad_output = torch.randn(B, sum(mixed_D_list)).float().cuda()
        num_feature_offsets_list = torch.tensor(
            [0] + np.cumsum(num_features_per_rank).tolist()
        )
        dim_sum_per_rank = [
            sum(
                mixed_D_list[
                    num_feature_offsets_list[i] : num_feature_offsets_list[i + 1]
                ]
            )
            for i in range(W)
        ]
        dim_sum_per_rank_tensor = torch.cuda.LongTensor(dim_sum_per_rank)
        cumsum_dim_sum_per_rank_tensor = torch.cuda.LongTensor(
            np.cumsum([0] + dim_sum_per_rank)[:-1]
        )

        grad_outputs_by_rank = grad_output.split(dim_sum_per_rank, dim=1)
        sharded_grad_output = torch.cat(
            [
                grad_output_by_rank.contiguous().view(-1)
                for grad_output_by_rank in grad_outputs_by_rank
            ],
            dim=0,
        )
        sharded_grad_output_impl = (
            torch.ops.fbgemm.recat_embedding_grad_output_mixed_D_batch(
                grad_output.cuda(),
                dim_sum_per_rank_tensor.cuda(),
                cumsum_dim_sum_per_rank_tensor.cuda(),
            )
        )
        torch.testing.assert_allclose(sharded_grad_output_impl.cpu(), sharded_grad_output.cpu())
        num_features_per_rank = np.random.randint(low=1, high=20, size=(W,)).tolist()
        global_T = sum(num_features_per_rank)
        mixed_D_list = np.random.randint(low=1, high=10, size=(global_T,))
        grad_output = torch.randn(B, sum(mixed_D_list)).float().cuda()
        num_feature_offsets_list = torch.tensor(
            [0] + np.cumsum(num_features_per_rank).tolist()
        )
        dim_sum_per_rank = [
            sum(
                mixed_D_list[
                    num_feature_offsets_list[i] : num_feature_offsets_list[i + 1]
                ]
            )
            for i in range(W)
        ]
        dim_sum_per_rank_tensor = torch.cuda.LongTensor(dim_sum_per_rank)
        cumsum_dim_sum_per_rank_tensor = torch.cuda.LongTensor(
            np.cumsum([0] + dim_sum_per_rank)[:-1]
        )

        grad_outputs_by_rank = grad_output.split(dim_sum_per_rank, dim=1)
        sharded_grad_output = torch.cat(
            [
                grad_output_by_rank.contiguous().view(-1)
                for grad_output_by_rank in grad_outputs_by_rank
            ],
            dim=0,
        )
        sharded_grad_output_impl = torch.ops.fbgemm.recat_embedding_grad_output_mixed_D_batch(
            grad_output, dim_sum_per_rank_tensor, cumsum_dim_sum_per_rank_tensor
        )
        torch.testing.assert_allclose(sharded_grad_output_impl.cpu(), sharded_grad_output.cpu())

if __name__ == "__main__":
    unittest.main()
