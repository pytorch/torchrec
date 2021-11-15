#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
import unittest
from itertools import accumulate
from typing import Optional, Tuple, Type, Union

import hypothesis.strategies as st
import numpy as np
import torch
from fbgemm_gpu.test.test_utils import gpu_available, gpu_unavailable
from hypothesis import Verbosity, given, settings

try:
    torch.ops.load_library("fbgemm_gpu_py.so")
except Exception:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")

np_int_types = Union[Type[np.int32], Type[np.int64]]


def unbucketize_indices_value(
    bucketized_indices: torch.Tensor,
    bucketized_lengths: torch.Tensor,
    block_sizes: torch.Tensor,
    W: int,
    B: int,
) -> torch.Tensor:
    block_size_expand = torch.empty_like(bucketized_indices)
    bucket_expand = torch.empty_like(bucketized_indices)
    T = block_sizes.size()[0]
    offset = 0
    for w in range(W):
        for t in range(T):
            for b in range(B):
                seg_length = bucketized_lengths[w * T * B + t * B + b]
                for i in range(offset, offset + seg_length):
                    block_size_expand[i] = block_sizes[t]
                    bucket_expand[i] = w
                offset += seg_length
    return bucket_expand * block_size_expand + bucketized_indices


def lengths_to_segment_ids(lengths: torch.Tensor) -> torch.Tensor:
    return torch.repeat_interleave(
        torch._dim_arange(lengths, 0).long(),
        lengths.long(),
    )


# Converts lengths + values format to COO format
# [B], [N, D] -> [B, N', D].
# pyre-ignore Missing return annotation [3]
def var_list_to_coo(lengths: torch.Tensor, values: torch.Tensor, N: int, D: int):
    rows = lengths_to_segment_ids(lengths)
    num_rows = lengths.size()[0]
    offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
    output_size = lengths.sum()
    # This does D&H sync
    cols = torch.ops.fbgemm.offsets_range(offsets, output_size)
    indices = torch.stack([rows, cols])
    dims = [num_rows, N, D]
    # torch.sparse_coo_tensor is not supported by torch.fx, wrap it.
    return torch.sparse_coo_tensor(
        indices=indices,
        values=values,
        size=dims,
    )


class SparseOpsTest(unittest.TestCase):
    @staticmethod
    def permute_indices_ref_(
        lengths: torch.Tensor,
        indices: torch.Tensor,
        weights: Optional[torch.Tensor],
        permute: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        T = lengths.size(0)
        permuted_lengths = torch.index_select(lengths.view(T, -1), 0, permute)

        original_segment_lengths = lengths.view(T, -1).sum(dim=1, dtype=torch.int32)
        original_segment_start = [0] + list(
            accumulate(original_segment_lengths.view(-1))
        )

        permuted_indices = []
        permuted_weights = []
        for i in range(permute.size(0)):
            start = original_segment_start[permute[i]]
            end = start + original_segment_lengths[permute[i]]
            permuted_indices.append(indices[start:end])
            if weights is not None:
                permuted_weights.append(weights[start:end])

        permuted_indices = torch.cat(permuted_indices, dim=0).flatten()

        if weights is None:
            permuted_weights = None
        else:
            permuted_weights = torch.cat(permuted_weights, dim=0).flatten()

        return permuted_lengths, permuted_indices, permuted_weights

    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
    @given(
        B=st.integers(min_value=1, max_value=20),
        T=st.integers(min_value=1, max_value=20),
        L=st.integers(min_value=2, max_value=20),
        long_index=st.booleans(),
        has_weight=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)
    def test_permute_indices(
        self, B: int, T: int, L: int, long_index: bool, has_weight: bool
    ) -> None:
        index_dtype = torch.int64 if long_index else torch.int32
        lengths = torch.randint(low=1, high=L, size=(T, B)).type(index_dtype)
        weights = torch.rand(lengths.sum().item()).float() if has_weight else None
        indices = torch.randint(
            low=1,
            high=int(1e5),
            # pyre-fixme[6]: Expected `Union[int, typing.Tuple[int, ...]]` for 3rd
            #  param but got `Tuple[typing.Union[float, int]]`.
            size=(lengths.sum().item(),),
        ).type(index_dtype)
        permute_list = list(range(T))
        random.shuffle(permute_list)
        permute = torch.IntTensor(permute_list)

        (
            permuted_lengths_cpu,
            permuted_indices_cpu,
            permuted_weights_cpu,
        ) = torch.ops.fbgemm.permute_sparse_data(permute, lengths, indices, weights)
        (
            permuted_lengths_ref,
            permuted_indices_ref,
            permuted_weights_ref,
        ) = self.permute_indices_ref_(lengths, indices, weights, permute.long())
        torch.testing.assert_allclose(permuted_indices_cpu, permuted_indices_ref)
        torch.testing.assert_allclose(permuted_lengths_cpu, permuted_lengths_ref)
        if has_weight:
            torch.testing.assert_allclose(permuted_weights_cpu, permuted_weights_ref)
        else:
            assert permuted_weights_cpu is None and permuted_weights_ref is None

        if gpu_available:
            (
                permuted_lengths_gpu,
                permuted_indices_gpu,
                permuted_weights_gpu,
            ) = torch.ops.fbgemm.permute_sparse_data(
                permute.cuda(),
                lengths.cuda(),
                indices.cuda(),
                weights.cuda() if has_weight else None,
            )
            torch.testing.assert_allclose(
                permuted_indices_gpu.cpu(), permuted_indices_cpu
            )
            torch.testing.assert_allclose(
                permuted_lengths_gpu.cpu(), permuted_lengths_cpu
            )
            if has_weight:
                torch.testing.assert_allclose(
                    permuted_weights_gpu.cpu(), permuted_weights_cpu
                )
            else:
                assert permuted_weights_gpu is None

    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
    @given(
        B=st.integers(min_value=1, max_value=20),
        T=st.integers(min_value=1, max_value=20),
        L=st.integers(min_value=2, max_value=20),
        long_index=st.booleans(),
        has_weight=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)
    def test_permute_indices_with_repeats(
        self, B: int, T: int, L: int, long_index: bool, has_weight: bool
    ) -> None:
        index_dtype = torch.int64 if long_index else torch.int32
        lengths = torch.randint(low=1, high=L, size=(T, B)).type(index_dtype)
        weights = torch.rand(lengths.sum().item()).float() if has_weight else None
        indices = torch.randint(
            low=1,
            high=int(1e5),
            # pyre-fixme[6]: Expected `Union[int, typing.Tuple[int, ...]]` for 3rd
            #  param but got `Tuple[typing.Union[float, int]]`.
            size=(lengths.sum().item(),),
        ).type(index_dtype)
        permute_list = list(range(T))

        num_repeats = random.randint(0, T)
        for _ in range(num_repeats):
            permute_list.append(random.randint(0, T - 1))

        random.shuffle(permute_list)
        permute = torch.IntTensor(permute_list)

        (
            permuted_lengths_cpu,
            permuted_indices_cpu,
            permuted_weights_cpu,
        ) = torch.ops.fbgemm.permute_sparse_data(permute, lengths, indices, weights)
        (
            permuted_lengths_ref,
            permuted_indices_ref,
            permuted_weights_ref,
        ) = self.permute_indices_ref_(lengths, indices, weights, permute.long())
        torch.testing.assert_allclose(permuted_indices_cpu, permuted_indices_ref)
        torch.testing.assert_allclose(permuted_lengths_cpu, permuted_lengths_ref)
        if has_weight:
            torch.testing.assert_allclose(permuted_weights_cpu, permuted_weights_ref)
        else:
            assert permuted_weights_cpu is None and permuted_weights_ref is None

        if gpu_available:
            (
                permuted_lengths_gpu,
                permuted_indices_gpu,
                permuted_weights_gpu,
            ) = torch.ops.fbgemm.permute_sparse_data(
                permute.cuda(),
                lengths.cuda(),
                indices.cuda(),
                weights.cuda() if has_weight else None,
            )
            torch.testing.assert_allclose(
                permuted_indices_gpu.cpu(), permuted_indices_cpu
            )
            torch.testing.assert_allclose(
                permuted_lengths_gpu.cpu(), permuted_lengths_cpu
            )
            if has_weight:
                torch.testing.assert_allclose(
                    permuted_weights_gpu.cpu(), permuted_weights_cpu
                )
            else:
                assert permuted_weights_cpu is None

    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
    @given(
        D=st.integers(min_value=5, max_value=20),
        B=st.integers(min_value=1, max_value=20),
        T=st.integers(min_value=1, max_value=20),
        L=st.integers(min_value=2, max_value=20),
        long_index=st.booleans(),
        has_weight=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)
    def test_permute_indices_multi_dimension(
        self, D: int, B: int, T: int, L: int, long_index: bool, has_weight: bool
    ) -> None:
        index_dtype = torch.int64 if long_index else torch.int32
        lengths = torch.randint(low=1, high=L, size=(T, B, D)).type(index_dtype)
        weights = torch.rand(lengths.sum().item()).float() if has_weight else None
        indices = torch.randint(
            low=1,
            high=int(1e5),
            # pyre-fixme[6]: Expected `Union[int, typing.Tuple[int, ...]]` for 3rd
            #  param but got `Tuple[typing.Union[float, int]]`.
            size=(lengths.sum().item(),),
        ).type(index_dtype)
        permute_list = list(range(T))
        random.shuffle(permute_list)
        permute = torch.IntTensor(permute_list)

        (
            permuted_lengths_cpu,
            permuted_indices_cpu,
            permuted_weights_cpu,
        ) = torch.ops.fbgemm.permute_sparse_data(permute, lengths, indices, weights)
        (
            permuted_lengths_ref,
            permuted_indices_ref,
            permuted_weights_ref,
        ) = self.permute_indices_ref_(lengths, indices, weights, permute.long())
        torch.testing.assert_allclose(permuted_indices_cpu, permuted_indices_ref)
        torch.testing.assert_allclose(permuted_lengths_cpu, permuted_lengths_ref)
        if has_weight:
            torch.testing.assert_allclose(permuted_weights_cpu, permuted_weights_ref)
        else:
            assert permuted_weights_cpu is None and permuted_weights_ref is None

        if gpu_available:
            (
                permuted_lengths_gpu,
                permuted_indices_gpu,
                permuted_weights_gpu,
            ) = torch.ops.fbgemm.permute_sparse_data(
                permute.cuda(),
                lengths.cuda(),
                indices.cuda(),
                weights.cuda() if has_weight else None,
            )
            torch.testing.assert_allclose(
                permuted_indices_gpu.cpu(), permuted_indices_cpu
            )
            torch.testing.assert_allclose(
                permuted_lengths_gpu.cpu(), permuted_lengths_cpu
            )
            if has_weight:
                torch.testing.assert_allclose(
                    permuted_weights_gpu.cpu(), permuted_weights_cpu
                )
            else:
                assert permuted_weights_cpu is None

    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
    @given(
        B=st.integers(min_value=1, max_value=20),
        T=st.integers(min_value=1, max_value=20),
        L=st.integers(min_value=2, max_value=20),
        long_index=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=10, deadline=None)
    def test_permute_embeddings(self, B: int, T: int, L: int, long_index: bool) -> None:
        index_dtype = torch.int32 if long_index else torch.int32
        lengths = torch.randint(low=1, high=L, size=(T, B)).type(index_dtype)
        embeddings = torch.rand(lengths.sum().item()).float()
        permute_list = list(range(T))
        random.shuffle(permute_list)
        permute = torch.IntTensor(permute_list)

        (
            permuted_lengths_cpu,
            permuted_embeddings_cpu,
            _,
        ) = torch.ops.fbgemm.permute_sparse_data(permute, lengths, embeddings, None)
        (
            permuted_lengths_ref,
            permuted_embeddings_ref,
            _,
        ) = self.permute_indices_ref_(lengths, embeddings, None, permute.long())
        torch.testing.assert_allclose(permuted_embeddings_cpu, permuted_embeddings_ref)
        torch.testing.assert_allclose(permuted_lengths_cpu, permuted_lengths_ref)

        if gpu_available:
            (
                permuted_lengths_gpu,
                permuted_embeddings_gpu,
                _,
            ) = torch.ops.fbgemm.permute_sparse_data(
                permute.cuda(),
                lengths.cuda(),
                embeddings.cuda(),
                None,
            )
            torch.testing.assert_allclose(
                permuted_embeddings_gpu.cpu(), permuted_embeddings_cpu
            )
            torch.testing.assert_allclose(
                permuted_lengths_gpu.cpu(), permuted_lengths_cpu
            )

    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
    @given(
        long_indices=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=16, deadline=None)
    def test_block_bucketize_sparse_features_long_indices(
        self,
        long_indices: bool,
    ) -> None:
        bucketize_pos = False
        sequence = False
        index_type = torch.long

        # 3 GPUs
        my_size = 3
        block_sizes = torch.tensor([3, 4, 5], dtype=index_type)

        if not long_indices:
            lengths = torch.tensor([0, 3, 2, 0, 1, 4], dtype=index_type)
            indices = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=index_type)
            new_lengths_ref = torch.tensor(
                [0, 2, 0, 0, 0, 0, 0, 1, 2, 0, 1, 3, 0, 0, 0, 0, 0, 1], dtype=index_type
            )
            new_indices_ref = torch.tensor(
                [1, 2, 0, 0, 1, 1, 2, 3, 4, 0], dtype=index_type
            )
        else:
            lengths = torch.tensor([0, 3, 2, 0, 1, 4], dtype=index_type)
            # Test long and negative indices: -8 will be casted to 18446644015555759292
            indices = torch.tensor(
                [1, 2, 3, 100061827127359, 5, 6, 7, -8, 100058153792324, 10],
                dtype=index_type,
            )
            new_lengths_ref = torch.tensor(
                [0, 2, 0, 0, 0, 0, 0, 1, 2, 0, 1, 1, 0, 0, 0, 0, 0, 3], dtype=index_type
            )
            new_indices_ref = torch.tensor(
                [
                    1,
                    2,
                    0,
                    33353942375786,  # 100061827127359/3 = 33353942375786
                    1,
                    1,
                    2,
                    6148914691236517202,  # -8 cast to 18446644015555759292, 18446644015555759292 /3 = 6148914691236517202
                    33352717930774,  # 100058153792324/3 = 33352717930774
                    0,
                ],
                dtype=index_type,
            )

        (
            new_lengths_gpu,
            new_indices_gpu,
            new_weights_gpu,
            new_pos_gpu,
            unbucketize_permute_gpu,
        ) = torch.ops.fbgemm.block_bucketize_sparse_features(
            lengths.cuda(),
            indices.cuda(),
            bucketize_pos,
            sequence,
            block_sizes.cuda(),
            my_size,
            None,
        )

        print(f"new_lengths_gpu={new_lengths_gpu}")
        print(f"new_indices_gpu={new_indices_gpu}")
        torch.testing.assert_allclose(new_lengths_gpu.cpu(), new_lengths_ref)
        torch.testing.assert_allclose(new_indices_gpu.cpu(), new_indices_ref)

    # pyre-ignore [56]
    @given(
        n=st.integers(min_value=1, max_value=100),
        long_index=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_cumsum(self, n: int, long_index: bool) -> None:
        index_dtype = torch.int64 if long_index else torch.int32
        np_index_dtype = np.int64 if long_index else np.int32

        # cpu tests
        x = torch.randint(low=0, high=100, size=(n,)).type(index_dtype)
        ze = torch.ops.fbgemm.asynchronous_exclusive_cumsum(x)
        zi = torch.ops.fbgemm.asynchronous_inclusive_cumsum(x)
        zc = torch.ops.fbgemm.asynchronous_complete_cumsum(x)
        torch.testing.assert_allclose(
            np.cumsum(x.cpu().numpy()).astype(np_index_dtype), zi.cpu()
        )
        torch.testing.assert_allclose(
            (np.cumsum([0] + x.cpu().numpy().tolist())[:-1]).astype(np_index_dtype),
            ze.cpu(),
        )
        torch.testing.assert_allclose(
            (np.cumsum([0] + x.cpu().numpy().tolist())).astype(np_index_dtype), zc.cpu()
        )

        if gpu_available:
            x = x.cuda()
            ze = torch.ops.fbgemm.asynchronous_exclusive_cumsum(x)
            zi = torch.ops.fbgemm.asynchronous_inclusive_cumsum(x)
            zc = torch.ops.fbgemm.asynchronous_complete_cumsum(x)
            torch.testing.assert_allclose(
                np.cumsum(x.cpu().numpy()).astype(np_index_dtype), zi.cpu()
            )
            torch.testing.assert_allclose(
                (np.cumsum([0] + x.cpu().numpy().tolist())[:-1]).astype(np_index_dtype),
                ze.cpu(),
            )
            torch.testing.assert_allclose(
                (np.cumsum([0] + x.cpu().numpy().tolist())).astype(np_index_dtype),
                zc.cpu(),
            )

    # pyre-ignore [56]
    @given(
        N=st.integers(min_value=1, max_value=20),
        offsets_type=st.sampled_from([torch.int32, torch.int64]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_offsets_range(
        self, N: int, offsets_type: "Union[Type[torch.int32], Type[torch.int64]]"
    ) -> None:
        lengths = np.array([np.random.randint(low=0, high=20) for _ in range(N)])
        offsets = np.cumsum(np.concatenate(([0], lengths)))[:-1]
        range_ref = np.concatenate([np.arange(size) for size in lengths])
        output_size = np.sum(lengths)

        offsets_cpu = torch.tensor(offsets, dtype=offsets_type)
        range_cpu = torch.ops.fbgemm.offsets_range(offsets_cpu, output_size)
        torch.testing.assert_allclose(range_cpu, range_ref, 0, 0)

        if gpu_available:
            range_gpu = torch.ops.fbgemm.offsets_range(offsets_cpu.cuda(), output_size)
            torch.testing.assert_allclose(range_gpu.cpu(), range_ref, 0, 0)

    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
    @given(
        index_type=st.sampled_from([torch.int, torch.long]),
        has_weight=st.booleans(),
        bucketize_pos=st.booleans(),
        sequence=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=16, deadline=None)
    def test_block_bucketize_sparse_features(
        self,
        index_type: Type[torch.dtype],
        has_weight: bool,
        bucketize_pos: bool,
        sequence: bool,
    ) -> None:
        B = 2
        # pyre-ignore [6]
        lengths = torch.tensor([0, 2, 1, 3, 2, 3, 3, 1], dtype=index_type)
        indices = torch.tensor(
            [3, 4, 15, 11, 28, 29, 1, 10, 11, 12, 13, 11, 22, 20, 20],
            # pyre-ignore [6]
            dtype=index_type,
        )
        weights = (
            torch.tensor(
                [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    7.0,
                    8.0,
                    9.0,
                    10.0,
                    11.0,
                    12.0,
                    13.0,
                    14.0,
                    15.0,
                ],
                dtype=torch.float,
            )
            if has_weight
            else None
        )
        # pyre-ignore [6]
        block_sizes = torch.tensor([5, 15, 10, 20], dtype=index_type)
        my_size = 2

        new_lengths_ref = torch.tensor(
            [0, 2, 0, 1, 1, 0, 1, 0, 0, 0, 1, 2, 1, 3, 2, 1],
            # pyre-ignore [6]
            dtype=index_type,
        )
        new_indices_ref = torch.tensor(
            [3, 4, 11, 1, 11, 0, 13, 14, 0, 1, 2, 3, 2, 0, 0],
            # pyre-ignore [6]
            dtype=index_type,
        )
        new_weights_ref = torch.tensor(
            [
                1.0,
                2.0,
                4.0,
                7.0,
                12.0,
                3.0,
                5.0,
                6.0,
                8.0,
                9.0,
                10.0,
                11.0,
                13.0,
                14.0,
                15.0,
            ],
            dtype=torch.float,
        )
        new_pos_ref = torch.tensor(
            [0, 1, 0, 0, 0, 0, 1, 2, 1, 0, 1, 2, 1, 2, 0],
            # pyre-ignore [6]
            dtype=index_type,
        )
        (
            new_lengths_cpu,
            new_indices_cpu,
            new_weights_cpu,
            new_pos_cpu,
            unbucketize_permute,
        ) = torch.ops.fbgemm.block_bucketize_sparse_features(
            lengths, indices, bucketize_pos, sequence, block_sizes, my_size, weights
        )
        torch.testing.assert_allclose(new_lengths_cpu, new_lengths_ref, 0, 0)
        torch.testing.assert_allclose(new_indices_cpu, new_indices_ref, 0, 0)
        if has_weight:
            torch.testing.assert_allclose(new_weights_cpu, new_weights_ref)
        if bucketize_pos:
            torch.testing.assert_allclose(new_pos_cpu, new_pos_ref)
        if sequence:
            value_unbucketized_indices = unbucketize_indices_value(
                new_indices_cpu, new_lengths_cpu, block_sizes, my_size, B
            )
            unbucketized_indices = torch.index_select(
                value_unbucketized_indices, 0, unbucketize_permute
            )
            torch.testing.assert_allclose(unbucketized_indices, indices, 0, 0)

        if gpu_available:
            (
                new_lengths_gpu,
                new_indices_gpu,
                new_weights_gpu,
                new_pos_gpu,
                unbucketize_permute_gpu,
            ) = torch.ops.fbgemm.block_bucketize_sparse_features(
                lengths.cuda(),
                indices.cuda(),
                bucketize_pos,
                sequence,
                block_sizes.cuda(),
                my_size,
                weights.cuda() if has_weight else None,
            )
            torch.testing.assert_allclose(new_lengths_gpu.cpu(), new_lengths_ref, 0, 0)
            torch.testing.assert_allclose(new_indices_gpu.cpu(), new_indices_ref, 0, 0)
            if has_weight:
                torch.testing.assert_allclose(new_weights_gpu.cpu(), new_weights_cpu)
            if bucketize_pos:
                torch.testing.assert_allclose(new_pos_gpu.cpu(), new_pos_cpu)
            if sequence:
                value_unbucketized_indices = unbucketize_indices_value(
                    new_indices_gpu.cpu(),
                    new_lengths_gpu.cpu(),
                    block_sizes,
                    my_size,
                    B,
                )
                unbucketized_indices = torch.index_select(
                    value_unbucketized_indices, 0, unbucketize_permute_gpu.cpu()
                )
                torch.testing.assert_allclose(unbucketized_indices, indices, 0, 0)

    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
    @given(
        B=st.integers(min_value=1, max_value=20),
        T=st.integers(min_value=1, max_value=20),
        L=st.integers(min_value=2, max_value=20),
        A=st.integers(min_value=1, max_value=20),
        Dtype=st.sampled_from([torch.int32, torch.float, torch.int64]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_reorder_batched_ad_lengths(self, B: int, T: int, L: int, A: int, Dtype: torch.dtype) -> None:
        cat_ad_lengths = (
            torch.cat([torch.tensor([L for _ in range(T * A)]) for _ in range(B)], 0)
            .cuda()
            .to(Dtype)
        )
        batch_offsets = torch.tensor([A * b for b in range(B + 1)]).int().cuda()
        num_ads_in_batch = B * A
        reordered_batched_ad_lengths = torch.ops.fbgemm.reorder_batched_ad_lengths(
            cat_ad_lengths, batch_offsets, num_ads_in_batch
        )
        torch.testing.assert_allclose(cat_ad_lengths, reordered_batched_ad_lengths)

        cat_ad_lengths_cpu = cat_ad_lengths.cpu()
        batch_offsets_cpu = batch_offsets.cpu()
        reordered_batched_ad_lengths_cpu = torch.ops.fbgemm.reorder_batched_ad_lengths(
            cat_ad_lengths_cpu, batch_offsets_cpu, num_ads_in_batch
        )
        torch.testing.assert_allclose(
            reordered_batched_ad_lengths_cpu, reordered_batched_ad_lengths.cpu()
        )

    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
    @given(
        B=st.integers(min_value=1, max_value=20),
        T=st.integers(min_value=1, max_value=20),
        L=st.integers(min_value=2, max_value=20),
        A=st.integers(min_value=1, max_value=20),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=40, deadline=None)
    def test_reorder_batched_ad_lengths_cpu(
        self, B: int, T: int, L: int, A: int
    ) -> None:
        cat_ad_lengths = torch.cat(
            [torch.tensor([L for _ in range(T * A)]) for _ in range(B)], 0
        ).int()
        batch_offsets = torch.tensor([A * b for b in range(B + 1)]).int()
        num_ads_in_batch = B * A
        reordered_batched_ad_lengths = torch.ops.fbgemm.reorder_batched_ad_lengths(
            cat_ad_lengths, batch_offsets, num_ads_in_batch
        )
        torch.testing.assert_allclose(cat_ad_lengths, reordered_batched_ad_lengths)

    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
    @given(
        B=st.integers(min_value=1, max_value=20),
        T=st.integers(min_value=1, max_value=20),
        L=st.integers(min_value=2, max_value=20),
        A=st.integers(min_value=1, max_value=20),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_reorder_batched_ad_indices(self, B: int, T: int, L: int, A: int) -> None:
        cat_ad_indices = (
            torch.randint(low=0, high=100, size=(B * T * A * L,)).int().cuda()
        )
        cat_ad_lengths = (
            torch.cat([torch.tensor([L for _ in range(T * A)]) for _ in range(B)], 0)
            .int()
            .cuda()
        )
        batch_offsets = torch.tensor([A * b for b in range(B + 1)]).int().cuda()
        num_ads_in_batch = B * A
        reordered_cat_ad_lengths = torch.ops.fbgemm.reorder_batched_ad_lengths(
            cat_ad_lengths, batch_offsets, num_ads_in_batch
        )
        torch.testing.assert_allclose(cat_ad_lengths, reordered_cat_ad_lengths)

        cat_ad_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(cat_ad_lengths)
        reordered_cat_ad_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
            reordered_cat_ad_lengths
        )
        reordered_cat_ad_indices = torch.ops.fbgemm.reorder_batched_ad_indices(
            cat_ad_offsets,
            cat_ad_indices,
            reordered_cat_ad_offsets,
            batch_offsets,
            num_ads_in_batch,
        )
        torch.testing.assert_allclose(
            reordered_cat_ad_indices.view(T, B, A, L).permute(1, 0, 2, 3),
            cat_ad_indices.view(B, T, A, L),
        )

        reordered_cat_ad_indices_cpu = torch.ops.fbgemm.reorder_batched_ad_indices(
            cat_ad_offsets.cpu(),
            cat_ad_indices.cpu(),
            reordered_cat_ad_offsets.cpu(),
            batch_offsets.cpu(),
            num_ads_in_batch,
        )
        torch.testing.assert_allclose(
            reordered_cat_ad_indices_cpu.view(T, B, A, L).permute(1, 0, 2, 3),
            cat_ad_indices.view(B, T, A, L).cpu(),
        )

    # pyre-ignore [56]: Invalid decoration, was not able to infer the type of argument
    @given(
        B=st.integers(min_value=1, max_value=20),
        T=st.integers(min_value=1, max_value=20),
        L=st.integers(min_value=2, max_value=20),
        A=st.integers(min_value=1, max_value=20),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=40, deadline=None)
    def test_reorder_batched_ad_indices_cpu(
        self, B: int, T: int, L: int, A: int
    ) -> None:
        cat_ad_indices = torch.randint(low=0, high=100, size=(B * T * A * L,)).int()
        cat_ad_lengths = torch.cat(
            [torch.tensor([L for _ in range(T * A)]) for _ in range(B)], 0
        ).int()
        batch_offsets = torch.tensor([A * b for b in range(B + 1)]).int()
        num_ads_in_batch = B * A
        reordered_cat_ad_lengths = torch.ops.fbgemm.reorder_batched_ad_lengths(
            cat_ad_lengths, batch_offsets, num_ads_in_batch
        )
        torch.testing.assert_allclose(cat_ad_lengths, reordered_cat_ad_lengths)
        cat_ad_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(cat_ad_lengths)
        reordered_cat_ad_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
            reordered_cat_ad_lengths
        )
        reordered_cat_ad_indices = torch.ops.fbgemm.reorder_batched_ad_indices(
            cat_ad_offsets,
            cat_ad_indices,
            reordered_cat_ad_offsets,
            batch_offsets,
            num_ads_in_batch,
        )
        torch.testing.assert_allclose(
            reordered_cat_ad_indices.view(T, B, A, L).permute(1, 0, 2, 3),
            cat_ad_indices.view(B, T, A, L),
        )

    @settings(
        verbosity=Verbosity.verbose,
        max_examples=20,
        deadline=None,
    )
    # pyre-ignore [56]
    @given(
        B=st.integers(min_value=1, max_value=128),
        D=st.integers(min_value=1, max_value=128),
        max_sequence_length=st.integers(min_value=1, max_value=200),
        is_half=st.booleans(),
    )
    def test_jagged_2d_to_dense(
        self,
        B: int,
        D: int,
        max_sequence_length: int,
        is_half: bool,
    ) -> None:
        D = D * 4
        lengths_ = np.random.randint(low=0, high=max_sequence_length, size=B)
        total_lengths = lengths_.sum()
        lengths = torch.from_numpy(lengths_)
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)

        ref_embeddings = torch.rand(total_lengths, D)
        ref_output_embeddings = var_list_to_coo(
            lengths,
            ref_embeddings,
            max_sequence_length,
            D,
        ).to_dense()

        # test cpu forward
        if is_half:
            embeddings = ref_embeddings.clone().half().detach().requires_grad_(True)
        else:
            embeddings = ref_embeddings.clone().detach().requires_grad_(True)
        output_embeddings = torch.ops.fbgemm.jagged_2d_to_dense(
            embeddings=embeddings,
            offsets=offsets,
            max_sequence_length=max_sequence_length,
        )
        torch.testing.assert_allclose(ref_output_embeddings, output_embeddings)

        if torch.cuda.is_available():
            # test gpu forward
            ref_embeddings = ref_embeddings.cuda()
            if is_half:
                embeddings = ref_embeddings.clone().half().detach().requires_grad_(True)
            else:
                embeddings = ref_embeddings.clone().detach().requires_grad_(True)
            offsets = offsets.cuda()
            ref_output_embeddings = ref_output_embeddings.cuda()
            output_embeddings = torch.ops.fbgemm.jagged_2d_to_dense(
                embeddings=embeddings,
                offsets=offsets,
                max_sequence_length=max_sequence_length,
            )
            torch.testing.assert_allclose(ref_output_embeddings, output_embeddings)

            # test gpu backward
            output_embeddings.backward(ref_output_embeddings)
            torch.testing.assert_allclose(ref_embeddings, embeddings.grad)

    def test_jagged_2d_to_dense_truncation(self) -> None:
        # Test the case where max_sequence_length < max(lengths[i])
        lengths_ = np.array([2, 3, 0, 1])
        lengths = torch.from_numpy(lengths_)
        total_lengths = lengths_.sum()
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)

        embedding_dim = 16
        max_sequence_length = 2
        ref_embeddings = torch.rand(total_lengths, embedding_dim)
        ref_output_embeddings = var_list_to_coo(
            lengths,
            ref_embeddings,
            3,
            embedding_dim,
        ).to_dense()[:, :max_sequence_length, :]

        # test cpu forward
        embeddings = ref_embeddings.clone().detach().requires_grad_(True)
        output_embeddings = torch.ops.fbgemm.jagged_2d_to_dense(
            embeddings=embeddings,
            offsets=offsets,
            max_sequence_length=max_sequence_length,
        )
        torch.testing.assert_allclose(ref_output_embeddings, output_embeddings)

        if torch.cuda.is_available():
            # test gpu forward
            ref_embeddings = ref_embeddings.cuda()
            embeddings = ref_embeddings.clone().detach().requires_grad_(True)
            offsets = offsets.cuda()
            ref_output_embeddings = ref_output_embeddings.cuda()
            output_embeddings = torch.ops.fbgemm.jagged_2d_to_dense(
                embeddings=embeddings,
                offsets=offsets,
                max_sequence_length=max_sequence_length,
            )
            torch.testing.assert_allclose(ref_output_embeddings, output_embeddings)

            # test gpu backward
            expected_grad = ref_embeddings
            expected_grad[4, :] = 0  # due to truncation
            expected_grad = expected_grad.cuda()
            output_embeddings.backward(ref_output_embeddings)
            torch.testing.assert_allclose(expected_grad, embeddings.grad)

    @settings(
        verbosity=Verbosity.verbose,
        max_examples=20,
        deadline=None,
    )
    # pyre-ignore [56]
    @given(
        B=st.integers(min_value=1, max_value=128),
        max_sequence_length=st.integers(min_value=1, max_value=500),
        padding_value=st.integers(min_value=-100000, max_value=100000),
    )
    def test_jagged_1d_to_dense(
        self,
        B: int,
        max_sequence_length: int,
        padding_value: int,
    ) -> None:
        def lengths_to_segment_ids(lengths: torch.Tensor) -> torch.Tensor:
            return torch.repeat_interleave(
                torch._dim_arange(lengths, 0).long(),
                lengths.long(),
            )

        # Converts lengths + values format to COO format
        # [B], [N] -> [B, N'].
        # pyre-ignore Missing return annotation [3]
        def var_list_to_coo(
            lengths: torch.Tensor,
            values: torch.Tensor,
            N: int,
        ):
            rows = lengths_to_segment_ids(lengths)
            num_rows = lengths.size()[0]
            # This does D&H sync
            offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
            output_size = lengths.sum()
            # This does D&H sync
            cols = torch.ops.fbgemm.offsets_range(offsets, output_size)
            indices = torch.stack([rows, cols])
            dims = [num_rows, N]
            # torch.sparse_coo_tensor is not supported by torch.fx, wrap it.
            return torch.sparse_coo_tensor(
                indices=indices,
                values=values,
                size=dims,
            )

        lengths_ = np.random.randint(low=0, high=max_sequence_length, size=B)
        total_lengths = lengths_.sum()
        lengths = torch.from_numpy(lengths_)
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)

        ref_values = torch.randint(low=0, high=1000000000, size=(total_lengths,))
        ref_values_mask = var_list_to_coo(
            lengths, torch.ones_like(ref_values), max_sequence_length
        ).to_dense()
        ref_output_values = (
            var_list_to_coo(
                lengths,
                ref_values,
                max_sequence_length,
            ).to_dense()
            + (1 - ref_values_mask) * torch.ones_like(ref_values_mask) * padding_value
        )

        # test cpu forward
        values = ref_values.clone().detach().requires_grad_(False)
        output_values = torch.ops.fbgemm.jagged_1d_to_dense(
            values=values,
            offsets=offsets,
            max_sequence_length=max_sequence_length,
            padding_value=padding_value,
        )
        torch.testing.assert_allclose(ref_output_values, output_values)

        if torch.cuda.is_available():
            # test gpu forward
            ref_values = ref_values.cuda()
            values = ref_values.clone().detach().requires_grad_(False)
            offsets = offsets.cuda()
            ref_output_values = ref_output_values.cuda()
            output_values = torch.ops.fbgemm.jagged_1d_to_dense(
                values=values,
                offsets=offsets,
                max_sequence_length=max_sequence_length,
                padding_value=padding_value,
            )
            torch.testing.assert_allclose(ref_output_values, output_values)

    def test_jagged_1d_to_dense_truncation(self) -> None:
        lengths_ = np.array([1, 3, 0, 1])
        lengths = torch.from_numpy(lengths_)
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)

        ref_values = torch.from_numpy(np.array([100, 3, 4, 5, 6]))
        ref_output = torch.from_numpy(np.array([100, 3, -1, 6])).reshape(-1, 1)

        # test cpu forward
        values = ref_values.clone().detach().requires_grad_(False)
        output = torch.ops.fbgemm.jagged_1d_to_dense(
            values=values,
            offsets=offsets,
            max_sequence_length=1,
            padding_value=-1,
        )
        torch.testing.assert_allclose(ref_output, output)

        if torch.cuda.is_available():
            # test gpu forward
            ref_values = ref_values.cuda()
            values = ref_values.clone().detach().requires_grad_(False)
            offsets = offsets.cuda()
            ref_output = ref_output.cuda()
            output = torch.ops.fbgemm.jagged_1d_to_dense(
                values=values,
                offsets=offsets,
                max_sequence_length=1,
                padding_value=-1,
            )
            torch.testing.assert_allclose(ref_output, output)


if __name__ == "__main__":
    unittest.main()
