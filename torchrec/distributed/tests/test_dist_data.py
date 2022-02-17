#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
import itertools
import multiprocessing
import os
import random
import unittest
from typing import List, Tuple, TypeVar, Any, Generator, Union

import hypothesis.strategies as st
import torch
import torch.distributed as dist
from hypothesis import given, settings

# @manual=//python/wheel/numpy:numpy
from numpy.testing import assert_array_equal
from torchrec.distributed.dist_data import (
    KJTAllToAll,
    PooledEmbeddingsAllToAll,
    PooledEmbeddingsReduceScatter,
    KJTAllToAllLengthsAwaitable,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.test_utils import get_free_port, seed_and_log


T = TypeVar("T", int, float)

# Lightly adapted from Stack Overflow #10823877
def _flatten(iterable: List[T]) -> Generator[T, None, None]:
    iterator, sentinel, stack = iter(iterable), object(), []
    while True:
        value = next(iterator, sentinel)
        if value is sentinel:
            if not stack:
                break
            iterator = stack.pop()
        else:
            try:
                new_iterator = iter(value)
            except TypeError:
                yield value
            else:
                stack.append(iterator)
                iterator = new_iterator


def _to_tensor(iterator: List[T], device_id: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(list(_flatten(iterator)), dtype=dtype).cuda(device_id)


def _generate_sparse_features_batch(
    keys: List[str], splits: List[int], B: int, is_weighted: bool = False
) -> Tuple[List[KeyedJaggedTensor], List[KeyedJaggedTensor]]:
    world_size = len(splits)
    offsets = [0] + list(itertools.accumulate(splits))
    values = {}
    lengths = {}
    weights = {} if is_weighted else None

    for key in keys:
        lengths[key] = [
            [random.randint(0, 10) for _ in range(B)] for i in range(world_size)
        ]
        values[key] = [
            [random.randint(0, 1000) for _ in range(sum(lengths[key][i]))]
            for i in range(world_size)
        ]

        if weights:
            weights[key] = [
                [random.random() for _ in range(sum(lengths[key][i]))]
                for i in range(world_size)
            ]

    in_jagged: List[KeyedJaggedTensor] = []
    out_jagged: List[KeyedJaggedTensor] = []
    for i in range(world_size):
        in_jagged.append(
            KeyedJaggedTensor.from_lengths_sync(
                keys=keys,
                lengths=_to_tensor([lengths[key][i] for key in keys], i, torch.int),
                values=_to_tensor([values[key][i] for key in keys], i, torch.int),
                weights=_to_tensor([weights[key][i] for key in keys], i, torch.float)
                if weights
                else None,
            )
        )
        key_index = []
        out_keys = keys[offsets[i] : offsets[i + 1]]
        for key in out_keys:
            for j in range(world_size):
                key_index.append((key, j))
        out_jagged.append(
            KeyedJaggedTensor.from_lengths_sync(
                keys=out_keys,
                lengths=_to_tensor(
                    [lengths[key][j] for key, j in key_index],
                    i,
                    torch.int,
                ),
                values=_to_tensor(
                    [values[key][j] for key, j in key_index],
                    i,
                    torch.int,
                ),
                weights=_to_tensor(
                    [weights[key][j] for key, j in key_index],
                    i,
                    torch.float,
                )
                if weights
                else None,
            )
        )
    return in_jagged, out_jagged


def _generate_pooled_embedding_batch(
    keys: List[str], dims: List[int], splits: List[int], batch_size_per_rank: List[int]
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    world_size = len(splits)
    offsets = [0] + list(itertools.accumulate(splits))
    local_emb = {}
    B_global = sum(batch_size_per_rank)
    B_offsets = [0] + list(itertools.accumulate(batch_size_per_rank))

    for key, dim in zip(keys, dims):
        local_emb[key] = [
            [random.random() for _ in range(dim)] for _ in range(B_global)
        ]

    in_tensor: List[torch.Tensor] = []
    out_tensor: List[torch.Tensor] = []
    for i in range(world_size):
        in_keys = keys[offsets[i] : offsets[i + 1]]
        in_tensor.append(
            _to_tensor(
                [local_emb[key][b] for b in range(B_global) for key in in_keys],
                i,
                torch.float,
            ).view(B_global, -1)
            if in_keys
            else torch.empty(B_global, 0, dtype=torch.float).cuda(i)
        )
        out_tensor.append(
            _to_tensor(
                [
                    local_emb[key][b]
                    for b in range(B_offsets[i], B_offsets[i + 1])
                    for key in keys
                ],
                i,
                torch.float,
            ).view(batch_size_per_rank[i], -1)
        )

    return in_tensor, out_tensor


class DistDataTestCase(abc.ABC, unittest.TestCase):
    @seed_and_log
    def setUp(self) -> None:
        torch.use_deterministic_algorithms(True)
        os.environ["MASTER_ADDR"] = str("localhost")
        os.environ["MASTER_PORT"] = str(get_free_port())
        os.environ["GLOO_DEVICE_TRANSPORT"] = "TCP"
        os.environ["NCCL_SOCKET_IFNAME"] = "lo"
        self.WORLD_SIZE = 2

    def tearDown(self) -> None:
        del os.environ["GLOO_DEVICE_TRANSPORT"]
        del os.environ["NCCL_SOCKET_IFNAME"]
        # pyre-fixme[16]
        super().tearDown()

    def _run_multi_process_test(
        self,
        _input: Union[List[KeyedJaggedTensor], List[torch.Tensor]],
        output: Union[List[KeyedJaggedTensor], List[torch.Tensor]],
        **kwargs: Any,
    ) -> None:
        processes = []
        ctx = multiprocessing.get_context("spawn")
        for rank in range(self.WORLD_SIZE):
            p = ctx.Process(
                target=self._run_test_dist,
                args=(
                    rank,
                    self.WORLD_SIZE,
                    _input[rank],
                    output[rank],
                ),
                kwargs=kwargs,
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            self.assertEqual(0, p.exitcode)

    @classmethod
    @abc.abstractmethod
    def _run_test_dist(cls) -> None:
        pass


class KJTAllToAllTest(DistDataTestCase):
    @classmethod
    def _validate(
        cls,
        actual_output_awaitable: Union[KJTAllToAllLengthsAwaitable, KeyedJaggedTensor],
        expected_output_awaitable: Union[
            KJTAllToAllLengthsAwaitable, KeyedJaggedTensor
        ],
    ) -> None:
        actual_output = (
            actual_output_awaitable
            if isinstance(actual_output_awaitable, KeyedJaggedTensor)
            else actual_output_awaitable.wait().wait()
        )
        expected_output = (
            expected_output_awaitable
            if isinstance(expected_output_awaitable, KeyedJaggedTensor)
            else expected_output_awaitable.wait().wait()
        )
        assert_array_equal(
            actual_output.values().cpu(),
            expected_output.values().cpu(),
        )
        assert_array_equal(
            actual_output.weights().cpu()
            if actual_output.weights_or_none() is not None
            else [],
            expected_output.weights().cpu()
            if expected_output.weights_or_none() is not None
            else [],
        )
        assert_array_equal(
            actual_output.lengths().cpu(),
            expected_output.lengths().cpu(),
        )
        assert_array_equal(
            actual_output.keys(),
            expected_output.keys(),
        )

    @classmethod
    def _run_test_dist(
        cls,
        rank: int,
        world_size: int,
        _input: KeyedJaggedTensor,
        output: KeyedJaggedTensor,
        backend: str,
        splits: List[int],
    ) -> None:
        dist.init_process_group(rank=rank, world_size=world_size, backend=backend)
        device = torch.device(f"cuda:{rank}")
        if backend == "gloo":
            device = torch.device("cpu")
        _input = _input.to(device=device)
        output = output.to(device=device)
        pg = dist.group.WORLD
        lengths_a2a = KJTAllToAll(pg=pg, splits=splits, device=device)
        cls._validate(lengths_a2a(_input), output)
        dist.destroy_process_group()

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    # pyre-fixme[56]
    @given(
        backend=st.sampled_from(["gloo", "nccl"]),
        B=st.integers(min_value=1, max_value=2),
        features=st.integers(min_value=3, max_value=4),
        is_weighted=st.booleans(),
    )
    @settings(max_examples=8, deadline=None)
    def test_features(
        self, backend: str, B: int, features: int, is_weighted: bool
    ) -> None:
        keys = [f"F{feature}" for feature in range(features)]
        rank0_split = random.randint(0, features)
        splits = [rank0_split, features - rank0_split]
        _input, output = _generate_sparse_features_batch(
            keys=keys, splits=splits, B=B, is_weighted=is_weighted
        )

        self._run_multi_process_test(
            _input=_input,
            output=output,
            backend=backend,
            splits=splits,
        )


class PooledEmbeddingsAllToAllTest(DistDataTestCase):
    @classmethod
    def _run_test_dist(
        cls,
        rank: int,
        world_size: int,
        _input: torch.Tensor,
        output: torch.Tensor,
        backend: str,
        dim_sum_per_rank: List[int],
        batch_size_per_rank: List[int],
    ) -> None:
        dist.init_process_group(rank=rank, world_size=world_size, backend=backend)
        pg = dist.group.WORLD
        if backend == "gloo":
            device = torch.device("cpu")
        else:
            device = torch.device(f"cuda:{rank}")
        _input = _input.to(device=device)
        output = output.to(device=device)
        a2a = PooledEmbeddingsAllToAll(
            pg=pg,
            dim_sum_per_rank=dim_sum_per_rank,
            device=device,
        )
        _input.requires_grad = True
        res = a2a(_input, batch_size_per_rank).wait()
        res.backward(res)
        assert_array_equal(
            res.cpu().detach(),
            output.cpu().detach(),
        )
        assert_array_equal(
            _input.cpu().detach().div_(world_size),
            _input.grad.cpu().detach(),
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    # pyre-fixme[56]
    @given(
        backend=st.sampled_from(["gloo", "nccl"]),
        B=st.integers(min_value=2, max_value=3),
        features=st.integers(min_value=3, max_value=4),
        is_reversed=st.booleans(),
    )
    @settings(max_examples=8, deadline=None)
    def test_pooled_embeddings(
        self, backend: str, B: int, features: int, is_reversed: bool
    ) -> None:
        keys = [f"F{feature}" for feature in range(features)]
        dims = random.sample([8, 16, 32] * features, features)
        rank0_split = random.randint(1, features - 1)
        splits = [rank0_split, features - rank0_split]
        if is_reversed:
            splits.reverse()
        dim_sum_per_rank = [sum(dims[: splits[0]]), sum(dims[splits[0] :])]
        batch_size_per_rank = [random.randint(B, B + 2), random.randint(B, B + 2)]

        _input, output = _generate_pooled_embedding_batch(
            keys=keys,
            dims=dims,
            splits=splits,
            batch_size_per_rank=batch_size_per_rank,
        )

        self._run_multi_process_test(
            _input=_input,
            output=output,
            backend=backend,
            dim_sum_per_rank=dim_sum_per_rank,
            batch_size_per_rank=batch_size_per_rank,
        )


class PooledEmbeddingsReduceScatterTest(DistDataTestCase):
    @classmethod
    def _validate(
        cls,
        actual_output: torch.Tensor,
        expected_output: torch.Tensor,
        input: torch.Tensor,
        world_size: int,
    ) -> None:
        assert_array_equal(actual_output.cpu().detach(), expected_output.cpu().detach())
        assert_array_equal(
            input.grad.cpu().detach(),
            torch.ones(input.size()).div_(world_size),
        )

    @classmethod
    def _run_test_dist(
        cls,
        rank: int,
        world_size: int,
        input: torch.Tensor,
        expected_output: torch.Tensor,
    ) -> None:
        dist.init_process_group(rank=rank, world_size=2, backend="nccl")
        pg = dist.group.WORLD
        input = input.cuda(rank)
        input.requires_grad = True
        rs = PooledEmbeddingsReduceScatter(pg).cuda(rank)
        actual_output = rs(input).wait()
        s = torch.sum(actual_output)
        s.backward()
        cls._validate(actual_output, expected_output, input, world_size)

    # pyre-fixme[56]
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    def test_pooled_embedding_reduce_scatter(self) -> None:
        embeddding_dim = 10
        batch_size = 2
        embeddings = torch.rand((batch_size * 2, embeddding_dim))
        embeddings_by_rank = list(torch.chunk(embeddings, batch_size, dim=0))
        expect_results = torch.chunk(
            torch.stack(embeddings_by_rank, dim=0).sum(dim=0),
            2,
            dim=0,
        )
        self._run_multi_process_test(
            embeddings_by_rank,
            expect_results,
        )
