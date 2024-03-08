#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import functools
import itertools
import multiprocessing
import os
import unittest
from typing import Callable, List, Optional, Tuple

import hypothesis.strategies as st

import torch
import torch.distributed as dist
import torchrec.distributed.comm_ops as comm_ops
from hypothesis import given, settings
from torch.distributed.distributed_c10d import GroupMember
from torchrec.distributed.test_utils.infer_utils import dynamo_skipfiles_allow
from torchrec.distributed.utils import none_throws
from torchrec.test_utils import get_free_port, seed_and_log


def torch_compile_args_to_fn_transform(
    torch_compile_args: Optional[Tuple[str, bool]]
    # pyre-ignore
) -> Callable:
    if torch_compile_args is None:
        return lambda x: x

    backend, fullgraph = torch_compile_args
    return functools.partial(
        torch.compile, backend=backend, fullgraph=fullgraph, dynamic=True
    )


class TestAllToAll(unittest.TestCase):
    @seed_and_log
    def setUp(self) -> None:
        os.environ["MASTER_ADDR"] = str("localhost")
        os.environ["MASTER_PORT"] = str(get_free_port())
        os.environ["GLOO_DEVICE_TRANSPORT"] = "TCP"
        os.environ["NCCL_SOCKET_IFNAME"] = "lo"
        self.WORLD_SIZE = 2

    def tearDown(self) -> None:
        del os.environ["GLOO_DEVICE_TRANSPORT"]
        del os.environ["NCCL_SOCKET_IFNAME"]
        super().tearDown()

    def _run_multi_process_test(
        self,
        world_size: int,
        backend: str,
        callable: Callable[[], None],
        # pyre-ignore
        *args,
        # pyre-ignore
        **kwargs,
    ) -> None:
        processes = []
        ctx = multiprocessing.get_context("spawn")
        for rank in range(world_size):
            p = ctx.Process(
                target=callable,
                args=(
                    rank,
                    world_size,
                    backend,
                    *args,
                ),
                kwargs=kwargs,
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            self.assertEqual(0, p.exitcode)

    @classmethod
    def _test_alltoallv(
        cls,
        rank: int,
        world_size: int,
        backend: str,
        torch_compile_args: Optional[Tuple[str, bool]],
        specify_pg: bool,
    ) -> None:
        dist.init_process_group(rank=rank, world_size=world_size, backend=backend)
        pg = GroupMember.WORLD
        assert pg is not None

        device = torch.device(f"cuda:{rank}")

        torch.cuda.set_device(device)

        B_global = 10
        D0 = 8
        D1 = 9

        input_embedding0 = torch.rand(
            (B_global, D0),
            device=device,
            requires_grad=True,
        )
        input_embedding1 = torch.rand(
            (B_global, D1),
            device=device,
            requires_grad=True,
        )

        input_embeddings = [input_embedding0, input_embedding1]
        out_split = [17, 17]

        # pyre-ignore
        def fn(*args, **kwargs) -> List[torch.Tensor]:
            return comm_ops.alltoallv(*args, **kwargs).wait()

        fn_transform = torch_compile_args_to_fn_transform(torch_compile_args)

        with dynamo_skipfiles_allow("torchrec"):
            v_embs_out = fn_transform(fn)(
                input_embeddings, out_split=out_split, group=pg if specify_pg else None
            )

        res = torch.cat(v_embs_out, dim=1).cpu()
        assert tuple(res.size()) == (5, 34)
        dist.destroy_process_group()

    @unittest.skipIf(
        torch.cuda.device_count() < 2, "Need at least two ranks to run this test"
    )
    # pyre-ignore
    @given(
        torch_compile_args=st.sampled_from([None, ("eager", True)]),
        specify_pg=st.sampled_from([True]),
    )
    @settings(deadline=None)
    def test_alltoallv(
        self,
        torch_compile_args: Optional[Tuple[str, bool]],
        specify_pg: bool,
    ) -> None:
        self._run_multi_process_test(
            world_size=self.WORLD_SIZE,
            backend="nccl",
            # pyre-ignore [6]
            callable=self._test_alltoallv,
            torch_compile_args=torch_compile_args,
            specify_pg=specify_pg,
        )

    @classmethod
    def _test_alltoall_sequence(
        cls,
        rank: int,
        world_size: int,
        backend: str,
        torch_compile_args: Optional[Tuple[str, bool]],
        specify_pg: bool,
        skip_dynamo_backwards: bool = False,
    ) -> None:
        dist.init_process_group(rank=rank, world_size=world_size, backend=backend)
        pg = GroupMember.WORLD
        if pg is None:
            dist.init_process_group(rank=rank, world_size=world_size, backend=backend)
            pg = GroupMember.WORLD

        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)

        ranks = 2
        tables_mp = [[0], [1, 2]]
        lengths_dp = [
            torch.tensor([[1, 2], [1, 1], [2, 1]], dtype=torch.int),
            torch.tensor([[1, 2], [2, 1], [3, 1]], dtype=torch.int),
        ]  # W, T_g, B_l
        lengths_a2a = [
            torch.tensor([[[1, 2]], [[1, 2]]], dtype=torch.int),  # Rank 0
            torch.tensor(
                [
                    [[1, 1], [2, 1]],  # from Rank 0
                    [[2, 1], [3, 1]],  # from rank 1
                ],
                dtype=torch.int,
            ),  # Rank 1
        ]  # W, W, T_l, B_l
        lengths_mp = [
            torch.tensor(
                [
                    [1, 2, 1, 2],
                ],
                dtype=torch.int,
            ),
            torch.tensor([[1, 1, 2, 1], [2, 1, 3, 1]], dtype=torch.int),
        ]  # w, t_l, b_g
        input_seg = list(itertools.accumulate([0] + [len(i) for i in tables_mp]))
        input_splits = [
            [
                int(lengths_dp[i][input_seg[j] : input_seg[j + 1], :].sum())
                for j in range(ranks)
            ]
            for i in range(ranks)
        ]
        output_splits = [lengths_a2a[i].sum(dim=(1, 2)).tolist() for i in range(ranks)]
        table_dim = 3
        num_features_per_rank = [len(features) for features in tables_mp]
        seq_all2all_forward_recat = []
        for j in range(ranks):
            for i in range(num_features_per_rank[rank]):
                seq_all2all_forward_recat.append(j + i * ranks)
        seq_all2all_forward_recat_tensor = torch.IntTensor(seq_all2all_forward_recat)
        seq_all2all_backward_recat = []
        for i in range(num_features_per_rank[rank]):
            for j in range(ranks):
                seq_all2all_backward_recat.append(i + j * num_features_per_rank[rank])

        seq_all2all_backward_recat_tensor = torch.IntTensor(seq_all2all_backward_recat)
        input_embeddings = torch.rand(
            int(lengths_mp[rank].sum()),
            table_dim,
            device=device,
            requires_grad=True,
        )
        lengths_after_sparse_data_all2all = torch.IntTensor(lengths_mp[rank])

        # pyre-ignore
        def fn(*args, **kwargs) -> torch.Tensor:
            return comm_ops.alltoall_sequence(*args, **kwargs).wait()

        fn_transform = torch_compile_args_to_fn_transform(torch_compile_args)

        with dynamo_skipfiles_allow("torchrec"):
            seq_embs_out = fn_transform(fn)(
                a2a_sequence_embs_tensor=input_embeddings.cuda(),
                forward_recat_tensor=seq_all2all_forward_recat_tensor.cuda(),
                backward_recat_tensor=seq_all2all_backward_recat_tensor.cuda(),
                lengths_after_sparse_data_all2all=lengths_after_sparse_data_all2all.cuda(),
                input_splits=input_splits[rank],
                output_splits=output_splits[rank],
                group=pg if specify_pg else None,
            )

        if torch_compile_args is not None and not skip_dynamo_backwards:
            seq_embs_out.backward(seq_embs_out)
            grad = input_embeddings.grad
            assert torch.equal(
                input_embeddings.cpu().detach(),
                # pyre-fixme[16]: Optional type has no attribute `cpu`.
                grad.cpu().detach() * world_size,
            )

        dist.destroy_process_group()

    @unittest.skipIf(
        torch.cuda.device_count() < 2, "Need at least two ranks to run this test"
    )
    # pyre-ignore
    @given(
        torch_compile_args=st.sampled_from([None, ("eager", True)]),
        specify_pg=st.sampled_from([True]),
    )
    @settings(deadline=None)
    def test_alltoall_sequence(
        self,
        torch_compile_args: Optional[Tuple[str, bool]],
        specify_pg: bool,
    ) -> None:
        self._run_multi_process_test(
            world_size=self.WORLD_SIZE,
            backend="nccl",
            # pyre-ignore [6]
            callable=self._test_alltoall_sequence,
            torch_compile_args=torch_compile_args,
            # TODO(ivankobzarev): Add backwards formula for fbgemm::permute_2D_sparse_data
            skip_dynamo_backwards=True,
            specify_pg=specify_pg,
        )

    @classmethod
    def _test_alltoall_pooled(
        cls,
        rank: int,
        world_size: int,
        backend: str,
        torch_compile_args: Optional[Tuple[str, bool]],
        specify_pg: bool,
    ) -> None:
        pg = GroupMember.WORLD
        if pg is None:
            dist.init_process_group(rank=rank, world_size=world_size, backend=backend)
            pg = GroupMember.WORLD

        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)

        pg = dist.distributed_c10d._get_default_group()

        # Each rank's local batch size

        batch_size_per_rank = [4] * world_size

        # Global batch size is the sum of all rank's local batch size
        B_global = sum(batch_size_per_rank)
        # sum of dimensions of the embedding tables hosted on each rank
        dim_sum_per_rank = [8] * world_size

        D_local_sum = dim_sum_per_rank[rank]

        # Construct pooled embeddings
        pooled_embeddings = torch.randn([B_global, D_local_sum], requires_grad=True).to(
            device
        )
        pooled_embeddings.retain_grad()

        # Save a copy for running again with gradient division
        pooled_embeddings_gradient_division = (
            pooled_embeddings.detach().clone().to(device)
        )
        pooled_embeddings_gradient_division.requires_grad = True
        pooled_embeddings_gradient_division.retain_grad()

        # Run alltoall_pooled with gradient division disabled
        comm_ops.set_gradient_division(False)

        # pyre-ignore
        def fn(*args, **kwargs) -> torch.Tensor:
            return comm_ops.alltoall_pooled(*args, **kwargs).wait()

        fn_transform = torch_compile_args_to_fn_transform(torch_compile_args)

        with dynamo_skipfiles_allow("torchrec"):
            a2a_embedding = fn_transform(fn)(
                pooled_embeddings,
                batch_size_per_rank,
                dim_sum_per_rank,
                group=pg if specify_pg else None,
            )

        a2a_embedding.retain_grad()
        a2a_embedding.backward(a2a_embedding)

        # Run alltoall_pooled with gradient division enabled
        comm_ops.set_gradient_division(True)

        with dynamo_skipfiles_allow("torchrec"):
            a2a_embedding_gradient_division = fn_transform(fn)(
                pooled_embeddings_gradient_division,
                batch_size_per_rank,
                dim_sum_per_rank,
                group=pg if specify_pg else None,
            )

        a2a_embedding_gradient_division.retain_grad()
        a2a_embedding_gradient_division.backward(a2a_embedding_gradient_division)

        assert torch.equal(
            none_throws(pooled_embeddings.grad),
            torch.mul(
                none_throws(pooled_embeddings_gradient_division.grad), world_size
            ),
        )
        dist.destroy_process_group()

    @unittest.skipIf(
        torch.cuda.device_count() < 2, "Need at least two ranks to run this test"
    )
    # pyre-ignore
    @given(
        torch_compile_args=st.sampled_from([None, ("eager", True)]),
        specify_pg=st.sampled_from([True]),
    )
    @settings(max_examples=1, deadline=None)
    def test_alltoall_pooled(
        self,
        torch_compile_args: Optional[Tuple[str, bool]],
        specify_pg: bool,
    ) -> None:
        self._run_multi_process_test(
            world_size=self.WORLD_SIZE,
            backend="nccl",
            # pyre-ignore [6]
            callable=self._test_alltoall_pooled,
            torch_compile_args=torch_compile_args,
            specify_pg=specify_pg,
        )

    @classmethod
    def _test_reduce_scatter_pooled(
        cls,
        rank: int,
        world_size: int,
        backend: str,
        torch_compile_args: Optional[Tuple[str, bool]],
        specify_pg: bool,
    ) -> None:
        pg = GroupMember.WORLD
        if pg is None:
            dist.init_process_group(rank=rank, world_size=world_size, backend=backend)
            pg = GroupMember.WORLD

        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)

        pg = dist.distributed_c10d._get_default_group()

        batch_size_per_rank = [4] * world_size
        B_global = sum(batch_size_per_rank)
        dim_sum_per_rank = [8] * world_size

        D_local_sum = dim_sum_per_rank[rank]

        inputs: List[torch.Tensor] = []
        for _ in range(world_size):
            input = torch.randn([B_global, D_local_sum], requires_grad=True).to(device)
            input.retain_grad()
            inputs.append(input)
        gradient_division: bool = False
        comm_ops.set_gradient_division(gradient_division)

        # pyre-ignore
        def fn(*args, **kwargs) -> torch.Tensor:
            return comm_ops.reduce_scatter_pooled(*args, **kwargs).wait()

        fn_transform = torch_compile_args_to_fn_transform(torch_compile_args)

        with dynamo_skipfiles_allow("torchrec"):
            output = fn_transform(fn)(
                inputs,
                group=pg if specify_pg else None,
            )

        output.retain_grad()
        output.backward(output)

        for input in inputs:
            assert input.grad is not None

        dist.destroy_process_group()

    @unittest.skipIf(
        torch.cuda.device_count() < 2, "Need at least two ranks to run this test"
    )
    # pyre-ignore
    @given(
        torch_compile_args=st.sampled_from([None, ("eager", True)]),
        specify_pg=st.sampled_from([True]),
    )
    @settings(max_examples=1, deadline=None)
    def test_reduce_scatter_pooled(
        self,
        torch_compile_args: Optional[Tuple[str, bool]],
        specify_pg: bool,
    ) -> None:
        self._run_multi_process_test(
            world_size=self.WORLD_SIZE,
            backend="nccl",
            # pyre-ignore [6]
            callable=self._test_reduce_scatter_pooled,
            torch_compile_args=torch_compile_args,
            specify_pg=specify_pg,
        )

    @classmethod
    def _test_reduce_scatter_v_pooled(
        cls,
        rank: int,
        world_size: int,
        backend: str,
        torch_compile_args: Optional[Tuple[str, bool]],
        specify_pg: bool,
    ) -> None:
        pg = GroupMember.WORLD
        if pg is None:
            dist.init_process_group(rank=rank, world_size=world_size, backend=backend)
            pg = GroupMember.WORLD

        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)

        pg = dist.distributed_c10d._get_default_group()

        src: List[int] = [1, 2, 3] * world_size
        input_splits: List[int] = src[:world_size]
        inputs_dim: int = sum(input_splits)

        input: torch.Tensor = torch.randn(inputs_dim, 2, requires_grad=True).to(device)
        input.retain_grad()

        gradient_division: bool = False
        comm_ops.set_gradient_division(gradient_division)

        # pyre-ignore
        def fn(*args, **kwargs) -> torch.Tensor:
            return comm_ops.reduce_scatter_v_pooled(*args, **kwargs).wait()

        fn_transform = torch_compile_args_to_fn_transform(torch_compile_args)

        with dynamo_skipfiles_allow("torchrec"):
            output = fn_transform(fn)(
                input,
                input_splits=input_splits,
                group=pg if specify_pg else None,
            )

        output.retain_grad()
        output.backward(output)

        input_splits_cumsum: List[int] = [0]
        cumsum = 0
        for s in input_splits:
            cumsum += s
            input_splits_cumsum.append(cumsum)

        from_idx = input_splits_cumsum[rank]
        to_idx = input_splits_cumsum[rank + 1]

        assert input.grad is not None
        input_grad_rank = input.grad[from_idx:to_idx]

        torch.testing.assert_close(
            input_grad_rank.cpu(),
            output.cpu(),
        )

        dist.destroy_process_group()

    @unittest.skipIf(
        torch.cuda.device_count() < 2, "Need at least two ranks to run this test"
    )
    # pyre-ignore
    @given(
        torch_compile_args=st.sampled_from([None, ("eager", True)]),
        specify_pg=st.sampled_from([True]),
    )
    @settings(max_examples=1, deadline=None)
    def test_reduce_scatter_v_pooled(
        self,
        torch_compile_args: Optional[Tuple[str, bool]],
        specify_pg: bool,
    ) -> None:
        self._run_multi_process_test(
            world_size=self.WORLD_SIZE,
            backend="nccl",
            # pyre-ignore [6]
            callable=self._test_reduce_scatter_v_pooled,
            torch_compile_args=torch_compile_args,
            specify_pg=specify_pg,
        )

    @classmethod
    def _test_all_gather_base_pooled(
        cls,
        rank: int,
        world_size: int,
        backend: str,
        torch_compile_args: Optional[Tuple[str, bool]],
        specify_pg: bool,
    ) -> None:
        pg = GroupMember.WORLD
        if pg is None:
            dist.init_process_group(rank=rank, world_size=world_size, backend=backend)
            pg = GroupMember.WORLD

        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)

        pg = dist.distributed_c10d._get_default_group()

        input = torch.randn([4, 4], requires_grad=True).to(device)
        input.retain_grad()

        # pyre-ignore
        def fn(*args, **kwargs) -> torch.Tensor:
            return comm_ops.all_gather_base_pooled(*args, **kwargs).wait()

        fn_transform = torch_compile_args_to_fn_transform(torch_compile_args)

        with dynamo_skipfiles_allow("torchrec"):
            output = fn_transform(fn)(
                input,
                group=pg if specify_pg else None,
            )

        output.retain_grad()
        output.backward(output)

        assert input.grad is not None
        torch.equal(none_throws(input.grad), output[4 * rank :])

        dist.destroy_process_group()

    @unittest.skipIf(
        torch.cuda.device_count() < 2, "Need at least two ranks to run this test"
    )
    # pyre-ignore
    @given(
        torch_compile_args=st.sampled_from([None, ("eager", True)]),
        specify_pg=st.sampled_from([True]),
    )
    @settings(max_examples=1, deadline=None)
    def test_all_gather_base_pooled(
        self,
        torch_compile_args: Optional[Tuple[str, bool]],
        specify_pg: bool,
    ) -> None:
        self._run_multi_process_test(
            world_size=self.WORLD_SIZE,
            backend="nccl",
            # pyre-ignore [6]
            callable=self._test_all_gather_base_pooled,
            torch_compile_args=torch_compile_args,
            specify_pg=specify_pg,
        )
