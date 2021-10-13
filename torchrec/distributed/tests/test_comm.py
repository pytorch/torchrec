#!/usr/bin/env python3

import itertools
import os
import unittest

import numpy
import torch
import torch.distributed as dist
import torchrec.distributed.comm_ops as comm_ops
from torch._utils_internal import TEST_MASTER_ADDR as MASTER_ADDR  # @manual
from torch.testing._internal.common_distributed import MultiProcessTestCase  # @manual
from torchrec.tests.utils import get_free_port


class TestAllToAll(MultiProcessTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ["MASTER_ADDR"] = str(MASTER_ADDR)
        os.environ["MASTER_PORT"] = str(get_free_port())
        super().setUpClass()

    def setUp(self) -> None:
        super(TestAllToAll, self).setUp()
        self._spawn_processes()

    def tearDown(self) -> None:
        super(TestAllToAll, self).tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @property
    def world_size(self) -> int:
        return 2

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `torch.cuda.device_count() = 0` to decorator factory `unittest.skipIf`.
    @unittest.skipIf(
        torch.cuda.device_count() < 2, "Need at least two ranks to run this test"
    )
    def test_alltoallv(self) -> None:
        dist.init_process_group(
            rank=self.rank, world_size=self.world_size, backend="nccl"
        )
        device = torch.device(f"cuda:{self.rank}")

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

        a2a_req = comm_ops.alltoallv(input_embeddings, out_split)
        v_embs_out = a2a_req.wait()
        res = torch.cat(v_embs_out, dim=1).cpu()
        self.assertEqual(tuple(res.size()), (5, 34))

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    #  `torch.cuda.device_count() = 0` to decorator factory `unittest.skipIf`.
    @unittest.skipIf(
        torch.cuda.device_count() < 2, "Need at least two ranks to run this test"
    )
    def test_alltoall_sequence(self) -> None:
        dist.init_process_group(
            rank=self.rank, world_size=self.world_size, backend="nccl"
        )
        device = torch.device(f"cuda:{self.rank}")
        torch.cuda.set_device(device)

        ranks = 2
        tables_mp = [[0], [1, 2]]
        lengths_dp = [
            numpy.array([[1, 2], [1, 1], [2, 1]]),
            numpy.array([[1, 2], [2, 1], [3, 1]]),
        ]  # W, T_g, B_l
        lengths_a2a = [
            numpy.array([[[1, 2]], [[1, 2]]]),  # Rank 0
            numpy.array(
                [
                    [[1, 1], [2, 1]],  # from Rank 0
                    [[2, 1], [3, 1]],  # from rank 1
                ]
            ),  # Rank 1
        ]  # W, W, T_l, B_l
        lengths_mp = [
            numpy.array(
                [
                    [1, 2, 1, 2],
                ]
            ),
            numpy.array([[1, 1, 2, 1], [2, 1, 3, 1]]),
        ]  # w, t_l, b_g
        input_seg = list(itertools.accumulate([0] + [len(i) for i in tables_mp]))
        input_splits = [
            [
                lengths_dp[i][input_seg[j] : input_seg[j + 1], :].sum()
                for j in range(ranks)
            ]
            for i in range(ranks)
        ]
        output_splits = [lengths_a2a[i].sum(axis=(1, 2)).tolist() for i in range(ranks)]
        table_dim = 3
        num_features_per_rank = [len(features) for features in tables_mp]
        seq_all2all_forward_recat = []
        for j in range(ranks):
            for i in range(num_features_per_rank[self.rank]):
                seq_all2all_forward_recat.append(j + i * ranks)
        seq_all2all_forward_recat_tensor = torch.IntTensor(seq_all2all_forward_recat)
        seq_all2all_backward_recat = []
        for i in range(num_features_per_rank[self.rank]):
            for j in range(ranks):
                seq_all2all_backward_recat.append(
                    i + j * num_features_per_rank[self.rank]
                )

        seq_all2all_backward_recat_tensor = torch.IntTensor(seq_all2all_backward_recat)
        input_embeddings = torch.rand(
            lengths_mp[self.rank].sum(),
            table_dim,
            device=device,
            requires_grad=True,
        )
        lengths_after_sparse_data_all2all = torch.IntTensor(lengths_mp[self.rank])
        a2a_req = comm_ops.alltoall_sequence(
            a2a_sequence_embs_tensor=input_embeddings.cuda(),
            forward_recat_tensor=seq_all2all_forward_recat_tensor.cuda(),
            backward_recat_tensor=seq_all2all_backward_recat_tensor.cuda(),
            lengths_after_sparse_data_all2all=lengths_after_sparse_data_all2all.cuda(),
            input_splits=input_splits[self.rank],
            output_splits=output_splits[self.rank],
        )
        seq_embs_out = a2a_req.wait()
        seq_embs_out.backward(seq_embs_out)
        grad = input_embeddings.grad
        self.assertEqual(input_embeddings.cpu().detach(), grad.cpu().detach())
