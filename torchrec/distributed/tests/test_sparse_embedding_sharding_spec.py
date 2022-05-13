#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
import random
import unittest
from typing import Callable

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed._shard.api import _collect_local_shard, _reshard_output
from torch.distributed._shard.replicated_tensor import ReplicatedTensor
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.nn.parallel._replicated_tensor_ddp_utils import _ddp_replicated_tensor
from torch.nn.parallel.distributed import DistributedDataParallel
from torchrec.distributed.sharding.sparse_embedding_sharding_spec import (
    SparseEmbeddingShardingSpec,
)
from torchrec.fb.feed.modules.user_history_arch import SparseSequenceEsuhmArch
from torchrec.test_utils import get_free_port, seed_and_log, skip_if_asan_class

TEST_GPU_NUM = int(os.getenv("TEST_GPU_NUM", 4))


@skip_if_asan_class
class SparseEmbeddingShardingSpecTest(unittest.TestCase):
    @seed_and_log
    def setUp(self) -> None:
        os.environ["MASTER_ADDR"] = str("localhost")
        os.environ["MASTER_PORT"] = str(get_free_port())
        os.environ["GLOO_DEVICE_TRANSPORT"] = "TCP"
        os.environ["NCCL_SOCKET_IFNAME"] = "lo"

    @classmethod
    def _test_sparse_sequence_esuhm_arch(cls, rank: int, world_size: int) -> None:
        device = torch.device(f"cuda:{rank}")
        torch.distributed.init_process_group(
            rank=rank, world_size=world_size, backend="nccl"
        )
        pg = torch.distributed.distributed_c10d._get_default_group()

        B = TEST_GPU_NUM
        N = 5
        D = 4

        # full
        torch.manual_seed(0)
        m = SparseSequenceEsuhmArch(
            d_model=D,
            hidden_dim=2,
            max_seq_len=N,
            pass_target_emb_from_seq=0,
            apply_softmax=True,
        ).to(device)
        LR = 0.01
        opt = torch.optim.SGD(m.parameters(), lr=LR)

        # sharded
        torch.manual_seed(0)
        m_sharded = SparseSequenceEsuhmArch(
            d_model=D,
            hidden_dim=2,
            max_seq_len=N,
            pass_target_emb_from_seq=0,
            apply_softmax=True,
        ).to(device)
        # Adjust learning rate for DDP.
        sharded_opt = torch.optim.SGD(m_sharded.parameters(), lr=LR * world_size)
        with _ddp_replicated_tensor(True):
            m_sharded = DistributedDataParallel(m_sharded)

        # Resharding for output.
        placements = [
            torch.distributed._remote_device(f"rank:{r}/cuda:{r}")
            for r in range(world_size)
        ]
        # pyre-ignore[28]
        reshard_spec = ChunkShardingSpec(
            dim=0,
            placements=placements,
        )
        m_sharded = _collect_local_shard(_reshard_output(m_sharded, reshard_spec))

        # Multiple iterations to test different splits of SparseEmbeddingShardingSpec.
        ITERATIONS = 1000
        for iter in range(ITERATIONS):
            torch.manual_seed(iter)
            source_embs = torch.rand(B, N, D, dtype=torch.float, device=device)
            target_embs = torch.rand(B, D, dtype=torch.float, device=device)

            # single train step
            res = m(source_embs, target_embs)
            loss = res.view(B, -1).sum(dim=1).mean()
            loss.backward()
            opt.step()
            opt.zero_grad()

            # sharded

            # split on N arbitrarily
            random.seed(iter)
            splits = [0 for _ in range(world_size)]
            for _ in range(N):
                idx = random.randint(0, world_size - 1)
                splits[idx] += 1

            lengths = copy.deepcopy(splits)
            splits.insert(0, 0)
            splits = np.cumsum(splits)

            start = splits[rank]
            end = splits[rank + 1]

            # re-order and split source_embs
            loc_source_embs = source_embs.clone()
            permutation = torch.randperm(N).to(device)
            loc_source_embs = loc_source_embs[:, permutation, :]
            loc_source_embs = loc_source_embs[:, start:end, :]
            loc_source_embs = loc_source_embs.contiguous().detach()

            # create sharded source_embs and replicated target_embs
            sharding_spec = SparseEmbeddingShardingSpec(
                dim=1,
                indices=permutation,
                lengths=torch.LongTensor(lengths).to(device),
                placements=placements,
            )
            sharded_source_embs = ShardedTensor._init_from_local_tensor(
                loc_source_embs, sharding_spec, (B, N, D), process_group=pg
            )
            replicated_target_embs = ReplicatedTensor(target_embs, process_group=pg)

            # single train step
            sharded_res = m_sharded(sharded_source_embs, replicated_target_embs)
            loss = sharded_res.view(B, -1).sum(dim=1).mean()
            loss.backward()
            sharded_opt.step()
            sharded_opt.zero_grad()

            sharded_res_list = [
                torch.zeros_like(sharded_res) for _ in range(world_size)
            ]
            dist.all_gather(sharded_res_list, sharded_res)

            # pyre-ignore[16]
            final_sharded_res = torch.cat(sharded_res_list).view_as(res)
            torch.testing.assert_allclose(res, final_sharded_res)

    # pyre-ignore[56]
    @unittest.skipIf(
        torch.cuda.device_count() < TEST_GPU_NUM,
        f"Not enough GPUs, this test requires at least {TEST_GPU_NUM} GPUs",
    )
    def test_sparse_sequence_esuhm_arch(self) -> None:
        self._run_multi_process_test(
            # pyre-ignore[6]
            SparseEmbeddingShardingSpecTest._test_sparse_sequence_esuhm_arch,
            TEST_GPU_NUM,
        )

    def _run_multi_process_test(
        self,
        callable: Callable[[], None],
        world_size: int,
    ) -> None:
        mp.spawn(callable, args=(world_size,), nprocs=world_size)
