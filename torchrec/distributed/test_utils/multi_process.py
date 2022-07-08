#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

import multiprocessing
import os
import unittest
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.distributed as dist
from torchrec.distributed.comm import _CROSS_PG, _INTRA_PG
from torchrec.test_utils import (
    get_free_port,
    init_distributed_single_host,
    seed_and_log,
)


class MultiProcessContext:
    def __init__(
        self,
        rank: int,
        world_size: int,
        backend: str = "gloo",
        local_size: Optional[int] = None,
    ) -> None:

        self.rank = rank
        self.world_size = world_size
        self.backend = backend
        self.local_size = local_size

        if backend == "nccl":
            device = torch.device(f"cuda:{rank}")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")
        self.device: torch.device = device
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
        self.pg: Optional[dist.ProcessGroup] = None

    # pyre-ignore
    def __enter__(self):
        """
        Override local_size after pg construction because unit test device count is
        larger than local_size setup. This can be problematic for twrw because we have
        ShardedTensor placement check.

        TODO (T108556130) Mock out functions in comm.py instead of overriding env vars
        """

        os.environ["LOCAL_WORLD_SIZE"] = str(self.local_size or self.world_size)
        if self.local_size is not None:
            os.environ["LOCAL_RANK"] = str(self.rank % self.local_size)

        self.pg = init_distributed_single_host(
            rank=self.rank,
            world_size=self.world_size,
            backend=self.backend,
            local_size=self.local_size,
        )
        return self

    # pyre-ignore
    def __exit__(self, exc_type, exc_instance, traceback) -> None:
        if _INTRA_PG is not None:
            dist.destroy_process_group(_INTRA_PG)
        if _CROSS_PG is not None:
            dist.destroy_process_group(_CROSS_PG)
        dist.destroy_process_group(self.pg)
        torch.use_deterministic_algorithms(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.allow_tf32 = True


class MultiProcessTestBase(unittest.TestCase):
    @seed_and_log
    def setUp(self) -> None:
        os.environ["MASTER_ADDR"] = str("localhost")
        os.environ["MASTER_PORT"] = str(get_free_port())
        os.environ["GLOO_DEVICE_TRANSPORT"] = "TCP"
        os.environ["NCCL_SOCKET_IFNAME"] = "lo"

        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    def tearDown(self) -> None:
        torch.use_deterministic_algorithms(False)
        del os.environ["GLOO_DEVICE_TRANSPORT"]
        del os.environ["NCCL_SOCKET_IFNAME"]
        if torch.cuda.is_available():
            os.unsetenv("CUBLAS_WORKSPACE_CONFIG")
        super().tearDown()

    def _run_multi_process_test(
        self,
        *,
        callable: Callable[
            ...,
            None,
        ],
        world_size: int,
        # pyre-ignore
        **kwargs,
    ) -> None:
        ctx = multiprocessing.get_context("forkserver")
        processes = []
        for rank in range(world_size):
            kwargs["rank"] = rank
            kwargs["world_size"] = world_size
            p = ctx.Process(
                target=callable,
                kwargs=kwargs,
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            self.assertEqual(0, p.exitcode)

    def _run_multi_process_test_per_rank(
        self,
        *,
        callable: Callable[
            ...,
            None,
        ],
        world_size: int,
        kwargs_per_rank: List[Dict[str, Any]],
    ) -> None:
        ctx = multiprocessing.get_context("forkserver")
        processes = []
        for rank in range(world_size):
            kwargs = {}
            kwargs["rank"] = rank
            kwargs["world_size"] = world_size
            kwargs.update(kwargs_per_rank[rank])
            p = ctx.Process(
                target=callable,
                kwargs=kwargs,
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            self.assertEqual(0, p.exitcode)
