#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from torchrec.distributed.sharding.rw_tensor_pool_sharding import TensorPoolRwSharding
from torchrec.distributed.tensor_sharding import TensorPoolRwShardingContext
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.types import ShardingEnv


class TestTensorPoolRwSharding(MultiProcessTestBase):
    @staticmethod
    def _test_update(
        rank: int,
        world_size: int,
    ) -> None:
        backend = "nccl"
        dtype = torch.float32
        with MultiProcessContext(
            rank, world_size, backend, local_size=world_size
        ) as ctx:
            # pyre-fixme[6]: For 1st argument expected `ProcessGroup` but got
            #  `Optional[ProcessGroup]`.
            sharding_env = ShardingEnv.from_process_group(ctx.pg)
            if ctx.rank == 0:
                ids = [4, 1]
                values = [0.1, 0.2, 0.3], [0.4, 0.5, 0.6]

            else:
                ids = [3, 0]
                values = [0.11, 0.21, 0.31], [0.41, 0.51, 0.61]

            ids = torch.tensor(ids, dtype=torch.int, device=ctx.device)
            values = torch.tensor(values, dtype=torch.float, device=ctx.device)

            block_size = torch.tensor([3], dtype=torch.int, device=ctx.device)
            update_ctx = TensorPoolRwShardingContext(block_size=block_size)
            rw_sharding = TensorPoolRwSharding(
                env=sharding_env, device=ctx.device, dim=3, pool_size=4
            )
            input_dist = rw_sharding.create_lookup_ids_dist()
            update_values_dist = rw_sharding.create_update_values_dist()
            dist_ids = input_dist(ctx=update_ctx, ids=ids).wait().wait()

            torch.testing.assert_close(
                dist_ids.cpu(),
                torch.tensor(
                    [1, 0],
                    device=torch.device("cpu"),
                    dtype=torch.int,
                ),
            )

            dist_values = update_values_dist(ctx=update_ctx, values=values).wait()
            if rank == 0:
                torch.testing.assert_close(
                    dist_values.cpu(),
                    torch.tensor(
                        [[0.4, 0.5, 0.6], [0.41, 0.51, 0.61]],
                        device=torch.device("cpu"),
                        dtype=dtype,
                    ),
                )
            else:
                torch.testing.assert_close(
                    dist_values.cpu(),
                    torch.tensor(
                        [[0.1, 0.2, 0.3], [0.11, 0.21, 0.31]],
                        device=torch.device("cpu"),
                        dtype=dtype,
                    ),
                )

    @staticmethod
    def _test_lookup(
        rank: int,
        world_size: int,
    ) -> None:
        backend = "nccl"
        dtype = torch.float32
        with MultiProcessContext(
            rank, world_size, backend, local_size=world_size
        ) as ctx:
            # pyre-fixme[6]: For 1st argument expected `ProcessGroup` but got
            #  `Optional[ProcessGroup]`.
            sharding_env = ShardingEnv.from_process_group(ctx.pg)

            block_size = torch.tensor([3], dtype=torch.int, device=ctx.device)
            lookup_ctx = TensorPoolRwShardingContext(block_size=block_size)
            rw_sharding = TensorPoolRwSharding(
                env=sharding_env, device=ctx.device, dim=3, pool_size=5
            )
            input_dist = rw_sharding.create_lookup_ids_dist()
            lookup_values_dist = rw_sharding.create_lookup_values_dist()

            ids = torch.tensor([0, 1, 2, 3], dtype=torch.int, device=ctx.device)
            dist_ids = input_dist(ctx=lookup_ctx, ids=ids).wait().wait()
            if rank == 0:
                torch.testing.assert_close(
                    dist_ids.cpu(),
                    torch.tensor(
                        [0, 1, 2, 0, 1, 2],
                        dtype=torch.int,
                        device=torch.device("cpu"),
                    ),
                )
            else:
                torch.testing.assert_close(
                    dist_ids.cpu(),
                    torch.tensor(
                        [0, 0],
                        dtype=torch.int,
                        device=torch.device("cpu"),
                    ),
                )

            # assume the _local_pool on rank 0 is
            # [
            # [0.41, 0.51, 0.61],
            # [0.4, 0.5, 0.6],
            # [0.0, 0.0, 0.0],
            # ]

            # on rank 1 is
            # [
            # [0.11, 0.21, 0.31],
            # [0.1, 0.2, 0.3],
            # ]

            if rank == 0:
                lookup_values = torch.tensor(
                    [
                        [0.41, 0.51, 0.61],
                        [0.4, 0.5, 0.6],
                        [0.0, 0.0, 0.0],
                        [0.41, 0.51, 0.61],
                        [0.4, 0.5, 0.6],
                        [0.0, 0.0, 0.0],
                    ],
                    dtype=dtype,
                    device=ctx.device,
                )

            else:
                lookup_values = torch.tensor(
                    [
                        [0.11, 0.21, 0.31],
                        [0.11, 0.21, 0.31],
                    ],
                    dtype=dtype,
                    device=ctx.device,
                )

            dist_output_values = lookup_values_dist(
                ctx=lookup_ctx, values=lookup_values
            ).wait()

            torch.testing.assert_close(
                dist_output_values.cpu(),
                torch.tensor(
                    [
                        [0.41, 0.51, 0.61],
                        [0.4, 0.5, 0.6],
                        [0.0, 0.0, 0.0],
                        [0.11, 0.21, 0.31],
                    ],
                    device=torch.device("cpu"),
                ),
            )

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    def test_update(
        self,
    ) -> None:
        world_size = 2
        self._run_multi_process_test(callable=self._test_update, world_size=world_size)

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    def test_lookup(
        self,
    ) -> None:
        world_size = 2
        self._run_multi_process_test(callable=self._test_lookup, world_size=world_size)
