#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torchrec.distributed.comm_ops import all_to_all_single, set_gradient_division

from torchrec.distributed.propagating_async_collective_tensor import (
    PropagatingAsyncCollectiveTensor,
)

from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.test_utils import skip_if_asan_class


def _test_propagating_async_collective_tensor(  # noqa C901
    rank: int,
    world_size: int,
    backend: str,
) -> None:
    with MultiProcessContext(rank, world_size, backend, local_size=world_size) as ctx:
        set_gradient_division(False)

        # output will be rank 0 [0,0,0,0 | 2,2,2,2] rank 1 [1,1,1,1 | 3,3,3,3 ]
        if ctx.rank == 0:
            input = torch.tensor(
                [0, 0, 0, 0, 1, 1, 1, 1],
                device=ctx.device,
                dtype=torch.float,
                requires_grad=True,
            )
        else:
            input = torch.tensor(
                [2, 2, 2, 2, 3, 3, 3, 3],
                device=ctx.device,
                dtype=torch.float,
                requires_grad=True,
            )

        a2a_out = all_to_all_single(
            input, output_split_sizes=[4, 4], input_split_sizes=[4, 4], group=ctx.pg
        )
        two_tensor = torch.tensor([2], dtype=torch.float, device=ctx.device)

        a2a_split = a2a_out.split([4, 4])
        assert isinstance(a2a_split[0], PropagatingAsyncCollectiveTensor) and (
            a2a_split[0].manifested_thunk is None
        )

        a2a_split_view = [x.view(2, 2) for x in a2a_split]
        a2a_split_cat = torch.cat(a2a_split_view, dim=0)
        assert isinstance(a2a_split_cat, PropagatingAsyncCollectiveTensor) and (
            a2a_split_cat.manifested_thunk is None
        )
        a2a_split_cat_sum = a2a_split_cat.sum()
        assert isinstance(a2a_split_cat_sum, PropagatingAsyncCollectiveTensor) and (
            a2a_split_cat_sum.manifested_thunk is None
        )
        a2a_split_cat_sum_twice = a2a_split_cat_sum * two_tensor
        assert isinstance(
            a2a_split_cat_sum_twice, PropagatingAsyncCollectiveTensor
        ) and (a2a_split_cat_sum_twice.manifested_thunk is None)

        a2a_split_cat_sum_twice.manifest_on_op_with_non_prop_async_tensor = True

        # Test to ensure that this gets manifested.

        a2a_split_cat_sum_twice_twice = a2a_split_cat_sum_twice * two_tensor
        assert isinstance(
            a2a_split_cat_sum_twice_twice, torch.Tensor
        ) and not isinstance(
            a2a_split_cat_sum_twice_twice, PropagatingAsyncCollectiveTensor
        )

        a2a_split_cat_sum_twice_twice.sum().backward()
        print("input grad", input.grad)
        torch.testing.assert_close(
            input.grad,
            torch.ones_like(input) * 4
            # [4,4,4,4,4,4,4,4] from multiplying two_tensor twice and summing
        )


@skip_if_asan_class
class TestPropagatingAsyncCollectiveTensor(MultiProcessTestBase):
    # pyre-ignore
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    def test_pact(self) -> None:
        WORLD_SIZE = 2
        self._run_multi_process_test(
            callable=_test_propagating_async_collective_tensor,
            world_size=WORLD_SIZE,
            backend="nccl",
        )
