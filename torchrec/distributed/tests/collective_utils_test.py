#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from unittest import mock

import caffe2.torch.fb.distributed.utils.log_utils as log_utils
import torch.distributed as dist
from torch._utils_internal import TEST_MASTER_ADDR as MASTER_ADDR  # @manual
from torch.testing._internal.common_distributed import MultiProcessTestCase  # @manual
from torchrec.distributed.collective_utils import (
    is_leader,
    invoke_on_rank_and_broadcast_result,
    run_on_leader,
)
from torchrec.tests.utils import get_free_port


logger: logging.Logger = log_utils.getLogger()


"""
buck test @mode/dev-nosan //torchrec/distributed/tests:collective_utils_test

Mirrors the test cases implemented for ExtendProcessGroup collective_utils located in:
fbcode/caffe2/torch/fb/hpc/tests/collective_utils_test.py
"""


class CollectiveUtilsTest(MultiProcessTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ["MASTER_ADDR"] = str(MASTER_ADDR)
        os.environ["MASTER_PORT"] = str(get_free_port())
        os.environ["GLOO_DEVICE_TRANSPORT"] = "TCP"
        super().setUpClass()

    def setUp(self) -> None:
        super(CollectiveUtilsTest, self).setUp()
        self._spawn_processes()

    def tearDown(self) -> None:
        super(CollectiveUtilsTest, self).tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @property
    def world_size(self) -> int:
        return 2

    def test_is_leader(self) -> None:
        dist.init_process_group(
            rank=self.rank, world_size=self.world_size, backend="gloo"
        )
        pg = dist.new_group(
            ranks=[0, 1],
            backend="gloo",
        )

        if pg.rank() == 0:
            assert is_leader(pg, 0) is True
            assert is_leader(pg, 1) is False
        else:
            assert is_leader(pg, 1) is True
            assert is_leader(pg, 0) is False

    def test_invoke_on_rank_and_broadcast_result(self) -> None:
        dist.init_process_group(
            rank=self.rank, world_size=self.world_size, backend="gloo"
        )
        pg = dist.new_group(
            ranks=[0, 1],
            backend="gloo",
        )

        func = mock.MagicMock()
        func.return_value = pg.rank()

        res = invoke_on_rank_and_broadcast_result(pg=pg, rank=0, func=func)
        assert res == 0, f"Expect res to be 0 (got {res})"

        if pg.rank() == 0:
            func.assert_called_once()
        else:
            func.assert_not_called()
        func.reset_mock()

        res = invoke_on_rank_and_broadcast_result(pg=pg, rank=1, func=func)
        assert res == 1, f"Expect res to be 1 (got {res})"

        if pg.rank() == 0:
            func.assert_not_called()
        else:
            func.assert_called_once()

    def test_run_on_leader_decorator(self) -> None:
        dist.init_process_group(
            rank=self.rank, world_size=self.world_size, backend="gloo"
        )
        pg = dist.new_group(
            ranks=[0, 1],
            backend="gloo",
        )

        @run_on_leader(pg, 0)
        def _test_run_on_0(rank: int) -> int:
            return rank

        res = _test_run_on_0(pg.rank())
        assert res == 0

        @run_on_leader(pg, 1)
        def _test_run_on_1(rank: int) -> int:
            return rank

        res = _test_run_on_1(pg.rank())
        assert res == 1
