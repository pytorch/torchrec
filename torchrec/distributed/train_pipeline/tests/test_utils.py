#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from functools import partial
from typing import List
from unittest.mock import MagicMock

import torch

from torchrec.distributed.embedding_sharding import (
    FusedKJTListSplitsAwaitable,
    KJTListAwaitable,
    KJTListSplitsAwaitable,
)
from torchrec.distributed.embedding_types import KJTList
from torchrec.distributed.train_pipeline.utils import (
    _fuse_input_dist_splits,
    TrainPipelineContext,
)
from torchrec.distributed.types import Awaitable, NoWait
from torchrec.distributed.utils import append_callback
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class TestFuseInputDist(unittest.TestCase):
    def test_fuse_input_dist_splits_no_callbacks(self) -> None:
        name = "ebc"
        context = TrainPipelineContext()
        kjt = KeyedJaggedTensor(
            values=torch.tensor([1.0]), lengths=torch.tensor(1), keys=["t1"]
        )
        # pyre-ignore
        awaitables: List[Awaitable[Awaitable[KeyedJaggedTensor]]] = [
            NoWait(NoWait(kjt))
        ]
        ebc_context = MagicMock()
        context.input_dist_splits_requests[name] = KJTListSplitsAwaitable(
            awaitables, ebc_context
        )
        context.module_contexts_next_batch[name] = MagicMock()

        _fuse_input_dist_splits(context)

        self.assertTrue(len(context.fused_splits_awaitables))

    def test_fuse_input_dist_splits_with_callbacks(self) -> None:
        name = "ebc"
        context: TrainPipelineContext = TrainPipelineContext()
        kjt: KeyedJaggedTensor = KeyedJaggedTensor(
            values=torch.tensor([1.0]), lengths=torch.tensor(1), keys=["t1"]
        )

        # pyre-ignore
        awaitable: Awaitable[Awaitable[KeyedJaggedTensor]] = NoWait(NoWait(kjt))
        ebc_context = MagicMock()
        splits_awaitable: Awaitable[Awaitable[KJTList]] = KJTListSplitsAwaitable(
            [awaitable], ebc_context
        )

        # append two layer callback
        def remap(kjtlist: KJTList) -> KJTList:
            for kjt in kjtlist:
                kjt._values += 1
            return kjtlist

        callback = partial(append_callback, callback=remap)
        splits_awaitable.callbacks.append(callback)

        # test fuse input dist splits
        context.input_dist_splits_requests[name] = splits_awaitable
        context.module_contexts_next_batch[name] = MagicMock()
        _fuse_input_dist_splits(context)
        self.assertEqual(len(context.fused_splits_awaitables), 1)

        # first FusedKJTListSplitsAwaitable, and then second position in a tuple
        fused_splits_awaitable: FusedKJTListSplitsAwaitable = (
            context.fused_splits_awaitables[0][1]
        )
        self.assertEqual(len(fused_splits_awaitable.callbacks), 1)

        fused_awaitables: List[KJTListAwaitable] = fused_splits_awaitable.wait()
        kjtlist: KJTList = fused_awaitables[0].wait()
        kjt = kjtlist[0]
        self.assertIsInstance(kjt, KeyedJaggedTensor)
        self.assertEqual(kjt._values, torch.tensor([2.0]))
