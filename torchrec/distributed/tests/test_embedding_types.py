#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Dict, List

import torch
from torchrec.distributed.embedding_types import KJTList, ShardedEmbeddingModule
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionContext
from torchrec.distributed.types import Awaitable, LazyAwaitable

Out = Dict[str, torch.Tensor]
CompIn = KJTList
DistOut = List[torch.Tensor]
ShrdCtx = EmbeddingBagCollectionContext


class FakeShardedEmbeddingModule(ShardedEmbeddingModule[CompIn, DistOut, Out, ShrdCtx]):
    def __init__(self) -> None:
        super().__init__()
        self._lookups = [
            torch.nn.Module(),
            torch.nn.Module(),
        ]

    # pyre-fixme[7]: Expected `EmbeddingBagCollectionContext` but got implicit
    #  return value of `None`.
    def create_context(self) -> ShrdCtx:
        pass

    def input_dist(
        self,
        ctx: ShrdCtx,
        # pyre-ignore[2]
        *input,
        # pyre-ignore[2]
        **kwargs,
        # pyre-fixme[7]: Expected `Awaitable[Awaitable[KJTList]]` but got implicit
        #  return value of `None`.
    ) -> Awaitable[Awaitable[CompIn]]:
        pass

    # pyre-fixme[7]: Expected `List[Tensor]` but got implicit return value of `None`.
    def compute(self, ctx: ShrdCtx, dist_input: CompIn) -> DistOut:
        pass

    # pyre-fixme[7]: Expected `LazyAwaitable[Dict[str, Tensor]]` but got implicit
    #  return value of `None`.
    def output_dist(self, ctx: ShrdCtx, output: DistOut) -> LazyAwaitable[Out]:
        pass


class TestShardedEmbeddingModule(unittest.TestCase):
    def test_train_mode(self) -> None:
        embedding_module = FakeShardedEmbeddingModule()
        for mode in [True, False]:
            with self.subTest(mode=mode):
                embedding_module.train(mode)
                self.assertEqual(embedding_module.training, mode)
                for lookup in embedding_module._lookups:
                    self.assertEqual(lookup.training, mode)
