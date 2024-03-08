#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Stress tests for planner to find problematic scaling behavior."""

import time
import unittest

from typing import List, Tuple

from torch import nn

from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.planner.constants import BATCH_SIZE
from torchrec.distributed.planner.enumerators import EmbeddingEnumerator
from torchrec.distributed.planner.types import Topology
from torchrec.distributed.test_utils.test_model import TestSparseNN
from torchrec.distributed.types import ModuleSharder, ShardingType
from torchrec.modules.embedding_configs import EmbeddingBagConfig


class TWSharder(EmbeddingBagCollectionSharder, ModuleSharder[nn.Module]):
    def sharding_types(self, compute_device_type: str) -> List[str]:
        return [ShardingType.TABLE_WISE.value]

    def compute_kernels(
        self, sharding_type: str, compute_device_type: str
    ) -> List[str]:
        return [EmbeddingComputeKernel.DENSE.value]


class TestEnumeratorBenchmark(unittest.TestCase):
    @staticmethod
    def build(
        world_size: int, num_tables: int
    ) -> Tuple[EmbeddingEnumerator, nn.Module]:
        compute_device = "cuda"
        topology = Topology(
            world_size=world_size, local_world_size=8, compute_device=compute_device
        )
        tables = [
            EmbeddingBagConfig(
                num_embeddings=100 + i,
                embedding_dim=128,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(num_tables)
        ]
        model = TestSparseNN(tables=tables, weighted_tables=[])
        enumerator = EmbeddingEnumerator(topology=topology, batch_size=BATCH_SIZE)
        return enumerator, model

    def measure(self, world_size: int, num_tables: int) -> float:
        enumerator, model = TestEnumeratorBenchmark.build(world_size, num_tables)

        start_time = time.time()
        sharding_options = enumerator.enumerate(module=model, sharders=[TWSharder()])
        end_time = time.time()

        self.assertEqual(len(sharding_options), num_tables)
        return end_time - start_time

    def test_benchmark(self) -> None:
        tests = [(2048, d) for d in [100, 200, 400, 800, 1600, 3200, 6400]]
        print("\nEnumerator benchmark:")
        for world_size, num_tables in tests:
            t = self.measure(world_size, num_tables)
            print(
                f"world_size={world_size:8} num_tables={num_tables:8} enumerate={t:4.2f}s"
            )


def main() -> None:
    unittest.main()


# This is structured as a unitttest like file so you can use its built-in command
# line argument parsing to control which benchmarks to run, e.g. "-k Enumerator"
if __name__ == "__main__":
    main()  # pragma: no cover
