#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from typing import cast, List

import torch

from torch import nn

from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder

from torchrec.distributed.planner.parallelized_planners import (
    ParallelizedEmbeddingShardingPlanner,
)
from torchrec.distributed.planner.planners import EmbeddingShardingPlanner

from torchrec.distributed.planner.types import Topology
from torchrec.distributed.test_utils.test_model import TestSparseNN
from torchrec.distributed.types import ModuleSharder
from torchrec.modules.embedding_configs import EmbeddingBagConfig

parser = argparse.ArgumentParser(description="custom model for running planner")

parser.add_argument(
    "-lws",
    "--local_world_size",
    type=int,
    default=8,
    help="local_world_size; local world size used in topolgy. Defaults to 8",
    required=False,
)
parser.add_argument(
    "-ws",
    "--world_size",
    type=int,
    default=16,
    help="world_size; number of ranks used in topology. Defaults to 16",
    required=False,
)
parser.add_argument(
    "-bs",
    "--batch_size",
    type=int,
    default=32,
    help="batch_size; batch_size used in topology. Defaults to 32",
    required=False,
)
parser.add_argument(
    "-hc",
    "--hbm_cap",
    type=int,
    default=16777216,
    help="hbm_cap; maximum storage used in topology. Defaults to 1024 * 1024 * 16",
    required=False,
)
parser.add_argument(
    "-cd",
    "--compute_device",
    type=str,
    default="cuda",
    help="compute_device; compute_device used in topology. Defaults to 'cuda'",
    required=False,
)
parser.add_argument(
    "-ne",
    "--num_embeddings",
    type=int,
    default=100,
    help="num_embeddings, number of embeddings used in creating tables. Defaults to 100",
    required=False,
)
parser.add_argument(
    "-ed",
    "--embedding_dim",
    type=int,
    default=64,
    help="embedding_dim: embedding dimension used in creating tables. Defaults to 64",
    required=False,
)
parser.add_argument(
    "-nt",
    "--num_tables",
    type=int,
    default=10,
    help="num_tables: number of tables used in creating tables. Defaults to 10",
    required=False,
)
parser.add_argument(
    "-pt",
    "--planner_type",
    type=str,
    default="parallelized",
    help="embedding_sharding_planner_type: type of embedding sharding planner used in creating a planner"
    "if need to use non_parallelized, type 'non_parallelized', otherwise defaults to parallelized",
    required=False,
)

args: argparse.Namespace = parser.parse_args()

logging.basicConfig(level=logging.INFO)


def main() -> None:
    """
    Generates the sharding plan for a SparseNN model.

    Purpose behind this function is to test planners quickly. This can be done by building the function with custom parameters
    such as local_world_size, num_embeddings, num_tables and more.

    Program outputs planner summary.
    """
    topology = Topology(
        local_world_size=args.local_world_size,
        world_size=args.world_size,
        batch_size=args.batch_size,
        hbm_cap=args.hbm_cap,
        compute_device=args.compute_device,
    )

    if args.embedding_sharding_planner_type == "non_parallelized":
        planner = EmbeddingShardingPlanner(topology=topology)
    else:
        planner = ParallelizedEmbeddingShardingPlanner(topology=topology)

    tables: List[EmbeddingBagConfig] = [
        EmbeddingBagConfig(
            num_embeddings=args.num_embeddings,
            embedding_dim=args.embedding_dim,
            name="table_" + str(i),
            feature_names=["feature_" + str(i)],
        )
        for i in range(args.num_tables)
    ]
    model = TestSparseNN(tables=tables, sparse_device=torch.device("meta"))

    Sharders: List[ModuleSharder[nn.Module]] = [
        cast(ModuleSharder[nn.Module], EmbeddingBagCollectionSharder()),
    ]

    planner.plan(
        module=model,
        sharders=Sharders,
    )


if __name__ == "__main__":
    main()
