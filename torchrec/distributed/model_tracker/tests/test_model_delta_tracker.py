#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import unittest
from dataclasses import dataclass
from typing import cast, Dict, List, Type, Union

import torch
import torchrec
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType

from parameterized import parameterized
from torch import nn
from torchrec.distributed import DistributedModelParallel
from torchrec.distributed.embedding import EmbeddingCollectionSharder
from torchrec.distributed.embedding_types import ModuleSharder, ShardingType
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.model_tracker.model_delta_tracker import ModelDeltaTracker
from torchrec.distributed.model_tracker.tests.utils import (
    EmbeddingTableProps,
    generate_planner_constraints,
    TestEBCModel,
    TestECModel,
)

from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.modules.embedding_configs import (
    EmbeddingBagConfig,
    EmbeddingConfig,
    PoolingType,
)
from torchrec.test_utils import skip_if_asan

NUM_EMBEDDINGS: int = 16
EMBEDDING_DIM: int = 256

HAS_2_GPU: bool = torch.cuda.device_count() >= 2
HAS_1_GPU: bool = torch.cuda.device_count() >= 1


# Helper function to create a model
def get_model(
    rank: int,
    world_size: int,
    ctx: MultiProcessContext,
    embedding_config_type: Union[Type[EmbeddingConfig], Type[EmbeddingBagConfig]],
    embedding_tables: Dict[str, EmbeddingTableProps],
) -> DistributedModelParallel:
    # Create the model
    test_model = (
        TestECModel(
            tables=[
                EmbeddingConfig(
                    name=table_name,
                    embedding_dim=table.embedding_dim,
                    num_embeddings=table.num_embeddings,
                    feature_names=table.feature_names,
                )
                for table_name, table in embedding_tables.items()
            ]
        )
        if embedding_config_type == EmbeddingConfig
        else TestEBCModel(
            tables=[
                EmbeddingBagConfig(
                    name=table_name,
                    embedding_dim=table.embedding_dim,
                    num_embeddings=table.num_embeddings,
                    feature_names=table.feature_names,
                    pooling=table.pooling,
                )
                for table_name, table in embedding_tables.items()
            ]
        )
    )

    # Set up device
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    # Create planner and sharders
    planner = EmbeddingShardingPlanner(
        topology=Topology(world_size, "cuda"),
        constraints=generate_planner_constraints(embedding_tables),
    )
    sharders = [
        cast(
            ModuleSharder[nn.Module],
            EmbeddingCollectionSharder(
                fused_params={
                    "optimizer": OptimType.ADAM,
                    "beta1": 0.9,
                    "beta2": 0.99,
                }
            ),
        ),
        cast(
            ModuleSharder[nn.Module],
            EmbeddingBagCollectionSharder(fused_params={"optimizer": OptimType.ADAM}),
        ),
    ]

    # Create plan
    plan = planner.collective_plan(test_model, sharders, ctx.pg)

    # Create DMP
    if ctx.pg is None:
        raise ValueError("Process group cannot be None")

    return DistributedModelParallel(
        module=test_model,
        device=device,
        env=torchrec.distributed.ShardingEnv.from_process_group(ctx.pg),
        plan=plan,
        sharders=sharders,
    )


@dataclass
class ModelDeltaTrackerInputTestParams:
    # input parameters
    embedding_config_type: Union[Type[EmbeddingConfig], Type[EmbeddingBagConfig]]
    embedding_tables: Dict[str, EmbeddingTableProps]
    fqns_to_skip: List[str]


@dataclass
class FqnToFeatureNamesOutputTestParams:
    # expected output parameters
    expected_fqn_to_feature_names: Dict[str, List[str]]


class ModelDeltaTrackerTest(MultiProcessTestBase):
    # pyre-fixme[2]: Parameter must be annotated.
    def __init__(self, methodName="runTest") -> None:
        super().__init__(methodName)
        self.world_size = 2

    @parameterized.expand(
        [
            (
                "EC_model_test",
                ModelDeltaTrackerInputTestParams(
                    embedding_config_type=EmbeddingConfig,
                    embedding_tables={
                        "sparse_table_1": EmbeddingTableProps(
                            num_embeddings=NUM_EMBEDDINGS,
                            embedding_dim=EMBEDDING_DIM,
                            sharding=ShardingType.ROW_WISE,
                            feature_names=["f1", "f2", "f3"],
                            pooling=PoolingType.NONE,
                        ),
                        "sparse_table_2": EmbeddingTableProps(
                            num_embeddings=NUM_EMBEDDINGS,
                            embedding_dim=EMBEDDING_DIM,
                            sharding=ShardingType.ROW_WISE,
                            feature_names=["f4", "f5", "f6"],
                            pooling=PoolingType.NONE,
                        ),
                    },
                    fqns_to_skip=[],
                ),
                FqnToFeatureNamesOutputTestParams(
                    expected_fqn_to_feature_names={
                        "ec.embeddings.sparse_table_1": ["f1", "f2", "f3"],
                        "ec.embeddings.sparse_table_2": ["f4", "f5", "f6"],
                    },
                ),
            ),
            (
                "EBC_model_test",
                ModelDeltaTrackerInputTestParams(
                    embedding_config_type=EmbeddingBagConfig,
                    embedding_tables={
                        "sparse_table_1": EmbeddingTableProps(
                            num_embeddings=NUM_EMBEDDINGS,
                            embedding_dim=EMBEDDING_DIM,
                            sharding=ShardingType.ROW_WISE,
                            feature_names=["f1", "f2", "f3"],
                            pooling=PoolingType.SUM,
                        ),
                        "sparse_table_2": EmbeddingTableProps(
                            num_embeddings=NUM_EMBEDDINGS,
                            embedding_dim=EMBEDDING_DIM,
                            sharding=ShardingType.ROW_WISE,
                            feature_names=["f4", "f5", "f6"],
                            pooling=PoolingType.SUM,
                        ),
                    },
                    fqns_to_skip=[],
                ),
                FqnToFeatureNamesOutputTestParams(
                    expected_fqn_to_feature_names={
                        "ebc.embedding_bags.sparse_table_1": ["f1", "f2", "f3"],
                        "ebc.embedding_bags.sparse_table_2": ["f4", "f5", "f6"],
                    },
                ),
            ),
            (
                "EC_model_test_with_duplicate_feature_names",
                ModelDeltaTrackerInputTestParams(
                    embedding_config_type=EmbeddingConfig,
                    embedding_tables={
                        "sparse_table_1": EmbeddingTableProps(
                            num_embeddings=NUM_EMBEDDINGS,
                            embedding_dim=EMBEDDING_DIM,
                            sharding=ShardingType.ROW_WISE,
                            feature_names=["f1", "f2", "f3"],
                            pooling=PoolingType.NONE,
                        ),
                        "sparse_table_2": EmbeddingTableProps(
                            num_embeddings=NUM_EMBEDDINGS,
                            embedding_dim=EMBEDDING_DIM,
                            sharding=ShardingType.ROW_WISE,
                            feature_names=["f3", "f4", "f5"],
                            pooling=PoolingType.NONE,
                        ),
                    },
                    fqns_to_skip=[],
                ),
                FqnToFeatureNamesOutputTestParams(
                    expected_fqn_to_feature_names={
                        "ec.embeddings.sparse_table_1": ["f1", "f2", "f3"],
                        "ec.embeddings.sparse_table_2": ["f3", "f4", "f5"],
                    },
                ),
            ),
            (
                "fqns_to_skip_table_name",
                ModelDeltaTrackerInputTestParams(
                    embedding_config_type=EmbeddingBagConfig,
                    embedding_tables={
                        "sparse_table_1": EmbeddingTableProps(
                            num_embeddings=NUM_EMBEDDINGS,
                            embedding_dim=EMBEDDING_DIM,
                            sharding=ShardingType.ROW_WISE,
                            feature_names=["f1", "f2", "f3"],
                            pooling=PoolingType.SUM,
                        ),
                        "sparse_table_2": EmbeddingTableProps(
                            num_embeddings=NUM_EMBEDDINGS,
                            embedding_dim=EMBEDDING_DIM,
                            sharding=ShardingType.ROW_WISE,
                            feature_names=["f4", "f5", "f6"],
                            pooling=PoolingType.SUM,
                        ),
                    },
                    fqns_to_skip=["sparse_table_1"],
                ),
                FqnToFeatureNamesOutputTestParams(
                    expected_fqn_to_feature_names={
                        "ebc.embedding_bags.sparse_table_2": ["f4", "f5", "f6"],
                    },
                ),
            ),
            (
                "fqns_to_skip_mid_fqn",
                ModelDeltaTrackerInputTestParams(
                    embedding_config_type=EmbeddingBagConfig,
                    embedding_tables={
                        "sparse_table_1": EmbeddingTableProps(
                            num_embeddings=NUM_EMBEDDINGS,
                            embedding_dim=EMBEDDING_DIM,
                            sharding=ShardingType.ROW_WISE,
                            feature_names=["f1", "f2", "f3"],
                            pooling=PoolingType.SUM,
                        ),
                        "sparse_table_2": EmbeddingTableProps(
                            num_embeddings=NUM_EMBEDDINGS,
                            embedding_dim=EMBEDDING_DIM,
                            sharding=ShardingType.ROW_WISE,
                            feature_names=["f4", "f5", "f6"],
                            pooling=PoolingType.SUM,
                        ),
                    },
                    fqns_to_skip=["embedding_bags"],
                ),
                FqnToFeatureNamesOutputTestParams(
                    expected_fqn_to_feature_names={},
                ),
            ),
            (
                "fqns_to_skip_parent_fqn",
                ModelDeltaTrackerInputTestParams(
                    embedding_config_type=EmbeddingConfig,
                    embedding_tables={
                        "sparse_table_1": EmbeddingTableProps(
                            num_embeddings=NUM_EMBEDDINGS,
                            embedding_dim=EMBEDDING_DIM,
                            sharding=ShardingType.ROW_WISE,
                            feature_names=["f1", "f2", "f3"],
                            pooling=PoolingType.NONE,
                        ),
                        "sparse_table_2": EmbeddingTableProps(
                            num_embeddings=NUM_EMBEDDINGS,
                            embedding_dim=EMBEDDING_DIM,
                            sharding=ShardingType.ROW_WISE,
                            feature_names=["f3", "f4", "f5"],
                            pooling=PoolingType.NONE,
                        ),
                    },
                    fqns_to_skip=["ec"],
                ),
                FqnToFeatureNamesOutputTestParams(
                    expected_fqn_to_feature_names={},
                ),
            ),
        ]
    )
    @skip_if_asan
    @unittest.skipUnless(HAS_1_GPU, reason="Test requires at least 1 GPU")
    def test_fqn_to_feature_names(
        self,
        _test_name: str,
        input_params: ModelDeltaTrackerInputTestParams,
        output_params: FqnToFeatureNamesOutputTestParams,
    ) -> None:
        self._run_multi_process_test(
            callable=_test_fqn_to_feature_names,
            world_size=self.world_size,
            input_params=input_params,
            output_params=output_params,
        )


def _test_fqn_to_feature_names(
    rank: int,
    world_size: int,
    input_params: ModelDeltaTrackerInputTestParams,
    output_params: FqnToFeatureNamesOutputTestParams,
) -> None:
    with MultiProcessContext(
        rank=rank,
        world_size=world_size,
        backend="nccl" if torch.cuda.is_available() else "gloo",
    ) as ctx:
        # Get the model using the helper function
        model = get_model(
            rank=rank,
            world_size=world_size,
            ctx=ctx,
            embedding_config_type=input_params.embedding_config_type,
            embedding_tables=input_params.embedding_tables,
        )

        model_dt = ModelDeltaTracker(model, fqns_to_skip=input_params.fqns_to_skip)
        actual_fqn_to_feature_names = model_dt.fqn_to_feature_names()

        unittest.TestCase().assertEqual(
            actual_fqn_to_feature_names,
            output_params.expected_fqn_to_feature_names,
        )
