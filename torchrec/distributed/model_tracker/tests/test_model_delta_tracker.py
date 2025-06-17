#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import unittest
from dataclasses import dataclass, field
from typing import cast, Dict, List, Optional, Tuple, Type, Union

import torch
import torchrec
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType

from parameterized import parameterized
from torch import nn
from torchrec import KeyedJaggedTensor
from torchrec.distributed import DistributedModelParallel
from torchrec.distributed.embedding import EmbeddingCollectionSharder
from torchrec.distributed.embedding_types import ModuleSharder, ShardingType
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.model_tracker.tests.utils import (
    EmbeddingTableProps,
    generate_planner_constraints,
    TestEBCModel,
    TestECModel,
)
from torchrec.distributed.model_tracker.types import ModelTrackerConfig, TrackingMode

from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.utils import none_throws
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


def generate_test_models(
    embedding_config_type: Union[Type[EmbeddingConfig], Type[EmbeddingBagConfig]],
    tables: Dict[str, EmbeddingTableProps],
) -> nn.Module:
    return (
        TestECModel(
            tables=[
                EmbeddingConfig(
                    name=table_name,
                    embedding_dim=table.embedding_dim,
                    num_embeddings=table.num_embeddings,
                    feature_names=table.feature_names,
                )
                for table_name, table in tables.items()
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
                for table_name, table in tables.items()
            ]
        )
    )


# input parameters for KJT generation
@dataclass
class ModelInput:
    keys: List[str]
    values: torch.Tensor
    offsets: torch.Tensor


@dataclass
class ModelDeltaTrackerInputTestParams:
    # input parameters
    embedding_config_type: Union[Type[EmbeddingConfig], Type[EmbeddingBagConfig]]
    embedding_tables: Dict[str, EmbeddingTableProps]
    fqns_to_skip: List[str] = field(default_factory=list)
    model_inputs: List[ModelInput] = field(default_factory=list)
    model_tracker_config: Optional[ModelTrackerConfig] = None


@dataclass
class FqnToFeatureNamesOutputTestParams:
    # expected output parameters
    expected_fqn_to_feature_names: Dict[str, List[str]]


@dataclass
class TrackerNotInitOutputTestParams:
    # DMP tracker Attribute
    dmp_tracker_atter: str


@dataclass
class EmbeddingModeOutputTestParams:
    # assert string
    assert_str: Optional[str]


def model_input_generator(
    model_inputs: List[ModelInput], rank: int
) -> List[KeyedJaggedTensor]:
    return [
        KeyedJaggedTensor.from_offsets_sync(
            model_input.keys, model_input.values + rank, model_input.offsets
        ).to(torch.device(f"cuda:{rank}"))
        for model_input in model_inputs
    ]


# Helper function to create a model
def get_models(
    rank: int,
    world_size: int,
    ctx: MultiProcessContext,
    embedding_config_type: Union[Type[EmbeddingConfig], Type[EmbeddingBagConfig]],
    embedding_tables: Dict[str, EmbeddingTableProps],
    optimizer_type: OptimType = OptimType.ADAM,
    config: Optional[ModelTrackerConfig] = None,
) -> Tuple[DistributedModelParallel, DistributedModelParallel]:
    # Create the model
    torch.manual_seed(0)
    test_model = generate_test_models(embedding_config_type, embedding_tables)

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
                    "optimizer": optimizer_type,
                    "beta1": 0.9,
                    "beta2": 0.99,
                }
            ),
        ),
        cast(
            ModuleSharder[nn.Module],
            EmbeddingBagCollectionSharder(fused_params={"optimizer": optimizer_type}),
        ),
    ]

    # Create plan
    plan = planner.collective_plan(test_model, sharders, ctx.pg)

    # Create DMP
    if ctx.pg is None:
        raise ValueError("Process group cannot be None")

    dt_dmp = DistributedModelParallel(
        module=test_model,
        device=device,
        env=torchrec.distributed.ShardingEnv.from_process_group(ctx.pg),
        plan=plan,
        sharders=sharders,
        model_tracker_config=config,
    )

    torch.manual_seed(0)
    baseline_module = generate_test_models(embedding_config_type, embedding_tables)
    baseline_dmp = DistributedModelParallel(
        module=baseline_module,
        device=device,
        # pyre-ignore[6]
        env=torchrec.distributed.ShardingEnv.from_process_group(ctx.pg),
        plan=plan,
        sharders=sharders,
    )

    return dt_dmp, baseline_dmp


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

    @parameterized.expand(
        [
            (
                "get_model_tracker",
                ModelDeltaTrackerInputTestParams(
                    embedding_config_type=EmbeddingConfig,
                    embedding_tables={
                        "table_fqn_1": EmbeddingTableProps(
                            num_embeddings=NUM_EMBEDDINGS,
                            embedding_dim=EMBEDDING_DIM,
                            sharding=ShardingType.ROW_WISE,
                            feature_names=["f1", "f2", "f3"],
                            pooling=PoolingType.NONE,
                        ),
                    },
                    fqns_to_skip=[],
                ),
                TrackerNotInitOutputTestParams(
                    dmp_tracker_atter="get_model_tracker",
                ),
            ),
            (
                "get_delta",
                ModelDeltaTrackerInputTestParams(
                    embedding_config_type=EmbeddingConfig,
                    embedding_tables={
                        "table_fqn_1": EmbeddingTableProps(
                            num_embeddings=NUM_EMBEDDINGS,
                            embedding_dim=EMBEDDING_DIM,
                            sharding=ShardingType.ROW_WISE,
                            feature_names=["f1", "f2", "f3"],
                            pooling=PoolingType.NONE,
                        ),
                    },
                    fqns_to_skip=[],
                ),
                TrackerNotInitOutputTestParams(
                    dmp_tracker_atter="get_delta",
                ),
            ),
        ]
    )
    @skip_if_asan
    @unittest.skipUnless(HAS_1_GPU, reason="Test requires at least 1 GPU")
    def test_tracker_not_initialized(
        self,
        _test_name: str,
        input_params: ModelDeltaTrackerInputTestParams,
        output_params: TrackerNotInitOutputTestParams,
    ) -> None:
        self._run_multi_process_test(
            callable=_test_tracker_init,
            world_size=self.world_size,
            input_params=input_params,
            output_params=output_params,
        )

    @parameterized.expand(
        [
            (
                "test_dup_with_EC_and_default_consumer",
                ModelDeltaTrackerInputTestParams(
                    embedding_config_type=EmbeddingConfig,
                    embedding_tables={
                        "sparse_table_1": EmbeddingTableProps(
                            num_embeddings=NUM_EMBEDDINGS,
                            embedding_dim=EMBEDDING_DIM,
                            sharding=ShardingType.ROW_WISE,
                            feature_names=["f1"],
                            pooling=PoolingType.NONE,
                        ),
                        "sparse_table_2": EmbeddingTableProps(
                            num_embeddings=NUM_EMBEDDINGS,
                            embedding_dim=EMBEDDING_DIM,
                            sharding=ShardingType.ROW_WISE,
                            feature_names=["f2"],
                            pooling=PoolingType.NONE,
                        ),
                    },
                    model_tracker_config=ModelTrackerConfig(
                        tracking_mode=TrackingMode.ID_ONLY,
                        delete_on_read=True,
                    ),
                    model_inputs=[
                        ModelInput(
                            keys=["f1", "f2"],
                            values=torch.tensor([0, 2, 4, 6, 8, 10, 12, 14]),
                            offsets=torch.tensor([0, 2, 2, 4, 6, 6, 8]),
                        ),
                        ModelInput(
                            keys=["f1", "f2"],
                            values=torch.tensor([8, 10, 12, 14, 0, 2, 4, 6]),
                            offsets=torch.tensor([0, 2, 2, 4, 6, 6, 8]),
                        ),
                        ModelInput(
                            keys=["f1", "f2"],
                            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
                            offsets=torch.tensor([0, 2, 2, 4, 6, 6, 8]),
                        ),
                    ],
                ),
            ),
            (
                "test_dup_with_EBC_and_default_consumer",
                ModelDeltaTrackerInputTestParams(
                    embedding_config_type=EmbeddingBagConfig,
                    embedding_tables={
                        "sparse_table_1": EmbeddingTableProps(
                            num_embeddings=NUM_EMBEDDINGS,
                            embedding_dim=EMBEDDING_DIM,
                            sharding=ShardingType.ROW_WISE,
                            feature_names=["f1"],
                            pooling=PoolingType.SUM,
                        ),
                        "sparse_table_2": EmbeddingTableProps(
                            num_embeddings=NUM_EMBEDDINGS,
                            embedding_dim=EMBEDDING_DIM,
                            sharding=ShardingType.ROW_WISE,
                            feature_names=["f2"],
                            pooling=PoolingType.SUM,
                        ),
                    },
                    model_tracker_config=ModelTrackerConfig(
                        tracking_mode=TrackingMode.ID_ONLY,
                        delete_on_read=True,
                    ),
                    model_inputs=[
                        ModelInput(
                            keys=["f1", "f2"],
                            values=torch.tensor([0, 2, 4, 6, 8, 10, 12, 14]),
                            offsets=torch.tensor([0, 2, 2, 4, 6, 6, 8]),
                        ),
                        ModelInput(
                            keys=["f1", "f2"],
                            values=torch.tensor([8, 10, 12, 14, 0, 2, 4, 6]),
                            offsets=torch.tensor([0, 2, 2, 4, 6, 6, 8]),
                        ),
                        ModelInput(
                            keys=["f1", "f2"],
                            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
                            offsets=torch.tensor([0, 2, 2, 4, 6, 6, 8]),
                        ),
                    ],
                ),
            ),
            (
                "test_multi_feature_per_table_EC",
                ModelDeltaTrackerInputTestParams(
                    embedding_config_type=EmbeddingConfig,
                    embedding_tables={
                        "sparse_table_1": EmbeddingTableProps(
                            num_embeddings=NUM_EMBEDDINGS,
                            embedding_dim=EMBEDDING_DIM,
                            sharding=ShardingType.ROW_WISE,
                            feature_names=["f1", "f2"],
                            pooling=PoolingType.NONE,
                        ),
                        "sparse_table_2": EmbeddingTableProps(
                            num_embeddings=NUM_EMBEDDINGS,
                            embedding_dim=EMBEDDING_DIM,
                            sharding=ShardingType.ROW_WISE,
                            feature_names=["f3", "f4"],
                            pooling=PoolingType.NONE,
                        ),
                    },
                    model_tracker_config=ModelTrackerConfig(
                        tracking_mode=TrackingMode.ID_ONLY,
                        delete_on_read=True,
                    ),
                    model_inputs=[
                        ModelInput(
                            keys=["f1", "f2", "f3", "f4"],
                            values=torch.tensor(
                                [0, 2, 4, 6, 0, 2, 4, 6, 8, 10, 12, 14, 8, 10, 12, 14]
                            ),
                            offsets=torch.tensor(
                                [0, 2, 2, 4, 6, 7, 8, 8, 10, 12, 15, 15, 16]
                            ),
                        ),
                    ],
                ),
            ),
            (
                "test_multi_feature_per_table_EBC",
                ModelDeltaTrackerInputTestParams(
                    embedding_config_type=EmbeddingBagConfig,
                    embedding_tables={
                        "sparse_table_1": EmbeddingTableProps(
                            num_embeddings=NUM_EMBEDDINGS,
                            embedding_dim=EMBEDDING_DIM,
                            sharding=ShardingType.ROW_WISE,
                            feature_names=["f1", "f2"],
                            pooling=PoolingType.SUM,
                        ),
                        "sparse_table_2": EmbeddingTableProps(
                            num_embeddings=NUM_EMBEDDINGS,
                            embedding_dim=EMBEDDING_DIM,
                            sharding=ShardingType.ROW_WISE,
                            feature_names=["f3", "f4"],
                            pooling=PoolingType.SUM,
                        ),
                    },
                    model_tracker_config=ModelTrackerConfig(
                        tracking_mode=TrackingMode.ID_ONLY,
                        delete_on_read=True,
                    ),
                    model_inputs=[
                        ModelInput(
                            keys=["f1", "f2", "f3", "f4"],
                            values=torch.tensor(
                                [0, 2, 4, 6, 0, 2, 4, 6, 8, 10, 12, 14, 8, 10, 12, 14]
                            ),
                            offsets=torch.tensor(
                                [0, 2, 2, 4, 6, 7, 8, 8, 10, 12, 15, 15, 16]
                            ),
                        ),
                    ],
                ),
            ),
        ]
    )
    @skip_if_asan
    @unittest.skipUnless(HAS_2_GPU, reason="Distributed test requires at least 2 GPUs")
    def test_tracker_id_mode(
        self,
        _test_name: str,
        test_params: ModelDeltaTrackerInputTestParams,
    ) -> None:
        self._run_multi_process_test(
            callable=_test_id_mode,
            world_size=self.world_size,
            test_params=test_params,
        )

    @parameterized.expand(
        [
            (
                "test_dup_with_EC_and_default_consumer",
                ModelDeltaTrackerInputTestParams(
                    embedding_config_type=EmbeddingConfig,
                    embedding_tables={
                        "sparse_table_1": EmbeddingTableProps(
                            num_embeddings=NUM_EMBEDDINGS,
                            embedding_dim=EMBEDDING_DIM,
                            sharding=ShardingType.ROW_WISE,
                            feature_names=["f1"],
                            pooling=PoolingType.NONE,
                        ),
                        "sparse_table_2": EmbeddingTableProps(
                            num_embeddings=NUM_EMBEDDINGS,
                            embedding_dim=EMBEDDING_DIM,
                            sharding=ShardingType.ROW_WISE,
                            feature_names=["f2"],
                            pooling=PoolingType.NONE,
                        ),
                    },
                    model_tracker_config=ModelTrackerConfig(
                        tracking_mode=TrackingMode.EMBEDDING,
                        delete_on_read=True,
                    ),
                    model_inputs=[
                        ModelInput(
                            keys=["f1", "f2"],
                            values=torch.tensor([0, 2, 4, 6, 8, 10, 12, 14]),
                            offsets=torch.tensor([0, 2, 2, 4, 6, 6, 8]),
                        ),
                        ModelInput(
                            keys=["f1", "f2"],
                            values=torch.tensor([8, 10, 12, 14, 0, 2, 4, 6]),
                            offsets=torch.tensor([0, 2, 2, 4, 6, 6, 8]),
                        ),
                        ModelInput(
                            keys=["f1", "f2"],
                            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
                            offsets=torch.tensor([0, 2, 2, 4, 6, 6, 8]),
                        ),
                    ],
                ),
                EmbeddingModeOutputTestParams(assert_str=None),
            ),
            (
                "test_multi_feature_per_table_EC",
                ModelDeltaTrackerInputTestParams(
                    embedding_config_type=EmbeddingConfig,
                    embedding_tables={
                        "sparse_table_1": EmbeddingTableProps(
                            num_embeddings=NUM_EMBEDDINGS,
                            embedding_dim=EMBEDDING_DIM,
                            sharding=ShardingType.ROW_WISE,
                            feature_names=["f1", "f2"],
                            pooling=PoolingType.NONE,
                        ),
                        "sparse_table_2": EmbeddingTableProps(
                            num_embeddings=NUM_EMBEDDINGS,
                            embedding_dim=EMBEDDING_DIM,
                            sharding=ShardingType.ROW_WISE,
                            feature_names=["f3", "f4"],
                            pooling=PoolingType.NONE,
                        ),
                    },
                    model_tracker_config=ModelTrackerConfig(
                        tracking_mode=TrackingMode.EMBEDDING,
                        delete_on_read=True,
                    ),
                    model_inputs=[
                        ModelInput(
                            keys=["f1", "f2", "f3", "f4"],
                            values=torch.tensor(
                                [0, 2, 4, 6, 0, 2, 4, 6, 8, 10, 12, 14, 8, 10, 12, 14]
                            ),
                            offsets=torch.tensor(
                                [0, 2, 2, 4, 6, 7, 8, 8, 10, 12, 15, 15, 16]
                            ),
                        ),
                    ],
                ),
                EmbeddingModeOutputTestParams(assert_str=None),
            ),
            # We don't support tracking of raw ids for EBC yet. This test validates that.
            (
                "assert_on_validation",
                ModelDeltaTrackerInputTestParams(
                    embedding_config_type=EmbeddingBagConfig,
                    embedding_tables={
                        "sparse_table_1": EmbeddingTableProps(
                            num_embeddings=NUM_EMBEDDINGS,
                            embedding_dim=EMBEDDING_DIM,
                            sharding=ShardingType.ROW_WISE,
                            feature_names=["f1"],
                            pooling=PoolingType.SUM,
                        ),
                        "sparse_table_2": EmbeddingTableProps(
                            num_embeddings=NUM_EMBEDDINGS,
                            embedding_dim=EMBEDDING_DIM,
                            sharding=ShardingType.ROW_WISE,
                            feature_names=["f2"],
                            pooling=PoolingType.SUM,
                        ),
                    },
                    model_tracker_config=ModelTrackerConfig(
                        tracking_mode=TrackingMode.EMBEDDING,
                        delete_on_read=True,
                    ),
                    model_inputs=[
                        ModelInput(
                            keys=["f1", "f2"],
                            values=torch.tensor([0, 2, 4, 6, 8, 10, 12, 14]),
                            offsets=torch.tensor([0, 2, 2, 4, 6, 6, 8]),
                        ),
                        ModelInput(
                            keys=["f1", "f2"],
                            values=torch.tensor([8, 10, 12, 14, 0, 2, 4, 6]),
                            offsets=torch.tensor([0, 2, 2, 4, 6, 6, 8]),
                        ),
                        ModelInput(
                            keys=["f1", "f2"],
                            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
                            offsets=torch.tensor([0, 2, 2, 4, 6, 6, 8]),
                        ),
                    ],
                ),
                EmbeddingModeOutputTestParams(
                    assert_str="EBC's lookup returns pooled embeddings and currently, we do not support tracking raw embeddings."
                ),
            ),
        ]
    )
    @skip_if_asan
    @unittest.skipUnless(HAS_2_GPU, reason="Distributed test requires at least 2 GPUs")
    def test_tracker_embedding_mode(
        self,
        _test_name: str,
        test_params: ModelDeltaTrackerInputTestParams,
        output_params: EmbeddingModeOutputTestParams,
    ) -> None:
        self._run_multi_process_test(
            callable=_test_embedding_mode,
            world_size=self.world_size,
            test_params=test_params,
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
        dt_model, _ = get_models(
            rank=rank,
            world_size=world_size,
            ctx=ctx,
            embedding_config_type=input_params.embedding_config_type,
            embedding_tables=input_params.embedding_tables,
            config=ModelTrackerConfig(
                tracking_mode=TrackingMode.ID_ONLY,
                delete_on_read=True,
                fqns_to_skip=input_params.fqns_to_skip,
            ),
        )

        dt = dt_model.get_model_tracker()
        unittest.TestCase().assertEqual(
            dt.fqn_to_feature_names(), output_params.expected_fqn_to_feature_names
        )


def _test_tracker_init(
    rank: int,
    world_size: int,
    input_params: ModelDeltaTrackerInputTestParams,
    output_params: TrackerNotInitOutputTestParams,
) -> None:
    with MultiProcessContext(
        rank=rank,
        world_size=world_size,
        backend="nccl" if torch.cuda.is_available() else "gloo",
    ) as ctx:
        # Get the model using the helper function
        dt_model, _ = get_models(
            rank=rank,
            world_size=world_size,
            ctx=ctx,
            embedding_config_type=input_params.embedding_config_type,
            embedding_tables=input_params.embedding_tables,
            config=None,
        )
        with unittest.TestCase().assertRaisesRegex(
            AssertionError,
            "Model tracker is not initialized. Add ModelTrackerConfig at DistributedModelParallel init.",
        ):
            getattr(dt_model, output_params.dmp_tracker_atter)()


def _test_id_mode(
    rank: int,
    world_size: int,
    test_params: ModelDeltaTrackerInputTestParams,
) -> None:
    with MultiProcessContext(
        rank=rank,
        world_size=world_size,
        backend="nccl" if torch.cuda.is_available() else "gloo",
    ) as ctx:
        # Get the model using the helper function
        dt_model, baseline_model = get_models(
            rank=rank,
            world_size=world_size,
            ctx=ctx,
            embedding_config_type=test_params.embedding_config_type,
            embedding_tables=test_params.embedding_tables,
            config=test_params.model_tracker_config,
        )
        features_list = model_input_generator(test_params.model_inputs, rank)
        dt = dt_model.get_model_tracker()
        for features in features_list:
            tracked_out = dt_model(features)
            baseline_out = baseline_model(features)
            unittest.TestCase().assertTrue(tracked_out.allclose(baseline_out))
            tracked_out.sum().backward()
            baseline_out.sum().backward()

        delta_ids = dt.get_delta_ids()

        table_fqns = dt.fqn_to_feature_names().keys()

        # Check if any table has multiple features
        has_multi_feature_tables = any(
            len(dt.fqn_to_feature_names()[fqn]) > 1 for fqn in table_fqns
        )

        table_fqns_list = list(table_fqns)

        if has_multi_feature_tables:
            # For multi-feature tables, each rank is responsible for one table
            # Rank 0 handles the first table, Rank 1 handles the second table
            if rank == 0:
                # Rank 0: First table has IDs, second table is empty
                unittest.TestCase().assertTrue(
                    delta_ids[table_fqns_list[0]].allclose(
                        torch.tensor(range(8), device=torch.device(f"cuda:{rank}"))
                    )
                )
                unittest.TestCase().assertEqual(
                    0, delta_ids[table_fqns_list[1]].numel()
                )
            elif rank == 1:
                # Rank 1: Second table has IDs, first table is empty
                unittest.TestCase().assertEqual(
                    0, delta_ids[table_fqns_list[0]].numel()
                )
                unittest.TestCase().assertTrue(
                    delta_ids[table_fqns_list[1]].allclose(
                        torch.tensor(range(8), device=torch.device(f"cuda:{rank}"))
                    )
                )
        else:
            # For single-feature tables, all tables have IDs on all ranks
            for table_fqn in table_fqns:
                unittest.TestCase().assertTrue(
                    delta_ids[table_fqn].allclose(
                        torch.tensor(range(8), device=torch.device(f"cuda:{rank}"))
                    ),
                    f"Table {table_fqn} on rank {rank} should have IDs",
                )


def _test_embedding_mode(
    rank: int,
    world_size: int,
    test_params: ModelDeltaTrackerInputTestParams,
    output_params: EmbeddingModeOutputTestParams,
) -> None:

    with MultiProcessContext(
        rank=rank,
        world_size=world_size,
        backend="nccl" if torch.cuda.is_available() else "gloo",
    ) as ctx:
        # Initialize variables to None
        dt_model = None
        baseline_model = None

        if output_params.assert_str is not None:
            with unittest.TestCase().assertRaisesRegex(
                AssertionError,
                # pyre-ignore[6]
                output_params.assert_str,
            ):
                dt_model, baseline_model = get_models(
                    rank=rank,
                    world_size=world_size,
                    ctx=ctx,
                    embedding_config_type=test_params.embedding_config_type,
                    embedding_tables=test_params.embedding_tables,
                    config=test_params.model_tracker_config,
                )
        else:
            dt_model, baseline_model = get_models(
                rank=rank,
                world_size=world_size,
                ctx=ctx,
                embedding_config_type=test_params.embedding_config_type,
                embedding_tables=test_params.embedding_tables,
                config=test_params.model_tracker_config,
            )

            # Only proceed with the rest of the test if models were created successfully
            features_list = model_input_generator(test_params.model_inputs, rank)
            dt = dt_model.get_model_tracker()

            orig_emb1 = (
                # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute `ec`.
                dt_model._dmp_wrapped_module.module.ec.embeddings.sparse_table_1.weight.detach().clone()
            )
            orig_emb2 = (
                # pyre-fixme[16]: Item `Tensor` of `Tensor | Module` has no attribute `ec`.
                dt_model._dmp_wrapped_module.module.ec.embeddings.sparse_table_2.weight.detach().clone()
            )

            for features in features_list:
                tracked_out = dt_model(features)
                baseline_out = baseline_model(features)
                unittest.TestCase().assertTrue(tracked_out.allclose(baseline_out))
                tracked_out.sum().backward()
                baseline_out.sum().backward()

            delta_rows = dt.get_delta()

            table_fqns = dt.fqn_to_feature_names().keys()
            table_fqns_list = list(table_fqns)

            # Check if any table has multiple features
            has_multi_feature_tables = any(
                len(dt.fqn_to_feature_names()[fqn]) > 1 for fqn in table_fqns
            )
            if has_multi_feature_tables:
                if rank == 0:
                    # Rank 0: First table has IDs and embeddings, second table is empty
                    expected_ids = torch.tensor(
                        range(8), device=torch.device(f"cuda:{rank}")
                    )
                    unittest.TestCase().assertTrue(
                        delta_rows[table_fqns_list[0]].ids.allclose(expected_ids)
                    )
                    unittest.TestCase().assertTrue(
                        none_throws(delta_rows[table_fqns_list[0]].embeddings).allclose(
                            orig_emb1[expected_ids]
                        )
                    )
                    # Second table should be empty
                    unittest.TestCase().assertEqual(
                        0, delta_rows[table_fqns_list[1]].ids.numel()
                    )
                    unittest.TestCase().assertTrue(
                        delta_rows[table_fqns_list[1]].embeddings is not None
                        # pyre-ignore[16]:
                        and delta_rows[table_fqns_list[1]].embeddings.numel() == 0,
                    )
                elif rank == 1:
                    # Rank 1: Second table has IDs and embeddings, first table is empty
                    expected_ids = torch.tensor(
                        range(8), device=torch.device(f"cuda:{rank}")
                    )
                    unittest.TestCase().assertTrue(
                        delta_rows[table_fqns_list[1]].ids.allclose(expected_ids)
                    )
                    unittest.TestCase().assertTrue(
                        none_throws(delta_rows[table_fqns_list[1]].embeddings).allclose(
                            orig_emb2[expected_ids]
                        )
                    )
                    # First table should be empty
                    unittest.TestCase().assertEqual(
                        0, delta_rows[table_fqns_list[0]].ids.numel()
                    )
                    unittest.TestCase().assertTrue(
                        delta_rows[table_fqns_list[0]].embeddings is not None
                        and delta_rows[table_fqns_list[0]].embeddings.numel() == 0,
                    )

            else:
                # For single-feature tables, all tables have IDs and embeddings on all ranks
                for table_fqn, orig_emb in zip(table_fqns, [orig_emb1, orig_emb2]):
                    expected_ids = torch.tensor(
                        range(8), device=torch.device(f"cuda:{rank}")
                    )
                    unittest.TestCase().assertTrue(
                        delta_rows[table_fqn].ids.allclose(expected_ids)
                    )
                    unittest.TestCase().assertTrue(
                        none_throws(delta_rows[table_fqn].embeddings).allclose(
                            orig_emb[expected_ids]
                        )
                    )
