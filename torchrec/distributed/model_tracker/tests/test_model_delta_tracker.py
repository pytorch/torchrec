#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import os
import unittest
from dataclasses import dataclass
from typing import cast, Dict, List, Type, Union

import torch
import torchrec
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType

from parameterized import parameterized
from torch import distributed as dist, nn
from torch.testing._internal.common_distributed import MultiProcessTestCase
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


class ModelDeltaTrackerTest(MultiProcessTestCase):
    # pyre-fixme[2]: Parameter must be annotated.
    def __init__(self, methodName="runTest") -> None:
        super().__init__(methodName)

    @property
    def world_size(self) -> int:
        return 2

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    def tearDown(self) -> None:
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def _get_store(self) -> dist.FileStore:
        return dist.FileStore(self.file_name, self.world_size)

    def _get_process_group(self) -> dist.ProcessGroup:
        if torch.cuda.is_available():
            backend = "nccl"
        else:
            backend = "gloo"
        store = self._get_store()
        dist.init_process_group(
            backend, store=store, rank=self.rank, world_size=self.world_size
        )
        return dist.distributed_c10d._get_default_group()

    def _get_models(
        self,
        embedding_config_type: Union[Type[EmbeddingConfig], Type[EmbeddingBagConfig]],
        tables: Dict[str, EmbeddingTableProps],
        optimizer_type: OptimType = OptimType.ADAM,
    ) -> DistributedModelParallel:
        torch.manual_seed(0)

        # Check if CUDA is available before setting device
        if torch.cuda.is_available():
            torch.cuda.set_device(self.rank)
            device = torch.device(f"cuda:{self.rank}")
        else:
            device = torch.device("cpu")

        pg = self._get_process_group()
        test_model = (
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
        planner = EmbeddingShardingPlanner(
            topology=Topology(self.world_size, "cuda"),
            constraints=generate_planner_constraints(tables),
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
                EmbeddingBagCollectionSharder(
                    fused_params={"optimizer": optimizer_type}
                ),
            ),
        ]
        plan = planner.collective_plan(test_model, sharders, pg)
        return DistributedModelParallel(
            module=test_model,
            device=device,
            env=torchrec.distributed.ShardingEnv.from_process_group(pg),
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
        model = self._get_models(
            input_params.embedding_config_type, input_params.embedding_tables
        )
        model_dt = ModelDeltaTracker(model, fqns_to_skip=input_params.fqns_to_skip)
        self.assertEqual(
            model_dt.fqn_to_feature_names(), output_params.expected_fqn_to_feature_names
        )
