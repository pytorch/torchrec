#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import Dict, List, Optional, cast, Union

from fbgemm_gpu.split_embedding_configs import EmbOptimType
from torch import nn
from torchrec.distributed.planner import ParameterConstraints
from torchrec.distributed.test_utils.test_model import (
    TestSparseNN,
    TestSparseNNBase,
    TestEBCSharder,
    TestEBSharder,
)
from torchrec.distributed.test_utils.test_model_parallel_base import (
    ModelParallelTestBase,
)
from torchrec.distributed.types import ModuleSharder
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.test_utils import seed_and_log


class SharderType(Enum):
    EMBEDDING_BAG = "embedding_bag"
    EMBEDDING_BAG_COLLECTION = "embedding_bag_collection"


def create_test_sharder(
    sharder_type: str, sharding_type: str, kernel_type: str
) -> Union[TestEBSharder, TestEBCSharder]:
    if sharder_type == SharderType.EMBEDDING_BAG.value:
        return TestEBSharder(sharding_type, kernel_type, {"learning_rate": 0.1})
    elif sharder_type == SharderType.EMBEDDING_BAG_COLLECTION.value:
        return TestEBCSharder(sharding_type, kernel_type, {"learning_rate": 0.1})
    else:
        raise ValueError(f"Sharder not supported {sharder_type}")


class ModelParallelTestShared(ModelParallelTestBase):
    @seed_and_log
    def setUp(self) -> None:
        super().setUp()

        num_features = 4
        num_weighted_features = 2

        self.tables = [
            EmbeddingBagConfig(
                num_embeddings=(i + 1) * 10,
                embedding_dim=(i + 2) * 4,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(num_features)
        ]
        self.weighted_tables = [
            EmbeddingBagConfig(
                num_embeddings=(i + 1) * 10,
                embedding_dim=(i + 2) * 4,
                name="weighted_table_" + str(i),
                feature_names=["weighted_feature_" + str(i)],
            )
            for i in range(num_weighted_features)
        ]

        self.embedding_groups = {
            "group_0": ["feature_" + str(i) for i in range(num_features)]
        }

    def _test_sharding(
        self,
        sharders: List[ModuleSharder[nn.Module]],
        backend: str = "gloo",
        world_size: int = 2,
        local_size: Optional[int] = None,
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
    ) -> None:
        self._run_multi_process_test(
            # pyre-ignore [6]
            callable=self._test_sharding_single_rank,
            world_size=world_size,
            local_size=local_size,
            model_class=cast(TestSparseNNBase, TestSparseNN),
            tables=self.tables,
            weighted_tables=self.weighted_tables,
            embedding_groups=self.embedding_groups,
            sharders=sharders,
            backend=backend,
            optim=EmbOptimType.EXACT_SGD,
            constraints=constraints,
        )
