#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Dict, List, Optional

import hypothesis.strategies as st
import torch
import torch.nn as nn
from hypothesis import given, settings, Verbosity
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner import (
    EmbeddingShardingPlanner,
    ParameterConstraints,
    Topology,
)
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)

from torchrec.distributed.test_utils.test_model import TestFusedEBCSharder
from torchrec.distributed.test_utils.test_sharding import copy_state_dict, SharderType
from torchrec.distributed.types import (
    ModuleSharder,
    ShardingEnv,
    ShardingPlan,
    ShardingType,
)
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.modules.fused_embedding_modules import (
    fuse_embedding_optimizer,
    FusedEmbeddingBagCollection,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.test_utils import skip_if_asan_class


def sharding_single_rank(
    rank: int,
    world_size: int,
    unsharded_model: nn.Module,
    kjt_input: KeyedJaggedTensor,
    sharders: List[ModuleSharder[nn.Module]],
    backend: str,
    constraints: Optional[Dict[str, ParameterConstraints]] = None,
    local_size: Optional[int] = None,
) -> None:

    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        kjt_input = kjt_input.to(ctx.device)
        unsharded_model = unsharded_model.to(ctx.device)

        # Shard model.
        planner = EmbeddingShardingPlanner(
            topology=Topology(
                world_size, ctx.device.type, local_world_size=ctx.local_size
            ),
            constraints=constraints,
        )
        plan: ShardingPlan = planner.collective_plan(unsharded_model, sharders, ctx.pg)

        sharded_model = DistributedModelParallel(
            unsharded_model,
            env=ShardingEnv.from_process_group(ctx.pg),
            plan=plan,
            sharders=sharders,
            device=ctx.device,
        )

        # Load model state from the global model.
        copy_state_dict(sharded_model.state_dict(), unsharded_model.state_dict())

        # cast to CPU because when casting unsharded_model.to on the same module, there could some race conditions
        # in normal author modelling code this won't be an issue because each rank would individually create
        # their model. output from sharded_pred is correctly on the correct device.
        unsharded_model_pred = (
            unsharded_model(kjt_input).values().detach().clone().cpu()
        )
        sharded_pred = sharded_model(kjt_input).values().detach().clone().cpu()

        # Compare predictions of sharded vs unsharded models.
        torch.testing.assert_close(sharded_pred, unsharded_model_pred)


@skip_if_asan_class
class FusedEmbeddingBagCollectionParallelTest(MultiProcessTestBase):
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    # pyre-fixme[56]
    @given(
        sharder_type=st.sampled_from(
            [
                SharderType.EMBEDDING_BAG_COLLECTION.value,
            ]
        ),
        sharding_type=st.sampled_from(
            [
                ShardingType.TABLE_WISE.value,
                ShardingType.ROW_WISE.value,
                ShardingType.COLUMN_WISE.value,
                # ShardingType.DATA_PARALLEL.value,
                # Data parallel checkpointing not yet supported
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=2, deadline=None)
    def test_sharding_fused_ebc(
        self,
        sharder_type: str,
        sharding_type: str,
    ) -> None:

        fused_ebc = FusedEmbeddingBagCollection(
            tables=[
                EmbeddingBagConfig(
                    name="table_0",
                    feature_names=["feature_0", "feature_1"],
                    embedding_dim=8,
                    num_embeddings=10,
                )
            ],
            optimizer_type=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.02},
            device=torch.device("cuda"),
        )

        #             instance 0   instance 1  instance 2
        # "feature_0"   [0, 1]       None        [2]
        # "feature_1"   [3]          [4]         [5,6,7]
        #

        kjt_input = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0", "feature_1"],
            values=torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7]),
            lengths=torch.LongTensor([2, 0, 1, 1, 1, 3]),
        )

        self._run_multi_process_test(
            callable=sharding_single_rank,
            world_size=2,
            unsharded_model=fused_ebc,
            kjt_input=kjt_input,
            sharders=[TestFusedEBCSharder(sharding_type=sharding_type)],
            backend="nccl",
        )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    # pyre-fixme[56]
    @given(
        sharding_type=st.sampled_from(
            [
                ShardingType.TABLE_WISE.value,
                ShardingType.ROW_WISE.value,
                ShardingType.COLUMN_WISE.value,
                # ShardingType.DATA_PARALLEL.value,
                # Data parallel checkpointing not yet supported
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=2, deadline=None)
    def test_sharding_fused_ebc_module_replace(
        self,
        sharding_type: str,
    ) -> None:

        ebc = EmbeddingBagCollection(
            tables=[
                EmbeddingBagConfig(
                    name="table_0",
                    feature_names=["feature_0", "feature_1"],
                    embedding_dim=8,
                    num_embeddings=10,
                )
            ],
        )

        fused_ebc = fuse_embedding_optimizer(
            ebc,
            optimizer_type=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.02},
            device=torch.device("cuda"),
        )

        #             instance 0   instance 1  instance 2
        # "feature_0"   [0, 1]       None        [2]
        # "feature_1"   [3]          [4]         [5,6,7]
        #

        kjt_input = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0", "feature_1"],
            values=torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7]),
            lengths=torch.LongTensor([2, 0, 1, 1, 1, 3]),
        )

        self._run_multi_process_test(
            callable=sharding_single_rank,
            world_size=2,
            unsharded_model=fused_ebc,
            kjt_input=kjt_input,
            sharders=[TestFusedEBCSharder(sharding_type=sharding_type)],
            backend="nccl",
        )
