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

from torchrec.distributed.test_utils.test_model import TestFusedECSharder
from torchrec.distributed.test_utils.test_sharding import copy_state_dict
from torchrec.distributed.types import (
    ModuleSharder,
    ShardingEnv,
    ShardingPlan,
    ShardingType,
)
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.modules.fused_embedding_modules import (
    fuse_embedding_optimizer,
    FusedEmbeddingCollection,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.test_utils import skip_if_asan_class, skipIfRocm

@skip_if_asan_class
class FusedEmbeddingBagCollectionParallelTest(MultiProcessTestBase):
    @classmethod
    def sharding_single_rank(
        cls,
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
            plan: ShardingPlan = planner.collective_plan(
                unsharded_model, sharders, ctx.pg
            )

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
            unsharded_model_pred = unsharded_model(kjt_input)
            sharded_pred = sharded_model(kjt_input)

            assert set(unsharded_model_pred.keys()) == set(sharded_pred.keys())

            for feature_name in unsharded_model_pred.keys():
                unsharded_jt = unsharded_model_pred[feature_name]
                sharded_jt = sharded_pred[feature_name]

                torch.testing.assert_close(
                    unsharded_jt.values().cpu(), sharded_jt.values().cpu()
                )
                torch.testing.assert_close(
                    unsharded_jt.lengths().cpu(), sharded_jt.lengths().cpu()
                )

    @skipIfRocm()
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
    def test_sharding_fused_ec(
        self,
        sharding_type: str,
    ) -> None:
        fused_ec = FusedEmbeddingCollection(
            tables=[
                EmbeddingConfig(
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
            callable=self.sharding_single_rank,
            world_size=2,
            unsharded_model=fused_ec,
            kjt_input=kjt_input,
            sharders=[TestFusedECSharder(sharding_type=sharding_type)],
            backend="nccl",
        )

    @skipIfRocm()
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
        ec = EmbeddingCollection(
            tables=[
                EmbeddingConfig(
                    name="table_0",
                    feature_names=["feature_0", "feature_1"],
                    embedding_dim=8,
                    num_embeddings=10,
                )
            ],
        )

        fused_ec = fuse_embedding_optimizer(
            ec,
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
            callable=self.sharding_single_rank,
            world_size=2,
            unsharded_model=fused_ec,
            kjt_input=kjt_input,
            sharders=[TestFusedECSharder(sharding_type=sharding_type)],
            backend="nccl",
        )
