#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest
from typing import Any, Dict, List, Optional

import hypothesis.strategies as st
import torch
from hypothesis import given, settings, Verbosity
from torchrec import distributed as trec_dist
from torchrec.distributed.parameter_sharding_utils import (
    column_wise_sharding,
    construct_module_sharding_plan,
    data_parallel_sharding,
    get_module_to_default_sharders,
    ParameterShardingGenerator,
    row_wise_sharding,
    table_row_wise_sharding,
    table_wise_sharding,
)

from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.test_utils.test_sharding import copy_state_dict
from torchrec.distributed.types import ParameterSharding, ShardingEnv
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection

from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.test_utils import skip_if_asan_class


def _test_sharding(
    tables: List[EmbeddingBagConfig],
    initial_state_dict: Dict[str, Any],
    rank: int,
    world_size: int,
    kjt_input_per_rank: List[KeyedJaggedTensor],
    backend: str,
    parameter_sharding_plan: Dict[str, ParameterSharding],
    local_size: Optional[int] = None,
) -> None:
    trec_dist.comm_ops.set_gradient_division(False)
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        kjt_input_per_rank = [kjt.to(ctx.device) for kjt in kjt_input_per_rank]
        initial_state_dict = {
            fqn: tensor.to(ctx.device) for fqn, tensor in initial_state_dict.items()
        }

        model = EmbeddingBagCollection(
            tables=tables,
            device=ctx.device,
        )

        sharder = get_module_to_default_sharders()[type(model)]

        unsharded_model = model
        sharded_model = sharder.shard(
            module=model,
            params=parameter_sharding_plan,
            env=ShardingEnv.from_process_group(ctx.pg),
            device=ctx.device,
        )

        unsharded_model.load_state_dict(copy.deepcopy(initial_state_dict))
        copy_state_dict(sharded_model.state_dict(), copy.deepcopy(initial_state_dict))

        feature_keys = []
        for table in tables:
            feature_keys.extend(table.feature_names)

        # each rank gets a subbatch
        sharded_model_pred_kt = sharded_model(kjt_input_per_rank[ctx.rank]).to_dict()
        _sharded_model_pred = torch.stack(  # noqa
            [sharded_model_pred_kt[feature] for feature in feature_keys]
        )


@skip_if_asan_class
class ConstructParameterShardingTest(MultiProcessTestBase):
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    # pyre-fixme[56]
    @given(
        per_param_sharding=st.sampled_from(
            [
                {
                    "table_0": data_parallel_sharding(),
                    "table_1": data_parallel_sharding(),
                },
                {
                    "table_0": table_wise_sharding(rank=0),
                    "table_1": table_wise_sharding(rank=1),
                },
                {
                    "table_0": row_wise_sharding(),
                    "table_1": row_wise_sharding(),
                },
                {
                    "table_0": column_wise_sharding(),
                    "table_1": column_wise_sharding(),
                },
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=8, deadline=None)
    def test_parameter_sharding_ebc(
        self,
        per_param_sharding: Dict[str, ParameterShardingGenerator],
    ) -> None:

        WORLD_SIZE = 2

        embedding_bag_config = [
            EmbeddingBagConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=16,
                num_embeddings=4,
            ),
            EmbeddingBagConfig(
                name="table_1",
                feature_names=["feature_1"],
                embedding_dim=16,
                num_embeddings=4,
            ),
        ]

        # Rank 0
        #             instance 0   instance 1  instance 2
        # "feature_0"   [0, 1]       None        [2]
        # "feature_1"   [0, 1]       None        [2]

        # Rank 1

        #             instance 0   instance 1  instance 2
        # "feature_0"   [3, 2]       [1,2]        [0, 1,2,3]
        # "feature_1"   [2,3]       None        [2]

        kjt_input_per_rank = [  # noqa
            KeyedJaggedTensor.from_lengths_sync(
                keys=["feature_0", "feature_1"],
                values=torch.LongTensor([0, 1, 2, 0, 1, 2]),
                lengths=torch.LongTensor([2, 0, 1, 2, 0, 1]),
            ),
            KeyedJaggedTensor.from_lengths_sync(
                keys=["feature_0", "feature_1"],
                values=torch.LongTensor([3, 2, 1, 2, 0, 1, 2, 3, 2, 3, 2]),
                lengths=torch.LongTensor([2, 2, 4, 2, 0, 1]),
            ),
        ]

        parameter_sharding_plan = construct_module_sharding_plan(
            EmbeddingBagCollection(tables=embedding_bag_config),
            per_param_sharding=per_param_sharding,
            local_size=2,
            world_size=2,
        )

        self._run_multi_process_test(
            callable=_test_sharding,
            world_size=WORLD_SIZE,
            tables=embedding_bag_config,
            initial_state_dict={
                "embedding_bags.table_0.weight": torch.Tensor(
                    [
                        [1] * 16,
                        [2] * 16,
                        [3] * 16,
                        [4] * 16,
                    ]
                ),
                "embedding_bags.table_1.weight": torch.Tensor(
                    [
                        [101] * 16,
                        [102] * 16,
                        [103] * 16,
                        [104] * 16,
                    ]
                ),
            },
            kjt_input_per_rank=kjt_input_per_rank,
            backend="nccl"
            if (torch.cuda.is_available() and torch.cuda.device_count() >= 2)
            else "gloo",
            parameter_sharding_plan=parameter_sharding_plan,
        )

    def test_construct_module_sharding_plan(self) -> None:
        embedding_bag_config = [
            EmbeddingBagConfig(
                name=f"table_{idx}",
                feature_names=[f"feature_{idx}"],
                embedding_dim=256,
                num_embeddings=32 * 32,
            )
            for idx in range(5)
        ]

        _parameter_sharding_plan = construct_module_sharding_plan(  # noqa
            EmbeddingBagCollection(tables=embedding_bag_config),
            per_param_sharding={
                "table_0": data_parallel_sharding(),
                "table_1": table_wise_sharding(rank=1),
                "table_2": row_wise_sharding(),
                "table_3": column_wise_sharding(shard_dim=128, ranks=[8, 9]),
                "table_4": table_row_wise_sharding(node_index=3),
            },
            local_size=8,
            world_size=32,
        )
