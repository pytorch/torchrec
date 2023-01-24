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
from torchrec.distributed.sharding_plan import (
    column_wise,
    construct_module_sharding_plan,
    data_parallel,
    get_module_to_default_sharders,
    ParameterShardingGenerator,
    row_wise,
    table_row_wise,
    table_wise,
)

from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.test_utils.test_sharding import copy_state_dict
from torchrec.distributed.types import (
    EnumerableShardingSpec,
    ModuleShardingPlan,
    ParameterSharding,
    ShardingEnv,
    ShardingPlan,
    ShardingType,
    ShardMetadata,
)
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
    module_sharding_plan: ModuleShardingPlan,
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
            params=module_sharding_plan,
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
class ConstructParameterShardingAndShardTest(MultiProcessTestBase):
    # pyre-fixme[56]
    @given(
        per_param_sharding=st.sampled_from(
            [
                {
                    "table_0": data_parallel(),
                    "table_1": data_parallel(),
                },
                {
                    "table_0": table_wise(rank=0),
                    "table_1": table_wise(rank=1),
                },
                {
                    "table_0": row_wise(),
                    "table_1": row_wise(),
                },
                {
                    "table_0": column_wise(ranks=[0, 1]),
                    "table_1": column_wise(ranks=[0, 1]),
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

        module_sharding_plan = construct_module_sharding_plan(
            EmbeddingBagCollection(tables=embedding_bag_config),
            per_param_sharding=per_param_sharding,
            local_size=WORLD_SIZE,
            world_size=WORLD_SIZE,
            device_type="cuda" if torch.cuda.is_available() else "cpu",
        )

        # Row-wise not supported on gloo
        if (
            not torch.cuda.is_available()
            and module_sharding_plan["table_0"].sharding_type
            == ShardingType.ROW_WISE.value
        ):
            return

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
            backend="nccl" if torch.cuda.is_available() else "gloo",
            module_sharding_plan=module_sharding_plan,
        )


class ConstructParameterShardingTest(unittest.TestCase):
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

        expected = {
            "table_0": ParameterSharding(
                sharding_type="data_parallel",
                compute_kernel="dense",
                ranks=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                    24,
                    25,
                    26,
                    27,
                    28,
                    29,
                    30,
                    31,
                ],
                sharding_spec=None,
            ),
            "table_1": ParameterSharding(
                sharding_type="table_wise",
                compute_kernel="fused",
                ranks=[1],
                sharding_spec=EnumerableShardingSpec(
                    shards=[
                        ShardMetadata(
                            shard_offsets=[0, 0],
                            shard_sizes=[1024, 256],
                            placement="rank:1/cuda:1",
                        )
                    ]
                ),
            ),
            "table_2": ParameterSharding(
                sharding_type="row_wise",
                compute_kernel="fused",
                ranks=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                    24,
                    25,
                    26,
                    27,
                    28,
                    29,
                    30,
                    31,
                ],
                sharding_spec=EnumerableShardingSpec(
                    shards=[
                        ShardMetadata(
                            shard_offsets=[0, 0],
                            shard_sizes=[32, 256],
                            placement="rank:0/cuda:0",
                        ),
                        ShardMetadata(
                            shard_offsets=[32, 0],
                            shard_sizes=[32, 256],
                            placement="rank:1/cuda:1",
                        ),
                        ShardMetadata(
                            shard_offsets=[64, 0],
                            shard_sizes=[32, 256],
                            placement="rank:2/cuda:2",
                        ),
                        ShardMetadata(
                            shard_offsets=[96, 0],
                            shard_sizes=[32, 256],
                            placement="rank:3/cuda:3",
                        ),
                        ShardMetadata(
                            shard_offsets=[128, 0],
                            shard_sizes=[32, 256],
                            placement="rank:4/cuda:4",
                        ),
                        ShardMetadata(
                            shard_offsets=[160, 0],
                            shard_sizes=[32, 256],
                            placement="rank:5/cuda:5",
                        ),
                        ShardMetadata(
                            shard_offsets=[192, 0],
                            shard_sizes=[32, 256],
                            placement="rank:6/cuda:6",
                        ),
                        ShardMetadata(
                            shard_offsets=[224, 0],
                            shard_sizes=[32, 256],
                            placement="rank:7/cuda:7",
                        ),
                        ShardMetadata(
                            shard_offsets=[256, 0],
                            shard_sizes=[32, 256],
                            placement="rank:8/cuda:0",
                        ),
                        ShardMetadata(
                            shard_offsets=[288, 0],
                            shard_sizes=[32, 256],
                            placement="rank:9/cuda:1",
                        ),
                        ShardMetadata(
                            shard_offsets=[320, 0],
                            shard_sizes=[32, 256],
                            placement="rank:10/cuda:2",
                        ),
                        ShardMetadata(
                            shard_offsets=[352, 0],
                            shard_sizes=[32, 256],
                            placement="rank:11/cuda:3",
                        ),
                        ShardMetadata(
                            shard_offsets=[384, 0],
                            shard_sizes=[32, 256],
                            placement="rank:12/cuda:4",
                        ),
                        ShardMetadata(
                            shard_offsets=[416, 0],
                            shard_sizes=[32, 256],
                            placement="rank:13/cuda:5",
                        ),
                        ShardMetadata(
                            shard_offsets=[448, 0],
                            shard_sizes=[32, 256],
                            placement="rank:14/cuda:6",
                        ),
                        ShardMetadata(
                            shard_offsets=[480, 0],
                            shard_sizes=[32, 256],
                            placement="rank:15/cuda:7",
                        ),
                        ShardMetadata(
                            shard_offsets=[512, 0],
                            shard_sizes=[32, 256],
                            placement="rank:16/cuda:0",
                        ),
                        ShardMetadata(
                            shard_offsets=[544, 0],
                            shard_sizes=[32, 256],
                            placement="rank:17/cuda:1",
                        ),
                        ShardMetadata(
                            shard_offsets=[576, 0],
                            shard_sizes=[32, 256],
                            placement="rank:18/cuda:2",
                        ),
                        ShardMetadata(
                            shard_offsets=[608, 0],
                            shard_sizes=[32, 256],
                            placement="rank:19/cuda:3",
                        ),
                        ShardMetadata(
                            shard_offsets=[640, 0],
                            shard_sizes=[32, 256],
                            placement="rank:20/cuda:4",
                        ),
                        ShardMetadata(
                            shard_offsets=[672, 0],
                            shard_sizes=[32, 256],
                            placement="rank:21/cuda:5",
                        ),
                        ShardMetadata(
                            shard_offsets=[704, 0],
                            shard_sizes=[32, 256],
                            placement="rank:22/cuda:6",
                        ),
                        ShardMetadata(
                            shard_offsets=[736, 0],
                            shard_sizes=[32, 256],
                            placement="rank:23/cuda:7",
                        ),
                        ShardMetadata(
                            shard_offsets=[768, 0],
                            shard_sizes=[32, 256],
                            placement="rank:24/cuda:0",
                        ),
                        ShardMetadata(
                            shard_offsets=[800, 0],
                            shard_sizes=[32, 256],
                            placement="rank:25/cuda:1",
                        ),
                        ShardMetadata(
                            shard_offsets=[832, 0],
                            shard_sizes=[32, 256],
                            placement="rank:26/cuda:2",
                        ),
                        ShardMetadata(
                            shard_offsets=[864, 0],
                            shard_sizes=[32, 256],
                            placement="rank:27/cuda:3",
                        ),
                        ShardMetadata(
                            shard_offsets=[896, 0],
                            shard_sizes=[32, 256],
                            placement="rank:28/cuda:4",
                        ),
                        ShardMetadata(
                            shard_offsets=[928, 0],
                            shard_sizes=[32, 256],
                            placement="rank:29/cuda:5",
                        ),
                        ShardMetadata(
                            shard_offsets=[960, 0],
                            shard_sizes=[32, 256],
                            placement="rank:30/cuda:6",
                        ),
                        ShardMetadata(
                            shard_offsets=[992, 0],
                            shard_sizes=[32, 256],
                            placement="rank:31/cuda:7",
                        ),
                    ]
                ),
            ),
            "table_3": ParameterSharding(
                sharding_type="column_wise",
                compute_kernel="fused",
                ranks=[8, 9],
                sharding_spec=EnumerableShardingSpec(
                    shards=[
                        ShardMetadata(
                            shard_offsets=[0, 0],
                            shard_sizes=[1024, 128],
                            placement="rank:8/cuda:0",
                        ),
                        ShardMetadata(
                            shard_offsets=[0, 128],
                            shard_sizes=[1024, 128],
                            placement="rank:9/cuda:1",
                        ),
                    ]
                ),
            ),
            "table_4": ParameterSharding(
                sharding_type="table_row_wise",
                compute_kernel="fused",
                ranks=[24, 25, 26, 27, 28, 29, 30, 31],
                sharding_spec=EnumerableShardingSpec(
                    shards=[
                        ShardMetadata(
                            shard_offsets=[0, 0],
                            shard_sizes=[128, 256],
                            placement="rank:24/cuda:0",
                        ),
                        ShardMetadata(
                            shard_offsets=[128, 0],
                            shard_sizes=[128, 256],
                            placement="rank:25/cuda:1",
                        ),
                        ShardMetadata(
                            shard_offsets=[256, 0],
                            shard_sizes=[128, 256],
                            placement="rank:26/cuda:2",
                        ),
                        ShardMetadata(
                            shard_offsets=[384, 0],
                            shard_sizes=[128, 256],
                            placement="rank:27/cuda:3",
                        ),
                        ShardMetadata(
                            shard_offsets=[512, 0],
                            shard_sizes=[128, 256],
                            placement="rank:28/cuda:4",
                        ),
                        ShardMetadata(
                            shard_offsets=[640, 0],
                            shard_sizes=[128, 256],
                            placement="rank:29/cuda:5",
                        ),
                        ShardMetadata(
                            shard_offsets=[768, 0],
                            shard_sizes=[128, 256],
                            placement="rank:30/cuda:6",
                        ),
                        ShardMetadata(
                            shard_offsets=[896, 0],
                            shard_sizes=[128, 256],
                            placement="rank:31/cuda:7",
                        ),
                    ]
                ),
            ),
        }

        module_sharding_plan = construct_module_sharding_plan(
            EmbeddingBagCollection(tables=embedding_bag_config),
            per_param_sharding={
                "table_0": data_parallel(),
                "table_1": table_wise(rank=1),
                "table_2": row_wise(),
                "table_3": column_wise(ranks=[8, 9]),
                "table_4": table_row_wise(host_index=3),
            },
            local_size=8,
            world_size=32,
            device_type="cuda",
        )
        self.assertDictEqual(expected, module_sharding_plan)

    def test_column_wise(self) -> None:
        embedding_bag_config = [
            EmbeddingBagConfig(
                name=f"table_{idx}",
                feature_names=[f"feature_{idx}"],
                embedding_dim=64,
                num_embeddings=4096,
            )
            for idx in range(2)
        ]
        module_sharding_plan = construct_module_sharding_plan(
            EmbeddingBagCollection(tables=embedding_bag_config),
            per_param_sharding={
                "table_0": column_wise(ranks=[0, 1]),
                "table_1": column_wise(ranks=[0, 1]),
            },
            local_size=2,
            world_size=2,
            device_type="cuda",
        )
        expected = {
            "table_0": ParameterSharding(
                sharding_type="column_wise",
                compute_kernel="fused",
                ranks=[0, 1],
                sharding_spec=EnumerableShardingSpec(
                    shards=[
                        ShardMetadata(
                            shard_offsets=[0, 0],
                            shard_sizes=[4096, 32],
                            placement="rank:0/cuda:0",
                        ),
                        ShardMetadata(
                            shard_offsets=[0, 32],
                            shard_sizes=[4096, 32],
                            placement="rank:1/cuda:1",
                        ),
                    ]
                ),
            ),
            "table_1": ParameterSharding(
                sharding_type="column_wise",
                compute_kernel="fused",
                ranks=[0, 1],
                sharding_spec=EnumerableShardingSpec(
                    shards=[
                        ShardMetadata(
                            shard_offsets=[0, 0],
                            shard_sizes=[4096, 32],
                            placement="rank:0/cuda:0",
                        ),
                        ShardMetadata(
                            shard_offsets=[0, 32],
                            shard_sizes=[4096, 32],
                            placement="rank:1/cuda:1",
                        ),
                    ]
                ),
            ),
        }
        self.assertDictEqual(expected, module_sharding_plan)


class ShardingPlanTest(unittest.TestCase):
    def test_str(self) -> None:
        plan = ShardingPlan(
            {
                "ebc": {
                    "user_id": ParameterSharding(
                        sharding_type="table_wise",
                        compute_kernel="fused",
                        ranks=[0],
                        sharding_spec=EnumerableShardingSpec(
                            [
                                ShardMetadata(
                                    shard_offsets=[0, 0],
                                    shard_sizes=[4096, 32],
                                    placement="rank:0/cuda:0",
                                ),
                            ]
                        ),
                    ),
                    "movie_id": ParameterSharding(
                        sharding_type="row_wise",
                        compute_kernel="dense",
                        ranks=[0, 1],
                        sharding_spec=EnumerableShardingSpec(
                            [
                                ShardMetadata(
                                    shard_offsets=[0, 0],
                                    shard_sizes=[2048, 32],
                                    placement="rank:0/cuda:0",
                                ),
                                ShardMetadata(
                                    shard_offsets=[2048, 0],
                                    shard_sizes=[2048, 32],
                                    placement="rank:0/cuda:1",
                                ),
                            ]
                        ),
                    ),
                }
            }
        )
        expected = """
module: ebc

param     sharding type    compute kernel    ranks
--------  ---------------  ----------------  -------
user_id   table_wise       fused             [0]
movie_id  row_wise         dense             [0, 1]

param     shard offsets    shard sizes    placement
--------  ---------------  -------------  -------------
user_id   [0, 0]           [4096, 32]     rank:0/cuda:0
movie_id  [0, 0]           [2048, 32]     rank:0/cuda:0
movie_id  [2048, 0]        [2048, 32]     rank:0/cuda:1
"""
        self.assertEqual(expected.strip(), str(plan))
