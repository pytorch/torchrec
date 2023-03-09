#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torchrec.distributed.planner.thrift_converter import sharding_plan_to_thrift
from torchrec.distributed.planner.types.thrift_types import (
    Device as ThriftDevice,
    EmbeddingModuleShardingPlan as ThriftEmbeddingModuleShardingPlan,
    ParameterSharding as ThriftParameterSharding,
    Placement as ThriftPlacement,
    ShardingPlan as ThriftShardingPlan,
    ShardMetadata as ThriftShardMetadata,
)

from torchrec.distributed.sharding_plan import (
    column_wise,
    construct_module_sharding_plan,
    row_wise,
)
from torchrec.distributed.types import (
    EmbeddingModuleShardingPlan,
    EnumerableShardingSpec,
    ParameterSharding,
    ShardingPlan,
    ShardMetadata,
)
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection


class ShardingPlanTest(unittest.TestCase):
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
        module_sharding_plan_2 = construct_module_sharding_plan(
            EmbeddingBagCollection(tables=embedding_bag_config),
            per_param_sharding={
                "table_0": column_wise(ranks=[0, 1]),
                "table_1": column_wise(ranks=[0, 1]),
            },
            local_size=2,
            world_size=2,
            device_type="cuda",
        )
        plan = ShardingPlan(
            {
                "ebc-1": module_sharding_plan,
                "ebc-2": module_sharding_plan_2,
            }
        )
        expected_thrift = ThriftShardingPlan(
            plan={
                "ebc-1": ThriftEmbeddingModuleShardingPlan(
                    shardingPlan={
                        "table_0": ThriftParameterSharding(
                            shardingType="column_wise",
                            computeKernel="fused",
                            ranks=[0, 1],
                            shardingSpec=[
                                ThriftShardMetadata(
                                    shardOffsets=[0, 0],
                                    shardSizes=[4096, 32],
                                    placement=ThriftPlacement(
                                        device=ThriftDevice(
                                            index=0,
                                            type="cuda",
                                        ),
                                        rank=0,
                                    ),
                                ),
                                ThriftShardMetadata(
                                    shardOffsets=[0, 32],
                                    shardSizes=[4096, 32],
                                    placement=ThriftPlacement(
                                        device=ThriftDevice(
                                            index=1,
                                            type="cuda",
                                        ),
                                        rank=1,
                                    ),
                                ),
                            ],
                        ),
                        "table_1": ThriftParameterSharding(
                            shardingType="column_wise",
                            computeKernel="fused",
                            ranks=[0, 1],
                            shardingSpec=[
                                ThriftShardMetadata(
                                    shardOffsets=[0, 0],
                                    shardSizes=[4096, 32],
                                    placement=ThriftPlacement(
                                        device=ThriftDevice(
                                            index=0,
                                            type="cuda",
                                        ),
                                        rank=0,
                                    ),
                                ),
                                ThriftShardMetadata(
                                    shardOffsets=[0, 32],
                                    shardSizes=[4096, 32],
                                    placement=ThriftPlacement(
                                        device=ThriftDevice(
                                            index=1,
                                            type="cuda",
                                        ),
                                        rank=1,
                                    ),
                                ),
                            ],
                        ),
                    }
                ),
                "ebc-2": ThriftEmbeddingModuleShardingPlan(
                    shardingPlan={
                        "table_0": ThriftParameterSharding(
                            shardingType="column_wise",
                            computeKernel="fused",
                            ranks=[0, 1],
                            shardingSpec=[
                                ThriftShardMetadata(
                                    shardOffsets=[0, 0],
                                    shardSizes=[4096, 32],
                                    placement=ThriftPlacement(
                                        device=ThriftDevice(
                                            index=0,
                                            type="cuda",
                                        ),
                                        rank=0,
                                    ),
                                ),
                                ThriftShardMetadata(
                                    shardOffsets=[0, 32],
                                    shardSizes=[4096, 32],
                                    placement=ThriftPlacement(
                                        device=ThriftDevice(
                                            index=1,
                                            type="cuda",
                                        ),
                                        rank=1,
                                    ),
                                ),
                            ],
                        ),
                        "table_1": ThriftParameterSharding(
                            shardingType="column_wise",
                            computeKernel="fused",
                            ranks=[0, 1],
                            shardingSpec=[
                                ThriftShardMetadata(
                                    shardOffsets=[0, 0],
                                    shardSizes=[4096, 32],
                                    placement=ThriftPlacement(
                                        device=ThriftDevice(
                                            index=0,
                                            type="cuda",
                                        ),
                                        rank=0,
                                    ),
                                ),
                                ThriftShardMetadata(
                                    shardOffsets=[0, 32],
                                    shardSizes=[4096, 32],
                                    placement=ThriftPlacement(
                                        device=ThriftDevice(
                                            index=1,
                                            type="cuda",
                                        ),
                                        rank=1,
                                    ),
                                ),
                            ],
                        ),
                    }
                ),
            }
        )
        column_wise_sharding_plan_thrift = sharding_plan_to_thrift(plan)
        self.assertEqual(expected_thrift, column_wise_sharding_plan_thrift)

    def test_row_wise(self) -> None:
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
                "table_0": row_wise(),
                "table_1": row_wise(),
            },
            local_size=2,
            world_size=2,
            device_type="cuda",
        )
        module_sharding_plan_2 = construct_module_sharding_plan(
            EmbeddingBagCollection(tables=embedding_bag_config),
            per_param_sharding={
                "table_0": row_wise(),
                "table_1": row_wise(),
            },
            local_size=2,
            world_size=2,
            device_type="cuda",
        )
        plan = ShardingPlan(
            {
                "ebc-1": module_sharding_plan,
                "ebc-2": module_sharding_plan_2,
            }
        )
        expected_thrift = ThriftShardingPlan(
            plan={
                "ebc-1": ThriftEmbeddingModuleShardingPlan(
                    shardingPlan={
                        "table_0": ThriftParameterSharding(
                            shardingType="row_wise",
                            computeKernel="fused",
                            ranks=[0, 1],
                            shardingSpec=[
                                ThriftShardMetadata(
                                    shardOffsets=[0, 0],
                                    shardSizes=[2048, 64],
                                    placement=ThriftPlacement(
                                        device=ThriftDevice(
                                            index=0,
                                            type="cuda",
                                        ),
                                        rank=0,
                                    ),
                                ),
                                ThriftShardMetadata(
                                    shardOffsets=[2048, 0],
                                    shardSizes=[2048, 64],
                                    placement=ThriftPlacement(
                                        device=ThriftDevice(
                                            index=1,
                                            type="cuda",
                                        ),
                                        rank=1,
                                    ),
                                ),
                            ],
                        ),
                        "table_1": ThriftParameterSharding(
                            shardingType="row_wise",
                            computeKernel="fused",
                            ranks=[0, 1],
                            shardingSpec=[
                                ThriftShardMetadata(
                                    shardOffsets=[0, 0],
                                    shardSizes=[2048, 64],
                                    placement=ThriftPlacement(
                                        device=ThriftDevice(
                                            index=0,
                                            type="cuda",
                                        ),
                                        rank=0,
                                    ),
                                ),
                                ThriftShardMetadata(
                                    shardOffsets=[2048, 0],
                                    shardSizes=[2048, 64],
                                    placement=ThriftPlacement(
                                        device=ThriftDevice(
                                            index=1,
                                            type="cuda",
                                        ),
                                        rank=1,
                                    ),
                                ),
                            ],
                        ),
                    }
                ),
                "ebc-2": ThriftEmbeddingModuleShardingPlan(
                    shardingPlan={
                        "table_0": ThriftParameterSharding(
                            shardingType="row_wise",
                            computeKernel="fused",
                            ranks=[0, 1],
                            shardingSpec=[
                                ThriftShardMetadata(
                                    shardOffsets=[0, 0],
                                    shardSizes=[2048, 64],
                                    placement=ThriftPlacement(
                                        device=ThriftDevice(
                                            index=0,
                                            type="cuda",
                                        ),
                                        rank=0,
                                    ),
                                ),
                                ThriftShardMetadata(
                                    shardOffsets=[2048, 0],
                                    shardSizes=[2048, 64],
                                    placement=ThriftPlacement(
                                        device=ThriftDevice(
                                            index=1,
                                            type="cuda",
                                        ),
                                        rank=1,
                                    ),
                                ),
                            ],
                        ),
                        "table_1": ThriftParameterSharding(
                            shardingType="row_wise",
                            computeKernel="fused",
                            ranks=[0, 1],
                            shardingSpec=[
                                ThriftShardMetadata(
                                    shardOffsets=[0, 0],
                                    shardSizes=[2048, 64],
                                    placement=ThriftPlacement(
                                        device=ThriftDevice(
                                            index=0,
                                            type="cuda",
                                        ),
                                        rank=0,
                                    ),
                                ),
                                ThriftShardMetadata(
                                    shardOffsets=[2048, 0],
                                    shardSizes=[2048, 64],
                                    placement=ThriftPlacement(
                                        device=ThriftDevice(
                                            index=1,
                                            type="cuda",
                                        ),
                                        rank=1,
                                    ),
                                ),
                            ],
                        ),
                    }
                ),
            }
        )
        row_wise_sharding_plan_thrift = sharding_plan_to_thrift(plan)
        self.assertEqual(expected_thrift, row_wise_sharding_plan_thrift)

    def test_table_wise_sharding(self) -> None:
        plan = ShardingPlan(
            {
                "ebc": EmbeddingModuleShardingPlan(
                    {
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
                ),
                "sparse": EmbeddingModuleShardingPlan(
                    {
                        "ads_id": ParameterSharding(
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
                        "data_id": ParameterSharding(
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
                ),
            }
        )
        expected_thrift = ThriftShardingPlan(
            plan={
                "ebc": ThriftEmbeddingModuleShardingPlan(
                    shardingPlan={
                        "user_id": ThriftParameterSharding(
                            shardingType="table_wise",
                            computeKernel="fused",
                            ranks=[0],
                            shardingSpec=[
                                ThriftShardMetadata(
                                    shardOffsets=[0, 0],
                                    shardSizes=[4096, 32],
                                    placement=ThriftPlacement(
                                        device=ThriftDevice(
                                            index=0,
                                            type="cuda",
                                        ),
                                        rank=0,
                                    ),
                                )
                            ],
                        ),
                        "movie_id": ThriftParameterSharding(
                            shardingType="row_wise",
                            computeKernel="dense",
                            ranks=[0, 1],
                            shardingSpec=[
                                ThriftShardMetadata(
                                    shardOffsets=[0, 0],
                                    shardSizes=[2048, 32],
                                    placement=ThriftPlacement(
                                        device=ThriftDevice(
                                            index=0,
                                            type="cuda",
                                        ),
                                        rank=0,
                                    ),
                                ),
                                ThriftShardMetadata(
                                    shardOffsets=[2048, 0],
                                    shardSizes=[2048, 32],
                                    placement=ThriftPlacement(
                                        device=ThriftDevice(
                                            index=1,
                                            type="cuda",
                                        ),
                                        rank=0,
                                    ),
                                ),
                            ],
                        ),
                    }
                ),
                "sparse": ThriftEmbeddingModuleShardingPlan(
                    shardingPlan={
                        "ads_id": ThriftParameterSharding(
                            shardingType="table_wise",
                            computeKernel="fused",
                            ranks=[0],
                            shardingSpec=[
                                ThriftShardMetadata(
                                    shardOffsets=[0, 0],
                                    shardSizes=[4096, 32],
                                    placement=ThriftPlacement(
                                        device=ThriftDevice(
                                            index=0,
                                            type="cuda",
                                        ),
                                        rank=0,
                                    ),
                                )
                            ],
                        ),
                        "data_id": ThriftParameterSharding(
                            shardingType="row_wise",
                            computeKernel="dense",
                            ranks=[0, 1],
                            shardingSpec=[
                                ThriftShardMetadata(
                                    shardOffsets=[0, 0],
                                    shardSizes=[2048, 32],
                                    placement=ThriftPlacement(
                                        device=ThriftDevice(
                                            index=0,
                                            type="cuda",
                                        ),
                                        rank=0,
                                    ),
                                ),
                                ThriftShardMetadata(
                                    shardOffsets=[2048, 0],
                                    shardSizes=[2048, 32],
                                    placement=ThriftPlacement(
                                        device=ThriftDevice(
                                            index=1,
                                            type="cuda",
                                        ),
                                        rank=0,
                                    ),
                                ),
                            ],
                        ),
                    }
                ),
            }
        )
        basic_sharding_plan_thrift = sharding_plan_to_thrift(plan)
        self.assertEqual(expected_thrift, basic_sharding_plan_thrift)
