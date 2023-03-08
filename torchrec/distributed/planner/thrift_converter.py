#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

from torch.distributed._shard.sharding_spec import ShardMetadata
from torch.distributed.remote_device import _remote_device
from torchrec.distributed.planner.types.thrift_types import (
    Device as ThriftDevice,
    EmbeddingModuleShardingPlan as ThriftEmbeddingModuleShardingPlan,
    ParameterSharding as ThriftParameterSharding,
    Placement as ThriftPlacement,
    ShardingPlan as ThriftShardingPlan,
    ShardMetadata as ThriftShardMetadata,
)
from torchrec.distributed.types import ParameterSharding, ShardingPlan


def remote_device_to_placement_thrift(remote_device: _remote_device) -> ThriftPlacement:
    # Retrieve device/rank from remote_device separately and serialize them into Device + Placement thrift
    ori_device = remote_device.device()
    thrift_device = ThriftDevice(type=ori_device.type, index=ori_device.index)
    return ThriftPlacement(
        device=thrift_device,
        rank=remote_device.rank(),
    )


def shard_metadata_to_thrift(shard_metadata: ShardMetadata) -> ThriftShardMetadata:
    thrift_placement = (
        remote_device_to_placement_thrift(shard_metadata.placement)
        if shard_metadata.placement
        else None
    )
    return ThriftShardMetadata(
        shardOffsets=shard_metadata.shard_offsets,
        shardSizes=shard_metadata.shard_sizes,
        placement=thrift_placement,
    )


def parameter_sharding_to_thrift(
    parameter_sharding: ParameterSharding,
) -> ThriftParameterSharding:
    thrift_sharding_spec = [
        shard_metadata_to_thrift(shard_metadata)
        # pyre-ignore Undefined attribute [16]: Optional type has no attribute `shards`
        for shard_metadata in parameter_sharding.sharding_spec.shards
    ]

    return ThriftParameterSharding(
        shardingType=parameter_sharding.sharding_type,
        computeKernel=parameter_sharding.compute_kernel,
        ranks=parameter_sharding.ranks,
        shardingSpec=thrift_sharding_spec,
    )


def sharding_plan_to_thrift(sharding_plan: ShardingPlan) -> ThriftShardingPlan:
    thrift_plan_map = {}
    for module_path, embedding_module_sharding_plan in sharding_plan.plan.items():
        embedding_module_sharding_plan_map = {}

        # pyre-ignore Undefined attribute [16]: `torchrec.distributed.types.ModuleShardingPlan` has no attribute `items`.
        for fqn, parameter_sharding in embedding_module_sharding_plan.items():
            embedding_module_sharding_plan_map[fqn] = parameter_sharding_to_thrift(
                parameter_sharding
            )
        thrift_embedding_module_sharding_plan = ThriftEmbeddingModuleShardingPlan(
            shardingPlan=embedding_module_sharding_plan_map
        )
        thrift_plan_map[module_path] = thrift_embedding_module_sharding_plan
    return ThriftShardingPlan(plan=thrift_plan_map)
