/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

namespace cpp2 torchrec.distributed.planner
namespace py3 torchrec.distributed.planner

struct Device {
  // Example: "cuda:0" will be transformed into type = "cuda", and index = 0
  // type of device: torch.device.type ("cpu" or "cuda")
  1: string type;
  // index of device
  2: i64 index;
}

struct Placement {
  // Example: "rank:1/cuda:1" will be transformed into corresponding Device
  // device = Device{type="cuda", index=1} and rank = 1
  1: Device device;
  2: i32 rank;
}

// This struct is a mapping from torch/distributed/_shard/metadata.py's ShardMetadata
struct ShardMetadata {
  1: list<i64> shardOffsets;
  // Offsets in the original tensor indicating
  // the start offsets for this shard. Should have the same rank as
  // the original tensor.
  2: list<i64> shardSizes;
  // Integers indicating the size of each
  // dimension for this shard. Should have the same rank as the
  // original tensor.
  3: optional Placement placement;
// Specifies the placement of this shard.
}

// This struct is a mapping from torchrec/distributed/types.py's ParameterSharding
struct ParameterSharding {
  1: string shardingType;
  // This is a mapping to torchrec/distributed/types.py's ShardingType
  // how this parameter is sharded. See ShardingType for well-known types.
  // There is no StrEnum supported in thrift.
  // Use string directly to support new sharding types.
  2: string computeKernel;
  // compute kernel to be used by this parameter.
  3: optional list<i64> ranks;
  // rank of each shard.
  4: optional list<ShardMetadata> shardingSpec;
// list of ShardMetadata for each shard
}

// This struct is a mapping from torchrec/distributed/types.py's EmbeddingModuleShardingPlan
struct EmbeddingModuleShardingPlan {
  1: map<string, ParameterSharding> shardingPlan;
// This simulates the multiple inheritance of ModuleShardingPlan and Dict[str, ParameterSharding]
}

// This struct is a mapping from torchrec/distributed/types.py's ShardingPlan
struct ShardingPlan {
  1: map<string, EmbeddingModuleShardingPlan> plan;
// dict keyed by module path of dict of parameter sharding specs keyed by parameter name.
}
