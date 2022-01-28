/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>

#include <ATen/ATen.h>

namespace torchrec {

struct ShardMetadata {
  std::vector<int64_t> shard_offsets;
  std::vector<int64_t> shard_lengths;

  bool operator==(const ShardMetadata& other) const {
    return shard_offsets == other.shard_offsets &&
        shard_lengths == other.shard_lengths;
  }
};

struct Shard {
  ShardMetadata metadata;
  at::Tensor tensor;
};

struct ShardedTensorMetadata {
  std::vector<ShardMetadata> shards_metadata;
};

struct ShardedTensor {
  std::vector<int64_t> sizes;
  std::vector<Shard> local_shards;
  ShardedTensorMetadata metadata;
};

struct ReplicatedTensor {
  ShardedTensor local_replica;
  int64_t local_replica_id;
  int64_t replica_count;
};

} // namespace torchrec
