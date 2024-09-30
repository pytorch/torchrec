/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <torchrec/csrc/dynamic_embedding/details/types.h>
#include <optional>

namespace torchrec {

class LXUStrategy {
 public:
  LXUStrategy() = default;
  LXUStrategy(const LXUStrategy&) = delete;
  LXUStrategy(LXUStrategy&& o) noexcept = default;

  virtual void update_time(uint32_t time) = 0;
  virtual int64_t time(lxu_record_t record) = 0;

  virtual lxu_record_t update(
      int64_t global_id,
      int64_t cache_id,
      std::optional<lxu_record_t> val) = 0;

  /**
   * Analysis all ids and returns the num_elems that are most need to evict.
   * @param iterator Returns each global_id to ExtValue pair. Returns nullopt
   * when at ends.
   * @param num_to_evict
   * @return
   */
  virtual std::vector<int64_t> evict(
      iterator_t iterator,
      uint64_t num_to_evict) = 0;
};

} // namespace torchrec
