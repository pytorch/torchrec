/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <c10/macros/Macros.h>
#include <torchrec/csrc/dynamic_embedding/details/mixed_lfu_lru_strategy.h>
#include <algorithm>

namespace torchrec {
MixedLFULRUStrategy::MixedLFULRUStrategy(uint16_t min_used_freq_power)
    : min_lfu_power_(min_used_freq_power), time_(new std::atomic<uint32_t>()) {}

void MixedLFULRUStrategy::update_time(uint32_t time) {
  time_->store(time);
}

MixedLFULRUStrategy::lxu_record_t MixedLFULRUStrategy::update(
    int64_t global_id,
    int64_t cache_id,
    std::optional<lxu_record_t> val) {
  Record r{};
  r.time_ = time_->load();

  if (C10_UNLIKELY(!val.has_value())) {
    r.freq_power_ = min_lfu_power_;
  } else {
    auto freq_power = reinterpret_cast<Record*>(&val.value())->freq_power_;
    bool should_carry = generator_.is_next_n_bits_all_zero(freq_power);
    if (should_carry) {
      ++freq_power;
    }
    r.freq_power_ = freq_power;
  }
  return *reinterpret_cast<lxu_record_t*>(&r);
}

} // namespace torchrec
