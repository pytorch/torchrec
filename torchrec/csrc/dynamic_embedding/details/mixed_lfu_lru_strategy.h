/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <torch/torch.h>
#include <torchrec/csrc/dynamic_embedding/details/lxu_strategy.h>
#include <torchrec/csrc/dynamic_embedding/details/random_bits_generator.h>
#include <atomic>
#include <optional>
#include <queue>
#include <string_view>
#include <vector>

namespace torchrec {

/**
 * Mixed LFU/LRU Eviction Strategy.
 *
 * It evict infrequence elements first, then evict least recent usage elements.
 *
 * The frequency is recorded as the power number of 2. i.e., The freq_power is
 * 1, then the global id is used approximately 2 times. And the freq_power is
 * 10, then the global id is used approximately 1024 times.
 *
 * Internally, if the current freq_power is n, then it use random bits to
 * generate n bits. If all these n bits are zero, then the freq_power++.
 *
 * All LFU/LRU information uses uint32_t to record. The uint32_t should be
 * recorded in IDTransformer's map/dictionary.
 *
 * Use `update_time` to update logical timer for LRU. It only uses lower 27 bits
 * of time.
 *
 * Use `update` to update extended value when every time global id that used.
 */
class MixedLFULRUStrategy : public LXUStrategy {
 public:
  /**
   * @param min_used_freq_power min usage is 2^min_used_freq_power. Set this to
   * avoid recent values evict too fast.
   */
  explicit MixedLFULRUStrategy(uint16_t min_used_freq_power = 5)
      : min_lfu_power_(min_used_freq_power),
        time_(new std::atomic<uint32_t>()) {}

  MixedLFULRUStrategy(const MixedLFULRUStrategy&) = delete;
  MixedLFULRUStrategy(MixedLFULRUStrategy&& o) noexcept = default;

  void update_time(lxu_record_t time) override {
    time_->store(time);
  }

  int64_t time(lxu_record_t record) override {
    return static_cast<int64_t>(reinterpret_cast<Record*>(&record)->time);
  }

  lxu_record_t update(
      int64_t global_id,
      int64_t cache_id,
      std::optional<lxu_record_t> val) override {
    Record r{};
    r.time = time_->load();

    if (C10_UNLIKELY(!val.has_value())) {
      r.freq_power = min_lfu_power_;
    } else {
      auto freq_power = reinterpret_cast<Record*>(&val.value())->freq_power;
      bool should_carry = generator_.is_next_n_bits_all_zero(freq_power);
      if (should_carry) {
        ++freq_power;
      }
      r.freq_power = freq_power;
    }
    return *reinterpret_cast<lxu_record_t*>(&r);
  }

  struct EvictItem {
    int64_t global_id;
    lxu_record_t record;
    bool operator<(const EvictItem& item) const {
      return record < item.record;
    }
  };

  std::vector<int64_t> evict(iterator_t iterator, uint64_t num_to_evict) {
    std::priority_queue<EvictItem> items;
    while (true) {
      auto val = iterator();
      if (!val.has_value()) [[unlikely]] {
        break;
      }
      EvictItem item{
          .global_id = val->global_id,
          .record = reinterpret_cast<Record*>(&val->lxu_record)->ToUint32(),
      };
      if (items.size() == num_to_evict) {
        if (!(item < items.top())) {
          continue;
        } else {
          items.pop();
          items.push(item);
        }
      } else {
        items.push(item);
      }
    }
    std::vector<int64_t> result;
    result.reserve(items.size());
    while (!items.empty()) {
      auto item = items.top();
      result.emplace_back(item.global_id);
      items.pop();
    }
    std::reverse(result.begin(), result.end());
    return result;
  }

  // Record should only be used in unittest or internally.
  struct Record {
    uint32_t time : 27;
    uint16_t freq_power : 5;

    [[nodiscard]] uint32_t ToUint32() const {
      return time | (freq_power << (32 - 5));
    }
  };
  static_assert(sizeof(Record) == sizeof(lxu_record_t));

 private:
  RandomBitsGenerator generator_;
  uint16_t min_lfu_power_;
  std::unique_ptr<std::atomic<uint32_t>> time_;
};

} // namespace torchrec
