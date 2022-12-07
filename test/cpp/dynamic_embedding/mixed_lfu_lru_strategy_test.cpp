/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <torchrec/csrc/dynamic_embedding/details/mixed_lfu_lru_strategy.h>

namespace torchrec {
TEST(TDE, order) {
  MixedLFULRUStrategy::Record a;
  a.time = 1;
  a.freq_power = 31;
  uint32_t i32 = a.ToUint32();
  ASSERT_EQ(0xF8000001, i32);
}

TEST(TDE, MixedLFULRUStrategy_Evict) {
  std::vector<std::pair<int64_t, MixedLFULRUStrategy::Record>> records;
  {
    records.emplace_back();
    records.back().first = 1;
    records.back().second.time = 100;
    records.back().second.freq_power = 2;
  }
  {
    records.emplace_back();
    records.back().first = 2;
    records.back().second.time = 10;
    records.back().second.freq_power = 2;
  }
  {
    records.emplace_back();
    records.back().first = 3;
    records.back().second.time = 100;
    records.back().second.freq_power = 1;
  }
  {
    records.emplace_back();
    records.back().first = 4;
    records.back().second.time = 150;
    records.back().second.freq_power = 2;
  }
  size_t offset_{0};
  MixedLFULRUStrategy strategy;
  auto ids = strategy.evict(
      [&offset_, &records]() -> std::optional<record_t> {
        if (offset_ == records.size()) {
          return std::nullopt;
        }
        auto record = records[offset_++];
        lxu_record_t ext_type =
            *reinterpret_cast<lxu_record_t*>(&record.second);
        return record_t{
            .global_id = record.first,
            .cache_id = 0,
            .lxu_record = ext_type,
        };
      },
      3);

  ASSERT_EQ(ids.size(), 3);
  ASSERT_EQ(ids[0], 3);
  ASSERT_EQ(ids[1], 2);
  ASSERT_EQ(ids[2], 1);
}

TEST(TDE, MixedLFULRUStrategy_Transform) {
  constexpr static size_t n_iter = 1000000;
  MixedLFULRUStrategy strategy;
  strategy.update_time(10);
  lxu_record_t val;
  {
    val = strategy.update(0, 0, std::nullopt);
    auto record = reinterpret_cast<MixedLFULRUStrategy::Record*>(&val);
    ASSERT_EQ(record->freq_power, 5);
    ASSERT_EQ(record->time, 10);
  }

  uint32_t freq_power_5_cnt = 0;
  uint32_t freq_power_6_cnt = 0;

  for (size_t i = 0; i < n_iter; ++i) {
    auto tmp = strategy.update(0, 0, val);
    auto record = reinterpret_cast<MixedLFULRUStrategy::Record*>(&tmp);
    ASSERT_EQ(record->time, 10);
    if (record->freq_power == 5) {
      ++freq_power_5_cnt;
    } else if (record->freq_power == 6) {
      ++freq_power_6_cnt;
    } else {
      ASSERT_TRUE(record->freq_power == 5 || record->freq_power == 6);
    }
  }

  double freq_6_prob = static_cast<double>(freq_power_6_cnt) /
      static_cast<double>(freq_power_5_cnt + freq_power_6_cnt);

  ASSERT_NEAR(freq_6_prob, 1 / 32.0f, 1e-3);
}

} // namespace torchrec
