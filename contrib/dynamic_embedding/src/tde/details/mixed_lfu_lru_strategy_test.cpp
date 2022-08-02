#include "gtest/gtest.h"
#include "tde/details/mixed_lfu_lru_strategy.h"

namespace tde::details {

TEST(TDE, MixedLFULRUStrategy_Evict) {
  std::vector<std::pair<int64_t, MixedLFULRUStrategy::Record>> records;
  {
    records.emplace_back();
    records.back().first = 1;
    records.back().second.time_ = 100;
    records.back().second.freq_power_ = 2;
  }
  {
    records.emplace_back();
    records.back().first = 2;
    records.back().second.time_ = 10;
    records.back().second.freq_power_ = 2;
  }
  {
    records.emplace_back();
    records.back().first = 3;
    records.back().second.time_ = 100;
    records.back().second.freq_power_ = 1;
  }
  size_t offset_{0};
  auto ids = MixedLFULRUStrategy::Evict(
      [&offset_, &records]()
          -> std::optional<
              std::pair<int64_t, MixedLFULRUStrategy::lxu_record_t>> {
        if (offset_ == records.size()) {
          return std::nullopt;
        }
        auto record = records[offset_++];
        MixedLFULRUStrategy::lxu_record_t ext_type =
            *reinterpret_cast<MixedLFULRUStrategy::lxu_record_t*>(
                &record.second);

        return std::pair<int64_t, MixedLFULRUStrategy::lxu_record_t>{
            record.first, ext_type};
      },
      2);

  ASSERT_EQ(ids.size(), 2);
  ASSERT_EQ(ids[0], 3);
  ASSERT_EQ(ids[1], 2);
}

TEST(TDE, MixedLFULRUStrategy_Transform) {
  constexpr static size_t n_iter = 1000000;
  MixedLFULRUStrategy strategy;
  strategy.UpdateTime(10);
  MixedLFULRUStrategy::lxu_record_t val;
  {
    val = strategy.Transform(std::nullopt);
    auto record = reinterpret_cast<MixedLFULRUStrategy::Record*>(&val);
    ASSERT_EQ(record->freq_power_, 5);
    ASSERT_EQ(record->time_, 10);
  }

  uint32_t freq_power_5_cnt = 0;
  uint32_t freq_power_6_cnt = 0;

  for (size_t i = 0; i < n_iter; ++i) {
    auto tmp = strategy.Transform(val);
    auto record = reinterpret_cast<MixedLFULRUStrategy::Record*>(&tmp);
    ASSERT_EQ(record->time_, 10);
    if (record->freq_power_ == 5) {
      ++freq_power_5_cnt;
    } else if (record->freq_power_ == 6) {
      ++freq_power_6_cnt;
    } else {
      ASSERT_TRUE(record->freq_power_ == 5 || record->freq_power_ == 6);
    }
  }

  double freq_6_prob = static_cast<double>(freq_power_6_cnt) /
      static_cast<double>(freq_power_5_cnt + freq_power_6_cnt);

  ASSERT_NEAR(freq_6_prob, 1 / 32.0f, 1e-3);
}

} // namespace tde::details
