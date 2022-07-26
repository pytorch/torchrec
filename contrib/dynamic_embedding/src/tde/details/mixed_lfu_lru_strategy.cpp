#include "mixed_lfu_lru_strategy.h"
#include <algorithm>
#include <queue>
#include "c10/macros/Macros.h"

namespace tde::details {
MixedLFULRUStrategy::MixedLFULRUStrategy(uint16_t min_used_freq_power)
    : min_lfu_power_(min_used_freq_power) {}

void MixedLFULRUStrategy::UpdateTime(uint32_t time) {
  time_.store(time);
}

MixedLFULRUStrategy::ExtendedValueType MixedLFULRUStrategy::Transform(
    std::optional<ExtendedValueType> val) {
  Record r{};
  r.time_ = time_.load();

  if (C10_UNLIKELY(!val.has_value())) {
    r.freq_power_ = min_lfu_power_;
  } else {
    auto freq_power = reinterpret_cast<Record*>(&val.value())->freq_power_;
    bool should_carry = generator_.IsNextNBitsAllZero(freq_power);
    if (should_carry) {
      ++freq_power;
    }
    r.freq_power_ = freq_power;
  }
  return *reinterpret_cast<ExtendedValueType*>(&r);
}

struct SortItems {
  int64_t global_id_;
  uint16_t freq_power_;
  uint32_t time_;

  bool operator<(const SortItems& item) const {
    bool freq_eq = freq_power_ == item.freq_power_;
    if (!freq_eq) { // infrequence global comes first.
      return freq_power_ < item.freq_power_;
    }

    // then, return least recent time comes first.
    return time_ < item.time_;
  }
};

std::vector<int64_t> MixedLFULRUStrategy::Evict(
    MoveOnlyFunction<std::optional<
        std::pair<int64_t, MixedLFULRUStrategy::ExtendedValueType>>()>
        id_visitor,
    uint64_t num_elems_to_evict) {
  std::priority_queue<SortItems> items;
  while (true) {
    auto val = id_visitor();
    if (!val.has_value()) {
      break;
    }

    auto& [global_id, ext_val] = *val;
    auto* record = reinterpret_cast<Record*>(&ext_val);
    SortItems item{
        .global_id_ = global_id,
        .freq_power_ = record->freq_power_,
        .time_ = record->time_,
    };
    items.push(item);
    if (items.size() > num_elems_to_evict) {
      items.pop();
    }
  }

  std::vector<int64_t> result;
  result.reserve(items.size());
  while (!items.empty()) {
    auto item = items.top();
    result.emplace_back(item.global_id_);
    items.pop();
  }
  std::reverse(result.begin(), result.end());
  return result;
}

} // namespace tde::details
