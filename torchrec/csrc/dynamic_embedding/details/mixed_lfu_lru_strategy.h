#pragma once
#include <torchrec/csrc/dynamic_embedding/details/naive_id_transformer.h>
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
 * The frequency is recorded as the power number of 2. i.e., The freq_power is 1,
 * then the global id is used approximately 2 times. And the freq_power is 10,
 * then the global id is used approximately 1024 times.
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
class MixedLFULRUStrategy {
 public:
  using lxu_record_t = uint32_t;
  using transformer_record_t = TransformerRecord<lxu_record_t>;

  static constexpr std::string_view type_ = "mixed_lru_lfu";

  /**
   * @param min_used_freq_power min usage is 2^min_used_freq_power. Set this to
   * avoid recent values evict too fast.
   */
  explicit MixedLFULRUStrategy(uint16_t min_used_freq_power = 5);

  MixedLFULRUStrategy(const MixedLFULRUStrategy&) = delete;
  MixedLFULRUStrategy(MixedLFULRUStrategy&& o) noexcept = default;

  void update_time(uint32_t time);
  template <typename T>
  static int64_t time(T record) {
    static_assert(sizeof(T) == sizeof(Record));
    return static_cast<int64_t>(reinterpret_cast<Record*>(&record)->time_);
  }

  lxu_record_t update(
      int64_t global_id,
      int64_t cache_id,
      std::optional<lxu_record_t> val);

  struct EvictItem {
    int64_t global_id_;
    lxu_record_t record_;
    bool operator<(const EvictItem& item) const {
      return record_ < item.record_;
    }
  };

  /**
   * Analysis all ids and returns the num_elems that are most need to evict.
   * @param iterator Returns each global_id to ExtValue pair. Returns nullopt
   * when at ends.
   * @param num_to_evict
   * @return
   */
  template <typename Iterator>
  static std::vector<int64_t> evict(Iterator iterator, uint64_t num_to_evict) {
    std::priority_queue<EvictItem> items;
    while (true) {
      auto val = iterator();
      if (!val.has_value()) [[unlikely]] {
        break;
      }
      EvictItem item{
          .global_id_ = val->global_id_,
          .record_ = reinterpret_cast<Record*>(&val->lxu_record_)->ToUint32(),
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
      result.emplace_back(item.global_id_);
      items.pop();
    }
    std::reverse(result.begin(), result.end());
    return result;
  }

  // Record should only be used in unittest or internally.
  struct Record {
    uint32_t time_ : 27;
    uint16_t freq_power_ : 5;

    [[nodiscard]] uint32_t ToUint32() const {
      return time_ | (freq_power_ << (32 - 5));
    }
  };

 private:
  static_assert(sizeof(Record) == sizeof(lxu_record_t));

  RandomBitsGenerator generator_;
  uint16_t min_lfu_power_;
  std::unique_ptr<std::atomic<uint32_t>> time_;
};

} // namespace torchrec
