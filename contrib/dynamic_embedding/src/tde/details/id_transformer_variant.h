#pragma once
#include <type_traits>
#include <variant>
#include "nlohmann/json.hpp"
#include "tde/details/cacheline_id_transformer.h"
#include "tde/details/mixed_lfu_lru_strategy.h"
#include "tde/details/naive_id_transformer.h"

namespace tde::details {

struct LXUStrategy {
 private:
  // use to indicate VisitUpdator 's result
  using _UpdateFunctor = std::function<MixedLFULRUStrategy::lxu_record_t(
      std::optional<MixedLFULRUStrategy::lxu_record_t>,
      int64_t,
      int64_t)>;

 public:
  explicit LXUStrategy(const nlohmann::json& json)
      : strategy_(MixedLFULRUStrategy(json.value("min_used_freq_power", 5))) {
    if (auto it = json.find("type"); it != json.end()) {
      TORCH_CHECK(
          static_cast<std::string>(it.value()) == MixedLFULRUStrategy::type_,
          "json type must be mixed_lru_lfu for now");
    } else {
      TORCH_CHECK(false, "type must set");
    }
  }

  void UpdateTime(uint32_t time);

  template <typename Visitor>
  auto VisitUpdator(Visitor visit)
      -> std::invoke_result_t<Visitor, _UpdateFunctor> {
    return std::visit(
        [&](auto& s) {
          using T = typename std::decay_t<decltype(s)>::lxu_record_t;
          auto update = [&](std::optional<T> record,
                            int64_t global_id,
                            int64_t cache_id) {
            return s.Update(global_id, cache_id, record);
          };
          return visit(update);
        },
        strategy_);
  }

  template <typename Iterator>
  std::vector<int64_t> Evict(Iterator iterator, uint64_t num_to_evict) {
    return std::visit(
        [&, iterator = std::move(iterator)](auto& s) mutable {
          return s.Evict(std::move(iterator), num_to_evict);
        },
        strategy_);
  }

 private:
  // Currently, only MixedLFULRUStrategy is the LXUStrategy. Add more strategy
  // when it is necessary.
  using Variant = std::variant<MixedLFULRUStrategy>;
  Variant strategy_;
};

class IDTransformer {
  using Variant = std::
      variant<NaiveIDTransformer<uint32_t>, CachelineIDTransformer<uint32_t>>;

 public:
  IDTransformer(
      LXUStrategy strategy,
      int64_t num_embeddings,
      const std::string& type)
      : strategy_(std::move(strategy)),
        var_(
            type == "naive"
                ? Variant(NaiveIDTransformer<uint32_t>(num_embeddings))
                : Variant(CachelineIDTransformer<uint32_t>(num_embeddings))) {}

  /**
   * Transform GlobalIDs to CacheIDs.
   *
   * @param global_ids
   * @param cache_ids
   * @param fetch Callback when need fetch. By default, do nothing.
   * @return number elems transformed. If the Transformer is full and need to be
   * evict. Then the return value is not equal to global_ids.size();
   */
  template <typename Fetch = decltype(transform_default::NoFetch)>
  bool Transform(
      tcb::span<const int64_t> global_ids,
      tcb::span<int64_t> cache_ids,
      Fetch fetch = transform_default::NoFetch);

  std::vector<int64_t> Evict(int64_t num_to_evict);

  LXUStrategy strategy_;

 private:
  Variant var_;
};

template <typename Fetch>
inline bool IDTransformer::Transform(
    tcb::span<const int64_t> global_ids,
    tcb::span<int64_t> cache_ids,
    Fetch fetch) {
  return strategy_.VisitUpdator([&](auto&& update) -> bool {
    return std::visit(
        [&](auto&& transformer) -> bool {
          return transformer.Transform(
              global_ids,
              cache_ids,
              std::forward<decltype(update)>(update),
              std::move(fetch));
        },
        var_);
  });
}

inline std::vector<int64_t> IDTransformer::Evict(int64_t num_to_evict) {
  // Get the ids to evict from lxu strategy.
  std::vector<int64_t> ids_to_evict = std::visit(
      [&](auto&& s) { return strategy_.Evict(s.Iterator(), num_to_evict); },
      var_);
  // get the cache id of the ids to evict.
  std::vector<int64_t> cache_ids(ids_to_evict.size());
  Transform(ids_to_evict, cache_ids);
  std::vector<int64_t> result;
  result.reserve(2 * ids_to_evict.size());
  for (size_t i = 0; i < ids_to_evict.size(); i++) {
    result.emplace_back(ids_to_evict[i]);
    result.emplace_back(cache_ids[i]);
  }
  // Evict ids from the ID transformer.
  std::visit([&](auto&& s) { s.Evict(ids_to_evict); }, var_);
  return result;
}

} // namespace tde::details
