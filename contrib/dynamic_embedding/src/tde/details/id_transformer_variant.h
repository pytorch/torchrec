#pragma once
#include <type_traits>
#include "nlohmann/json.hpp"
#include "tde/details/id_transformer_registry.h"
namespace tde::details {

struct LXUStrategy {
 private:
  using _FirstLXURecord = std::tuple_element_t<0, to_tuple_t<LXURecordTypes>>;
  using _UpdateFunctor = std::function<
      _FirstLXURecord(std::optional<_FirstLXURecord>, int64_t, int64_t)>;

 public:
  using Variant = to_variant_t<LXUStrategies>;
  using lxu_record_t = to_variant_t<LXURecordTypes>;

  explicit LXUStrategy(const nlohmann::json& json);

  void UpdateTime(uint32_t time);

  lxu_record_t DefaultRecordValue();

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

  Variant strategy_;
};

struct IDTransformer {
  using Variant = to_variant_t<Transformers>;

  IDTransformer(
      LXUStrategy strategy,
      int64_t num_embeddings,
      const nlohmann::json& json);

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
  std::vector<int64_t> result(2 * ids_to_evict.size());
  for (size_t i = 0; i < ids_to_evict.size(); i++) {
    result[2 * i] = ids_to_evict[i];
    result[2 * i + 1] = cache_ids[i];
  }
  // Evict ids from the ID transformer.
  std::visit([&](auto&& s) { s.Evict(ids_to_evict); }, var_);
  return result;
}

} // namespace tde::details
