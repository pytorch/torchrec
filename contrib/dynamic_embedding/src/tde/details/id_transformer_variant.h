#pragma once
#include <type_traits>
#include <variant>
#include "nlohmann/json.hpp"
#include "tde/details/cacheline_id_transformer.h"
#include "tde/details/mixed_lfu_lru_strategy.h"
#include "tde/details/naive_id_transformer.h"

namespace tde::details {

class IDTransformer {
  using Variant = std::
      variant<NaiveIDTransformer<uint32_t>, CachelineIDTransformer<uint32_t>>;

 public:
  IDTransformer(int64_t num_embeddings, nlohmann::json json);

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

  struct LXUStrategy {
   private:
    // use to indicate VisitUpdator 's result
    using _UpdateFunctor = std::function<MixedLFULRUStrategy::lxu_record_t(
        std::optional<MixedLFULRUStrategy::lxu_record_t>,
        int64_t,
        int64_t)>;

   public:
    explicit LXUStrategy(const nlohmann::json& json);

    void UpdateTime(uint32_t time);

    template <typename Visitor>
    auto VisitUpdator(Visitor visit)
        -> std::invoke_result_t<Visitor, _UpdateFunctor>;

    template <typename Iterator>
    std::vector<int64_t> Evict(Iterator iterator, uint64_t num_to_evict);

   private:
    // Currently, only MixedLFULRUStrategy is the LXUStrategy. Add more strategy
    // when it is necessary.
    using Variant = std::variant<MixedLFULRUStrategy>;
    Variant strategy_;
  };

  LXUStrategy strategy_;

 private:
  Variant var_;
};

} // namespace tde::details

#include "tde/details/id_transformer_variant_impl.h"
