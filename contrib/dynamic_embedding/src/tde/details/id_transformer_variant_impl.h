#pragma once

namespace tde::details {

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

template <typename Visitor>
inline auto IDTransformer::LXUStrategy::VisitUpdator(Visitor visit)
    -> std::invoke_result_t<Visitor, _UpdateFunctor> {
  return std::visit(
      [&](auto& s) {
        using T = typename std::decay_t<decltype(s)>::lxu_record_t;
        auto update =
            [&](std::optional<T> record, int64_t global_id, int64_t cache_id) {
              return s.Update(global_id, cache_id, record);
            };
        return visit(update);
      },
      strategy_);
}

template <typename T>
int64_t IDTransformer::LXUStrategy::Time(T record) {
  return std::visit(
      [&](auto& s) -> int64_t { return s.Time(record); }, strategy_);
}

template <typename Iterator>
inline std::vector<int64_t> IDTransformer::LXUStrategy::Evict(
    Iterator iterator,
    uint64_t num_to_evict) {
  return std::visit(
      [&, iterator = std::move(iterator)](auto& s) mutable {
        return s.Evict(std::move(iterator), num_to_evict);
      },
      strategy_);
}

} // namespace tde::details
