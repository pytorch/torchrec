#pragma once
#include <optional>
#include <vector>
#include "tcb/span.hpp"
namespace tde::details {

template <typename ExtendValueType>
class NaiveIDTransformer {
 public:
  template <typename ExtendValueTransformer, typename OnFetch, typename Skip>
  int64_t Transform(
      tcb::span<const int64_t> global_ids,
      tcb::span<int64_t> cache_ids,
      ExtendValueTransformer transform_ext_value =
          [](std::optional<ExtendValueType> ext,
             int64_t global_id,
             int64_t cache_id) -> ExtendValueType { return ExtendValueType{}; },
      OnFetch on_fetch = [](int64_t global_id, int64_t cache_id) {},
      Skip skip = [](int64_t global_id) -> bool { return false; });

  template <typename Callback>
  void ForEach(
      Callback callback = [](int64_t global_id,
                             int64_t cache_id,
                             ExtendValueType ext_value) {});

  void Evict(tcb::span<const int64_t> global_ids);
};

} // namespace tde::details
