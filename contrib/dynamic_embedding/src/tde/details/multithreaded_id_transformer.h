#pragma once
#include <vector>
#include "tcb/span.hpp"
namespace tde::details {

class MultiThreadedIDTransformer {
 public:
  /**
   * Transform GlobalIDs to CacheIDs.
   *
   * @param global_ids
   * @param cache_ids
   * @param on_fetch Callback when need fetch. By default, do nothing.
   * @return number elems transformed. If the Transformer is full and need to be
   * evict. Then the return value is not equal to global_ids.size();
   */
  template <typename OnFetch>
  int64_t Transform(
      tcb::span<const int64_t> global_ids,
      tcb::span<int64_t> cache_ids,
      OnFetch on_fetch = [](int64_t global_id, int64_t cache_id) {});

  int64_t Lookup(int64_t global_id) const;

  void Evict(tcb::span<const int64_t> global_ids);
};

} // namespace tde::details

#include "tde/details/multithreaded_id_transformer_impl.h"
