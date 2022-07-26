#pragma once
#include <optional>
#include <vector>
#include "tcb/span.hpp"
#include "tde/details/naive_id_transformer.h"
#include "tde/details/thread_pool.h"

namespace tde::details {

template <typename Tag>
class MultiThreadedIDTransformer {
 public:
  MultiThreadedIDTransformer(int64_t num_embedding, size_t num_threads);

  /**
   * Transform GlobalIDs to CacheIDs.
   *
   * @param global_ids
   * @param cache_ids
   * @param fetch Callback when need fetch. By default, do nothing.
   * @return number elems transformed. If the Transformer is full and need to be
   * evict. Then the return value is not equal to global_ids.size();
   */
  template <
      typename Update = decltype(transform_default::NoUpdate<Tag>),
      typename Fetch = decltype(transform_default::NoFetch)>
  int64_t Transform(
      tcb::span<const int64_t> global_ids,
      tcb::span<int64_t> cache_ids,
      Update update = transform_default::NoUpdate<Tag>,
      Fetch fetch = transform_default::NoFetch);

  template <typename Callback>
  void ForEach(
      Callback callback = [](int64_t global_id, int64_t cache_id, Tag tag) {});

  void Evict(tcb::span<const int64_t> global_ids);

 private:
  size_t num_threads_;
  ThreadPool thread_pool_;
  std::vector<NaiveIDTransformer<Tag, uint32_t>> transformers_;
};

} // namespace tde::details

#include "tde/details/multithreaded_id_transformer_impl.h"
