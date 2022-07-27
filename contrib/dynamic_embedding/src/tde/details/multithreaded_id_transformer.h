#pragma once
#include <optional>
#include <vector>
#include "tcb/span.hpp"
#include "tde/details/naive_id_transformer.h"
#include "tde/details/thread_pool.h"

namespace tde::details {

template <typename UnderlyingTransformer>
class MultiThreadedIDTransformer {
 public:
  using lxu_record_t = typename UnderlyingTransformer::lxu_record_t;
  static_assert(UnderlyingTransformer::TransformHasFilter);
  static_assert(UnderlyingTransformer::TransformerHasCacheIDTransformer);
  enum {
    TransformUpdateNeedThreadSafe = 1,
    TransformFetchNeedThreadSafe = 1,
    TransformHasFilter = 0,
    TransformerHasCacheIDTransformer = 0,
    TransformCanContinue = 0,
  };

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
      typename Update = decltype(transform_default::NoUpdate<lxu_record_t>),
      typename Fetch = decltype(transform_default::NoFetch)>
  int64_t Transform(
      tcb::span<const int64_t> global_ids,
      tcb::span<int64_t> cache_ids,
      Update update = transform_default::NoUpdate<lxu_record_t>,
      Fetch fetch = transform_default::NoFetch);

  template <typename Callback>
  void ForEach(
      Callback callback =
          [](int64_t global_id, int64_t cache_id, lxu_record_t tag) {});

  void Evict(tcb::span<const int64_t> global_ids);

 private:
  size_t num_threads_;
  ThreadPool thread_pool_;
  std::vector<UnderlyingTransformer> transformers_;
  std::vector<int64_t> embedding_offsets_;
};

} // namespace tde::details

#include "tde/details/multithreaded_id_transformer_impl.h"
