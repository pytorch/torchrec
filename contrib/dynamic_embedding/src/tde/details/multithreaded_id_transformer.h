#pragma once
#include <optional>
#include <vector>
#include "tcb/span.hpp"
#include "tde/details/move_only_function.h"
#include "tde/details/naive_id_transformer.h"
#include "tde/details/thread_pool.h"

namespace tde::details {

namespace transform_default {
template <typename Transformer>
Transformer DefaultCreator(int64_t num_embedding) {
  return Transformer(num_embedding);
}
} // namespace transform_default
template <typename UnderlyingTransformer>
class MultiThreadedIDTransformer {
 public:
  using lxu_record_t = typename UnderlyingTransformer::lxu_record_t;
  using record_t = typename UnderlyingTransformer::record_t;
  static_assert(UnderlyingTransformer::TransformHasFilter);
  static_assert(UnderlyingTransformer::TransformerHasCacheIDTransformer);
  enum {
    TransformUpdateNeedThreadSafe = 1,
    TransformFetchNeedThreadSafe = 1,
    TransformHasFilter = 0,
    TransformerHasCacheIDTransformer = 0,
    TransformCanContinue = 0,
    IsCompose = 1,
  };
  static constexpr std::string_view type_ = "thread";
  using underlying_t = UnderlyingTransformer;

  template <
      typename UnderlyingTransformerCreator =
          decltype(transform_default::DefaultCreator<UnderlyingTransformer>)>
  MultiThreadedIDTransformer(
      int64_t num_embedding,
      size_t num_threads,
      UnderlyingTransformerCreator creator =
          transform_default::DefaultCreator<UnderlyingTransformer>);

  MultiThreadedIDTransformer(
      const MultiThreadedIDTransformer<UnderlyingTransformer>&) = delete;
  MultiThreadedIDTransformer(
      MultiThreadedIDTransformer<UnderlyingTransformer>&&) noexcept = default;

  template <typename UnderlyingCreator>
  static MultiThreadedIDTransformer<underlying_t> Create(
      int64_t num_embedding,
      const nlohmann::json& json,
      UnderlyingCreator creator) {
    auto num_threads = static_cast<size_t>(json["num_threads"]);
    return MultiThreadedIDTransformer<underlying_t>(
        num_embedding, num_threads, std::move(creator));
  }

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

  MoveOnlyFunction<std::optional<record_t>()> Iterator();

 private:
  size_t num_threads_;
  std::unique_ptr<ThreadPool> thread_pool_;
  std::vector<UnderlyingTransformer> transformers_;
  std::vector<int64_t> embedding_offsets_;
};

} // namespace tde::details

#include "tde/details/multithreaded_id_transformer_impl.h"
