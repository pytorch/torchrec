#pragma once

namespace tde::details {

template <typename Tag>
MultiThreadedIDTransformer<Tag>::MultiThreadedIDTransformer(
    int64_t num_embedding,
    size_t num_threads)
    : num_threads_(num_threads), thread_pool_(num_threads) {
  transformers_.reserve(num_threads);
  int64_t embedding_per_transformer = num_embedding / num_threads;
  int64_t embedding_offset = 0;
  for (size_t i = 0; i < num_threads; i++) {
    transformers_.emplace_back(
        i == num_threads - 1 ? num_embedding - embedding_offset
                             : embedding_per_transformer,
        embedding_offset);
    embedding_offset += embedding_per_transformer;
  }
}

template <typename Tag>
template <typename Update, typename Fetch>
int64_t MultiThreadedIDTransformer<Tag>::Transform(
    tcb::span<const int64_t> global_ids,
    tcb::span<int64_t> cache_ids,
    Update update,
    Fetch fetch) {
  std::vector<std::future<int64_t>> futures;
  futures.reserve(num_threads_);
  for (size_t i = 0; i < num_threads_; ++i) {
    futures.emplace_back(std::move(thread_pool_.Enqueue([&, this, i] {
      return transformers_[i].Transform(
          global_ids,
          cache_ids,
          [n = num_threads_, i](int64_t global_ids) {
            return global_ids % static_cast<int64_t>(n) == i;
          },
          update,
          fetch);
    })));
  }
  int64_t num_transformed = 0;
  for (size_t i = 0; i < num_threads_; ++i) {
    num_transformed += futures[i].get();
  }
  return num_transformed;
}

template <typename Tag>
void MultiThreadedIDTransformer<Tag>::Evict(
    tcb::span<const int64_t> global_ids) {
  for (size_t i = 0; i < num_threads_; ++i) {
    transformers_[i].Evict(global_ids);
  }
}

} // namespace tde::details
