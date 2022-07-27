#include "tde/id_transformer.h"

namespace tde {

IDTransformer::IDTransformer(int64_t num_embedding, size_t num_threads)
    : transformer_(num_embedding, num_threads) {}

int64_t IDTransformer::Transform(
    torch::Tensor global_ids,
    torch::Tensor cache_ids) {
  return transformer_.Transform(
      tcb::span{
          global_ids.template data_ptr<int64_t>(),
          static_cast<size_t>(global_ids.numel())},
      tcb::span{
          cache_ids.template data_ptr<int64_t>(),
          static_cast<size_t>(cache_ids.numel())},
      [this](
          std::optional<LXURecord> record,
          int64_t global_id,
          int64_t cache_id) { return strategy_.Transform(record); },
      [this](int64_t global_id, int64_t cache_id) {
        return fetcher_.Fetch(global_id, cache_id);
      });
}

} // namespace tde
