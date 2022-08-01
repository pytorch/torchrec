#include "tde/id_transformer.h"

namespace tde {

IDTransformer::IDTransformer(int64_t num_embedding, nlohmann::json json)
    : json_(std::move(json)),
      transformer_(
          std::move(details::LXUStrategy(json_["lxu_strategy"])),
          num_embedding,
          json_["id_transformer"]),
      num_ids_to_fetch_(0) {}

std::tuple<int64_t, torch::Tensor> IDTransformer::Transform(
    torch::Tensor global_ids,
    torch::Tensor cache_ids,
    int64_t time) {
  TORCH_CHECK(time >= 0);
  transformer_.strategy_.UpdateTime(static_cast<uint32_t>(time));
  ids_to_fetch_.resize(2 * global_ids.numel());
  int64_t num_transformed = transformer_.Transform(
      tcb::span{
          global_ids.template data_ptr<int64_t>(),
          static_cast<size_t>(global_ids.numel())},
      tcb::span{
          cache_ids.template data_ptr<int64_t>(),
          static_cast<size_t>(cache_ids.numel())},
      [this](int64_t global_id, int64_t cache_id) {
        int64_t idx = num_ids_to_fetch_.fetch_add(1);
        ids_to_fetch_[2 * idx] = global_id;
        ids_to_fetch_[2 * idx + 1] = cache_id;
      });
  int64_t num_ids_to_fetch = num_ids_to_fetch_.load();
  if (num_ids_to_fetch == 0) {
    return {num_transformed, torch::Tensor{}};
  }
  torch::Tensor ids_to_fetch = torch::from_blob(
                                   ids_to_fetch_.data(),
                                   {num_ids_to_fetch, 2},
                                   torch::dtype(torch::kLong))
                                   .clone();
  num_ids_to_fetch_.store(0);
  return {num_transformed, ids_to_fetch};
}

} // namespace tde
