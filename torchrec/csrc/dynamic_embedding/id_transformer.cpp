/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torchrec/csrc/dynamic_embedding/id_transformer.h>

namespace torchrec {

IDTransformer::IDTransformer(IDTransformerVariant transformer)
    : transformer_(std::move(transformer)), time_(-1), last_save_time_(-1) {}

c10::intrusive_ptr<TransformResult> IDTransformer::transform(
    c10::intrusive_ptr<TensorList> global_id_list,
    c10::intrusive_ptr<TensorList> cache_id_list,
    int64_t time) {
  std::lock_guard<std::mutex> lock(mu_);
  torch::NoGradGuard no_grad;
  TORCH_CHECK(time >= 0);
  TORCH_CHECK(time >= time_, "Time cannot go backward");
  time_ = time;
  TORCH_CHECK(global_id_list->size() == cache_id_list->size());
  transformer_.strategy_.update_time(static_cast<uint32_t>(time));
  {
    int64_t total_num_embeddings = std::accumulate(
        global_id_list->begin(),
        global_id_list->end(),
        int64_t(0),
        [](int64_t v, auto&& tensor) -> int64_t { return v + tensor.numel(); });
    ids_to_fetch_.resize(2 * total_num_embeddings);
  }

  std::atomic<int64_t> next_fetch_offset{0};
  bool ok = true;
  for (int64_t i = 0; i < global_id_list->size(); ++i) {
    auto& global_ids = (*global_id_list)[i];
    auto& cache_ids = (*cache_id_list)[i];
    ok = transformer_.transform(
        std::span{
            global_ids.data_ptr<int64_t>(),
            static_cast<size_t>(global_ids.numel())},
        std::span{
            cache_ids.data_ptr<int64_t>(),
            static_cast<size_t>(cache_ids.numel())},
        [&](int64_t global_id, int64_t cache_id) {
          int64_t offset = next_fetch_offset.fetch_add(1);
          ids_to_fetch_[2 * offset] = global_id;
          ids_to_fetch_[2 * offset + 1] = cache_id;
        });
    if (!ok) {
      break;
    }
  }

  return c10::make_intrusive<TransformResult>(
      ok,
      at::from_blob(
          ids_to_fetch_.data(),
          {next_fetch_offset.load(), 2},
          torch::TensorOptions().dtype(c10::kLong).device(c10::kCPU)));
}

torch::Tensor IDTransformer::evict(int64_t num_to_evict) {
  std::lock_guard<std::mutex> lock(mu_);
  torch::NoGradGuard no_grad;
  std::vector<int64_t> ids_to_evict = transformer_.evict(num_to_evict);
  int64_t num_ids_to_evict = ids_to_evict.size() / 2;
  return torch::tensor(ids_to_evict, torch::dtype(torch::kLong))
      .reshape({num_ids_to_evict, 2});
}

torch::Tensor IDTransformer::save() {
  std::lock_guard<std::mutex> lock(mu_);
  torch::NoGradGuard no_grad;
  std::vector<int64_t> ids = transformer_.save(last_save_time_);
  last_save_time_ = time_;
  int64_t num_ids = ids.size() / 2;
  return torch::tensor(ids, torch::dtype(torch::kLong)).reshape({num_ids, 2});
}

} // namespace torchrec
