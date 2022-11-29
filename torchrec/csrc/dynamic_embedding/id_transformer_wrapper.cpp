/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torchrec/csrc/dynamic_embedding/details/mixed_lfu_lru_strategy.h>
#include <torchrec/csrc/dynamic_embedding/details/naive_id_transformer.h>
#include <torchrec/csrc/dynamic_embedding/id_transformer_wrapper.h>

namespace torchrec {

IDTransformerWrapper::IDTransformerWrapper(
    int64_t num_embedding,
    const std::string& id_transformer_type,
    const std::string& lxu_strategy_type,
    int64_t min_used_freq_power)
    : time_(-1), last_save_time_(-1) {
  TORCH_CHECK(id_transformer_type == "naive");
  TORCH_CHECK(lxu_strategy_type == "mixed_lru_lfu");
  transformer_ =
      std::unique_ptr<IDTransformer>(new NaiveIDTransformer(num_embedding));
  strategy_ = std::unique_ptr<LXUStrategy>(
      new MixedLFULRUStrategy(min_used_freq_power));
}

c10::intrusive_ptr<TransformResult> IDTransformerWrapper::transform(
    std::vector<torch::Tensor> global_id_list,
    std::vector<torch::Tensor> cache_id_list,
    int64_t time) {
  std::lock_guard<std::mutex> lock(mu_);
  torch::NoGradGuard no_grad;
  TORCH_CHECK(time >= 0);
  TORCH_CHECK(time >= time_, "Time cannot go backward");
  time_ = time;
  TORCH_CHECK(global_id_list.size() == cache_id_list.size());
  strategy_->update_time(static_cast<uint32_t>(time));
  {
    int64_t total_num_embeddings = std::accumulate(
        global_id_list.begin(),
        global_id_list.end(),
        int64_t(0),
        [](int64_t v, auto&& tensor) -> int64_t { return v + tensor.numel(); });
    ids_to_fetch_.resize(2 * total_num_embeddings);
  }

  update_t update = [this](
                        int64_t global_id,
                        int64_t cache_id,
                        std::optional<lxu_record_t> lxu_record) {
    return strategy_->update(global_id, cache_id, lxu_record);
  };
  std::atomic<int64_t> next_fetch_offset{0};
  fetch_t fetch = [&, this](int64_t global_id, int64_t cache_id) {
    int64_t offset = next_fetch_offset.fetch_add(1);
    ids_to_fetch_[2 * offset] = global_id;
    ids_to_fetch_[2 * offset + 1] = cache_id;
  };

  bool ok = true;
  for (int64_t i = 0; i < global_id_list.size(); ++i) {
    auto& global_ids = global_id_list[i];
    auto& cache_ids = cache_id_list[i];
    ok = transformer_->transform(
        std::span{
            global_ids.data_ptr<int64_t>(),
            static_cast<size_t>(global_ids.numel())},
        std::span{
            cache_ids.data_ptr<int64_t>(),
            static_cast<size_t>(cache_ids.numel())},
        update,
        fetch);
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

torch::Tensor IDTransformerWrapper::evict(int64_t num_to_evict) {
  std::lock_guard<std::mutex> lock(mu_);
  torch::NoGradGuard no_grad;
  // get the global ids to evict.
  std::vector<int64_t> global_ids_to_evict =
      strategy_->evict(transformer_->iterator(), num_to_evict);
  int64_t num_ids_to_evict = global_ids_to_evict.size();
  // get the cache id from transformer_
  std::vector<int64_t> cache_ids_to_evict(num_ids_to_evict);
  transformer_->transform(global_ids_to_evict, cache_ids_to_evict);
  // evict the global ids from transformer_
  transformer_->evict(global_ids_to_evict);

  std::vector<int64_t> ids_to_evict(num_ids_to_evict * 2);
  for (int64_t i = 0; i < num_ids_to_evict; ++i) {
    ids_to_evict[2 * i] = global_ids_to_evict[i];
    ids_to_evict[2 * i + 1] = cache_ids_to_evict[i];
  }
  return torch::tensor(ids_to_evict, torch::dtype(torch::kLong))
      .reshape({num_ids_to_evict, 2});
}

torch::Tensor IDTransformerWrapper::save() {
  std::lock_guard<std::mutex> lock(mu_);
  torch::NoGradGuard no_grad;
  // traverse transformer_ and get the id with new timestamp.
  std::vector<int64_t> ids;
  iterator_t iterator = transformer_->iterator();
  while (true) {
    auto val = iterator();
    if (!val.has_value()) [[unlikely]] {
      break;
    }
    if (strategy_->time(val->lxu_record) > last_save_time_) {
      ids.emplace_back(val->global_id);
      ids.emplace_back(val->cache_id);
    }
  }

  last_save_time_ = time_;
  int64_t num_ids = ids.size() / 2;
  return torch::tensor(ids, torch::dtype(torch::kLong)).reshape({num_ids, 2});
}

} // namespace torchrec
