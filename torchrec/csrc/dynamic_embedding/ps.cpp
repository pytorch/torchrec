/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torchrec/csrc/dynamic_embedding/details/io.h>
#include <torchrec/csrc/dynamic_embedding/ps.h>

namespace torchrec {

c10::intrusive_ptr<FetchHandle> PS::fetch(
    torch::Tensor ids_to_fetch,
    int64_t time,
    bool reinit,
    double weight_init_min,
    double weight_init_max) {
  std::lock_guard<std::mutex> lock(mu_);
  torch::NoGradGuard no_grad;

  auto [local_global_ids, local_cache_ids] = filter_local_ids(ids_to_fetch);
  if (local_global_ids.empty()) {
    return c10::make_intrusive<FetchHandle>(time, c10::intrusive_ptr<PS>());
  }

  fetch_notifications_.emplace_back(time, c10::make_intrusive<Notification>());
  c10::intrusive_ptr<Notification> notification =
      fetch_notifications_.back().second;
  // Does not support multiple col ids at the moment.
  std::vector<int64_t> col_ids{0};
  uint32_t num_os_ids = os_ids_.size();
  io_.fetch(
      table_name_,
      std::move(local_global_ids),
      col_ids,
      num_os_ids,
      torch::kF32,
      [=, this, cache_ids_to_fetch = std::move(local_cache_ids)](auto&& val) {
        TORCH_CHECK(val.size() == cache_ids_to_fetch.size());
        for (uint32_t i = 0; i < cache_ids_to_fetch.size(); ++i) {
          int64_t cache_id = cache_ids_to_fetch[i];
          auto& fetched = val[i];
          if (!fetched.defined()) {
            if (reinit) {
              std::vector<torch::Tensor> tensors = get_tensor_views(cache_id);
              tensors[0].uniform_(weight_init_min, weight_init_max);
              // optimizer states will be set to zero
              for (uint32_t j = 1; j < num_os_ids; ++j) {
                tensors[j].zero_();
              }
            }
            continue;
          }

          std::vector<torch::Tensor> tensors = get_tensor_views(cache_id);
          for (uint32_t j = 0; j < num_os_ids; ++j) {
            tensors[j].copy_(fetched.slice(0, j, j + 1));
          }
        }
        notification->done();
      });
  // `unsafe_reclain_from_nonowning` is the `instrusive_ptr` version of
  // `enable_shared_from_this`
  return c10::make_intrusive<FetchHandle>(
      time, c10::intrusive_ptr<PS>::unsafe_reclaim_from_nonowning(this));
}

void PS::evict(torch::Tensor ids_to_evict) {
  std::lock_guard<std::mutex> lock(mu_);
  torch::NoGradGuard no_grad;
  // make sure all previous fetches are done.
  synchronize_fetch();

  auto [local_global_ids, local_cache_ids] = filter_local_ids(ids_to_evict);
  if (local_global_ids.empty()) {
    return;
  }

  // Does not support multiple col ids at the moment.
  std::vector<int64_t> col_ids{0};
  uint32_t num_os_ids = os_ids_.size();
  uint32_t num_ids_to_fetch = local_global_ids.size();

  Notification notification;
  // Done first so that the Wait after preparing the first chunk won't stuck.
  notification.done();
  // The shared data for all chunks.
  std::vector<uint64_t> offsets;
  offsets.resize(num_ids_per_chunk_ * num_os_ids * col_ids.size() + 1);
  // Evict by chunks
  for (uint32_t i = 0; i < num_ids_to_fetch; i += num_ids_per_chunk_) {
    uint32_t num_ids_in_chunk = std::min(
        static_cast<uint32_t>(num_ids_per_chunk_), num_ids_to_fetch - i);
    uint32_t data_size = num_ids_in_chunk * num_os_ids * col_ids.size();
    uint32_t offsets_size = num_ids_in_chunk * num_os_ids * col_ids.size() + 1;

    std::vector<torch::Tensor> all_tensors;
    for (uint32_t j = i; j < i + num_ids_in_chunk; ++j) {
      int64_t cache_id = local_cache_ids[j];
      std::vector<torch::Tensor> tensors = get_tensor_views(cache_id);
      all_tensors.insert(all_tensors.end(), tensors.begin(), tensors.end());
    }
    torch::Tensor data = torch::cat(all_tensors, 0).cpu();
    TORCH_CHECK(data.numel() == data_size * col_size_);

    offsets[0] = 0;
    for (uint32_t j = 0; j < all_tensors.size(); ++j) {
      offsets[j + 1] =
          offsets[j] + all_tensors[j].numel() * all_tensors[j].element_size();
    }
    // waiting for the Push of last chunk finishes.
    notification.wait();
    notification.clear();
    io_.push(
        table_name_,
        std::span{local_global_ids.data() + i, num_ids_in_chunk},
        col_ids,
        os_ids_,
        std::span{
            reinterpret_cast<uint8_t*>(data.data_ptr<float>()),
            data_size * sizeof(float)},
        std::span{offsets.data(), offsets_size},
        [&notification] { notification.done(); });
  }
  notification.wait();
}

void PS::synchronize_fetch(int64_t time) {
  while (!fetch_notifications_.empty()) {
    auto& [t, notification] = fetch_notifications_.front();
    if (t != time && time >= 0) {
      break;
    }
    notification->wait();
    fetch_notifications_.pop_front();
  }
}

std::vector<torch::Tensor> PS::get_tensor_views(int64_t cache_id) {
  for (auto& shard : *shards_) {
    if (shard.has(cache_id)) {
      return shard.get_tensor_view(cache_id);
    }
  }
  TORCH_CHECK(false, "all local shards do not contain cache id ", cache_id);
}

std::tuple<std::vector<int64_t>, std::vector<int64_t>> PS::filter_local_ids(
    const torch::Tensor& ids) {
  std::vector<int64_t> local_global_ids;
  std::vector<int64_t> local_cache_ids;
  TORCH_CHECK(ids.is_contiguous());
  TORCH_CHECK(ids.dim() == 2);
  auto* ids_ptr = ids.data_ptr<int64_t>();
  int64_t numel = ids.numel();
  for (int64_t i = 0; i < numel; i += 2) {
    auto cache_id = ids_ptr[i + 1];
    if (std::any_of(shards_->begin(), shards_->end(), [&](auto&& shard) {
          return shard.has(cache_id);
        })) {
      auto global_id = ids_ptr[i];
      local_global_ids.emplace_back(global_id);
      local_cache_ids.emplace_back(cache_id);
    }
  }
  return {std::move(local_global_ids), std::move(local_cache_ids)};
}

} // namespace torchrec
