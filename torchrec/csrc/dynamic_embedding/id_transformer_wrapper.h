/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <torch/custom_class.h>
#include <torch/torch.h>
#include <torchrec/csrc/dynamic_embedding/details/id_transformer.h>
#include <torchrec/csrc/dynamic_embedding/details/lxu_strategy.h>

namespace torchrec {

struct TransformResult : public torch::CustomClassHolder {
  TransformResult(bool success, torch::Tensor ids_to_fetch)
      : success(success), ids_to_fetch(ids_to_fetch) {}

  // Whether the fetch succeeded (if evicted is not necessary)
  bool success;
  // new ids to fetch from PS.
  // shape of [num_to_fetch, 2], where each row is consist of
  // the global id and cache id of each ID.
  torch::Tensor ids_to_fetch;
};

class IDTransformerWrapper : public torch::CustomClassHolder {
 public:
  IDTransformerWrapper(
      int64_t num_embedding,
      const std::string& id_transformer_type,
      const std::string& lxu_strategy_type,
      int64_t min_used_freq_power = 5);

  c10::intrusive_ptr<TransformResult> transform(
      std::vector<torch::Tensor> global_ids,
      std::vector<torch::Tensor> cache_ids,
      int64_t time);
  torch::Tensor evict(int64_t num_to_evict);
  torch::Tensor save();

 private:
  std::mutex mu_;
  std::unique_ptr<IDTransformer> transformer_;
  std::unique_ptr<LXUStrategy> strategy_;
  std::vector<int64_t> ids_to_fetch_;
  int64_t time_;
  int64_t last_save_time_;
};

} // namespace torchrec
