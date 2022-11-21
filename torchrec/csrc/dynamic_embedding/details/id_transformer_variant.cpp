/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torchrec/csrc/dynamic_embedding/details/id_transformer_variant.h>
#include <vector>

namespace torchrec {

IDTransformerVariant::IDTransformerVariant(
    LXUStrategyVariant strategy,
    int64_t num_embeddings,
    const std::string& id_transformer_type)
    : strategy_(std::move(strategy)),
      var_(NaiveIDTransformer<uint32_t>(num_embeddings)) {
  if (id_transformer_type != "naive") {
    throw std::invalid_argument(
        "unknown id_transformer_type: " + id_transformer_type);
  }
}

std::vector<int64_t> IDTransformerVariant::evict(int64_t num_to_evict) {
  // Get the ids to evict from lxu strategy.
  std::vector<int64_t> ids_to_evict = std::visit(
      [&](auto&& s) { return strategy_.evict(s.iterator(), num_to_evict); },
      var_);
  // get the cache id of the ids to evict.
  std::vector<int64_t> cache_ids(ids_to_evict.size());
  transform(ids_to_evict, cache_ids);
  std::vector<int64_t> result;
  result.reserve(2 * ids_to_evict.size());
  for (size_t i = 0; i < ids_to_evict.size(); i++) {
    result.emplace_back(ids_to_evict[i]);
    result.emplace_back(cache_ids[i]);
  }
  // Evict ids from the ID transformer.
  std::visit([&](auto&& s) { s.evict(ids_to_evict); }, var_);
  return result;
}

std::vector<int64_t> IDTransformerVariant::save(int64_t time) {
  return std::visit(
      [=, this](auto&& s) {
        std::vector<int64_t> result;
        auto iterator = s.iterator();
        while (true) {
          auto val = iterator();
          if (!val.has_value()) [[unlikely]] {
            break;
          }
          if (strategy_.time(val->lxu_record_) > time) {
            result.emplace_back(val->global_id_);
            result.emplace_back(val->cache_id_);
          }
        }
        return result;
      },
      var_);
}

LXUStrategyVariant::LXUStrategyVariant(
    const std::string& lxu_strategy_type,
    uint16_t min_used_freq_power)
    : strategy_(MixedLFULRUStrategy(min_used_freq_power)) {
  if (lxu_strategy_type != "mixed_lru_lfu") {
    throw std::invalid_argument(
        "unknown lxu_strategy_type: " + lxu_strategy_type);
  }
}

void LXUStrategyVariant::update_time(uint32_t time) {
  return std::visit(
      [&](auto& strategy) { return strategy.update_time(time); }, strategy_);
}

} // namespace torchrec
