/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <vector>

namespace torchrec {

template <typename T>
NaiveIDTransformer<T>::NaiveIDTransformer(int64_t num_embedding)
    : bitmap_(num_embedding) {
  global_id2cache_value_.reserve(num_embedding);
}

template <typename T>
bool NaiveIDTransformer<T>::transform(
    std::span<const int64_t> global_ids,
    std::span<int64_t> cache_ids,
    update_t update,
    fetch_t fetch) {
  for (size_t i = 0; i < global_ids.size(); ++i) {
    int64_t global_id = global_ids[i];
    auto iter = global_id2cache_value_.find(global_id);
    // cache_id is in [0, num_embedding)
    int64_t cache_id;
    if (iter != global_id2cache_value_.end()) {
      cache_id = iter->second.cache_id;
      iter->second.lxu_record =
          update(iter->second.lxu_record, global_id, cache_id);
    } else {
      // The transformer is full.
      if (C10_UNLIKELY(bitmap_.full())) {
        return false;
      }
      auto stored_cache_id = bitmap_.next_free_bit();
      cache_id = stored_cache_id;
      lxu_record_t record = update(std::nullopt, global_id, cache_id);
      global_id2cache_value_.emplace(
          global_id, CacheValue{stored_cache_id, record});
      fetch(global_id, cache_id);
    }
    cache_ids[i] = cache_id;
  }
  return true;
}

template <typename T>
void NaiveIDTransformer<T>::evict(std::span<const int64_t> global_ids) {
  for (const int64_t global_id : global_ids) {
    auto iter = global_id2cache_value_.find(global_id);
    if (iter == global_id2cache_value_.end()) {
      continue;
    }
    int64_t cache_id = iter->second.cache_id;
    global_id2cache_value_.erase(iter);
    bitmap_.free_bit(cache_id);
  }
}

template <typename T>
iterator_t NaiveIDTransformer<T>::iterator() const {
  auto iter = global_id2cache_value_.begin();
  return [iter, this]() mutable -> std::optional<record_t> {
    if (iter != global_id2cache_value_.end()) {
      auto record = record_t{
          .global_id = iter->first,
          .cache_id = iter->second.cache_id,
          .lxu_record = iter->second.lxu_record,
      };
      iter++;
      return record;
    } else {
      return {};
    }
  };
}

} // namespace torchrec
