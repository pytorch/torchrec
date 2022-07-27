#pragma once
#include <algorithm>
#include <vector>
#include "tde/details/bits_op.h"

namespace tde::details {

template <typename T>
inline Bitmap<T>::Bitmap(int64_t num_bits)
    : num_total_bits_(num_bits),
      num_values_((num_bits + num_bits_per_value - 1) / num_bits_per_value),
      values_(new T[num_values_]),
      next_free_bit_(0) {
  std::fill(values_.get(), values_.get() + num_values_, -1);
}

template <typename T>
inline int64_t Bitmap<T>::NextFreeBit() {
  int64_t result = next_free_bit_;
  int64_t offset = result / num_bits_per_value;
  T value = values_[offset];
  // set the last 1 bit to zero
  values_[offset] = value & (value - 1);
  while (values_[offset] == 0 && offset < num_values_) {
    offset++;
  }
  value = values_[offset];
  if (C10_LIKELY(value)) {
    next_free_bit_ = offset * num_bits_per_value + Ctz(value);
  } else {
    next_free_bit_ = num_total_bits_;
  }

  return result;
}

template <typename T>
inline void Bitmap<T>::FreeBit(int64_t offset) {
  int64_t mask_offset = offset / num_bits_per_value;
  int64_t bit_offset = offset % num_bits_per_value;
  values_[mask_offset] |= 1 << bit_offset;
  next_free_bit_ = std::min(offset, next_free_bit_);
}
template <typename T>
inline bool Bitmap<T>::Full() const {
  return next_free_bit_ >= num_total_bits_;
}

template <typename Tag, typename T>
inline NaiveIDTransformer<Tag, T>::NaiveIDTransformer(int64_t num_embedding)
    : bitmap_(num_embedding) {
  global_id2cache_value_.reserve(num_embedding);
}

template <typename Tag, typename T>
template <
    typename Filter,
    typename CacheIDTransformer,
    typename Update,
    typename Fetch>
inline int64_t NaiveIDTransformer<Tag, T>::Transform(
    tcb::span<const int64_t> global_ids,
    tcb::span<int64_t> cache_ids,
    Filter filter,
    CacheIDTransformer cache_id_transformer,
    Update update,
    Fetch fetch) {
  int64_t num_transformed = 0;
  for (size_t i = 0; i < global_ids.size(); ++i) {
    int64_t global_id = global_ids[i];
    if (!filter(global_id)) {
      continue;
    }
    auto iter = global_id2cache_value_.find(global_id);
    // cache_id is in [0, num_embedding)
    int64_t cache_id;
    if (iter != global_id2cache_value_.end()) {
      cache_id = cache_id_transformer(iter->second.cache_id_);
      iter->second.lxu_record_ =
          update(iter->second.lxu_record_, global_id, cache_id);
    } else {
      // The transformer is full.
      if (C10_UNLIKELY(bitmap_.Full())) {
        break;
      }
      auto stored_cache_id = bitmap_.NextFreeBit();
      cache_id = cache_id_transformer(stored_cache_id);
      Tag tag = update(std::nullopt, global_id, cache_id);
      global_id2cache_value_.emplace(
          global_id, CacheValue{stored_cache_id, tag});
      fetch(global_id, cache_id);
    }
    cache_ids[i] = cache_id;
    num_transformed++;
  }
  return num_transformed;
}

template <typename Tag, typename T>
template <typename Callback>
inline void NaiveIDTransformer<Tag, T>::ForEach(Callback callback) {
  for (auto&& [global_id, value] : global_id2cache_value_) {
    callback(global_id, value.cache_id_, value.lxu_record_);
  }
}

template <typename Tag, typename T>
inline void NaiveIDTransformer<Tag, T>::Evict(
    tcb::span<const int64_t> global_ids) {
  for (const int64_t global_id : global_ids) {
    auto iter = global_id2cache_value_.find(global_id);
    if (iter == global_id2cache_value_.end()) {
      continue;
    }
    int64_t cache_id = iter->second.cache_id_;
    global_id2cache_value_.erase(iter);
    bitmap_.FreeBit(cache_id);
  }
}

} // namespace tde::details
