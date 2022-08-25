#pragma once
#include <algorithm>
#include <vector>
#include "tde/details/bits_op.h"

namespace tde::details {

template <
    typename LXURecord,
    int64_t num_cacheline,
    int64_t cacheline_size,
    typename BitMap,
    typename Hash>
inline CachelineIDTransformer<
    LXURecord,
    num_cacheline,
    cacheline_size,
    BitMap,
    Hash>::CachelineIDTransformer(int64_t num_embedding, int64_t capacity)
    : num_groups_(
          ((capacity == 0 ? 2 * num_embedding : capacity) + group_size_ - 1) /
          group_size_),
      cache_values_(new CacheValue[num_groups_ * group_size_]),
      bitmap_(num_embedding) {
  std::fill(
      cache_values_.get(),
      cache_values_.get() + num_groups_ * group_size_,
      CacheValue{});
}

template <
    typename LXURecord,
    int64_t num_cacheline,
    int64_t cacheline_size,
    typename BitMap,
    typename Hash>
template <
    typename Filter,
    typename CacheIDTransformer,
    typename Update,
    typename Fetch>
inline bool CachelineIDTransformer<
    LXURecord,
    num_cacheline,
    cacheline_size,
    BitMap,
    Hash>::
    Transform(
        tcb::span<const int64_t> global_ids,
        tcb::span<int64_t> cache_ids,
        Filter filter,
        CacheIDTransformer cache_id_transformer,
        Update update,
        Fetch fetch) {
  for (size_t i = 0; i < global_ids.size(); ++i) {
    int64_t global_id = global_ids[i];
    if (!filter(global_id)) {
      continue;
    }
    auto [group_id, intra_id] = FindGroupIndex(global_id);

    bool need_eviction = true;
    // cache_id is in [0, num_embedding)
    int64_t cache_id;
    for (int64_t k = 0; k < group_size_; k++) {
      int64_t offset = group_id * group_size_ + (intra_id + k) % group_size_;
      auto& cache_value = cache_values_[offset];
      // tricky but fast :p
      int64_t neg_global_id = -global_id - 1;
      int64_t xor_value = cache_value.neg_global_id_ ^ neg_global_id;
      if (xor_value == 0) { // found exist
        cache_id = cache_value.cache_id();
        cache_value.lxu_record_ =
            update(cache_value.lxu_record_, global_id, cache_id);
        need_eviction = false;
        break;
      } else if (xor_value < 0) { // empty slot
        // The transformer is full.
        if (C10_UNLIKELY(bitmap_.Full())) {
          break;
        }
        auto stored_cache_id = bitmap_.NextFreeBit();
        cache_id = cache_id_transformer(stored_cache_id);
        cache_value.neg_global_id_ = neg_global_id;
        cache_value.set_cache_id(cache_id);
        cache_value.lxu_record_ =
            update(cache_value.lxu_record_, global_id, cache_id);
        fetch(global_id, cache_id);
        need_eviction = false;
        break;
      }
    }
    if (need_eviction) {
      return false;
    }
    cache_ids[i] = cache_id;
  }
  return true;
}

template <
    typename LXURecord,
    int64_t num_cacheline,
    int64_t cacheline_size,
    typename BitMap,
    typename Hash>
template <typename Callback>
inline void CachelineIDTransformer<
    LXURecord,
    num_cacheline,
    cacheline_size,
    BitMap,
    Hash>::ForEach(Callback callback) {
  for (int64_t i = 0; i < num_groups_; ++i) {
    for (int64_t j = 0; j < group_size_; ++j) {
      int64_t offset = i * group_size_ + j;
      auto& cache_value = cache_values_[offset];
      if (cache_value.is_filled()) {
        callback(
            cache_value.global_id(),
            cache_value.cache_id(),
            cache_value.lxu_record_);
      }
    }
  }
}

template <
    typename LXURecord,
    int64_t num_cacheline,
    int64_t cacheline_size,
    typename BitMap,
    typename Hash>
inline void CachelineIDTransformer<
    LXURecord,
    num_cacheline,
    cacheline_size,
    BitMap,
    Hash>::Evict(tcb::span<const int64_t> global_ids) {
  for (const int64_t global_id : global_ids) {
    auto [group_id, intra_id] = FindGroupIndex(global_id);

    for (int64_t k = 0; k < group_size_; k++) {
      int64_t offset = group_id * group_size_ + (intra_id + k) % group_size_;
      auto& cache_value = cache_values_[offset];
      // tricky but fast :p
      int64_t neg_global_id_ = -global_id - 1;
      int64_t xor_value = neg_global_id_ ^ cache_value.neg_global_id_;
      if (xor_value < 0) { // not exist
        break;
      } else if (xor_value == 0) { // found slot
        bitmap_.FreeBit(cache_value.cache_id());
        cache_value.set_empty();
        break;
      }
    }
  }
}

template <
    typename LXURecord,
    int64_t num_cacheline,
    int64_t cacheline_size,
    typename BitMap,
    typename Hash>
inline auto CachelineIDTransformer<
    LXURecord,
    num_cacheline,
    cacheline_size,
    BitMap,
    Hash>::Iterator() const -> MoveOnlyFunction<std::optional<record_t>()> {
  int64_t i = 0;
  return [i, this]() mutable -> std::optional<record_t> {
    for (; i < num_groups_ * group_size_; ++i) {
      auto& cache_value = cache_values_[i];
      if (cache_value.is_filled()) {
        auto record = record_t{
            .global_id_ = cache_value.global_id(),
            .cache_id_ = cache_value.cache_id(),
            .lxu_record_ = cache_value.lxu_record_,
        };
        ++i;
        return record;
      }
    }
    return {};
  };
}

} // namespace tde::details
