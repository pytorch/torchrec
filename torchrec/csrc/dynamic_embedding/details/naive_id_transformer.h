/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <c10/util/flat_hash_map.h>
#include <torchrec/csrc/dynamic_embedding/details/bitmap.h>
#include <memory>
#include <optional>
#include <span>

namespace torchrec {

namespace transform_default {

template <typename LXURecord>
inline LXURecord no_update(
    std::optional<LXURecord> record,
    int64_t global_id,
    int64_t cache_id) {
  return record.value_or(LXURecord{});
};

inline void no_fetch(int64_t global_id, int64_t cache_id) {}

} // namespace transform_default

template <typename LXURecord>
struct TransformerRecord {
  int64_t global_id_;
  int64_t cache_id_;
  LXURecord lxu_record_;
};

/**
 * NaiveIDTransformer
 *
 * transform GlobalID to CacheID by naive flat hash map
 * @tparam LXURecord The extension type used for eviction strategy.
 * @tparam Bitmap The bitmap class to record the free cache ids.
 */
template <typename LXURecord, typename Bitmap = Bitmap<uint32_t>>
class NaiveIDTransformer {
 public:
  using lxu_record_t = LXURecord;
  using record_t = TransformerRecord<lxu_record_t>;

  explicit NaiveIDTransformer(int64_t num_embedding);
  NaiveIDTransformer(const NaiveIDTransformer<LXURecord, Bitmap>&) = delete;
  NaiveIDTransformer(NaiveIDTransformer<LXURecord, Bitmap>&&) noexcept =
      default;

  /**
   * Transform global ids to cache ids
   *
   * @tparam Update Update the eviction strategy tag type. Update LXU Record
   * @tparam Fetch Fetch the not existing global-id/cache-id pair. It is used
   * by dynamic embedding parameter server.
   *
   * @param global_ids Global ID vector
   * @param cache_ids [out] Cache ID vector
   * @param update update lambda. See `Update` doc.
   * @param fetch fetch lambda. See `Fetch` doc.
   * @return true if all transformed, otherwise need eviction.
   */
  template <
      typename Update = decltype(transform_default::no_update<LXURecord>),
      typename Fetch = decltype(transform_default::no_fetch)>
  bool transform(
      std::span<const int64_t> global_ids,
      std::span<int64_t> cache_ids,
      Update update = transform_default::no_update<LXURecord>,
      Fetch fetch = transform_default::no_fetch);

  /**
   * Evict global ids from the transformer
   *
   * @param global_ids Global IDs to evict.
   */
  void evict(std::span<const int64_t> global_ids);

  /**
   * Create an iterator of the id transformer, a possible usecase is:
   *
   *   auto iterator = transformer.iterator();
   *   auto record = iterator();
   *   while (record.has_value()) {
   *     // do sth with the record
   *     // ...
   *     // get next record
   *     auto record = iterator();
   *   }
   *
   * @return the iterator created.
   */
  std::function<std::optional<record_t>()> iterator() const;

 private:
  struct CacheValue {
    int64_t cache_id_;
    LXURecord lxu_record_;
  };

  ska::flat_hash_map<int64_t, CacheValue> global_id2cache_value_;
  Bitmap bitmap_;
};

} // namespace torchrec

#include <torchrec/csrc/dynamic_embedding/details/naive_id_transformer_impl.h>
