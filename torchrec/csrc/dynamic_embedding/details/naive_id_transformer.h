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
#include <torchrec/csrc/dynamic_embedding/details/id_transformer.h>
#include <memory>
#include <optional>
#include <span>

namespace torchrec {

/**
 * NaiveIDTransformer
 *
 * transform GlobalID to CacheID by naive flat hash map
 * @tparam LXURecord The extension type used for eviction strategy.
 * @tparam Bitmap The bitmap class to record the free cache ids.
 */
template <typename Bitmap = Bitmap<uint32_t>>
class NaiveIDTransformer : public IDTransformer {
 public:
  explicit NaiveIDTransformer(int64_t num_embedding);
  NaiveIDTransformer(const NaiveIDTransformer<Bitmap>&) = delete;
  NaiveIDTransformer(NaiveIDTransformer<Bitmap>&&) noexcept = default;

  bool transform(
      std::span<const int64_t> global_ids,
      std::span<int64_t> cache_ids,
      update_t update = transform_default::no_update,
      fetch_t fetch = transform_default::no_fetch) override;

  void evict(std::span<const int64_t> global_ids) override;

  iterator_t iterator() const override;

 private:
  struct CacheValue {
    int64_t cache_id;
    lxu_record_t lxu_record;
  };

  ska::flat_hash_map<int64_t, CacheValue> global_id2cache_value_;
  Bitmap bitmap_;
};

} // namespace torchrec

#include <torchrec/csrc/dynamic_embedding/details/naive_id_transformer_impl.h>
