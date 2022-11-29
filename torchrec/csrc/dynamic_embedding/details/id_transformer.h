/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <torchrec/csrc/dynamic_embedding/details/types.h>
#include <functional>
#include <span>
#include <type_traits>

namespace torchrec {

namespace transform_default {
inline lxu_record_t no_update(
    std::optional<lxu_record_t> record,
    int64_t global_id,
    int64_t cache_id) {
  return record.value_or(lxu_record_t{});
};

inline void no_fetch(int64_t global_id, int64_t cache_id) {}
} // namespace transform_default

class IDTransformer {
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
  virtual bool transform(
      std::span<const int64_t> global_ids,
      std::span<int64_t> cache_ids,
      update_t update = transform_default::no_update,
      fetch_t fetch = transform_default::no_fetch) = 0;

  /**
   * Evict global ids from the transformer
   *
   * @param global_ids Global IDs to evict.
   */
  virtual void evict(std::span<const int64_t> global_ids) = 0;

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
  virtual iterator_t iterator() const = 0;
};

} // namespace torchrec
