/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <torchrec/csrc/dynamic_embedding/details/mixed_lfu_lru_strategy.h>
#include <torchrec/csrc/dynamic_embedding/details/naive_id_transformer.h>
#include <span>
#include <type_traits>
#include <variant>

namespace torchrec {

class LXUStrategyVariant {
  // Currently, only MixedLFULRUStrategy is the LXUStrategyVariant. Add more
  // strategy when it is necessary.
  using Variant = std::variant<MixedLFULRUStrategy>;

  // use to indicate VisitUpdator 's result
  using _UpdateFunctor = std::function<MixedLFULRUStrategy::lxu_record_t(
      std::optional<MixedLFULRUStrategy::lxu_record_t>,
      int64_t,
      int64_t)>;

 public:
  explicit LXUStrategyVariant(
      const std::string& lxu_strategy_type,
      uint16_t min_used_freq_power);

  void update_time(uint32_t time);
  template <typename T>
  int64_t time(T record);

  template <typename Visitor>
  auto visit_updator(Visitor visit)
      -> std::invoke_result_t<Visitor, _UpdateFunctor>;

  template <typename Iterator>
  std::vector<int64_t> evict(Iterator iterator, uint64_t num_to_evict);

 private:
  Variant strategy_;
};

class IDTransformerVariant {
  // Currently, only NaiveIDTransformer is the IDTransformerVariant. Add more
  // strategy when it is necessary.
  using Variant = std::variant<NaiveIDTransformer<uint32_t>>;

 public:
  IDTransformerVariant(
      LXUStrategyVariant strategy,
      int64_t num_embeddings,
      const std::string& id_transformer_type);

  /**
   * Transform GlobalIDs to CacheIDs.
   *
   * @param fetch Callback when need fetch. By default, do nothing.
   * @return number elems transformed. If the Transformer is full and need to be
   * evict. Then the return value is not equal to global_ids.size();
   */
  template <typename Fetch = decltype(transform_default::no_fetch)>
  bool transform(
      std::span<const int64_t> global_ids,
      std::span<int64_t> cache_ids,
      Fetch fetch = transform_default::no_fetch);

  /**
   * Evict `num_to_evict` global ids from the transformer.
   *
   * @return A vector of size 2 * `num_to_evict`, records the
   * global id and cache id of each evicted ids.
   */
  std::vector<int64_t> evict(int64_t num_to_evict);

  /**
   * Save the ids by timestamp `time`.
   *
   * @return Return the global id and cache id of the ids that
   * are newly inserted or updated from last saving timestamp to
   * `time`
   */
  std::vector<int64_t> save(int64_t time);

  LXUStrategyVariant strategy_;

 private:
  Variant var_;
};

} // namespace torchrec

#include <torchrec/csrc/dynamic_embedding/details/id_transformer_variant_impl.h>
