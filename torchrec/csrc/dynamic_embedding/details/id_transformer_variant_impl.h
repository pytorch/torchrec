/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

namespace torchrec {

template <typename Visitor>
inline auto LXUStrategyVariant::visit_updator(Visitor visit)
    -> std::invoke_result_t<Visitor, _UpdateFunctor> {
  return std::visit(
      [&](auto& s) {
        using T = typename std::decay_t<decltype(s)>::lxu_record_t;
        auto update =
            [&](std::optional<T> record, int64_t global_id, int64_t cache_id) {
              return s.update(global_id, cache_id, record);
            };
        return visit(update);
      },
      strategy_);
}

template <typename T>
int64_t LXUStrategyVariant::time(T record) {
  return std::visit(
      [&](auto& s) -> int64_t { return s.time(record); }, strategy_);
}

template <typename Iterator>
inline std::vector<int64_t> LXUStrategyVariant::evict(
    Iterator iterator,
    uint64_t num_to_evict) {
  return std::visit(
      [&, iterator = std::move(iterator)](auto& s) mutable {
        return s.evict(std::move(iterator), num_to_evict);
      },
      strategy_);
}

template <typename Fetch>
inline bool IDTransformerVariant::transform(
    std::span<const int64_t> global_ids,
    std::span<int64_t> cache_ids,
    Fetch fetch) {
  return strategy_.visit_updator([&](auto&& update) -> bool {
    return std::visit(
        [&](auto&& transformer) -> bool {
          return transformer.transform(
              global_ids,
              cache_ids,
              std::forward<decltype(update)>(update),
              std::move(fetch));
        },
        var_);
  });
}

} // namespace torchrec
