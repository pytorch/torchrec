/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <benchmark/benchmark.h>
#include <torchrec/csrc/dynamic_embedding/details/mixed_lfu_lru_strategy.h>

namespace torchrec {
void BM_MixedLFULRUStrategy(benchmark::State& state) {
  size_t num_ext_values = state.range(0);
  std::vector<MixedLFULRUStrategy::lxu_record_t> ext_values(num_ext_values);

  MixedLFULRUStrategy strategy;
  for (auto& v : ext_values) {
    v = strategy.update(0, 0, std::nullopt);
  }

  size_t num_elems = state.range(1);
  std::default_random_engine engine((std::random_device())());
  size_t time = 0;
  for (auto _ : state) {
    state.PauseTiming();
    std::vector<size_t> offsets;
    offsets.reserve(num_elems);
    for (size_t i = 0; i < num_elems; ++i) {
      std::uniform_int_distribution<size_t> dist(0, num_elems - 1);
      offsets.emplace_back(dist(engine));
    }
    state.ResumeTiming();

    ++time;
    strategy.update_time(time);
    for (auto& v : offsets) {
      ext_values[v] = strategy.update(0, 0, ext_values[v]);
    }
  }
}

BENCHMARK(BM_MixedLFULRUStrategy)
    ->ArgNames({"num_ext_values", "num_elems_per_iter"})
    ->Args({30000000, 1024 * 1024})
    ->Args({300000000, 1024 * 1024})
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

} // namespace torchrec
