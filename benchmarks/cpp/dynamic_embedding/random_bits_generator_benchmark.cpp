/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <benchmark/benchmark.h>
#include <torchrec/csrc/dynamic_embedding/details/random_bits_generator.h>
#include <span>

namespace torchrec {

void BMRandomBitsGenerator(benchmark::State& state) {
  auto n = state.range(0);
  auto n_bits_limit = state.range(1);
  BitScanner scanner(n);
  std::mt19937_64 engine((std::random_device())());
  std::uniform_int_distribution<uint16_t> dist(1, n_bits_limit);
  uint16_t n_bits = dist(engine);
  for (auto _ : state) {
    if (n_bits != 0) {
      scanner.reset_array([&](std::span<uint64_t> span) {
        for (auto& v : span) {
          v = engine();
        }
      });
    } else {
      n_bits = dist(engine);
    }
    benchmark::DoNotOptimize(scanner.is_next_n_bits_all_zero(n_bits));
  }
}

BENCHMARK(BMRandomBitsGenerator)
    ->ArgNames({"n", "limit"})
    ->Args({1024, 32})
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1024 * 1024);

} // namespace torchrec
