/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <benchmark/benchmark.h>
#include <torch/torch.h>
#include <torchrec/csrc/dynamic_embedding/details/naive_id_transformer.h>

namespace torchrec {

static void BM_NaiveIDTransformer(benchmark::State& state) {
  using Tag = int32_t;
  NaiveIDTransformer<Tag> transformer(2e8);
  torch::Tensor global_ids = torch::empty({1024, 1024}, torch::kLong);
  torch::Tensor cache_ids = torch::empty_like(global_ids);
  for (auto _ : state) {
    state.PauseTiming();
    global_ids.random_(state.range(0), state.range(1));
    state.ResumeTiming();
    transformer.transform(
        std::span{
            global_ids.template data_ptr<int64_t>(),
            static_cast<size_t>(global_ids.numel())},
        std::span{
            cache_ids.template data_ptr<int64_t>(),
            static_cast<size_t>(cache_ids.numel())});
  }
}

BENCHMARK(BM_NaiveIDTransformer)
    ->Iterations(100)
    ->Unit(benchmark::kMillisecond)
    ->ArgNames({"rand_from", "rand_to"})
    ->Args({static_cast<long long>(1e10), static_cast<long long>(2e10)})
    ->Args({static_cast<long long>(1e6), static_cast<long long>(2e6)});

} // namespace torchrec
