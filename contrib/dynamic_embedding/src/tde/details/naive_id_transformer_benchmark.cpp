#include <torch/torch.h>
#include "benchmark/benchmark.h"
#include "tde/details/naive_id_transformer.h"

namespace tde::details {

static void BM_NaiveIDTransformer(benchmark::State& state) {
  using Tag = int32_t;
  NaiveIDTransformer<Tag> transformer(1e8);
  torch::Tensor global_ids = torch::empty({1024, 1024}, torch::kLong);
  torch::Tensor cache_ids = torch::empty_like(global_ids);
  for (auto _ : state) {
    state.PauseTiming();
    global_ids.random_(state.range(0), state.range(1));
    state.ResumeTiming();
    transformer.Transform(
        tcb::span{
            global_ids.template data_ptr<int64_t>(),
            static_cast<size_t>(global_ids.numel())},
        tcb::span{
            cache_ids.template data_ptr<int64_t>(),
            static_cast<size_t>(cache_ids.numel())},
        transform_default::All,
        [](int64_t cid) { return cid + 1e8; });
  }
}

BENCHMARK(BM_NaiveIDTransformer)
    ->Iterations(100)
    ->Unit(benchmark::kMillisecond)
    ->ArgNames({"rand_from", "rand_to"})
    ->Args({static_cast<long long>(1e10), static_cast<long long>(2e10)})
    ->Args({static_cast<long long>(1e6), static_cast<long long>(2e6)});

} // namespace tde::details
