#include "benchmark/benchmark.h"
#include "tde/details/mixed_lfu_lru_strategy.h"

namespace tde::details {
void BM_MixedLFULRUStrategy(benchmark::State& state) {
  size_t num_ext_values = state.range(0);
  std::vector<MixedLFULRUStrategy::lxu_record_t> ext_values(
      num_ext_values);

  MixedLFULRUStrategy strategy;
  for (auto& v : ext_values) {
    v = strategy.Transform(std::nullopt);
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
    strategy.UpdateTime(time);
    for (auto& v : offsets) {
      ext_values[v] = strategy.Transform(ext_values[v]);
    }
  }
}

BENCHMARK(BM_MixedLFULRUStrategy)
    ->ArgNames({"num_ext_values", "num_elems_per_iter"})
    ->Args({30000000, 1024 * 1024})
    ->Args({300000000, 1024 * 1024})
    ->Unit(benchmark::kMillisecond)
    ->Iterations(100);

} // namespace tde::details
