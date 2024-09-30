#include "benchmark/benchmark.h"
#include "tde/details/random_bits_generator.h"

namespace tde::details {

void BMRandomBitsGenerator(benchmark::State& state) {
  auto n = state.range(0);
  auto n_bits_limit = state.range(1);
  BitScanner scanner(n);
  std::mt19937_64 engine((std::random_device())());
  std::uniform_int_distribution<uint16_t> dist(1, n_bits_limit);
  uint16_t n_bits = dist(engine);
  for (auto _ : state) {
    if (n_bits != 0) {
      scanner.ResetArray([&](tcb::span<uint64_t> span) {
        for (auto& v : span) {
          v = engine();
        }
      });
    } else {
      n_bits = dist(engine);
    }
    benchmark::DoNotOptimize(scanner.IsNextNBitsAllZero(n_bits));
  }
}

BENCHMARK(BMRandomBitsGenerator)
    ->ArgNames({"n", "limit"})
    ->Args({1024, 32})
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1024 * 1024);

} // namespace tde::details
