#include "benchmark/benchmark.h"
#include "tde/details/multithreaded_id_transformer.h"

namespace tde::details {

static void BM_MultiThreadedIDTransformer(benchmark::State& state) {}

BENCHMARK(BM_MultiThreadedIDTransformer);

} // namespace tde::details
