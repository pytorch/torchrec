#include "gtest/gtest.h"
#include "tde/details/id_transformer_variant.h"

namespace tde::details {

TEST(TDE, CreateLXUStrategy) {
  auto strategy =
      LXUStrategy({{"min_used_freq_power", 6}, {"type", "mixed_lru_lfu"}});
}

TEST(TDE, IDTransformer) {
  IDTransformer transformer(
      LXUStrategy(nlohmann::json::parse(R"({"type": "mixed_lru_lfu"})")),
      1000,
      "naive");
  std::vector<int64_t> vec{0, 1, 2};
  std::vector<int64_t> result;
  result.resize(vec.size());
  transformer.Transform(vec, result);
}

} // namespace tde::details
