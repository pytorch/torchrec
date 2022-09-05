#include "tde/details/id_transformer_variant.h"
#include <vector>

namespace tde::details {

IDTransformer::IDTransformer(int64_t num_embeddings, nlohmann::json json)
    : strategy_(json["lxu_strategy"]),
      var_(
          json["id_transformer"]["type"] == "naive"
              ? Variant(NaiveIDTransformer<uint32_t>(num_embeddings))
              : Variant(CachelineIDTransformer<uint32_t>(num_embeddings))) {}

std::vector<int64_t> IDTransformer::Evict(int64_t num_to_evict) {
  // Get the ids to evict from lxu strategy.
  std::vector<int64_t> ids_to_evict = std::visit(
      [&](auto&& s) { return strategy_.Evict(s.Iterator(), num_to_evict); },
      var_);
  // get the cache id of the ids to evict.
  std::vector<int64_t> cache_ids(ids_to_evict.size());
  Transform(ids_to_evict, cache_ids);
  std::vector<int64_t> result;
  result.reserve(2 * ids_to_evict.size());
  for (size_t i = 0; i < ids_to_evict.size(); i++) {
    result.emplace_back(ids_to_evict[i]);
    result.emplace_back(cache_ids[i]);
  }
  // Evict ids from the ID transformer.
  std::visit([&](auto&& s) { s.Evict(ids_to_evict); }, var_);
  return result;
}

IDTransformer::LXUStrategy::LXUStrategy(const nlohmann::json& json)
    : strategy_(MixedLFULRUStrategy(json.value("min_used_freq_power", 5))) {
  if (auto it = json.find("type"); it != json.end()) {
    TORCH_CHECK(
        static_cast<std::string>(it.value()) == MixedLFULRUStrategy::type_,
        "json type must be mixed_lru_lfu for now");
  } else {
    TORCH_CHECK(false, "type must set");
  }
}

void IDTransformer::LXUStrategy::UpdateTime(uint32_t time) {
  return std::visit(
      [&](auto& strategy) { return strategy.UpdateTime(time); }, strategy_);
}

} // namespace tde::details
