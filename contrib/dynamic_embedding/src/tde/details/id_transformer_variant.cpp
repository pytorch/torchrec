#include "tde/details/id_transformer_variant.h"
#include <vector>

namespace tde::details {

void LXUStrategy::UpdateTime(uint32_t time) {
  return std::visit(
      [&](auto& strategy) { return strategy.UpdateTime(time); }, strategy_);
}

LXUStrategy::lxu_record_t LXUStrategy::DefaultRecordValue() {
  return std::visit(
      [&](auto& strategy) -> LXUStrategy::lxu_record_t {
        using T = typename std::decay_t<decltype(strategy)>::lxu_record_t;
        return T{};
      },
      strategy_);
}

} // namespace tde::details
