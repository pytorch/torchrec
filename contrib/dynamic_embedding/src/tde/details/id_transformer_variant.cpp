#include "tde/details/id_transformer_variant.h"
#include <vector>

namespace tde::details {

void LXUStrategy::UpdateTime(uint32_t time) {
  return std::visit(
      [&](auto& strategy) { return strategy.UpdateTime(time); }, strategy_);
}

} // namespace tde::details
