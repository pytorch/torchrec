#pragma once

namespace tde::details {

class NoFetcher {
 public:
  void Fetch(int64_t global_id, int64_t cache_id) {}
};

} // namespace tde::details
