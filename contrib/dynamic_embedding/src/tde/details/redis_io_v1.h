#pragma once
#include <string>
#include <string_view>

namespace tde::details::redis_v1 {

struct Option {
 private:
  struct ParseTag {};

 public:
  std::string host_;
  std::string username_;
  std::string password_;
  uint16_t port_{6379};
  uint16_t db_{0};
  uint32_t num_io_threads_{1};
  std::string prefix_;

  Option() = default;

  static Option Parse(std::string_view config_str) {
    return Option(ParseTag{}, config_str);
  }

 private:
  Option(ParseTag tag, std::string_view config_str);
};

} // namespace tde::details::redis_v1
