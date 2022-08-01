#include "tde/details/redis_io_v1.h"
#include <variant>
#include <vector>
#include "lexy/callback.hpp"
#include "lexy/dsl.hpp"
#include "tde/details/url.h"

namespace tde::details::redis_v1 {

struct NumThreadsOpt {
  uint32_t num_threads_;
};
struct DBOpt {
  uint32_t db_;
};
struct PrefixOpt {
  std::string prefix_;
};

using OptVar = std::variant<NumThreadsOpt, DBOpt, PrefixOpt>;

namespace option_rules {
namespace dsl = lexy::dsl;
struct NumThreads {
  constexpr static auto rule = LEXY_LIT("num_threads=") >>
      dsl::integer<uint32_t>(dsl::digits<>);
  constexpr static auto value = lexy::construct<NumThreadsOpt>;
};

struct DB {
  constexpr static auto rule = LEXY_LIT("db=") >>
      dsl::integer<uint32_t>(dsl::digits<>);
  constexpr static auto value = lexy::construct<DBOpt>;
};

struct Prefix {
  constexpr static auto rule = LEXY_LIT("prefix=") >>
      dsl::capture(dsl::token(dsl::identifier(
          dsl::ascii::alpha_underscore,
          dsl::ascii::alpha_digit_underscore)));
  constexpr static auto value =
      lexy::callback<PrefixOpt>([](auto&& str) -> PrefixOpt {
        return PrefixOpt{std::string(str.data(), str.size())};
      });
};

struct Option {
  constexpr static auto rule = dsl::p<NumThreads> | dsl::p<DB> | dsl::p<Prefix>;
  constexpr static auto value = lexy::construct<OptVar>;
};

struct Param {
  constexpr static auto rule =
      dsl::list(dsl::p<Option>, dsl::sep(LEXY_LIT("&&")));

  constexpr static auto value = lexy::as_list<std::vector<OptVar>>;
};

} // namespace option_rules

Option::Option(Option::ParseTag tag, std::string_view config_str) {
  auto url = url_parser::ParseUrl(config_str);
  if (url.auth_.has_value()) {
    username_ = std::move(url.auth_->username_);
    if (url.auth_->password_.has_value()) {
      password_ = std::move(url.auth_->password_.value());
    }
  }

  host_ = std::move(url.host_);

  if (url.port_.has_value()) {
    port_ = url.port_.value();
  }

  if (url.param_.has_value()) {
    std::ostringstream err_oss_;
    url_parser::ErrorCollector collector{err_oss_};

    auto result = lexy::parse<option_rules::Param>(
        lexy::string_input(url.param_.value()), collector);
    TORCH_CHECK(result.has_value(), "parse param error %s", err_oss_.str());

    for (auto&& opt_var : result.value()) {
      std::visit(
          [this](auto&& opt) {
            using T = std::decay_t<decltype(opt)>;

            if constexpr (std::is_same_v<T, NumThreadsOpt>) {
              num_io_threads_ = opt.num_threads_;
            } else if constexpr (std::is_same_v<T, DBOpt>) {
              db_ = opt.db_;
            } else {
              prefix_ = std::move(opt.prefix_);
            }
          },
          opt_var);
    }
  }
}
} // namespace tde::details::redis_v1
