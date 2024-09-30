#include "gtest/gtest.h"
#include "tde/details/url.h"

namespace tde::details::url_parser::rules {

TEST(TDE, url_token) {
  auto ipt = lexy::string_input(std::string_view("%61"));
  auto parse = lexy::parse<UrlToken>(ipt, lexy_ext::report_error);
  ASSERT_TRUE(parse.has_value());
  ASSERT_EQ('a', parse.value());
}

TEST(TDE, url_string) {
  auto ipt = lexy::string_input(std::string_view("%61bc"));
  auto parse = lexy::parse<UrlString>(ipt, lexy_ext::report_error);
  ASSERT_TRUE(parse.has_value());
  ASSERT_EQ("abc", parse.value());
}

TEST(TDE, url_normal) {
  auto ipt = lexy::string_input(std::string_view("a@"));
  auto parse = lexy::parse<rules::Auth>(ipt, lexy_ext::report_error);
  ASSERT_TRUE(parse.has_value());
  ASSERT_EQ("a", parse.value().username_);
  ASSERT_FALSE(parse.value().password_.has_value());
  //  ASSERT_EQ("abc", parse.value());
}

TEST(TDE, url_host) {
  auto ipt = lexy::string_input(std::string_view("www.qq.com"));
  auto parse = lexy::parse<rules::Host>(ipt, lexy_ext::report_error);
  ASSERT_TRUE(parse.has_value());
  ASSERT_EQ("www.qq.com", parse.value());
}

TEST(TDE, url) {
  auto url = ParseUrl("www.qq.com/?a=b&&c=d");
  ASSERT_EQ(url.host_, "www.qq.com");
  ASSERT_TRUE(url.param_.has_value());
  ASSERT_EQ("a=b&&c=d", url.param_.value());
}

TEST(TDE, bad_url) {
  ASSERT_ANY_THROW([] { ParseUrl("blablah!@@"); }());
}

} // namespace tde::details::url_parser::rules
