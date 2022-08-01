#include "gtest/gtest.h"
#include "tde/details/redis_io_v1.h"

namespace tde::details::redis_v1 {

TEST(TDE, redis_v1_Option) {
  auto opt = Option::Parse("192.168.3.1:3948/?db=3&&num_threads=2");
  ASSERT_EQ(opt.host_, "192.168.3.1");
  ASSERT_EQ(opt.port_, 3948);
  ASSERT_EQ(opt.db_, 3);
  ASSERT_EQ(opt.num_io_threads_, 2);
  ASSERT_TRUE(opt.prefix_.empty());
}
} // namespace tde::details::redis_v1
