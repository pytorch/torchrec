#include "gtest/gtest.h"
#include "tde/details/move_only_function.h"
namespace tde::details {

TEST(tde, move_only_function) {
  MoveOnlyFunction<int()> foo = +[] { return 0; };
  ASSERT_EQ(foo(), 0);
  ASSERT_TRUE(foo);
  foo = {};
  ASSERT_FALSE(foo);
}

} // namespace tde::details
