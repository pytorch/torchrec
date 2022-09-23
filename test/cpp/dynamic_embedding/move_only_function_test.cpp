#include "gtest/gtest.h"
#include <torchrec/csrc/dynamic_embedding/details/move_only_function.h>

namespace torchrec {

TEST(tde, move_only_function) {
  MoveOnlyFunction<int()> foo = +[] { return 0; };
  ASSERT_EQ(foo(), 0);
  ASSERT_TRUE(foo);
  foo = {};
  ASSERT_FALSE(foo);
}

} // namespace torchrec
