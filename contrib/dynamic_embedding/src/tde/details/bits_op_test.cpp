#include "gtest/gtest.h"
#include "tde/details/bits_op.h"

namespace tde::details {
TEST(TDE, bits_op_Clz) {
  ASSERT_EQ(Clz(int32_t(0x7FFFFFFF)), 1);
  ASSERT_EQ(Clz(int64_t(0x7FFFFFFFFFFFFFFF)), 1);
  ASSERT_EQ(Clz(int8_t(0x7F)), 1);
}

TEST(TDE, bits_op_Ctz) {
  ASSERT_EQ(Ctz(int32_t(0x2)), 1);
  ASSERT_EQ(Ctz(int64_t(0xF00)), 8);
  ASSERT_EQ(Ctz(int8_t(0x4)), 2);
}
} // namespace tde::details
