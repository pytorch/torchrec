#include "gtest/gtest.h"
#include "tde/details/bits_op.h"

namespace tde::details {
TEST(TDE, bits_op) {
  ASSERT_EQ(Clz(int32_t(0x7FFFFFFF)), 1);
  ASSERT_EQ(Clz(int64_t(0x7FFFFFFFFFFFFFFF)), 1);
  ASSERT_EQ(Clz(int8_t(0x7F)), 1);
}
} // namespace tde::details
