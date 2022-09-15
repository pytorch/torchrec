#include <gtest/gtest.h>
#include <torchrec/csrc/dynamic_embedding/details/bits_op.h>

namespace torchrec::dynamic_embedding::details {
TEST(TDE, bits_op_clz) {
  ASSERT_EQ(clz(int32_t(0x7FFFFFFF)), 1);
  ASSERT_EQ(clz(int64_t(0x7FFFFFFFFFFFFFFF)), 1);
  ASSERT_EQ(clz(int8_t(0x7F)), 1);
}

TEST(TDE, bits_op_ctz) {
  ASSERT_EQ(ctz(int32_t(0x2)), 1);
  ASSERT_EQ(ctz(int64_t(0xF00)), 8);
  ASSERT_EQ(ctz(int8_t(0x4)), 2);
}
} // namespace torchrec::dynamic_embedding::details
