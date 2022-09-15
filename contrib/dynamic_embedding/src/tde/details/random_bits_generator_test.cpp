#include "gtest/gtest.h"
#include "random_bits_generator.h"

namespace tde::details {
TEST(TDE, BitScanner1Elem) {
  BitScanner scanner(1);
  scanner.ResetArray([](auto span) { span[0] = 0; });

  for (size_t i = 0; i < 64; ++i) {
    uint16_t n_bits = 1;
    ASSERT_EQ(scanner.bit_idx, i);
    ASSERT_EQ(scanner.array_idx_, 0);
    ASSERT_TRUE(scanner.IsNextNBitsAllZero(n_bits));
    ASSERT_EQ(n_bits, 0);
  }
  scanner.ResetArray([](auto span) { span[0] = 0x80FFFFFFFFFFFFFF; });
  {
    uint16_t n_bits = 1;
    ASSERT_FALSE(scanner.IsNextNBitsAllZero(n_bits));
    ASSERT_EQ(n_bits, 0);
    ASSERT_EQ(scanner.bit_idx, 1);
    n_bits = 5;
    ASSERT_TRUE(scanner.IsNextNBitsAllZero(n_bits));
    ASSERT_EQ(n_bits, 0);
    ASSERT_EQ(scanner.bit_idx, 6);

    n_bits = 4;
    ASSERT_FALSE(scanner.IsNextNBitsAllZero(n_bits));
    ASSERT_EQ(n_bits, 0);
    ASSERT_EQ(scanner.bit_idx, 9);

    n_bits = 64 - 9 + 17;
    ASSERT_FALSE(scanner.IsNextNBitsAllZero(n_bits));
    ASSERT_EQ(n_bits, 17);
    ASSERT_EQ(scanner.bit_idx, 10);
    ASSERT_EQ(scanner.array_idx_, 0);
  }
}

TEST(TDE, BitScanner2Elems) {
  BitScanner scanner(2);
  scanner.ResetArray([](auto span) {
    span[0] = 0;
    span[1] = 0x8000000000000000;
  });

  uint16_t n_bits = 32;
  ASSERT_TRUE(scanner.IsNextNBitsAllZero(n_bits));
  ASSERT_EQ(n_bits, 0);
  ASSERT_EQ(scanner.bit_idx, 32);
  ASSERT_EQ(scanner.array_idx_, 0);

  n_bits = 64;
  ASSERT_FALSE(scanner.IsNextNBitsAllZero(n_bits));
  ASSERT_EQ(n_bits, 0);
  ASSERT_EQ(scanner.bit_idx, 1);
  ASSERT_EQ(scanner.array_idx_, 1);

  n_bits = 64;
  ASSERT_TRUE(scanner.IsNextNBitsAllZero(n_bits));
  ASSERT_EQ(n_bits, 1);
  ASSERT_EQ(scanner.bit_idx, 0);
  ASSERT_EQ(scanner.array_idx_, 2);
}

TEST(TDE, RandomBitsGenerator) {
  RandomBitsGenerator generator;
  size_t true_cnt_{0};
  constexpr static size_t n_iter = 10000000;
  for (size_t i = 0; i < n_iter; ++i) {
    if (generator.IsNextNBitsAllZero(10)) {
      ++true_cnt_;
    }
  }

  ASSERT_NEAR(double(true_cnt_) / double(n_iter), 1 / 1024.f, 1e-4);
}

} // namespace tde::details
