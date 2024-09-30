#include "gtest/gtest.h"
#include "tde/details/naive_id_transformer.h"

namespace tde::details {

TEST(tde, NaiveThreadedIDTransformer_NoFilter) {
  using Tag = int32_t;
  NaiveIDTransformer<Tag, Bitmap<uint8_t>> transformer(16);
  const int64_t global_ids[5] = {100, 101, 100, 102, 101};
  int64_t cache_ids[5];
  int64_t expected_cache_ids[5] = {0, 1, 0, 2, 1};
  ASSERT_TRUE(transformer.Transform(global_ids, cache_ids));
  for (size_t i = 0; i < 5; i++) {
    ASSERT_EQ(expected_cache_ids[i], cache_ids[i]);
  }
}

TEST(tde, NaiveThreadedIDTransformer_Full) {
  using Tag = int32_t;
  NaiveIDTransformer<Tag, Bitmap<uint8_t>> transformer(4);
  const int64_t global_ids[5] = {100, 101, 102, 103, 104};
  int64_t cache_ids[5];
  int64_t expected_cache_ids[5] = {0, 1, 2, 3, -1};

  ASSERT_FALSE(transformer.Transform(global_ids, cache_ids));
  for (size_t i = 0; i < 4; i++) {
    EXPECT_EQ(expected_cache_ids[i], cache_ids[i]);
  }
}

TEST(tde, NaiveThreadedIDTransformer_Evict) {
  using Tag = int32_t;
  NaiveIDTransformer<Tag, Bitmap<uint8_t>> transformer(4);
  const int64_t global_ids[5] = {100, 101, 102, 103, 104};
  int64_t cache_ids[5];

  ASSERT_FALSE(transformer.Transform(global_ids, cache_ids));

  const int64_t evict_global_ids[2] = {100, 102};
  transformer.Evict(evict_global_ids);

  const int64_t new_global_ids[4] = {101, 102, 103, 104};
  int64_t new_cache_ids[4];

  ASSERT_TRUE(transformer.Transform(new_global_ids, new_cache_ids));

  int64_t expected_cache_ids[4] = {1, 0, 3, 2};

  for (size_t i = 0; i < 4; i++) {
    EXPECT_EQ(expected_cache_ids[i], new_cache_ids[i]);
  }
}

TEST(tde, NaiveThreadedIDTransformer_Iterator) {
  using Tag = int32_t;
  NaiveIDTransformer<Tag, Bitmap<uint8_t>> transformer(16);
  const int64_t global_ids[5] = {100, 101, 100, 102, 101};
  int64_t cache_ids[5];
  int64_t expected_cache_ids[5] = {3, 4, 3, 5, 4};
  ASSERT_TRUE(transformer.Transform(global_ids, cache_ids));

  auto iterator = transformer.Iterator();
  for (size_t i = 0; i < 3; i++) {
    EXPECT_TRUE(iterator().has_value());
  }
  EXPECT_TRUE(!iterator().has_value());
}

} // namespace tde::details
