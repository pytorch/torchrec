#include "gtest/gtest.h"
#include "tde/details/cacheline_id_transformer.h"

namespace tde::details {

TEST(tde, CachelineThreadedIDTransformer_NoFilter) {
  CachelineIDTransformer<int32_t, 4, 64, Bitmap<uint8_t>> transformer(16);
  const int64_t global_ids[5] = {100, 101, 100, 102, 101};
  int64_t cache_ids[5];
  int64_t expected_cache_ids[5] = {0, 1, 0, 2, 1};
  ASSERT_TRUE(transformer.Transform(global_ids, cache_ids));
  for (size_t i = 0; i < 5; i++) {
    ASSERT_EQ(expected_cache_ids[i], cache_ids[i]);
  }
}

TEST(tde, CachelineThreadedIDTransformer_Full) {
  CachelineIDTransformer<int32_t, 4, 64, Bitmap<uint8_t>> transformer(4);
  const int64_t global_ids[5] = {100, 101, 102, 103, 104};
  int64_t cache_ids[5];
  int64_t expected_cache_ids[5] = {0, 1, 2, 3, -1};
  EXPECT_FALSE(transformer.Transform(global_ids, cache_ids));
  for (size_t i = 0; i < 4; i++) {
    EXPECT_EQ(expected_cache_ids[i], cache_ids[i]);
  }
}

TEST(tde, CachelineThreadedIDTransformer_Evict) {
  CachelineIDTransformer<int32_t, 4, 64, Bitmap<uint8_t>> transformer(4);
  const int64_t global_ids[5] = {100, 101, 102, 103, 104};
  int64_t cache_ids[5];

  EXPECT_FALSE(transformer.Transform(global_ids, cache_ids));

  const int64_t evict_global_ids[2] = {100, 102};
  transformer.Evict(evict_global_ids);

  const int64_t new_global_ids[4] = {101, 102, 103, 104};
  int64_t new_cache_ids[4];

  EXPECT_TRUE(transformer.Transform(new_global_ids, new_cache_ids));

  int64_t expected_cache_ids[4] = {1, 0, 3, 2};
  for (size_t i = 0; i < 4; i++) {
    EXPECT_EQ(expected_cache_ids[i], new_cache_ids[i]);
  }
}

TEST(tde, CachelineThreadedIDTransformer_Iterator) {
  CachelineIDTransformer<int32_t, 4, 64, Bitmap<uint8_t>> transformer(16);
  const int64_t global_ids[5] = {100, 101, 100, 102, 101};
  int64_t cache_ids[5];
  int64_t expected_cache_ids[5] = {0, 1, 0, 2, 0};
  ASSERT_TRUE(transformer.Transform(global_ids, cache_ids));

  auto iterator = transformer.Iterator();
  for (size_t i = 0; i < 3; i++) {
    ASSERT_TRUE(iterator().has_value());
  }
  ASSERT_TRUE(!iterator().has_value());
}

} // namespace tde::details
