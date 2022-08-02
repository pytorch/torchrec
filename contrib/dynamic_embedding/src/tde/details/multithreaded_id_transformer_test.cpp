#include "gtest/gtest.h"
#include "tde/details/multithreaded_id_transformer.h"

namespace tde::details {

TEST(tde, MultiThreadedIDTransformer) {
  using Tag = int32_t;
  MultiThreadedIDTransformer<NaiveIDTransformer<Tag>> transformer(8, 2);
  const int64_t global_ids[6] = {100, 101, 100, 102, 101, 103};
  int64_t cache_ids[6];
  int64_t expected_cache_ids[6] = {0, 4, 0, 1, 4, 5};
  int64_t num_transformed = transformer.Transform(global_ids, cache_ids);
  EXPECT_EQ(6, num_transformed);
  for (size_t i = 0; i < 6; i++) {
    EXPECT_EQ(expected_cache_ids[i], cache_ids[i]);
  }
}

TEST(tde, MultiThreadedIDTransformer_Iterator) {
  using Tag = int32_t;
  MultiThreadedIDTransformer<NaiveIDTransformer<Tag>> transformer(8, 3);
  const int64_t global_ids[6] = {100, 101, 100, 104, 101, 103};
  int64_t cache_ids[6];
  int64_t expected_cache_ids[6] = {0, 4, 0, 1, 4, 5};
  int64_t num_transformed = transformer.Transform(global_ids, cache_ids);
  EXPECT_EQ(6, num_transformed);

  auto iterator = transformer.Iterator();
  for (size_t i = 0; i < 4; i++) {
    EXPECT_TRUE(iterator().has_value());
  }
  EXPECT_TRUE(!iterator().has_value());
}

} // namespace tde::details
