/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <torchrec/csrc/dynamic_embedding/details/id_transformer_variant.h>

namespace torchrec {

TEST(TDE, IDTransformerVariant) {
  LXUStrategyVariant strategy("mixed_lru_lfu", 5);
  IDTransformerVariant transformer(std::move(strategy), 1000, "naive");
  std::vector<int64_t> vec{2, 1, 0};
  std::vector<int64_t> result;
  result.resize(vec.size());
  transformer.transform(vec, result);
  std::vector<int64_t> expected_result{0, 1, 2};
  ASSERT_EQ(result.size(), 3);
  for (uint32_t i = 0; i < 3; ++i) {
    ASSERT_EQ(result[i], expected_result[i]);
  }
}

} // namespace torchrec
