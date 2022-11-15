/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <torchrec/csrc/dynamic_embedding/details/redis/url.h>

namespace torchrec::url_parser::rules {

TEST(TDE, url) {
  auto url = parse_url("www.qq.com/?a=b&&c=d");
  ASSERT_EQ(url.host, "www.qq.com");
  ASSERT_TRUE(url.param.has_value());
  ASSERT_EQ("a=b&&c=d", url.param.value());
}

} // namespace torchrec::url_parser::rules
