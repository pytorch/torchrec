/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <torchrec/csrc/dynamic_embedding/details/notification.h>
#include <thread>

namespace torchrec {
TEST(TDE, notification) {
  Notification notification;
  std::thread th([&] { notification.done(); });
  notification.wait();
  th.join();
}
} // namespace torchrec
