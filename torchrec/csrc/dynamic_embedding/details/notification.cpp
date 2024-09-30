/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "notification.h"

namespace torchrec {
void Notification::done() {
  {
    std::lock_guard<std::mutex> guard(mu_);
    set_ = true;
  }
  cv_.notify_all();
}
void Notification::wait() {
  std::unique_lock<std::mutex> lock(mu_);
  cv_.wait(lock, [this] { return set_; });
}

void Notification::clear() {
  set_ = false;
}
} // namespace torchrec
