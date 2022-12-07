/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <condition_variable>
#include <mutex>
#include <torch/torch.h>

namespace torchrec {

/**
 * Multi-thread notification
 */
class Notification : public torch::CustomClassHolder {
 public:
  Notification() = default;

  void done();
  void wait();

  /**
   * Clear the set status.
   *
   * NOTE: Clear is not thread-safe.
   */
  void clear();

 private:
  bool set_{false};
  std::mutex mu_;
  std::condition_variable cv_;
};

} // namespace torchrec
