/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "torchrec/inference/GPUExecutor.h"

#include <iostream>

int main(int argc, const char* argv[]) {
  std::vector<std::unique_ptr<torchrec::GPUExecutor>> executors;

  // run server

  std::cout << executors.size() << std::endl;
}
