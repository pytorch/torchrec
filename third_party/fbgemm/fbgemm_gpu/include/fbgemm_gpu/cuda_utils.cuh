/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <cuda.h>
#include <cassert>

#define CUDA_CHECK(X)                      \
  do {                                     \
    cudaError_t err = X;                   \
    assert(err == cudaError::cudaSuccess); \
  } while (0)
