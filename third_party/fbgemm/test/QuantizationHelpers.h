/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <cinttypes>

namespace fbgemm {

/*
 * @brief Make sure we won't have overflows from vpmaddubsw instruction.
 */
template <typename T>
void avoidOverflow(
    int m,
    int n,
    int k,
    const uint8_t* Aint8,
    int lda,
    T* B,
    int ldb);

template <typename T>
void avoidOverflow(int m, int n, int k, const uint8_t* Aint8, T* B);

} // namespace fbgemm
