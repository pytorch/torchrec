/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * Copyright (c) Intel Corporation.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <cstdint>
#include <utility>

namespace fbgemm {

template <typename K, typename V>
std::pair<K*, V*> radix_sort_parallel(
    K* inp_key_buf,
    V* inp_value_buf,
    K* tmp_key_buf,
    V* tmp_value_buf,
    int64_t elements_count,
    int64_t max_value);

} // namespace fbgemm
