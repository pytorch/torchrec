/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <stdint.h>
#include <memory>

namespace torchrec {

/**
 * Bitmap
 *
 * A bitmap for recording whether num_bits of slots are
 *  occupied or free.
 */
template <typename T = uint32_t>
struct Bitmap {
  explicit Bitmap(int64_t num_bits);
  Bitmap(const Bitmap&) = delete;
  Bitmap(Bitmap&&) noexcept = default;

  /**
   * Returns the position of the next free slot.
   * If the bitmap is full, return `num_total_bits_`.
   */
  int64_t next_free_bit();

  /**
   * Set the slot of position `offset` to free.
   */
  void free_bit(int64_t offset);

  /**
   * Returns if all slots in the bitmap is occupied.
   */
  bool full() const;

  static constexpr int64_t num_bits_per_value = sizeof(T) * 8;

  const int64_t num_total_bits_;
  const int64_t num_values_;
  std::unique_ptr<T[]> values_;

  int64_t next_free_bit_;
};

} // namespace torchrec

#include <torchrec/csrc/dynamic_embedding/details/bitmap_impl.h>
