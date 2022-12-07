/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torchrec/csrc/dynamic_embedding/details/bits_op.h>
#include <torchrec/csrc/dynamic_embedding/details/random_bits_generator.h>

namespace torchrec {

bool BitScanner::is_next_n_bits_all_zero(uint16_t& n_bits) {
  if ((n_bits == 0)) [[unlikely]] {
    return true;
  }
  if (array_idx_ == size_) [[unlikely]] {
    return true;
  }

  auto val = array[array_idx_];
  val &= static_cast<uint64_t>(-1) >>
      bit_idx; // mask higher bits to zeros if already scan.
  uint16_t remaining_bits =
      static_cast<uint16_t>(sizeof(uint64_t) * 8) - bit_idx;

  if (val == 0) { // already all zero
    auto scanned_bits = std::min(remaining_bits, n_bits);
    n_bits -= scanned_bits;
    bit_idx += scanned_bits;
    could_carry_bit_index_to_array_index();

    // if n_bits is scanned, return true. otherwise, scan remaining n_bits
    return n_bits == 0 || is_next_n_bits_all_zero(n_bits);
  } else { // val is not zero
    uint16_t positive_bit_position = clz(val);
    if (positive_bit_position >= n_bits + bit_idx) { // n_bits are all zero
      bit_idx += n_bits;
      // bit_idx must less than sizeof(uint64_t) * 8,
      // because the val is not zero but n_bits are zero.
      // so do not call could_carry_bit_index_to_array_index() here.

      // n_bits are all scanned
      n_bits = 0;
      return true;
    }

    // n_bits -= scanned bits
    n_bits -= std::min(remaining_bits, n_bits);
    bit_idx = positive_bit_position + 1;
    could_carry_bit_index_to_array_index();

    return false;
  }
}
void BitScanner::could_carry_bit_index_to_array_index() {
  if (bit_idx == sizeof(uint64_t) * 8) {
    bit_idx = 0;
    ++array_idx_;
  }
}

BitScanner::BitScanner(size_t n) : array(new uint64_t[n]), size_(n) {}

constexpr static size_t k_n_random_elems =
    8; // 64 Byte is just x86 L1 cache-line size

RandomBitsGenerator::RandomBitsGenerator()
    : engine_(std::random_device()()), scanner_(k_n_random_elems) {
  reset_scanner();
}
void RandomBitsGenerator::reset_scanner() {
  scanner_.reset_array([this](std::span<uint64_t> elems) {
    for (auto& elem : elems) {
      elem = engine_();
    }
  });
}

bool RandomBitsGenerator::is_next_n_bits_all_zero(uint16_t n_bits) {
  bool ok = scanner_.is_next_n_bits_all_zero(n_bits);
  if (n_bits != 0) { // scanner is end.
    reset_scanner();
  }
  if (!ok) {
    return false;
  }

  if (n_bits != 0) [[unlikely]] {
    return is_next_n_bits_all_zero(n_bits);
  } else {
    return true;
  }
}

RandomBitsGenerator::~RandomBitsGenerator() = default;
} // namespace torchrec
