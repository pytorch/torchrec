#pragma once
#include <memory>
#include <random>
#include <utility>
#include "tcb/span.hpp"

namespace tde::details {

/**
 * BitScanner holds n uint64_t values as bit stream.
 * And It can detect next n bits are all zero or not.
 * If n_bits is larger than remaining, the return
 * n_bits will be as remainder.
 */
class BitScanner {
 public:
  explicit BitScanner(size_t n);

  /**
   * Reset array data
   * @tparam Callback (tcb::span<uint64_t>) -> void
   * @param callback
   */
  template <typename Callback>
  void ResetArray(Callback callback) {
    callback(tcb::span<uint64_t>(array.get(), size_));
    array_idx_ = 0;
    bit_idx = 0;
  }

  bool IsNextNBitsAllZero(uint16_t& n_bits);

  // used by unittest only
  uint16_t array_idx_{0};
  uint16_t bit_idx{0};

 private:
  std::unique_ptr<uint64_t[]> array;
  uint16_t size_;

  // if bit_idx > 64, incr array_idx
  void CouldCarryBitIndexToArrayIndex();
};

class RandomBitsGenerator {
 public:
  RandomBitsGenerator();
  ~RandomBitsGenerator();

  /**
   * Is next N random bits are all zero or not.
   * i.e., the true prob is approximately 1/(2^n_bits).
   *
   * @param n_bits
   * @return
   */
  bool IsNextNBitsAllZero(uint16_t n_bits);

 private:
  BitScanner scanner_;
  std::mt19937_64 engine_;
  void ResetScanner();
};

} // namespace tde::details
