#pragma once
#include <memory>
#include <random>
#include <span>
#include <utility>

namespace torchrec {

/**
 * BitScanner holds n uint64_t values as bit stream.
 * And It can detect next n bits are all zero or not.
 * If n_bits is larger than remaining, the return
 * n_bits will be as remainder.
 */
class BitScanner {
 public:
  explicit BitScanner(size_t n);
  BitScanner(const BitScanner&) = delete;
  BitScanner(BitScanner&&) noexcept = default;

  /**
   * Reset array data
   * @tparam Callback (std::span<uint64_t>) -> void
   * @param callback
   */
  template <typename Callback>
  void reset_array(Callback callback) {
    callback(std::span<uint64_t>(array.get(), size_));
    array_idx_ = 0;
    bit_idx = 0;
  }

  bool is_next_n_bits_all_zero(uint16_t& n_bits);

  // used by unittest only
  uint16_t array_idx_{0};
  uint16_t bit_idx{0};

 private:
  std::unique_ptr<uint64_t[]> array;
  uint16_t size_;

  // if bit_idx > 64, incr array_idx
  void could_carry_bit_index_to_array_index();
};

class RandomBitsGenerator {
 public:
  RandomBitsGenerator();
  ~RandomBitsGenerator();
  RandomBitsGenerator(const RandomBitsGenerator&) = delete;
  RandomBitsGenerator(RandomBitsGenerator&&) noexcept = default;

  /**
   * Is next N random bits are all zero or not.
   * i.e., the true prob is approximately 1/(2^n_bits).
   *
   * @param n_bits
   * @return
   */
  bool is_next_n_bits_all_zero(uint16_t n_bits);

 private:
  BitScanner scanner_;
  std::mt19937_64 engine_;
  void reset_scanner();
};

} // namespace torchrec
