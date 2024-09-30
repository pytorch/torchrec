#pragma once
#include <cstdint>
#include <type_traits>

namespace tde::details {

namespace bits_impl {
template <typename T>
struct Clz {
  int operator()(T v) const;
};

template <typename T>
struct Ctz {
  int operator()(T v) const;
};
} // namespace bits_impl

/**
 * Returns the number of leading 0-bits in t, starting at the most significant
 * bit position. If t is 0, the result is undefined.
 */
template <typename T>
inline int Clz(T t) {
  bits_impl::Clz<T> clz;
  return clz(t);
}

/**
 * Returns the number of trailing 0-bits in t, starting at the least significant
 * bit position. If t is 0, the result is undefined.
 */
template <typename T>
inline int Ctz(T t) {
  bits_impl::Ctz<T> ctz;
  return ctz(t);
}

} // namespace tde::details
