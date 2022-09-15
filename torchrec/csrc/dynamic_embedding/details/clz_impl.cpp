#include <torchrec/csrc/dynamic_embedding/details/bits_op.h>

namespace torchrec::dynamic_embedding::details::bits_impl {

template <typename T>
inline static bool get_bit(T n, int k) {
  int mask = 1 << k;
  return static_cast<bool>(n & mask);
}

template <typename T>
struct ClzImpl {
  /**
   * Naive implementation for no __builtin_clz
   * @param v
   * @return
   */
  int operator()(T v) const {
    int result = 0;
    for (uint16_t num_bits = sizeof(T) * 8; num_bits != 0;
         --num_bits, ++result) {
      if (get_bit(v, num_bits - 1)) {
        break;
      }
    }
    return result;
  }
};

#if defined(__GNUC__) || defined(__clang__)

template <>
struct ClzImpl<unsigned int> {
  int operator()(unsigned int v) const {
    return __builtin_clz(v);
  }
};

template <>
struct ClzImpl<int> {
  int operator()(int v) const {
    return __builtin_clz(static_cast<unsigned int>(v));
  }
};

template <>
struct ClzImpl<unsigned long> {
  int operator()(unsigned long v) const {
    return __builtin_clzl(v);
  }
};

template <>
struct ClzImpl<long> {
  int operator()(long v) const {
    return __builtin_clzl(static_cast<unsigned long>(v));
  }
};

template <>
struct ClzImpl<unsigned long long> {
  int operator()(unsigned long long v) const {
    return __builtin_clzll(v);
  }
};

template <>
struct ClzImpl<long long> {
  int operator()(long long v) const {
    return __builtin_clzll(static_cast<unsigned long long>(v));
  }
};

#endif

template <typename T>
int Clz<T>::operator()(T v) const {
  ClzImpl<T> clz;
  return clz(v);
}

template struct Clz<int>;
template struct Clz<unsigned int>;
template struct Clz<long>;
template struct Clz<unsigned long>;
template struct Clz<long long>;
template struct Clz<unsigned long long>;

// only for unittests
template struct Clz<int8_t>;
template struct Clz<uint8_t>;
} // namespace torchrec::dynamic_embedding::details::bits_impl
