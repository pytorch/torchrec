#define FBGEMM_EXPORTS
#include <algorithm>
#include <iterator>
#include <numeric>
#include <type_traits>

#include "fbgemm/QuantUtils.h"

#include <cpuinfo.h>

#include "fbgemm/Fbgemm.h"

#include "fbgemm/Types.h"

namespace fbgemm {

using namespace std;

// Use fp16_min as the small scale cutoff because we don't want to use scales in
// fp16 subnormal range. This is to be consistent with Glow and FakeLowP
// implementation for NNPI.
constexpr float SMALL_SCALE_THRESHOLD = 6.1e-5f;

float TensorQuantizationParams::Min() const {
  return Dequantize(0, *this);
}

float TensorQuantizationParams::Max() const {
  return Dequantize((1 << precision) - 1, *this);
}

TensorQuantizationParams ChooseQuantizationParams(
    float min,
    float max,
    int32_t qmin,
    int32_t qmax,
    bool preserve_sparsity,
    bool force_scale_power_of_two) {
  if (min < 0 && max > 0 && preserve_sparsity) {
    int symmetric_qmin = -((qmax - qmin) / 2 + 1);
    int symmetric_qmax = (qmax - qmin) / 2;
    double max_scale =
        std::max(fabs(min / symmetric_qmin), fabs(max / symmetric_qmax));
    min = max_scale * symmetric_qmin;
    max = max_scale * symmetric_qmax;
  }

  // We extend the [min, max] interval to ensure that it contains 0.
  // Otherwise, we would not meet the requirement that 0 be an exactly
  // representable value.
  min = std::min(min, 0.f);
  max = std::max(max, 0.f);

  // Use double precision for intermediate computation but use single precision
  // in final number to reflect the actual number used during quantization.
  float scale = (static_cast<double>(max) - min) / (qmax - qmin);
  // If scale is 0 or too small so its reciprocal is infinity, we arbitrary
  // adjust the scale to 0.1 . We want to avoid scale's reciprocal being
  // infinity because some of fbgemm code pre-computes scale's reciprocal to do
  // multiplication instead of division in the time critical part of code.
  if (scale == 0.0f || isinf(1.0f / scale)) {
    scale = 0.1;
  }
  assert(scale > 0);

  if (force_scale_power_of_two) {
    if (scale < 1) {
      scale = 1.0 / (1 << static_cast<int>(floor(log2(1.0 / scale))));
    } else {
      scale = 1 << static_cast<int>(ceil(log2(scale)));
    }
  }

  // Cut off small scale
  if (scale < SMALL_SCALE_THRESHOLD) {
    float org_scale = scale;
    scale = SMALL_SCALE_THRESHOLD;
    // Adjust the min and max based on the new scale
    if (min == 0.0f) {
      max = SMALL_SCALE_THRESHOLD * (qmax - qmin);
    } else if (max == 0.0f) {
      min = -SMALL_SCALE_THRESHOLD * (qmax - qmin);
    } else {
      float amplifier = SMALL_SCALE_THRESHOLD / org_scale;
      min *= amplifier;
      max *= amplifier;
    }
  }

  // Zero-point computation.
  // First the initial floating-point computation. The zero-point can be
  // determined from solving an affine equation for any known pair
  // (real value, corresponding quantized value).
  // We know two such pairs: (rmin, qmin) and (rmax, qmax).
  // The arithmetic error on the zero point computed from either pair
  // will be roughly machine_epsilon * (sum of absolute values of terms)
  // so we want to use the variant that adds the smaller terms.
  double zero_point_from_min = qmin - min / static_cast<double>(scale);
  double zero_point_from_max = qmax - max / static_cast<double>(scale);
  double zero_point_from_min_error =
      std::abs(qmin) + std::abs(min / static_cast<double>(scale));
  double zero_point_from_max_error =
      std::abs(qmax) + std::abs(max / static_cast<double>(scale));
  double initial_zero_point =
      zero_point_from_min_error < zero_point_from_max_error
      ? zero_point_from_min
      : zero_point_from_max;

  // Note: preserve_sparsity here means symmetric quantization.
  // for symmetric quantization, we force zero_point
  // to be a middle value between qmin and qmax.
  // If either min or max is 0, then we just use 0 as zero_point.
  if (min < 0 && max > 0 && preserve_sparsity) {
    initial_zero_point = static_cast<double>(qmin + qmax) / 2;
  }

  // Now we need to nudge the zero point to be an integer
  // (our zero points are integer, and this is motivated by the requirement
  // to be able to represent the real value "0" exactly as a quantized value,
  // which is required in multiple places, for example in Im2col with zero
  // padding).
  int32_t nudged_zero_point = 0;
  if (initial_zero_point < qmin) {
    nudged_zero_point = qmin;
  } else if (initial_zero_point > qmax) {
    nudged_zero_point = qmax;
  } else {
    nudged_zero_point = nearbyint(initial_zero_point);
  }

  TensorQuantizationParams result;
  result.scale = scale;
  result.zero_point = nudged_zero_point;
  return result;
}

void ChooseRequantizationMultiplier(
    float real_multiplier,
    int32_t* quantized_multiplier,
    int* right_shift,
    int requantization_multiplier_precision) {
  assert(real_multiplier != 0.f);

  // Assuming requantization_multiplier_precision_ = 31,
  // the default right shift is 31 when the real multiplier is already
  // in interval [1/2, 1).
  // Multiplying a 32-bit signed integer with all 31 bits except the sign bit
  // is used followed by 31-bit right shift implements multiplying with a real
  // number in [1/2, 1).
  // We want to utilize all 31 bits except the sign bit in the 32-bit signed
  // integer to get the best accuracy.
  int s = 31;

  // We want to bring the real multiplier into the interval [1/2, 1).
  // We can do so by multiplying it by two, and recording how many times
  // we multiplied by two so that we can compensate that by a right
  // shift by the same amount.
  if (real_multiplier > 0.f) {
    while (real_multiplier < 0.5f) {
      real_multiplier *= 2.f;
      s++;
    }
    while (real_multiplier > 1.f) {
      real_multiplier /= 2.f;
      s--;
    }
  }
  // Now that the real multiplier is in [1/2, 1), we convert it
  // into a fixed-point number.
  int64_t q = nearbyint(
      real_multiplier * (1ll << (requantization_multiplier_precision - 1)));
  assert(q <= (1ll << (requantization_multiplier_precision - 1)));
  // Handle the special case when the real multiplier was so close to 1
  // that its fixed-point approximation was undistinguishable from 1.
  // We handle this by dividing it by two, and remembering to decrement
  // the right shift amount.
  if (q == (1ll << (requantization_multiplier_precision - 1))) {
    q /= 2;
    s--;
  }
  assert(s >= 0);
  assert(q >= 0);
  assert(q <= numeric_limits<int32_t>::max());
  *quantized_multiplier = static_cast<int32_t>(q);
  *right_shift = s;
  assert(s < 64);
}

////////////////////////////////////////////////////////////////////////////////
// Utility functions

#define FBGEMM_SPECIALIZED_QUANTIZE(T, LEGACY)                      \
  template <>                                                       \
  FBGEMM_API void Quantize<T, LEGACY>(                              \
      const float* src,                                             \
      T* dst,                                                       \
      const int len,                                                \
      const TensorQuantizationParams& qparams,                      \
      int thread_id,                                                \
      int num_threads) {                                            \
    int i_begin, i_end;                                             \
    fbgemmPartition1D(thread_id, num_threads, len, i_begin, i_end); \
    for (int i = i_begin; i < i_end; ++i) {                         \
      dst[i] = Quantize<T, LEGACY>(src[i], qparams);                \
    }                                                               \
  }
FBGEMM_SPECIALIZED_QUANTIZE(uint16_t, true)
FBGEMM_SPECIALIZED_QUANTIZE(int16_t, true)
FBGEMM_SPECIALIZED_QUANTIZE(int32_t, true)
FBGEMM_SPECIALIZED_QUANTIZE(uint16_t, false)
FBGEMM_SPECIALIZED_QUANTIZE(int16_t, false)
FBGEMM_SPECIALIZED_QUANTIZE(int32_t, false)
#undef FBGEMM_SPECIALIZED_QUANTIZE

#define FBGEMM_SPECIALIZED_QUANTIZE_AVX2(T, LEGACY)                     \
  template <>                                                           \
  FBGEMM_API void Quantize<T, LEGACY>(                                  \
      const float* src,                                                 \
      T* dst,                                                           \
      int len,                                                          \
      const TensorQuantizationParams& qparams,                          \
      int thread_id,                                                    \
      int num_threads) {                                                \
    bool avx2_support = cpuinfo_initialize() && fbgemmHasAvx2Support(); \
    bool fma_support = cpuinfo_has_x86_fma3();                          \
    int i_begin, i_end;                                                 \
    fbgemmPartition1D(thread_id, num_threads, len, i_begin, i_end);     \
    if (avx2_support && fma_support && qparams.precision == 8) {        \
      /* fast path  */                                                  \
      QuantizeAvx2<T, LEGACY>(                                          \
          &src[i_begin], &dst[i_begin], i_end - i_begin, qparams);      \
    } else {                                                            \
      for (int i = i_begin; i < i_end; ++i) {                           \
        dst[i] = Quantize<T, LEGACY>(src[i], qparams);                  \
      }                                                                 \
    }                                                                   \
  }

FBGEMM_SPECIALIZED_QUANTIZE_AVX2(int8_t, true)
FBGEMM_SPECIALIZED_QUANTIZE_AVX2(uint8_t, true)
FBGEMM_SPECIALIZED_QUANTIZE_AVX2(int8_t, false)
FBGEMM_SPECIALIZED_QUANTIZE_AVX2(uint8_t, false)
#undef FBGEMM_SPECIALIZED_QUANTIZE_AVX2

#define FBGEMM_SPECIALIZED_FUSED_QUANTIZE_DEQUANTIZE_AVX2(T)            \
  template <>                                                           \
  FBGEMM_API void FusedQuantizeDequantize<T>(                           \
      const float* src,                                                 \
      float* dst,                                                       \
      int len,                                                          \
      const TensorQuantizationParams& qparams,                          \
      int thread_id,                                                    \
      int num_threads,                                                  \
      float noise_ratio) {                                              \
    bool avx2_support = cpuinfo_initialize() && fbgemmHasAvx2Support(); \
    bool fma_support = cpuinfo_has_x86_fma3();                          \
    int i_begin, i_end;                                                 \
    fbgemmPartition1D(thread_id, num_threads, len, i_begin, i_end);     \
    if (avx2_support && fma_support && qparams.precision == 8) {        \
      /* fast path  */                                                  \
      FusedQuantizeDequantizeAvx2<T>(                                   \
          &src[i_begin], &dst[i_begin], i_end - i_begin, qparams);      \
    } else if (noise_ratio <= 0.0f) {                                   \
      for (int i = i_begin; i < i_end; ++i) {                           \
        dst[i] = FusedQuantizeDequantize<T>(src[i], qparams);           \
      }                                                                 \
    } else {                                                            \
      throw std::runtime_error("Failed to initialize cpuinfo!");        \
    }                                                                   \
  }

FBGEMM_SPECIALIZED_FUSED_QUANTIZE_DEQUANTIZE_AVX2(int8_t)
FBGEMM_SPECIALIZED_FUSED_QUANTIZE_DEQUANTIZE_AVX2(uint8_t)
#undef FBGEMM_SPECIALIZED_FUSED_QUANTIZE_DEQUANTIZE_AVX2

#define FBGEMM_SPECIALIZED_QUANTIZEGROUPWISEKCX(T)                \
  template <>                                                     \
  FBGEMM_API void QuantizeGroupwise<T, layout_t::KCX>(            \
      const float* src,                                           \
      int N,                                                      \
      int C,                                                      \
      int X,                                                      \
      int G,                                                      \
      const float* scales,                                        \
      const std::int32_t* zero_points,                            \
      T* dst) {                                                   \
    assert(C % G == 0);                                           \
    int C_per_G = C / G;                                          \
    for (int i = 0; i < N; ++i) {                                 \
      for (int g = 0; g < G; ++g) {                               \
        float scale = scales[g];                                  \
        int32_t zero_point = zero_points[g];                      \
        for (int c = 0; c < C / G; ++c) {                         \
          for (int x = 0; x < X; ++x) {                           \
            dst[(i * C + g * C_per_G + c) * X + x] = Quantize<T>( \
                src[(i * C + g * C_per_G + c) * X + x],           \
                zero_point,                                       \
                scale,                                            \
                8 * sizeof(T));                                   \
          }                                                       \
        }                                                         \
      }                                                           \
    }                                                             \
  }
FBGEMM_SPECIALIZED_QUANTIZEGROUPWISEKCX(int8_t)
FBGEMM_SPECIALIZED_QUANTIZEGROUPWISEKCX(int32_t)
#undef FBGEMM_SPECIALIZED_QUANTIZEGROUPWISEKCX

template <>
FBGEMM_API void QuantizeGroupwise<uint8_t, layout_t::KCX>(
    const float* src,
    int K,
    int C,
    int X,
    int G,
    const float* scales,
    const std::int32_t* zero_points,
    uint8_t* dst) {
  assert(C % G == 0);
  int C_per_G = C / G;
  fbgemm::TensorQuantizationParams qparams;
  qparams.precision = 8 * sizeof(uint8_t);
  bool takeFastPath =
      cpuinfo_initialize() && fbgemmHasAvx2Support() && cpuinfo_has_x86_fma3();

  for (int i = 0; i < K; ++i) {
    for (int g = 0; g < G; ++g) {
      qparams.scale = scales[g];
      qparams.zero_point = zero_points[g];
      if (takeFastPath) {
        QuantizeAvx2(
            src + (i * C + g * C_per_G) * X,
            dst + (i * C + g * C_per_G) * X,
            C_per_G * X,
            qparams);
      } else {
        for (int c = 0; c < C / G; ++c) {
          for (int x = 0; x < X; ++x) {
            dst[(i * C + g * C_per_G + c) * X + x] = Quantize<uint8_t>(
                src[(i * C + g * C_per_G + c) * X + x],
                qparams.zero_point,
                qparams.scale,
                qparams.precision);
          }
        }
      }
    }
  }
}

#define FBGEMM_SPECIALIZED_QUANTIZEGROUPWISEKXC(T)                \
  template <>                                                     \
  FBGEMM_API void QuantizeGroupwise<T, layout_t::KXC>(            \
      const float* src,                                           \
      int K,                                                      \
      int C,                                                      \
      int X,                                                      \
      int G,                                                      \
      const float* scales,                                        \
      const std::int32_t* zero_points,                            \
      T* dst) {                                                   \
    assert(C % G == 0);                                           \
    int C_per_G = C / G;                                          \
    for (int i = 0; i < K; ++i) {                                 \
      for (int x = 0; x < X; ++x) {                               \
        for (int g = 0; g < G; ++g) {                             \
          float scale = scales[g];                                \
          int32_t zero_point = zero_points[g];                    \
          for (int c = 0; c < C / G; ++c) {                       \
            dst[(i * X + x) * C + g * C_per_G + c] = Quantize<T>( \
                src[(i * X + x) * C + g * C_per_G + c],           \
                zero_point,                                       \
                scale,                                            \
                8 * sizeof(T));                                   \
          }                                                       \
        }                                                         \
      }                                                           \
    }                                                             \
  }
FBGEMM_SPECIALIZED_QUANTIZEGROUPWISEKXC(int8_t)
FBGEMM_SPECIALIZED_QUANTIZEGROUPWISEKXC(uint8_t)
FBGEMM_SPECIALIZED_QUANTIZEGROUPWISEKXC(int32_t)
#undef FBGEMM_SPECIALIZED_QUANTIZEGROUPWISEKXC

////////////////////////////////////////////////////////////////////////////////
// Requantization (pure fixed-point)

int64_t SaturatingRoundingMulWithShift(int32_t a, int32_t b, int right_shift) {
  int64_t a_64(a);
  int64_t b_64(b);
  int64_t ab_64 = a_64 * b_64;

  int64_t nudge = 1ll << (right_shift - 1);
  return (ab_64 + nudge) >> right_shift;
}

#define FBGEMM_SPECIALIZED_REQUANTIZE(T)                            \
  template <>                                                       \
  FBGEMM_API void Requantize<T>(                                    \
      const int32_t* src,                                           \
      T* dst,                                                       \
      const int len,                                                \
      const RequantizationParams& params,                           \
      int thread_id,                                                \
      int num_threads) {                                            \
    int i_begin, i_end;                                             \
    fbgemmPartition1D(thread_id, num_threads, len, i_begin, i_end); \
    for (int i = i_begin; i < i_end; ++i) {                         \
      dst[i] = Requantize<T>(src[i], params);                       \
    }                                                               \
  }
FBGEMM_SPECIALIZED_REQUANTIZE(uint16_t)
FBGEMM_SPECIALIZED_REQUANTIZE(int32_t)
#undef FBGEMM_SPECIALIZED_REQUANTIZE

template <>
FBGEMM_API void Requantize<uint8_t>(
    const int32_t* src,
    uint8_t* dst,
    const int len,
    const RequantizationParams& params,
    int thread_id,
    int num_threads) {
  int i_begin, i_end;
  fbgemmPartition1D(thread_id, num_threads, len, i_begin, i_end);
  if (params.target_qparams.precision == 8 && cpuinfo_initialize() &&
      fbgemmHasAvx2Support()) {
    RequantizeAvx2(&src[i_begin], &dst[i_begin], i_end - i_begin, params);
  } else {
    for (int i = i_begin; i < i_end; ++i) {
      dst[i] = Requantize<uint8_t>(src[i], params);
    }
  }
}

template <typename T>
FBGEMM_API void RequantizeFixedPoint(
    const std::int32_t* src,
    T* dst,
    int len,
    const RequantizationParams& params,
    int thread_id,
    int num_threads) {
  int i_begin, i_end;
  fbgemmPartition1D(thread_id, num_threads, len, i_begin, i_end);
  if (std::is_same<T, uint8_t>::value && params.target_qparams.precision == 8 &&
      cpuinfo_initialize() && fbgemmHasAvx2Support()) {
    RequantizeFixedPointAvx2(
        &src[i_begin], &dst[i_begin], i_end - i_begin, params);
  } else {
    for (int i = i_begin; i < i_end; ++i) {
      dst[i] = RequantizeFixedPoint<T>(src[i], params);
    }
  }
}

#define FBGEMM_SPECIALIZED_REQUANTIZE(T)                            \
  template <>                                                       \
  FBGEMM_API void RequantizeFixedPoint<T>(                          \
      const int32_t* src,                                           \
      T* dst,                                                       \
      const int len,                                                \
      const RequantizationParams& params,                           \
      int thread_id,                                                \
      int num_threads) {                                            \
    int i_begin, i_end;                                             \
    fbgemmPartition1D(thread_id, num_threads, len, i_begin, i_end); \
    for (int i = i_begin; i < i_end; ++i) {                         \
      dst[i] = RequantizeFixedPoint<T>(src[i], params);             \
    }                                                               \
  }
FBGEMM_SPECIALIZED_REQUANTIZE(uint16_t)
FBGEMM_SPECIALIZED_REQUANTIZE(int32_t)
#undef FBGEMM_SPECIALIZED_REQUANTIZE

template <>
FBGEMM_API void RequantizeFixedPoint<uint8_t>(
    const int32_t* src,
    uint8_t* dst,
    const int len,
    const RequantizationParams& params,
    int thread_id,
    int num_threads) {
  int i_begin, i_end;
  fbgemmPartition1D(thread_id, num_threads, len, i_begin, i_end);

  if (params.target_qparams.precision == 8 && cpuinfo_initialize() &&
      fbgemmHasAvx2Support()) {
    RequantizeFixedPointAvx2(
        &src[i_begin], &dst[i_begin], i_end - i_begin, params);
  } else {
    for (int i = i_begin; i < i_end; ++i) {
      dst[i] = RequantizeFixedPoint<uint8_t>(src[i], params);
    }
  }
}

template <typename InputType>
void FloatOrHalfToFusedNBitRowwiseQuantizedSBHalfRef(
    int bit_rate,
    const InputType* input,
    int input_rows,
    int input_columns,
    std::uint8_t* output) {
  static_assert(
      std::is_same<InputType, float>() || std::is_same<InputType, float16>(),
      "Only float and float16 types are allowed.");
  int num_elem_per_byte = 8 / bit_rate;
  int output_columns =
      (input_columns + num_elem_per_byte - 1) / num_elem_per_byte +
      2 * sizeof(float16);
  std::vector<float> input_row_float(input_columns);
  for (int row = 0; row < input_rows; ++row) {
    const InputType* input_row = input + row * input_columns;
    std::uint8_t* output_row = output + row * output_columns;
    float16* output_row_scale_bias = reinterpret_cast<float16*>(
        output_row +
        (input_columns + num_elem_per_byte - 1) / num_elem_per_byte);

    // NOTE: this can be optimized, however we don't care much about performance
    // for reference implementation.
    for (int col = 0; col < input_columns; ++col) {
      if (std::is_same<InputType, float>()) {
        input_row_float[col] = input_row[col];
      } else {
        input_row_float[col] = cpu_half2float(input_row[col]);
      }
    }

    float minimum_element =
        *std::min_element(input_row_float.begin(), input_row_float.end());
    float maximum_element =
        *std::max_element(input_row_float.begin(), input_row_float.end());
    // Truncate since bias will be represented by fp16. Keep higher precision
    // max untouched.
    float16 minimum_element_fp16 = cpu_float2half_rn(minimum_element);
    minimum_element = cpu_half2float(minimum_element_fp16);
    const float range = maximum_element - minimum_element;

    float scale = range == 0 ? 1.0f : range / ((1 << bit_rate) - 1);
    float16 scale_fp16 = cpu_float2half_rn(scale);
    scale = cpu_half2float(scale_fp16);
    if (scale == 0) {
      // Corner case handling when maximum_element == minimum_element
      // Any scale would work because X - minimum_element will be 0 for all X
      scale = 1.0f;
    }
    float inverse_scale = 1.0f / scale;
    if (std::isinf(inverse_scale)) {
      scale = 1.0f;
      inverse_scale = 1.0f;
    }

    output_row_scale_bias[0] = cpu_float2half_rn(scale);
    output_row_scale_bias[1] = minimum_element_fp16;
    for (int col = 0; col < input_columns; ++col) {
      float X = input_row_float[col];
      std::uint8_t quantized = std::max(
          0,
          std::min<int>(
              std::lrintf((X - minimum_element) * inverse_scale),
              (1 << bit_rate) - 1));
      if (col % num_elem_per_byte == 0) {
        output_row[col / num_elem_per_byte] = quantized;
      } else {
        output_row[col / num_elem_per_byte] |=
            (quantized << ((col % num_elem_per_byte) * bit_rate));
      }
    }
  }
}

template <typename InputType>
void FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf(
    int bit_rate,
    const InputType* input,
    int input_rows,
    int input_columns,
    std::uint8_t* output) {
  // Currenlty we can only dequantize if the number of input columns
  // is a multiple of number of elements_per_byte

  int num_elem_per_byte = 8 / bit_rate;
  if (input_columns % num_elem_per_byte != 0) {
    throw std::runtime_error("Unsupported number of columns");
  }

  if (cpuinfo_initialize() && fbgemmHasAvx2Support()) {
    switch (bit_rate) {
      case 2:
        FloatOrHalfToFusedNBitRowwiseQuantizedSBHalfAvx2<InputType, 2>(
            input, input_rows, input_columns, output);
        break;
      case 4:
        FloatOrHalfToFusedNBitRowwiseQuantizedSBHalfAvx2<InputType, 4>(
            input, input_rows, input_columns, output);
        break;
      case 8:
        FloatOrHalfToFusedNBitRowwiseQuantizedSBHalfAvx2<InputType, 8>(
            input, input_rows, input_columns, output);
        break;
      default:
        FloatOrHalfToFusedNBitRowwiseQuantizedSBHalfRef<InputType>(
            bit_rate, input, input_rows, input_columns, output);
    }
  } else {
    FloatOrHalfToFusedNBitRowwiseQuantizedSBHalfRef<InputType>(
        bit_rate, input, input_rows, input_columns, output);
  }
}

template <typename InputType>
void FloatOrHalfToFused8BitRowwiseQuantizedSBFloatRef(
    const InputType* input,
    int input_rows,
    int input_columns,
    std::uint8_t* output) {
  constexpr float kEpsilon = 1e-8f;

  int output_columns = input_columns + 2 * sizeof(float);
  std::vector<float> input_row_float(input_columns);
  for (int row = 0; row < input_rows; ++row) {
    const InputType* input_row = input + row * input_columns;
    std::uint8_t* output_row = output + row * output_columns;
    float* output_row_scale_bias =
        reinterpret_cast<float*>(output_row + input_columns);

    for (int col = 0; col < input_columns; ++col) {
      if (std::is_same<InputType, float>()) {
        input_row_float[col] = input_row[col];
      } else {
        input_row_float[col] = cpu_half2float(input_row[col]);
      }
    }

    float minimum_element =
        *std::min_element(input_row_float.begin(), input_row_float.end());
    float maximum_element =
        *std::max_element(input_row_float.begin(), input_row_float.end());
    float range = maximum_element - minimum_element;

    output_row_scale_bias[0] = range / 255.0f;
    output_row_scale_bias[1] = minimum_element;
    const auto inverse_scale = 255.0f / (range + kEpsilon);
    for (int col = 0; col < input_columns; ++col) {
      output_row[col] =
          std::lrintf((input_row_float[col] - minimum_element) * inverse_scale);
    }
  }
}

template <typename InputType>
void FloatOrHalfToFused8BitRowwiseQuantizedSBFloat(
    const InputType* input,
    int input_rows,
    int input_columns,
    std::uint8_t* output) {
  if (cpuinfo_initialize() && fbgemmHasAvx2Support()) {
    FloatOrHalfToFused8BitRowwiseQuantizedSBFloatAvx2<InputType>(
        input, input_rows, input_columns, output);
  } else {
    FloatOrHalfToFused8BitRowwiseQuantizedSBFloatRef<InputType>(
        input, input_rows, input_columns, output);
  }
}

template <typename OutputType>
void FusedNBitRowwiseQuantizedSBHalfToFloatOrHalfRef(
    int bit_rate,
    const uint8_t* input,
    int input_rows,
    int input_columns,
    OutputType* output) {
  static_assert(
      std::is_same<OutputType, float>() || std::is_same<OutputType, float16>(),
      "Only float and float16 types are allowed.");
  int num_elem_per_byte = 8 / bit_rate;
  int output_columns =
      (input_columns - 2 * sizeof(float16)) * num_elem_per_byte;

  for (int row = 0; row < input_rows; ++row) {
    const std::uint8_t* input_row = input + row * input_columns;
    const float16* input_row_scale_bias = reinterpret_cast<const float16*>(
        input_row +
        (output_columns + num_elem_per_byte - 1) / num_elem_per_byte);
    float scale = cpu_half2float(input_row_scale_bias[0]);
    float bias = cpu_half2float(input_row_scale_bias[1]);
    OutputType* output_row = output + row * output_columns;

    for (int col = 0; col < output_columns; ++col) {
      std::uint8_t quantized = input_row[col / num_elem_per_byte];
      quantized >>= (col % num_elem_per_byte) * bit_rate;
      quantized &= (1 << bit_rate) - 1;
      float output_value = scale * quantized + bias;
      if (std::is_same<OutputType, float>()) {
        output_row[col] = output_value;
      } else {
        output_row[col] = cpu_float2half_rn(output_value);
      }
    }
  }
}

template <typename OutputType>
void FusedNBitRowwiseQuantizedSBHalfToFloatOrHalf(
    int bit_rate,
    const uint8_t* input,
    int input_rows,
    int input_columns,
    OutputType* output) {
  if (cpuinfo_initialize() && fbgemmHasAvx2Support()) {
    switch (bit_rate) {
      case 2:
        FusedNBitRowwiseQuantizedSBHalfToFloatOrHalfAvx2<OutputType, 2>(
            input, input_rows, input_columns, output);
        break;
      case 4:
        FusedNBitRowwiseQuantizedSBHalfToFloatOrHalfAvx2<OutputType, 4>(
            input, input_rows, input_columns, output);
        break;
      case 8:
        FusedNBitRowwiseQuantizedSBHalfToFloatOrHalfAvx2<OutputType, 8>(
            input, input_rows, input_columns, output);
        break;
      default:
        FusedNBitRowwiseQuantizedSBHalfToFloatOrHalfRef<OutputType>(
            bit_rate, input, input_rows, input_columns, output);
    }
  } else {
    FusedNBitRowwiseQuantizedSBHalfToFloatOrHalfRef<OutputType>(
        bit_rate, input, input_rows, input_columns, output);
  }
}

template <typename OutputType>
void Fused8BitRowwiseQuantizedSBFloatToFloatOrHalfRef(
    const std::uint8_t* input,
    int input_rows,
    int input_columns,
    OutputType* output) {
  int output_columns = input_columns - 2 * sizeof(float);

  for (int row = 0; row < input_rows; ++row) {
    const std::uint8_t* input_row = input + row * input_columns;
    const float* input_row_scale_bias =
        reinterpret_cast<const float*>(input_row + output_columns);
    OutputType* output_row = output + row * output_columns;

    for (int col = 0; col < output_columns; ++col) {
      float output_value =
          input_row[col] * input_row_scale_bias[0] + input_row_scale_bias[1];
      if (std::is_same<OutputType, float>()) {
        output_row[col] = output_value;
      } else {
        output_row[col] = cpu_float2half_rn(output_value);
      }
    }
  }
}

template <typename OutputType>
void Fused8BitRowwiseQuantizedSBFloatToFloatOrHalf(
    const std::uint8_t* input,
    int input_rows,
    int input_columns,
    OutputType* output) {
  if (cpuinfo_initialize() && fbgemmHasAvx2Support()) {
    Fused8BitRowwiseQuantizedSBFloatToFloatOrHalfAvx2<OutputType>(
        input, input_rows, input_columns, output);
  } else {
    Fused8BitRowwiseQuantizedSBFloatToFloatOrHalfRef<OutputType>(
        input, input_rows, input_columns, output);
  }
}

#define INSTANTIATE_QuantizationFunctions(type)                                \
  template FBGEMM_API void                                                     \
  FloatOrHalfToFusedNBitRowwiseQuantizedSBHalfRef<type>(                       \
      int bit_rate,                                                            \
      const type* input,                                                       \
      int input_rows,                                                          \
      int input_columns,                                                       \
      std::uint8_t* output);                                                   \
  template FBGEMM_API void FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf<type>( \
      int bit_rate,                                                            \
      const type* input,                                                       \
      int input_rows,                                                          \
      int input_columns,                                                       \
      std::uint8_t* output);                                                   \
  template FBGEMM_API void                                                     \
  FusedNBitRowwiseQuantizedSBHalfToFloatOrHalfRef<type>(                       \
      int bit_rate,                                                            \
      const uint8_t* input,                                                    \
      int input_rows,                                                          \
      int input_columns,                                                       \
      type* output);                                                           \
  template FBGEMM_API void FusedNBitRowwiseQuantizedSBHalfToFloatOrHalf<type>( \
      int bit_rate,                                                            \
      const uint8_t* input,                                                    \
      int input_rows,                                                          \
      int input_columns,                                                       \
      type* output);                                                           \
  template FBGEMM_API void                                                     \
  FloatOrHalfToFused8BitRowwiseQuantizedSBFloatRef<type>(                      \
      const type* input,                                                       \
      int input_rows,                                                          \
      int input_columns,                                                       \
      std::uint8_t* output);                                                   \
  template FBGEMM_API void                                                     \
  FloatOrHalfToFused8BitRowwiseQuantizedSBFloat<type>(                         \
      const type* input,                                                       \
      int input_rows,                                                          \
      int input_columns,                                                       \
      std::uint8_t* output);                                                   \
  template FBGEMM_API void                                                     \
  Fused8BitRowwiseQuantizedSBFloatToFloatOrHalfRef<type>(                      \
      const uint8_t* input, int input_rows, int input_columns, type* output);  \
  template FBGEMM_API void                                                     \
  Fused8BitRowwiseQuantizedSBFloatToFloatOrHalf<type>(                         \
      const uint8_t* input, int input_rows, int input_columns, type* output);

// clang-format off
INSTANTIATE_QuantizationFunctions(float)
INSTANTIATE_QuantizationFunctions(float16)
// clang-format on

#undef INSTANTIATE_QuantizationFunctions

} // namespace fbgemm
