#pragma once

#include "./FbgemmBuild.h"
#include "./QuantUtilsAvx2.h"
#include "./Types.h"
#include "./Utils.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>

namespace fbgemm {

FBGEMM_API TensorQuantizationParams ChooseQuantizationParams(
    float min,
    float max,
    std::int32_t qmin,
    std::int32_t qmax,
    bool preserve_sparsity = false,
    bool force_scale_power_of_two = false);

FBGEMM_API void ChooseRequantizationMultiplier(
    float real_multiplier,
    std::int32_t* quantized_multiplier,
    int* right_shift,
    int requantization_multiplier_precision = 32);

////////////////////////////////////////////////////////////////////////////////
// Utility functions

// Clamp src in T1 to the desired precision and convert it to T2
// TODO: T26263653 fix signed-integer-overflow undefined behavior
template <typename T1, typename T2 = std::uint8_t>
NO_SANITIZE("signed-integer-overflow")
T2 clamp(T1 src, int precision, bool is_signed = false) {
  std::int32_t min = is_signed ? -(1LL << (precision - 1)) : 0;
  std::int32_t max =
      is_signed ? ((1LL << (precision - 1)) - 1) : (1LL << precision) - 1;

  // Make sure T1 and T2 can represent the precision
  assert(min >= std::numeric_limits<T1>::lowest());
  assert(min >= std::numeric_limits<T2>::lowest());
  assert(max <= std::numeric_limits<T1>::max());
  assert(max <= std::numeric_limits<T2>::max());

  return std::min<T1>(std::max<T1>(src, min), max);
}

/// Quantize src using zero_point and scale, clamp to the specified precision,
/// and convert it to type T
template <typename T, bool LEGACY = true>
T Quantize(
    float src,
    std::int32_t zero_point,
    float scale,
    int result_precision,
    bool result_is_signed = std::is_signed<T>::value) {
  // Note: We want to multiply with src with inv_scale instead of
  // dividing src by scale. The same is done in vector code and
  // at other places.
  //
  // Example:
  // With scale = 0.00214854861f, zero_point = 0 and src = 0.273939937f
  // transformed_val is 127.5 for src * inv_scale while
  // transformed_val is 127.499992 for src / scale.
  // Eventually 127.5 gets rounded to 128 while 127.499992 gets rounded to 127.
  float inv_scale = 1.0f / scale;

  float transformed_val = src * inv_scale;
  // nearbyint here performs round-to-nearest-ties-to-even with
  // default rounding mode.
  // For example, nearbyint(1.4) is 1.0, nearbyint(1.5) is 2.0
  // and nearbyint(2.5) is 2.0
  // Adding zero_point before or after rounding can make a difference
  // in exactly halfway cases.
  if (LEGACY) {
    transformed_val = std::nearbyint(zero_point + transformed_val);
  } else {
    transformed_val = zero_point + std::nearbyint(transformed_val);
  }
  // Please note the use of double. Unlike float, a double can represent
  // all int32 values exactly. Using a float results in a float value >
  // INT32_MAX conversion to int32 in clamp function and hence an UBSAN error.
  return clamp<double, T>(transformed_val, result_precision, result_is_signed);
}

template <typename T, bool LEGACY = true>
T Quantize(float src, const TensorQuantizationParams& qparams) {
  return Quantize<T, LEGACY>(
      src, qparams.zero_point, qparams.scale, qparams.precision);
}

template <typename T, bool LEGACY = true>
FBGEMM_API void Quantize(
    const float* src,
    T* dst,
    int len,
    const TensorQuantizationParams& qparams,
    int thread_id = 0,
    int num_threads = 1);

/*
 * @brief Quantize floating point data in src to type T
 *
 * @tparam T output quantized data type (int8_t, uint8_t and int32_t are
 *                  supported)
 *
 * @tparam T LAYOUT layout of input tensor in src. (KCX and KXC are supported)
 *                  KCX corresponds to KCRS or KCTRS (for weight tensors with
 *                  time dimension)
 *                  KXC corresponds to KRSC or KTRSC (for weight tensors with
 *                  time dimension)
 *
 * @param K Output channels for weight tensors
 * @param C Number of channels
 * @param X R*S or T*R*S
 * @param G Groups (if G == C the function performs channelwise quantization;
 *                  if 1 < G < C the function performs groupwise quantization;
 *                  if G == 1 the function performs per tensor quantization;)
 * @param scales floating point scales.
 *               Size should be equal G
 * @param zero_points zero points (should be reprsentable in type T).
 *                    Size should be equal G
 */
template <typename T, layout_t LAYOUT = layout_t::KCX>
FBGEMM_API void QuantizeGroupwise(
    const float* src,
    int K,
    int C,
    int X,
    int G,
    const float* scales,
    const std::int32_t* zero_points,
    T* dst);

template <typename T>
float Dequantize(T src, const TensorQuantizationParams& qparams) {
  return qparams.scale * (src - qparams.zero_point);
}

template <typename T>
void Dequantize(
    const T* src,
    float* dst,
    int len,
    const TensorQuantizationParams& qparams,
    int thread_id = 0,
    int num_threads = 1) {
  int i_begin, i_end;
  fbgemmPartition1D(thread_id, num_threads, len, i_begin, i_end);
  for (auto i = i_begin; i < i_end; i++) {
    dst[i] = Dequantize(src[i], qparams);
  }
}

template <typename T>
float FusedQuantizeDequantize(
    float src,
    const TensorQuantizationParams& qparams) {
  T q = Quantize<T, false>(
      src, qparams.zero_point, qparams.scale, qparams.precision);
  return Dequantize<T>(q, qparams);
}

/*
Fused integer quantization dequantization kernel to accelerate
quantization-aware training. Quantize fp32 values in src to (u)int8 using the
provided qparams, and dequantize quantized integer values back into fp32.
*/
template <typename T>
FBGEMM_API void FusedQuantizeDequantize(
    const float* src,
    float* dst,
    int len,
    const TensorQuantizationParams& qparams,
    int thread_id = 0,
    int num_threads = 1,
    float noise_ratio = 0.0f);

////////////////////////////////////////////////////////////////////////////////
// Requantization (pure fixed-point)

FBGEMM_API std::int64_t
SaturatingRoundingMulWithShift(std::int32_t a, std::int32_t b, int right_shift);

template <typename T>
T Requantize(
    std::int32_t src, // int32 input before requantization
    std::int32_t zero_point,
    std::int32_t multiplier,
    int right_shift,
    int result_precision,
    bool result_is_signed = false) {
  std::int64_t quantized_down =
      zero_point + SaturatingRoundingMulWithShift(src, multiplier, right_shift);
  return clamp<std::int64_t, T>(
      quantized_down, result_precision, result_is_signed);
}

template <typename T>
T RequantizeFixedPoint(
    std::int32_t src, // int32 input before requantization
    const RequantizationParams& params) {
  return Requantize<T>(
      src,
      params.target_qparams.zero_point,
      params.multiplier,
      params.right_shift,
      params.target_qparams.precision);
}

template <typename T>
FBGEMM_API void RequantizeFixedPoint(
    const std::int32_t* src,
    T* dst,
    int len,
    const RequantizationParams& params,
    int thread_id = 0,
    int num_threads = 1);

////////////////////////////////////////////////////////////////////////////////
// Requantization (with floats)

template <typename T>
T Requantize(
    std::int32_t src, // int32 input before requantization
    std::int32_t zero_point,
    float multiplier,
    int result_precision,
    bool result_is_signed = false) {
  long quantized_down = zero_point + std::lrintf(src * multiplier);
  return clamp<long, T>(quantized_down, result_precision, result_is_signed);
}

template <typename T>
T Requantize(
    std::int32_t src, // int32 input before requantization
    const RequantizationParams& params) {
  return Requantize<T>(
      src,
      params.target_qparams.zero_point,
      params.real_multiplier,
      params.target_qparams.precision);
}

template <typename T>
FBGEMM_API void Requantize(
    const std::int32_t* src,
    T* dst,
    int len,
    const RequantizationParams& params,
    int thread_id = 0,
    int num_threads = 1);

/**
 * Convert float (fp32 or fp16) inputs to rowwise quantized outputs.
 * bitrate specifies the number of bits in quantized output.
 * Scale and Bias are in fp16. Each row's Scale and Bias are stored in
 * the row itself (fused) at the end.
 *
 * @param bit_rate can be 2, 4, or 8
 */
template <typename InputType>
FBGEMM_API void FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf(
    int bit_rate,
    const InputType* input,
    int input_rows,
    int input_columns,
    std::uint8_t* output);

/**
 * Convert fused rowwise quantized inputs to float (fp32 or fp16).
 * bitrate specifies the number of bits in quantized input.
 * Scale and Bias are in fp16. Each row's Scale and Bias are stored in
 * the row itself (fused) at the end.
 *
 * @param bit_rate can be 2, 4, or 8
 */
template <typename OutputType>
FBGEMM_API void FusedNBitRowwiseQuantizedSBHalfToFloatOrHalf(
    int bit_rate,
    const uint8_t* input,
    int input_rows,
    int input_columns,
    OutputType* output);

/**
 * Convert float or half inputs to rowwise quantized (8-bit) outputs.
 * Scale and Bias are in float. Each row's Scale and Bias are stored in
 * the row itself (fused) at the end.
 *
 * This version intentionally supports only 8-bit because we want to discourage
 * the usage of float scale and bias with 2 and 4 bit cases as that diminishes
 * the overall memory savings.
 */
template <typename InputType>
FBGEMM_API void FloatOrHalfToFused8BitRowwiseQuantizedSBFloat(
    const InputType* input,
    int input_rows,
    int input_columns,
    std::uint8_t* output);

/**
 * Convert fused rowwise quantized (8-bit) inputs to float or half outputs.
 * Scale and Bias are in float. Each row's Scale and Bias are stored in
 * the row itself (fused) at the end.
 *
 * This version intentionally supports only 8-bit because
 * the corresponding quantize version only supports 8-bit.
 */
template <typename OutputType>
FBGEMM_API void Fused8BitRowwiseQuantizedSBFloatToFloatOrHalf(
    const uint8_t* input,
    int input_rows,
    int input_columns,
    OutputType* output);

/**
 * Same as ToFusedNBitRowwiseQuantizedSBHalf but unoptimized.
 * This should not be called directly except in testing.
 */
template <typename InputType>
FBGEMM_API void FloatOrHalfToFusedNBitRowwiseQuantizedSBHalfRef(
    int bit_rate,
    const InputType* input,
    int input_rows,
    int input_columns,
    std::uint8_t* output);

/**
 * Same as FloatOrHalfToFused8BitRowwiseQuantizedSBFloat but unoptimized.
 * This should not be called directly except in testing.
 */
template <typename InputType>
FBGEMM_API void FloatOrHalfToFused8BitRowwiseQuantizedSBFloatRef(
    const InputType* input,
    int input_rows,
    int input_columns,
    std::uint8_t* output);

/**
 * Same as FusedNBitRowwiseQuantizedSBHalfToFloat but unoptimized.
 * This should not be called directly except in testing.
 */
template <typename OutputType>
FBGEMM_API void FusedNBitRowwiseQuantizedSBHalfToFloatOrHalfRef(
    int bit_rate,
    const uint8_t* input,
    int input_rows,
    int input_columns,
    OutputType* output);

/**
 * Same as Fused8BitRowwiseQuantizedSBFloatToFloatOrHalf but unoptimized.
 * This should not be called directly except in testing.
 */
template <typename OutputType>
FBGEMM_API void Fused8BitRowwiseQuantizedSBFloatToFloatOrHalfRef(
    const uint8_t* input,
    int input_rows,
    int input_columns,
    OutputType* output);

} // namespace fbgemm
