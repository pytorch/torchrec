#pragma once

#include <cstdint>
#include "./FbgemmBuild.h"
#include "./UtilsAvx2.h"

namespace fbgemm {

// Structs from gemmlowp
//
// A structure to hold quantization parameters 'scale' and 'zero_point'.
// The meaning of these values is as the constants in the quantization equation
//
//   real_value = scale * (quantized_value - zero_point)
//
// In other words, 'zero_point' is the quantized value that corresponds
// to the real value 0, and 'scale' is the difference of real values
// corresponding to consecutive quantized values.
struct FBGEMM_API TensorQuantizationParams {
  float scale;
  std::int32_t zero_point;
  int precision;
  float Min() const;
  float Max() const;
};

// Parameters when we scale from int32 intermediate matrix multiplication
// results to 8-bit integers
struct FBGEMM_API RequantizationParams {
  // For floating-point requantization
  float real_multiplier;

  // For fixed-point requantization
  std::int32_t multiplier;
  int right_shift;

  TensorQuantizationParams target_qparams;
};

////////////////////////////////////////////////////////////////////////////////
// Utility functions

template <typename T = std::uint8_t, bool LEGACY = true>
void QuantizeAvx2(
    const float* src,
    T* dst,
    int len,
    const TensorQuantizationParams& qparams);

template <typename T = std::uint8_t>
void FusedQuantizeDequantizeAvx2(
    const float* src,
    float* dst,
    int len,
    const TensorQuantizationParams& qparams,
    float noise_ratio=0.0f);

/*
 * Random number generator in [0, 9]: https://www.jstatsoft.org/v08/i14/paper
 */
uint32_t FBGEMM_API Xor128(void);

/**
 * @brief Find the min and max value in a float matrix.
 */
void FBGEMM_API FindMinMax(const float* m, float* min, float* max, int len);

void RequantizeFixedPointAvx2(
    const std::int32_t* src,
    std::uint8_t* dst,
    int len,
    const RequantizationParams& params);

void RequantizeAvx2(
    const std::int32_t* src,
    std::uint8_t* dst,
    int len,
    const RequantizationParams& params);

/**
 * @brief Requantize with avx2 and bias is fused.
 */
template <
    bool A_SYMMETRIC,
    bool B_SYMMETRIC,
    QuantizationGranularity Q_GRAN,
    bool HAS_BIAS,
    bool FUSE_RELU,
    typename BIAS_TYPE = std::int32_t>
FBGEMM_API void requantizeOutputProcessingAvx2(
    std::uint8_t* out,
    const std::int32_t* inp,
    const block_type_t& block,
    int ld_out,
    int ld_in,
    const requantizationParams_t<BIAS_TYPE>& r);

template <
    bool A_SYMMETRIC,
    bool B_SYMMETRIC,
    QuantizationGranularity Q_GRAN,
    bool HAS_BIAS,
    bool FUSE_RELU,
    int C_PER_G,
    typename BIAS_TYPE = std::int32_t>
FBGEMM_API void requantizeOutputProcessingGConvAvx2(
    std::uint8_t* out,
    const std::int32_t* inp,
    const block_type_t& block,
    int ld_out,
    int ld_in,
    const requantizationParams_t<BIAS_TYPE>& r);

template <
    bool A_SYMMETRIC,
    bool B_SYMMETRIC,
    QuantizationGranularity Q_GRAN,
    bool HAS_BIAS,
    bool FUSE_RELU>
FBGEMM_API void requantizeForFloatAvx2(
    float* out,
    const std::int32_t* inp,
    const block_type_t& block,
    int ld_out,
    int ld_in,
    const requantizationForFloatParams_t& r);

template <typename InputType, int BIT_RATE>
void FloatOrHalfToFusedNBitRowwiseQuantizedSBHalfAvx2(
    const InputType* input,
    int input_rows,
    int input_columns,
    std::uint8_t* output);

template <typename InputType>
void FloatOrHalfToFused8BitRowwiseQuantizedSBFloatAvx2(
    const InputType* input,
    int input_rows,
    int input_columns,
    std::uint8_t* output);

template <typename OutputType, int BIT_RATE>
void FusedNBitRowwiseQuantizedSBHalfToFloatOrHalfAvx2(
    const std::uint8_t* input,
    int input_rows,
    int input_columns,
    OutputType* output);

template <typename OutputType>
void Fused8BitRowwiseQuantizedSBFloatToFloatOrHalfAvx2(
    const std::uint8_t* input,
    int input_rows,
    int input_columns,
    OutputType* output);

} // namespace fbgemm
