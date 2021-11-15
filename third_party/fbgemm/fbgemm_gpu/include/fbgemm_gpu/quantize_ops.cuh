/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <math_constants.h>

#define QUANTIZE_OPS_MAX(a, b) ((a) > (b) ? (a) : (b))
#define QUANTIZE_OPS_MIN(a, b) ((a) < (b) ? (a) : (b))

template <typename T>
__device__ inline __attribute__((always_inline)) T quantize_ops_shfl_xor(const T val, int laneMask, int width) {
#if CUDA_VERSION >= 9000
  return __shfl_xor_sync(0xffffffff, val, laneMask, width);
#else
  return __shfl_xor(val, laneMask, width);
#endif
}

__global__ inline void _get_8bit_qparam_cuda_kernel(
    const float* __restrict__ input,
    int nrows,
    int ncols,
    uint8_t* __restrict__ output,
    float* __restrict__ range_list) {
  const int row = (int)blockIdx.x * blockDim.y + threadIdx.y;

  const int ncols_aligned = (ncols + 4 - 1) / 4 * 4;
  const int output_columns = ncols_aligned + 2 * sizeof(float);

  // starting values for future reductions
  float minimum_element = CUDART_INF_F;
  float maximum_element = -CUDART_INF_F;

  // always a power of 2 up to size 32. Multiple rows can share the same warp when
  // smaller than 32.
  const int lane_width = blockDim.x;

  // March warp-wise through the row, doing thread local min and max reductions.
  // This loop will only execute once when ncol <= 32
  if (row < nrows) {
    const float* const input_row = input + row * ncols;

    for (int col = threadIdx.x; col < ncols; col += lane_width) {
      // Get thread-local minmax. These are the smallest min and max ever seen
      // by this thread.
      minimum_element = fminf(minimum_element, input_row[col]);
      maximum_element = fmaxf(maximum_element, input_row[col]);
    }
  }

  // Perform warp-wide min and max reductions. All threads in the warp participate,
  // even if they aren't assigned to a row, since we can't assume the existence of
  // the `*_sync` warp primitives with support for masking.
  for (int offset = lane_width >> 1; offset > 0; offset >>= 1) {
    minimum_element = fminf(minimum_element, quantize_ops_shfl_xor(minimum_element, offset, lane_width));
    maximum_element = fmaxf(maximum_element, quantize_ops_shfl_xor(maximum_element, offset, lane_width));
  }

  // only the leading thread in the warp is needed to return the final result in output.
  // Additionally, threads mapped to non-existent rows do not write to the output array.
  if (threadIdx.x != 0 || row >= nrows) {
    return;
  }

  const float range = maximum_element - minimum_element;
  float* const output_row_qparams =
      reinterpret_cast<float*>(output + row * output_columns + ncols_aligned);

  output_row_qparams[0] = range / 255.0f;
  output_row_qparams[1] = minimum_element;
  range_list[row] = range;
}

__global__ inline void _compute_8bit_quantize_cuda_kernel(
    const float* const __restrict__ input,
    const float* const __restrict__ range_list,
    const int nrows,
    const int ncols,
    std::uint8_t* const __restrict__ output) {
  constexpr float kEpsilon = 1e-8f;

  const int ncols_aligned = (ncols + 4 - 1) / 4 * 4;
  const int output_columns = ncols_aligned + 2 * sizeof(float);

  int row = (int)blockIdx.y * blockDim.y + threadIdx.y;
  const int col = (int)blockIdx.x * blockDim.x + threadIdx.x;
  const int row_incre = blockDim.y * gridDim.y;
  for (/*row*/; row < nrows; row += row_incre) {
    if (col < ncols) {
      // load scale, bias
      float* row_qparams = reinterpret_cast<float*>(
          output + row * output_columns + ncols_aligned);
      float bias = row_qparams[1];

      int input_idx = row * ncols + col;
      uint8_t* output_addr = output + row * output_columns + col;
      // TODO: lift range_list into shared memory. However, when nrows is large,
      // it might exceed the size of shared memory.
      const auto inverse_scale = 255.0f / (range_list[row] + kEpsilon);
      output_addr[0] = std::lrintf((input[input_idx] - bias) * inverse_scale);
    }
  }
}

// FP32 -> Fused 8-bit rowwise kernel
__global__ inline void _float_to_fused8bitrowwise_cuda_kernel(
    const float* __restrict__ input,
    int nrows,
    int ncols,
    std::uint8_t* __restrict__ output) {
  constexpr float kEpsilon = 1e-8f;

  int ncols_aligned = (ncols + 4 - 1) / 4 * 4;
  int output_columns = ncols_aligned + 2 * sizeof(float);

  int64_t row = (int)blockIdx.x * blockDim.x + threadIdx.x;

  if (row < nrows) {
    const float* input_row = input + row * ncols;
    std::uint8_t* output_row = output + row * output_columns;
    float* output_row_scale_bias =
        reinterpret_cast<float*>(output_row + ncols_aligned);

    float minimum_element =
        *thrust::min_element(thrust::device, input_row, input_row + ncols);
    float maximum_element =
        *thrust::max_element(thrust::device, input_row, input_row + ncols);
    float range = maximum_element - minimum_element;

    output_row_scale_bias[0] = range / 255.0f;
    output_row_scale_bias[1] = minimum_element;
    const auto inverse_scale = 255.0f / (range + kEpsilon);
    for (std::size_t col = 0; col < ncols; ++col) {
      output_row[col] =
          std::lrintf((input_row[col] - minimum_element) * inverse_scale);
    }
  }
}

// Fused 8-bit rowwise -> FP32 kernel
__global__ inline void _fused8bitrowwise_to_float_cuda_kernel(
    const std::uint8_t* const __restrict__ input,
    const int nrows,
    const int ncols,
    float* const __restrict__ output) {
  const int output_columns = ncols - 2 * sizeof(float);

  int row = (int)blockIdx.y * blockDim.y + threadIdx.y;
  const int col = (int)blockIdx.x * blockDim.x + threadIdx.x;
  const int row_incre = blockDim.y * gridDim.y;
  for (/*row*/; row < nrows; row += row_incre) {
    if (col < output_columns) {
      const std::uint8_t* input_row = input + row * ncols;
      const float* input_row_scale_bias =
          reinterpret_cast<const float*>(input_row + output_columns);
      float* output_row = output + row * output_columns;

      output_row[col] =
          input_row[col] * input_row_scale_bias[0] + input_row_scale_bias[1];
    }
  }
}

// Fake 8-bit quantize kernel: FP32 -> UINT8 rowwise -> FP32
__global__ inline void _fake_8bit_quantize_cuda_kernel(
    const float* __restrict__ input,
    int nrows,
    int ncols,
    float* __restrict__ output) {
  constexpr float kEpsilon = 1e-8f;
  const int row_incre = blockDim.y * gridDim.y;
  for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < nrows;
       row += row_incre) {
    const float* input_row = input + row * ncols;
    float* output_row = output + row * ncols;
    const int col_incre = blockDim.x * gridDim.x;
    for (int col = blockIdx.y * blockDim.y + threadIdx.y; col < ncols;
         col += col_incre) {
      float minimum_element =
          *thrust::min_element(thrust::device, input_row, input_row + ncols);
      float maximum_element =
          *thrust::max_element(thrust::device, input_row, input_row + ncols);
      float range = maximum_element - minimum_element;
      const auto inverse_scale = 255.0f / (range + kEpsilon);
      std::uint8_t quantized_val =
          std::lrintf((input_row[col] - minimum_element) * inverse_scale);
      output_row[col] = quantized_val * (range / 255.0f) + minimum_element;
    }
  }
}

// FP32 -> Fused 4/2-bit rowwise kernel
__global__ inline void _float_to_fusednbitrowwise_cuda_kernel(
    int bit_rate,
    const float* __restrict__ input,
    int nrows,
    int ncols,
    std::uint8_t* __restrict__ output) {
  int num_elem_per_byte = 8 / bit_rate;
  int output_columns =
      (ncols + num_elem_per_byte - 1) / num_elem_per_byte + 2 * sizeof(__half);

  int row = (int)blockIdx.x * blockDim.x + threadIdx.x;
  const int row_incre = blockDim.x * gridDim.x;
  for (/*row*/; row < nrows; row += row_incre) {
    const float* input_row = input + row * ncols;
    std::uint8_t* output_row = output + row * output_columns;
    __half* output_row_scale_bias = reinterpret_cast<__half*>(
        output_row + (ncols + num_elem_per_byte - 1) / num_elem_per_byte);

    float minimum_element =
        *thrust::min_element(thrust::device, input_row, input_row + ncols);
    float maximum_element =
        *thrust::max_element(thrust::device, input_row, input_row + ncols);

    minimum_element = __half2float(__float2half(minimum_element));
    const float range = maximum_element - minimum_element;

    float scale = __half2float(
        __float2half(range == 0 ? 1.0f : range / ((1 << bit_rate) - 1)));
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

    output_row_scale_bias[0] = __float2half(scale);
    output_row_scale_bias[1] = __float2half(minimum_element);
    for (std::size_t col = 0; col < ncols; ++col) {
      float X = input_row[col];

      std::uint8_t quantized = QUANTIZE_OPS_MAX(
          0,
          QUANTIZE_OPS_MIN(
              static_cast<int>(
                  std::lrintf((X - minimum_element) * inverse_scale)),
              static_cast<int>((1 << bit_rate) - 1)));

      if (col % num_elem_per_byte == 0) {
        output_row[col / num_elem_per_byte] = quantized;
      } else {
        output_row[col / num_elem_per_byte] |=
            (quantized << ((col & (num_elem_per_byte - 1)) * bit_rate));
      }
    }
  }
}

// Fused 4/2-bit rowwise -> FP32 kernel
__global__ inline void _fusednbitrowwise_to_float_cuda_kernel(
    const int bit_rate,
    const std::uint8_t* input,
    const int nrows,
    const int ncols,
    float* const output) {
  const int num_elem_per_byte = 8 / bit_rate;
  const int output_columns = (ncols - 2 * sizeof(__half)) * num_elem_per_byte;

  int row = (int)blockIdx.y * blockDim.y + threadIdx.y;
  const int col = (int)blockIdx.x * blockDim.x + threadIdx.x;
  const int row_incre = blockDim.y * gridDim.y;
  for (/*row*/; row < nrows; row += row_incre) {
    if (row < nrows && col < output_columns) {
      const std::uint8_t* input_row = input + row * ncols;
      const __half* input_row_scale_bias = reinterpret_cast<const __half*>(
          input_row +
          (output_columns + num_elem_per_byte - 1) / num_elem_per_byte);
      float scale = __half2float(input_row_scale_bias[0]);
      float bias = __half2float(input_row_scale_bias[1]);
      float* output_row = output + row * output_columns;

      std::uint8_t quantized = input_row[col / num_elem_per_byte];
      quantized >>= (col % num_elem_per_byte) * bit_rate;
      quantized &= (1 << bit_rate) - 1;
      output_row[col] = scale * quantized + bias;
    }
  }
}

// FP32 -> BF16 kernel
__global__ inline void _float_to_bfloat16_cuda_kernel(
    const float* __restrict__ input,
    const int nrows,
    const int ncols,
    uint16_t* __restrict__ output) {
  const int row_incre = blockDim.y * gridDim.y;
  const int col_incre = blockDim.x * gridDim.x;
  for (int row = blockIdx.y * blockDim.y + threadIdx.y; row < nrows;
       row += row_incre) {
    const float* input_row = input + row * ncols;
    uint16_t* output_row = output + row * ncols;
    for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < ncols;
         col += col_incre) {
      // Add 2^15 and right shift 16 to do round-nearest
      output_row[col] =
          (*reinterpret_cast<const uint32_t*>(input_row + col) + (1 << 15)) >>
          16;
    }
  }
}

// BF16 -> FP32 kernel
__global__ inline void _bfloat16_to_float_cuda_kernel(
    const uint16_t* __restrict__ input,
    const int nrows,
    const int ncols,
    float* __restrict__ output) {
  const int row_incre = blockDim.y * gridDim.y;
  const int col_incre = blockDim.x * gridDim.x;
  for (int row = blockIdx.y * blockDim.y + threadIdx.y; row < nrows;
       row += row_incre) {
    for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < ncols;
         col += col_incre) {
      const uint16_t* input_row = input + row * ncols;
      float* output_row = output + row * ncols;
      uint32_t val_fp32 = static_cast<uint32_t>(
                              reinterpret_cast<const uint16_t*>(input_row)[col])
          << 16;
      reinterpret_cast<uint32_t*>(output_row)[col] = val_fp32;
    }
  }
}

typedef union {
  uint32_t I;
  float F;
} fint32;

// TODO: add a flag later to control whether underflow
// flushes to 0 or clips to smallest denorm number.
__device__ inline uint8_t float_to_hfp8(
    float val_fp,
    int ebits,
    int mbits,
    int bias,
    float min_pos,
    float max_pos) {
  fint32 val_out, bouncer, smallest_normal;
  uint32_t sign_bit;

  val_out.F = val_fp;
  sign_bit = val_out.I & 0x80000000;
  val_out.I = val_out.I & 0x7FFFFFFF;
  val_out.F = min(val_out.F, max_pos);

  smallest_normal.I = (127 - bias + 1)
      << 23; // smallest hfp8 normal number in FP32
  // I don't know if the input "min_pos" is the smallest denormalized number
  // or the smallest normalized number. The test below needs to be done with
  // the smallest normal number, which is the numerical value 2^(1-bias)

  // The conversion for denormalized values are slightly different. HFP8 is so
  // low precision that gradual underflow is probably crucial
  if (val_out.F >= smallest_normal.F) {
    // Use round to nearest even. We make use of the standard rounding mechanism
    // in FP32 rather than rounding the mantissa and handling tie-to-even and
    // incrementing exponent We want to round of 23-mbits of the FP32 value
    // val_in This can be done by adding a power of 2 exactly 23-mbits larger
    // than the exponent of val_in This forces val_in to be moved to the right
    // and rounding exact at the location corresponding to having mbits of
    // explicit mantissa left
    bouncer.I = (val_out.I & 0xFF800000) + ((23 - mbits) << 23);
    val_out.F = (bouncer.F + val_out.F) - bouncer.F;
    // adding the bouncer rounds off bits, and subtracting bouncer
    // leaves the desired value, albeit in FP32 encoding
    // All we need is to change the exponent encoding to using "bias"
    val_out.I = uint32_t(val_out.I - ((127 - bias) << 23)) << (8 - ebits);
    val_out.I =
        ((val_out.I | sign_bit) >>
         24); // the 8 lsbs is the desired HFP8 encoding

  } else {
    // When the value is in the denormal range, IEEE numbers essentially becomes
    // a fixed point number. The lsb is the smallest non-zero number
    // 2^(1-bias-mbits) Hence, we define the bouncer so that its lsb is this
    // smallest non-zero number Adding the input to this bouncer forces rounding
    // to occur appropriately Also, in this situation, after adding the bouncer,
    // the 8 least significant bits of the sum is already the HFP8 encoding of
    // the desired result. Just need to restore the sign bit
    bouncer.I = (127 + (23 + (1 - bias - mbits))) << 23;
    val_out.F = bouncer.F + val_out.F;
    val_out.I = val_out.I | (sign_bit >> 24);
    ;
  }

  uint8_t bfp8_val = val_out.I; // get the 8 lsbs
  return bfp8_val;
}

__device__ inline float
hfp8_to_float(uint8_t hfp8_val, int ebits, int mbits, int bias) {
  fint32 val_out, sign, multiplier;

  sign.I = (hfp8_val & 0x80) << 24;
  val_out.I = (hfp8_val & 0x7F) << (24 - (8 - ebits));
  // printf("val_out %d %d\n", val_out.I, hfp8_val);
  // so that the mantissa bits start at the mantissa bit positions of FP32
  // encoding

  // Let the hfp8 mantissa bits correspond to the value frac, 0 <= frac < 1
  // So if the hfp8 value is a normal number, it's value is 2^e x (1+frac)
  // where e is its (true, unbiased) exponent
  // If the hfp8 value is denormal, the value is 2^(1-bias) x frac

  // However, the bit pattern in the 8-bit exponent field of val_out.F
  // is bias+e when hfp8 is normal, and 0 when hfp8 is subnormal.
  // So, as an FP32 value, when hfp8 is normal, val_out.F represents the value
  // of 2^(bias+e-127) * (1+frac)
  // And when hfp8 is subnormal, val_out.F is also subnormal, and represents the
  // value of 2^(-126) * frac In either case, val_out.F corresponds to
  // 2^(bias-127) * (value of hfp8 input) Thus, if we multiply val_out.F by
  // 2^(127-bias), we obtain the hfp8 value as an FP32 number

  multiplier.I = (127 + (127 - bias)) << 23; // multiplier.F is 2^(127-bias)
  val_out.F *= multiplier.F;
  val_out.I |= sign.I;
  return val_out.F;
}

__global__ inline void _float_to_hfp8_cuda_kernel(
    const float* __restrict__ input,
    const int nrows,
    const int ncols,
    uint8_t* __restrict__ output,
    int ebits,
    int mbits,
    int bias,
    float min_pos,
    float max_pos) {
  const int row_incre = blockDim.y * gridDim.y;
  const int col_incre = blockDim.x * gridDim.x;
  for (int row = blockIdx.y * blockDim.y + threadIdx.y; row < nrows;
       row += row_incre) {
    const float* input_row = input + row * ncols;
    uint8_t* output_row = output + row * ncols;
    for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < ncols;
         col += col_incre) {
      output_row[col] = float_to_hfp8(
          input_row[col], ebits, mbits, bias, min_pos, max_pos);
    }
  }
}

__global__ inline void _hfp8_to_float_cuda_kernel(
    const uint8_t* __restrict__ input,
    const int nrows,
    const int ncols,
    float* __restrict__ output,
    int ebits,
    int mbits,
    int bias) {
  const int row_incre = blockDim.y * gridDim.y;
  const int col_incre = blockDim.x * gridDim.x;
  for (int row = blockIdx.y * blockDim.y + threadIdx.y; row < nrows;
       row += row_incre) {
    for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < ncols;
         col += col_incre) {
      const uint8_t* input_row = input + row * ncols;
      float* output_row = output + row * ncols;
      output_row[col] = hfp8_to_float(input_row[col], ebits, mbits, bias);
    }
  }
}

#undef QUANTIZE_OPS_MAX
#undef QUANTIZE_OPS_MIN
