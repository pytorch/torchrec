/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#define FBGEMM_EXPORTS
#include "./RefImplementations.h"

#include "fbgemm/FbgemmBuild.h"
#include "fbgemm/FbgemmConvert.h"
#include "fbgemm/Types.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <thread>

using namespace std;

namespace fbgemm {

// Thread-safe random number generator
//
// Return a random 32bit integer using xoshiro128++
// http://prng.di.unimi.it/xoshiro128plusplus.c
inline uint32_t rnd128_next(int idx, int vlen) {
  constexpr int VLEN_MAX = 16; // max vector size
  alignas(64) static thread_local uint32_t g_rnd128_buffer[4 * VLEN_MAX];
  static thread_local bool g_rnd128_initialized = false;

  // Splitmix64: http://prng.di.unimi.it/splitmix64.c
  auto rnd128_init_next = [](uint64_t& x) {
    uint64_t z = (x += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
  };

  auto rotl = [](const uint32_t x, int k) {
    return (x << k) | (x >> (32 - k));
  };

  if (!g_rnd128_initialized) {
    // Initialize rand buffer with uniq values per thread
    uint64_t h0 = std::hash<std::thread::id>{}(std::this_thread::get_id());
    for (auto i = 0; i < 4; ++i) {
      // Use thread hash as seed
      g_rnd128_buffer[i * VLEN_MAX] = rnd128_init_next(h0);
      uint64_t h1 = g_rnd128_buffer[i * VLEN_MAX];
      for (auto v = 1; v < VLEN_MAX; ++v) {
        g_rnd128_buffer[i * VLEN_MAX + v] = rnd128_init_next(h1);
      }
    }
    g_rnd128_initialized = true;
  }

  const uint32_t result =
      rotl(g_rnd128_buffer[idx] + g_rnd128_buffer[3 * vlen + idx], 7) +
      g_rnd128_buffer[idx];

  const uint32_t t = g_rnd128_buffer[1 * vlen + idx] << 9;

  g_rnd128_buffer[2 * vlen + idx] ^= g_rnd128_buffer[0 * vlen + idx];
  g_rnd128_buffer[3 * vlen + idx] ^= g_rnd128_buffer[1 * vlen + idx];
  g_rnd128_buffer[1 * vlen + idx] ^= g_rnd128_buffer[2 * vlen + idx];
  g_rnd128_buffer[0 * vlen + idx] ^= g_rnd128_buffer[3 * vlen + idx];

  g_rnd128_buffer[2 * vlen + idx] ^= t;

  g_rnd128_buffer[3 * vlen + idx] = rotl(g_rnd128_buffer[3 * vlen + idx], 11);

  return result;
}

void FloatToFloat16_ref(
    const float* src,
    float16* dst,
    size_t size,
    bool do_clip) {
  constexpr float FP16_MAX = 65504.f;
  if (do_clip) {
    for (size_t i = 0; i < size; i++) {
      float cur_src = std::max(-FP16_MAX, std::min(src[i], FP16_MAX));
      dst[i] = cpu_float2half_rn(cur_src);
    }
  } else {
    for (size_t i = 0; i < size; i++) {
      dst[i] = cpu_float2half_rn(src[i]);
    }
  }
}

void Float16ToFloat_ref(const float16* src, float* dst, size_t size) {
  for (size_t i = 0; i < size; i++) {
    dst[i] = cpu_half2float(src[i]);
  }
}

void FloatToBfloat16_ref(const float* src, bfloat16* dst, size_t size) {
  for (size_t i = 0; i < size; i++) {
    // Add 2^15 and right shift 16 to do round-nearest
    dst[i] = (*reinterpret_cast<const uint32_t*>(src + i) + (1 << 15)) >> 16;
  }
}

void Bfloat16ToFloat_ref(const bfloat16* src, float* dst, size_t size) {
  for (size_t i = 0; i < size; i++) {
    uint32_t val_fp32 =
        static_cast<uint32_t>(reinterpret_cast<const uint16_t*>(src)[i]) << 16;
    reinterpret_cast<uint32_t*>(dst)[i] = val_fp32;
  }
}

void requantize_u8acc32_ref(
    int M,
    int N,
    int ld,
    const int32_t* inp,
    uint8_t* out,
    int32_t C_multiplier,
    int32_t C_right_shift,
    int32_t C_zero_point,
    int32_t A_zero_point,
    int32_t B_zero_point,
    const int32_t* row_offsets,
    const int32_t* col_offsets,
    const int32_t* bias,
    bool fuse_relu) {
  int64_t nudge = 1ll << std::max(0, C_right_shift - 1);
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      int32_t raw = inp[i * ld + j];
      if (A_zero_point) {
        raw -= A_zero_point * col_offsets[j];
      }
      if (B_zero_point) {
        raw -= B_zero_point * row_offsets[i];
      }
      if (bias) {
        raw += bias[j];
      }

      int64_t ab_64 =
          static_cast<int64_t>(raw) * static_cast<int64_t>(C_multiplier);
      int64_t rounded = ((ab_64 + nudge) >> C_right_shift) + C_zero_point;

      out[i * ld + j] = std::max(
          fuse_relu ? static_cast<int64_t>(C_zero_point) : 0l,
          std::min(static_cast<int64_t>(255l), rounded));
    }
  }
}

void requantize_u8acc32_ref(
    int M,
    int N,
    int ld,
    const int32_t* inp,
    uint8_t* out,
    const float* C_multiplier,
    int32_t C_zero_point,
    int32_t A_zero_point,
    const int32_t* B_zero_point,
    const int32_t* row_offsets,
    const int32_t* col_offsets,
    const int32_t* bias,
    int ncols_per_quant_group,
    bool fuse_relu) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      int32_t raw = inp[i * ld + j];
      if (A_zero_point) {
        raw -= A_zero_point * col_offsets[j];
      }
      raw -= B_zero_point[j / ncols_per_quant_group] * row_offsets[i];
      if (bias) {
        raw += bias[j];
      }

      float result = raw * C_multiplier[j / ncols_per_quant_group];
      long rounded = lrintf(result) + C_zero_point;
      out[i * ld + j] = std::max(
          fuse_relu ? static_cast<long>(C_zero_point) : 0l,
          std::min(255l, rounded));
    }
  }
}

void matmul_u8i8acc32_ref(
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc,
    const uint8_t* Aint8,
    const int8_t* Bint8,
    int32_t* Cint32) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      int32_t sum = 0;
      for (int k = 0; k < K; ++k) {
        sum += static_cast<int32_t>(Aint8[i * lda + k]) *
            static_cast<int32_t>(Bint8[k * ldb + j]);
      }
      Cint32[i * ldc + j] = sum;
    }
  }
}

void matmul_u8i8acc16_ref(
    int M,
    int N,
    int K,
    int lda,
    int ldb,
    int ldc,
    int brow,
    const uint8_t* Aint8,
    const int8_t* Bint8,
    int32_t* Cint32) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      int32_t sum = 0, sum_32bit = 0;
      for (int k = 0; k < K; k += 2) {
        int a0 = Aint8[i * lda + k];
        int b0 = Bint8[k * ldb + j];
        int a1 = 0, b1 = 0;
        if (k + 1 < K) {
          a1 = Aint8[i * lda + k + 1];
          b1 = Bint8[(k + 1) * ldb + j];
        }
        sum = clip_16bit(sum + clip_16bit(a0 * b0 + a1 * b1));
        if ((k % brow) == (brow - 2)) {
          sum_32bit += sum;
          sum = 0;
        }
      }
      Cint32[i * ldc + j] = sum_32bit + sum;
    }
  }
}

void cblas_sgemm_ref(
    const matrix_op_t transa,
    const matrix_op_t transb,
    const int m,
    const int n,
    const int k,
    float alpha,
    const float* Afp32,
    int lda,
    const float* Bfp32,
    int ldb,
    float beta,
    float* Cfp32,
    int ldc) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      float sum = 0;
      for (int p = 0; p < k; ++p) {
        float a =
            (transa == matrix_op_t::NoTranspose ? Afp32[i * lda + p]
                                                : Afp32[p * lda + i]);
        float b =
            (transb == matrix_op_t::NoTranspose ? Bfp32[p * ldb + j]
                                                : Bfp32[j * ldb + p]);
        sum += a * b;
      }
      if (beta == 0) {
        Cfp32[i * ldc + j] = alpha * sum;
      } else {
        Cfp32[i * ldc + j] = alpha * sum + beta * Cfp32[i * ldc + j];
      }
    }
  }
}

namespace {
// From https://stackoverflow.com/questions/31652875
uint64_t umul64wide(uint64_t a, uint64_t b) {
  uint64_t a_lo = static_cast<uint32_t>(a);
  uint64_t a_hi = a >> 32;
  uint64_t b_lo = static_cast<uint32_t>(b);
  uint64_t b_hi = b >> 32;

  uint64_t p0 = a_lo * b_lo;
  uint64_t p1 = a_lo * b_hi;
  uint64_t p2 = a_hi * b_lo;

  return p0 + (p1 << 32) + (p2 << 32);
}
} // namespace

// Expected to have overflows
NO_SANITIZE("undefined")
void cblas_gemm_i64_i64acc_ref(
    matrix_op_t transa,
    matrix_op_t transb,
    int M,
    int N,
    int K,
    const int64_t* A,
    int lda,
    const int64_t* B,
    int ldb,
    bool accumulate,
    int64_t* C,
    int ldc) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      int64_t acc;
      if (accumulate) {
        acc = C[i * ldc + j];
      } else {
        acc = 0;
      }
      for (int k = 0; k < K; ++k) {
        int64_t a =
            A[transa == matrix_op_t::Transpose ? i + k * lda : i * lda + k];
        int64_t b =
            B[transb == matrix_op_t::Transpose ? k + j * ldb : k * ldb + j];
        int64_t lo = umul64wide(a, b);
        acc += lo;
      }
      C[i * ldc + j] = acc;
    } // j
  } // i
}

void row_offsets_u8acc32_ref(
    int M,
    int K,
    int ld,
    const uint8_t* Aint8,
    int32_t* row_offsets) {
  // row offset
  for (int i = 0; i < M; ++i) {
    int32_t sum = 0;
    for (int k = 0; k < K; ++k) {
      sum += static_cast<int32_t>(Aint8[i * ld + k]);
    }
    row_offsets[i] = sum;
  }
}

void col_offsets_with_zero_pt_s8acc32_ref(
    int K,
    int N,
    int ld,
    const int8_t* Bint8,
    const int32_t* B_zero_point,
    int32_t* col_offsets,
    int ncols_per_quant_group) {
  for (int j = 0; j < N; ++j) {
    int32_t sum = 0;
    for (int k = 0; k < K; ++k) {
      sum += Bint8[k * ld + j];
    }
    col_offsets[j] = sum - B_zero_point[j / ncols_per_quant_group] * K;
  }
}

void spmdm_ref(
    int M,
    const uint8_t* A,
    int lda,
    fbgemm::CompressedSparseColumn& B,
    bool accumulation,
    int32_t* C,
    int ldc,
    int groups /*=1*/) {
  int N = B.NumOfCols();
  assert(N % groups == 0);
  if (!accumulation) {
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        C[i * ldc + j] = 0;
      }
    }
  }
  for (int g = 0; g < groups; ++g) {
    for (int j = g * (N / groups); j < (g + 1) * (N / groups); ++j) {
      for (int k = B.ColPtr()[j]; k < B.ColPtr()[j + 1]; ++k) {
        int row = g * B.NumOfRows() + B.RowIdx()[k];
        int w = B.Values()[k];
        for (int i = 0; i < M; ++i) {
          C[i * ldc + j] += A[i * lda + row] * w;
        }
      }
    } // for each column of B
  } // for each group
}

int32_t clip_16bit(int32_t x) {
  if (x > numeric_limits<int16_t>::max()) {
    return std::min<int>(numeric_limits<int16_t>::max(), x);
  } else if (x < numeric_limits<int16_t>::min()) {
    return std::max<int>(numeric_limits<int16_t>::min(), x);
  } else {
    return x;
  }
}

/* Imitate the Im2Col<float, CPUContext, StorageOrder::NWC> function
 * from caffe2/utils/math_cpu.cc
 * NWC StorageOrder/Layout
 * A:  NWC: NW_0 x C_0
 * Ao: NWC: NW_1 x G KW C_0/G
 */
template <>
FBGEMM_API void im2col_ref(
    const conv_param_t<1>& conv_p,
    const uint8_t* A,
    int32_t A_zero_point,
    uint8_t* Ao) {
  int IC = conv_p.IC;
  int G = conv_p.G;
  assert(IC % G == 0);
  array<int, 1> IN_DIM = conv_p.IN_DIM;
  array<int, 1> OUT_DIM = conv_p.OUT_DIM;
  array<int, 1> K = conv_p.K;

  if (conv_p.transposed) {
    for (int n = 0; n < conv_p.MB; ++n) {
      for (int ow = 0; ow < OUT_DIM[0]; ++ow) {
        for (int s = 0; s < K[0]; ++s) {
          int w = ow + conv_p.pad[0] - s * conv_p.dilation[0];
          int w_in = w / conv_p.stride[0];
          if (w_in * conv_p.stride[0] == w && w_in >= 0 && w_in < IN_DIM[0]) {
            for (int g = 0; g < G; ++g) {
              memcpy(
                  Ao + (((n * OUT_DIM[0] + ow) * G + g) * K[0] + s) * (IC / G),
                  A + (n * IN_DIM[0] + w_in) * IC + g * (IC / G),
                  sizeof(uint8_t) * (IC / G));
            }
          } else {
            for (int g = 0; g < G; ++g) {
              memset(
                  Ao + (((n * OUT_DIM[0] + ow) * G + g) * K[0] + s) * (IC / G),
                  A_zero_point,
                  sizeof(uint8_t) * (IC / G));
            }
          }
        } // for each s
      } // for each ow
    } // for each n
  } else {
    for (int n = 0; n < conv_p.MB; ++n) {
      for (int w = 0; w < OUT_DIM[0]; ++w) {
        for (int s = 0; s < K[0]; ++s) {
          int w_in =
              -conv_p.pad[0] + w * conv_p.stride[0] + s * conv_p.dilation[0];
          if (w_in < 0 || w_in >= IN_DIM[0]) {
            for (int g = 0; g < G; ++g) {
              memset(
                  Ao + (((n * OUT_DIM[0] + w) * G + g) * K[0] + s) * (IC / G),
                  A_zero_point,
                  sizeof(uint8_t) * (IC / G));
            }
          } else {
            for (int g = 0; g < G; ++g) {
              memcpy(
                  Ao + (((n * OUT_DIM[0] + w) * G + g) * K[0] + s) * (IC / G),
                  A + (n * IN_DIM[0] + w_in) * IC + g * (IC / G),
                  sizeof(uint8_t) * (IC / G));
            }
          }
        } // for each s
      } // for each w
    } // for each n
  }
}

/* Imitate the Im2Col<float, CPUContext, StorageOrder::NHWC> function
 * from caffe2/utils/math_cpu.cc
 * NHWC StorageOrder/Layout
 * A:  NHWC: NH_0W_0 x C_0
 * Ao: NHWC: NH_1W_1 x G RS C_0/G
 */
template <>
FBGEMM_API void im2col_ref(
    const conv_param_t<2>& conv_p,
    const uint8_t* A,
    int32_t A_zero_point,
    uint8_t* Ao) {
  int IC = conv_p.IC;
  int G = conv_p.G;
  assert(IC % G == 0);
  array<int, 2> IN_DIM = conv_p.IN_DIM;
  array<int, 2> OUT_DIM = conv_p.OUT_DIM;
  array<int, 2> K = conv_p.K;

  if (conv_p.transposed) {
    for (int n = 0; n < conv_p.MB; ++n) {
      for (int oh = 0; oh < OUT_DIM[0]; ++oh) {
        for (int ow = 0; ow < OUT_DIM[1]; ++ow) {
          for (int r = 0; r < K[0]; ++r) {
            for (int s = 0; s < K[1]; ++s) {
              int h = oh + conv_p.pad[0] - r * conv_p.dilation[0];
              int w = ow + conv_p.pad[1] - s * conv_p.dilation[1];
              int h_in = h / conv_p.stride[0];
              int w_in = w / conv_p.stride[1];
              if (h_in * conv_p.stride[0] == h && h_in >= 0 &&
                  h_in < IN_DIM[0] && w_in * conv_p.stride[1] == w &&
                  w_in >= 0 && w_in < IN_DIM[1]) {
                for (int g = 0; g < G; ++g) {
                  memcpy(
                      Ao +
                          (((((n * OUT_DIM[0] + oh) * OUT_DIM[1] + ow) * G +
                             g) *
                                K[0] +
                            r) *
                               K[1] +
                           s) *
                              (IC / G),
                      A + ((n * IN_DIM[0] + h_in) * IN_DIM[1] + w_in) * IC +
                          g * (IC / G),
                      sizeof(uint8_t) * (IC / G));
                }
              } else {
                for (int g = 0; g < G; ++g) {
                  memset(
                      Ao +
                          (((((n * OUT_DIM[0] + oh) * OUT_DIM[1] + ow) * G +
                             g) *
                                K[0] +
                            r) *
                               K[1] +
                           s) *
                              (IC / G),
                      A_zero_point,
                      sizeof(uint8_t) * (IC / G));
                }
              }
            } // for each s
          } // for each r
        } // for each ow
      } // for each oh
    } // for each n
  } else {
    for (int n = 0; n < conv_p.MB; ++n) {
      for (int h = 0; h < OUT_DIM[0]; ++h) {
        for (int w = 0; w < OUT_DIM[1]; ++w) {
          for (int r = 0; r < K[0]; ++r) {
            int h_in =
                -conv_p.pad[0] + h * conv_p.stride[0] + r * conv_p.dilation[0];
            for (int s = 0; s < K[1]; ++s) {
              int w_in = -conv_p.pad[1] + w * conv_p.stride[1] +
                  s * conv_p.dilation[1];
              if (h_in < 0 || h_in >= IN_DIM[0] || w_in < 0 ||
                  w_in >= IN_DIM[1]) {
                for (int g = 0; g < G; ++g) {
                  memset(
                      Ao +
                          (((((n * OUT_DIM[0] + h) * OUT_DIM[1] + w) * G + g) *
                                K[0] +
                            r) *
                               K[1] +
                           s) *
                              (IC / G),
                      A_zero_point,
                      sizeof(uint8_t) * (IC / G));
                }
              } else {
                for (int g = 0; g < G; ++g) {
                  memcpy(
                      Ao +
                          (((((n * OUT_DIM[0] + h) * OUT_DIM[1] + w) * G + g) *
                                K[0] +
                            r) *
                               K[1] +
                           s) *
                              (IC / G),
                      A + ((n * IN_DIM[0] + h_in) * IN_DIM[1] + w_in) * IC +
                          g * (IC / G),
                      sizeof(uint8_t) * (IC / G));
                }
              }
            } // for each s
          } // for each r
        } // for each w
      } // for each h
    } // for each n
  }
}

/* Imitate the Im2Col<float, CPUContext, StorageOrder::NHWC> function
 * from caffe2/utils/math_cpu.cc
 * NHWC StorageOrder/Layout
 * A:  NHWC: NT_0H_0W_0 x C_0
 * Ao: NHWC: NT_1H_1W_1 x G QRS C_0/G
 */
template <>
FBGEMM_API void im2col_ref(
    const conv_param_t<3>& conv_p,
    const uint8_t* A,
    int32_t A_zero_point,
    uint8_t* Ao) {
  int IC = conv_p.IC;
  int G = conv_p.G;
  assert(IC % G == 0);
  array<int, 3> IN_DIM = conv_p.IN_DIM;
  array<int, 3> OUT_DIM = conv_p.OUT_DIM;
  array<int, 3> K = conv_p.K;

  if (conv_p.transposed) {
    for (int n = 0; n < conv_p.MB; ++n) {
      for (int ot = 0; ot < OUT_DIM[0]; ++ot) {
        for (int oh = 0; oh < OUT_DIM[1]; ++oh) {
          for (int ow = 0; ow < OUT_DIM[2]; ++ow) {
            for (int q = 0; q < K[0]; ++q) {
              for (int r = 0; r < K[1]; ++r) {
                for (int s = 0; s < K[2]; ++s) {
                  int t = ot + conv_p.pad[0] - q * conv_p.dilation[0];
                  int h = oh + conv_p.pad[1] - r * conv_p.dilation[1];
                  int w = ow + conv_p.pad[2] - s * conv_p.dilation[2];
                  int t_in = t / conv_p.stride[0];
                  int h_in = h / conv_p.stride[1];
                  int w_in = w / conv_p.stride[2];
                  if (t_in * conv_p.stride[0] == t && t_in >= 0 &&
                      t_in < IN_DIM[0] && h_in * conv_p.stride[1] == h &&
                      h_in >= 0 && h_in < IN_DIM[1] &&
                      w_in * conv_p.stride[2] == w && w_in >= 0 &&
                      w_in < IN_DIM[2]) {
                    for (int g = 0; g < G; ++g) {
                      memcpy(
                          Ao +
                              (((((((n * OUT_DIM[0] + ot) * OUT_DIM[1] + oh) *
                                       OUT_DIM[2] +
                                   ow) *
                                      G +
                                  g) *
                                     K[0] +
                                 q) *
                                    K[1] +
                                r) *
                                   K[2] +
                               s) *
                                  (IC / G),
                          A +
                              (((n * IN_DIM[0] + t_in) * IN_DIM[1] + h_in) *
                                   IN_DIM[2] +
                               w_in) *
                                  IC +
                              g * (IC / G),
                          sizeof(uint8_t) * (IC / G));
                    }
                  } else {
                    for (int g = 0; g < G; ++g) {
                      memset(
                          Ao +
                              (((((((n * OUT_DIM[0] + ot) * OUT_DIM[1] + oh) *
                                       OUT_DIM[2] +
                                   ow) *
                                      G +
                                  g) *
                                     K[0] +
                                 q) *
                                    K[1] +
                                r) *
                                   K[2] +
                               s) *
                                  (IC / G),
                          A_zero_point,
                          sizeof(uint8_t) * (IC / G));
                    }
                  }
                } // for each s
              } // for each r
            } // for each q
          } // for each ow
        } // for each oh
      } // for each ot
    } // for each n
  } else {
    for (int n = 0; n < conv_p.MB; ++n) {
      for (int t = 0; t < OUT_DIM[0]; ++t) {
        for (int h = 0; h < OUT_DIM[1]; ++h) {
          for (int w = 0; w < OUT_DIM[2]; ++w) {
            for (int q = 0; q < K[0]; ++q) {
              int t_in = -conv_p.pad[0] + t * conv_p.stride[0] +
                  q * conv_p.dilation[0];
              for (int r = 0; r < K[1]; ++r) {
                int h_in = -conv_p.pad[1] + h * conv_p.stride[1] +
                    r * conv_p.dilation[1];
                for (int s = 0; s < K[2]; ++s) {
                  int w_in = -conv_p.pad[2] + w * conv_p.stride[2] +
                      s * conv_p.dilation[2];
                  if (t_in < 0 || t_in >= IN_DIM[0] || h_in < 0 ||
                      h_in >= IN_DIM[1] || w_in < 0 || w_in >= IN_DIM[2]) {
                    for (int g = 0; g < G; ++g) {
                      memset(
                          Ao +
                              (((((((n * OUT_DIM[0] + t) * OUT_DIM[1] + h) *
                                       OUT_DIM[2] +
                                   w) *
                                      G +
                                  g) *
                                     K[0] +
                                 q) *
                                    K[1] +
                                r) *
                                   K[2] +
                               s) *
                                  (IC / G),
                          A_zero_point,
                          sizeof(uint8_t) * (IC / G));
                    }
                  } else {
                    for (int g = 0; g < G; ++g) {
                      memcpy(
                          Ao +
                              (((((((n * OUT_DIM[0] + t) * OUT_DIM[1] + h) *
                                       OUT_DIM[2] +
                                   w) *
                                      G +
                                  g) *
                                     K[0] +
                                 q) *
                                    K[1] +
                                r) *
                                   K[2] +
                               s) *
                                  (IC / G),
                          A +
                              (((n * IN_DIM[0] + t_in) * IN_DIM[1] + h_in) *
                                   IN_DIM[2] +
                               w_in) *
                                  IC +
                              g * (IC / G),
                          sizeof(uint8_t) * (IC / G));
                    }
                  }
                } // for each s
              } // for each r
            } // for each q
          } // for each w
        } // for each h
      } // for each t
    } // for each n
  }
}

// 1D Conv
template <>
FBGEMM_API void conv_ref(
    const conv_param_t<1>& conv_p,
    const uint8_t* A,
    int32_t A_zero_point,
    const int8_t* B,
    int32_t* C) {
  // A is assumed to be (N Lin Cin)
  // B is assumed to be (G K Cin/G Cout/G)
  // C is assumed to be (N Lout Cout)
  int IC = conv_p.IC;
  int OC = conv_p.OC;
  int G = conv_p.G;
  assert(IC % G == 0);
  assert(OC % G == 0);
  array<int, 1> IN_DIM = conv_p.IN_DIM;
  array<int, 1> OUT_DIM = conv_p.OUT_DIM;
  array<int, 1> K = conv_p.K;

  if (conv_p.transposed) {
    // for ref implementation, there is no padding on the input buffer,
    // padding specifies how much we remove from the output buffers
    for (int n = 0; n < conv_p.MB; ++n) {
      for (int ow = 0; ow < OUT_DIM[0]; ++ow) {
        // stride on output is fractional stride on input
        // conv index is
        // int w_in = -conv_p.pad[0] + w* conv_p.stride[0] + r*
        // conv_p.dilation[0];
        // so we reverse it
        for (int g = 0; g < G; ++g) {
          for (int oc = 0; oc < OC / G; ++oc) {
            int sum = 0;
            for (int r = 0; r < K[0]; ++r) {
              int w = ow + conv_p.pad[0] - r * conv_p.dilation[0];
              int w_in = w / conv_p.stride[0];
              for (int ic = 0; ic < IC / G; ++ic) {
                int a = (w_in * conv_p.stride[0] == w && w_in >= 0 &&
                         w_in < IN_DIM[0])
                    ? A[(n * IN_DIM[0] + w_in) * IC + g * (IC / G) + ic]
                    : A_zero_point;
                int b =
                    B[((g * K[0] + r) * IC / G + ic) * (OC / G) +
                      oc]; // G K IC/G OC/G after  transpose
                sum += a * b;
              } // for each ic
            } // for each r
            C[(n * OUT_DIM[0] + ow) * OC + g * (OC / G) + oc] = sum;
          } // for each oc
        } // for each g
      } // for each w
    } // for each n
  } else {
    for (int n = 0; n < conv_p.MB; ++n) {
      for (int w = 0; w < OUT_DIM[0]; ++w) {
        for (int g = 0; g < G; ++g) {
          for (int m = 0; m < OC / G; ++m) {
            int sum = 0;
            for (int r = 0; r < K[0]; ++r) {
              int w_in = -conv_p.pad[0] + w * conv_p.stride[0] +
                  r * conv_p.dilation[0];
              for (int c = 0; c < IC / G; ++c) {
                int a = w_in < 0 || w_in >= IN_DIM[0]
                    ? A_zero_point
                    : A[(n * IN_DIM[0] + w_in) * IC + g * (IC / G) + c];
                int b =
                    B[((g * K[0] + r) * (IC / G) + c) * (OC / G) +
                      m]; // G K IC/G OC/G  after  transpose
                sum += a * b;
              } // for each c
            } // for each r
            C[(n * OUT_DIM[0] + w) * OC + g * (OC / G) + m] = sum;
          } // for each w
        } // for each m
      } // for each group
    } // for each n
  }
}

// 2D Conv
template <>
FBGEMM_API void conv_ref(
    const conv_param_t<2>& conv_p,
    const uint8_t* A,
    int32_t A_zero_point,
    const int8_t* B,
    int32_t* C) {
  // filters are assumed to be in G RS C/G x K format
  int IC = conv_p.IC;
  int OC = conv_p.OC;
  int G = conv_p.G;
  assert(IC % G == 0);
  assert(OC % G == 0);
  array<int, 2> IN_DIM = conv_p.IN_DIM;
  array<int, 2> OUT_DIM = conv_p.OUT_DIM;
  array<int, 2> K = conv_p.K;

  if (conv_p.transposed) {
    // for ref implementation, there is no padding on the input buffer,
    // padding specifies how much we remove from the output buffers
    for (int n = 0; n < conv_p.MB; ++n) {
      for (int oh = 0; oh < OUT_DIM[0]; ++oh) {
        for (int ow = 0; ow < OUT_DIM[1]; ++ow) {
          // stride on output is fractional stride on input
          // conv index is
          // int h_in =
          //     -conv_p.pad[0] + h * conv_p.stride[0] + r * conv_p.dilation[0];
          // int w_in =
          //     -conv_p.pad[1] + w * conv_p.stride[1] + s * conv_p.dilation[1];
          // so we reverse it
          for (int g = 0; g < G; ++g) {
            for (int oc = 0; oc < OC / G; ++oc) {
              int sum = 0;
              for (int r = 0; r < K[0]; ++r) {
                for (int s = 0; s < K[1]; ++s) {
                  int h = oh + conv_p.pad[0] - r * conv_p.dilation[0];
                  int w = ow + conv_p.pad[1] - s * conv_p.dilation[1];
                  int h_in = h / conv_p.stride[0];
                  int w_in = w / conv_p.stride[1];
                  for (int ic = 0; ic < IC / G; ++ic) {
                    int a = (h_in * conv_p.stride[0] == h && h_in >= 0 &&
                             h_in < IN_DIM[0] && w_in * conv_p.stride[1] == w &&
                             w_in >= 0 && w_in < IN_DIM[1])
                        ? A[((n * IN_DIM[0] + h_in) * IN_DIM[1] + w_in) * IC +
                            g * (IC / G) + ic]
                        : A_zero_point;
                    int b =
                        B[((((g * K[0] + r) * K[1] + s) * (IC / G) + ic) * OC /
                           G) +
                          oc]; // G R S IC OC after  transpose
                    sum += a * b;
                  } // for each ic
                } // for each s
              } // for each r
              C[((n * OUT_DIM[0] + oh) * OUT_DIM[1] + ow) * OC + g * (OC / G) +
                oc] = sum;
            } // for each oc
          } // for each g
        } // for each w
      } // for each h
    } // for each n
  } else {
    for (int n = 0; n < conv_p.MB; ++n) {
      for (int h = 0; h < OUT_DIM[0]; ++h) {
        for (int w = 0; w < OUT_DIM[1]; ++w) {
          for (int g = 0; g < G; ++g) {
            for (int m = 0; m < OC / G; ++m) {
              int sum = 0;
              for (int r = 0; r < K[0]; ++r) {
                int h_in = -conv_p.pad[0] + h * conv_p.stride[0] +
                    r * conv_p.dilation[0];
                for (int s = 0; s < K[1]; ++s) {
                  int w_in = -conv_p.pad[1] + w * conv_p.stride[1] +
                      s * conv_p.dilation[1];
                  for (int c = 0; c < IC / G; ++c) {
                    int a = h_in < 0 || h_in >= IN_DIM[0] || w_in < 0 ||
                            w_in >= IN_DIM[1]
                        ? A_zero_point
                        : A[((n * IN_DIM[0] + h_in) * IN_DIM[1] + w_in) * IC +
                            g * (IC / G) + c];
                    int b =
                        B[(((g * K[0] + r) * K[1] + s) * (IC / G) + c) *
                              (OC / G) +
                          m];
                    sum += a * b;
                  } // for each c
                } // for each s
              } // for each r
              C[((n * OUT_DIM[0] + h) * OUT_DIM[1] + w) * OC + g * (OC / G) +
                m] = sum;
            } // for each m
          } // for each group
        } // for each w
      } // for each h
    } // for each n
  }
}

// 3D Conv
template <>
FBGEMM_API void conv_ref(
    const conv_param_t<3>& conv_p,
    const uint8_t* A,
    int32_t A_zero_point,
    const int8_t* B,
    int32_t* C) {
  // filters are assumed to be in G QRS C/G x K format
  int IC = conv_p.IC;
  int OC = conv_p.OC;
  int G = conv_p.G;
  assert(IC % G == 0);
  assert(OC % G == 0);
  array<int, 3> IN_DIM = conv_p.IN_DIM;
  array<int, 3> OUT_DIM = conv_p.OUT_DIM;
  array<int, 3> K = conv_p.K;

  if (conv_p.transposed) {
    // for ref implementation, there is no padding on the input buffer,
    // padding specifies how much we remove from the output buffers
    for (int n = 0; n < conv_p.MB; ++n) {
      for (int ot = 0; ot < OUT_DIM[0]; ++ot) {
        for (int oh = 0; oh < OUT_DIM[1]; ++oh) {
          for (int ow = 0; ow < OUT_DIM[2]; ++ow) {
            // stride on output is fractional stride on input
            // conv index is
            // int t_in =
            //     -conv_p.pad[0] + t * conv_p.stride[0] + q *
            //     conv_p.dilation[0];
            // int h_in =
            //     -conv_p.pad[1] + h * conv_p.stride[1] + r *
            //     conv_p.dilation[1];
            // int w_in =
            //     -conv_p.pad[2] + w * conv_p.stride[2] + s *
            //     conv_p.dilation[2];
            // so we reverse it
            for (int g = 0; g < G; ++g) {
              for (int oc = 0; oc < OC / G; ++oc) {
                int sum = 0;
                for (int q = 0; q < K[0]; ++q) {
                  for (int r = 0; r < K[1]; ++r) {
                    for (int s = 0; s < K[2]; ++s) {
                      int t = ot + conv_p.pad[0] - q * conv_p.dilation[0];
                      int h = oh + conv_p.pad[1] - r * conv_p.dilation[1];
                      int w = ow + conv_p.pad[2] - s * conv_p.dilation[2];
                      int t_in = t / conv_p.stride[0];
                      int h_in = h / conv_p.stride[1];
                      int w_in = w / conv_p.stride[2];
                      for (int ic = 0; ic < IC / G; ++ic) {
                        int a =
                            (t_in * conv_p.stride[0] == t && t_in >= 0 &&
                             t_in < IN_DIM[0] && h_in * conv_p.stride[1] == h &&
                             h_in >= 0 && h_in < IN_DIM[1] &&
                             w_in * conv_p.stride[2] == w && w_in >= 0 &&
                             w_in < IN_DIM[2])
                            ? A[((((n * IN_DIM[0] + t_in) * IN_DIM[1] + h_in) *
                                  IN_DIM[2]) +
                                 w_in) *
                                    IC +
                                g * (IC / G) + ic]
                            : A_zero_point;
                        int b =
                            B[((((((g * K[0] + q)) * K[1] + r) * K[2] + s) *
                                    (IC / G) +
                                ic) *
                               (OC / G)) +
                              oc]; // G Q R S Cin/G Cout/G after transpose
                        sum += a * b;
                      } // for each ic
                    } // for each s
                  } // for each r
                } // for each q
                C[(((n * OUT_DIM[0] + ot) * OUT_DIM[1] + oh) * OUT_DIM[2] +
                   ow) *
                      OC +
                  g * (OC / G) + oc] = sum;
              } // for each oc
            } // for each g
          } // for each ow
        } // for each oh
      } // for each ot
    } // for each n
  } else {
    for (int n = 0; n < conv_p.MB; ++n) {
      for (int t = 0; t < OUT_DIM[0]; ++t) {
        for (int h = 0; h < OUT_DIM[1]; ++h) {
          for (int w = 0; w < OUT_DIM[2]; ++w) {
            for (int g = 0; g < G; ++g) {
              for (int m = 0; m < OC / G; ++m) {
                int sum = 0;
                for (int q = 0; q < K[0]; ++q) {
                  int t_in = -conv_p.pad[0] + t * conv_p.stride[0] +
                      q * conv_p.dilation[0];
                  for (int r = 0; r < K[1]; ++r) {
                    int h_in = -conv_p.pad[1] + h * conv_p.stride[1] +
                        r * conv_p.dilation[1];
                    for (int s = 0; s < K[2]; ++s) {
                      int w_in = -conv_p.pad[2] + w * conv_p.stride[2] +
                          s * conv_p.dilation[2];
                      for (int c = 0; c < IC / G; ++c) {
                        int a = t_in < 0 || t_in >= IN_DIM[0] || h_in < 0 ||
                                h_in >= IN_DIM[1] || w_in < 0 ||
                                w_in >= IN_DIM[2]
                            ? A_zero_point
                            : A[(((n * IN_DIM[0] + t_in) * IN_DIM[1] + h_in) *
                                     IN_DIM[2] +
                                 w_in) *
                                    IC +
                                g * (IC / G) + c];
                        int b =
                            B[((((g * K[0] + q) * K[1] + r) * K[2] + s) *
                                   (IC / G) +
                               c) *
                                  (OC / G) +
                              m];
                        sum += a * b;
                      } // for each c
                    } // for each s
                  } // for each r
                } // for each q
                C[(((n * OUT_DIM[0] + t) * OUT_DIM[1] + h) * OUT_DIM[2] + w) *
                      OC +
                  g * (OC / G) + m] = sum;
              } // for each m
            } // for each group
          } // for each w
        } // for each h
      } // for each t
    } // for each n
  }
}

template <int SPATIAL_DIM>
void transposeConvWeights(
    const conv_param_t<SPATIAL_DIM>& conv_p,
    const std::int8_t* src,
    std::int8_t* dest) {
  int G = conv_p.G;
  int IC_per_G = conv_p.IC / conv_p.G;
  int OC_per_G = conv_p.OC / conv_p.G;

  int filter_prod = std::accumulate(
      conv_p.K.begin(),
      conv_p.K.begin() + SPATIAL_DIM,
      1,
      std::multiplies<int>());
  // Transforms weights from  G K/G (T R S C/G) to G (T R S C/G) K/G format.
  for (int g = 0; g < G; ++g) {
    for (int k = 0; k < OC_per_G; ++k) {
      for (int f = 0; f < filter_prod; ++f) {
        for (int c = 0; c < IC_per_G; ++c) {
          dest[((g * filter_prod + f) * IC_per_G + c) * OC_per_G + k] =
              src[((g * OC_per_G + k) * filter_prod + f) * IC_per_G + c];
        }
      }
    }
  }
}

template <typename InType, typename IndexType, typename OffsetType>
bool EmbeddingSpMDM_ref(
    const int64_t block_size,
    const int64_t output_size,
    const int64_t index_size,
    const int64_t data_size,
    const InType* input,
    const IndexType* indices,
    const OffsetType* offsets_or_lengths,
    const float* weights, // optional, can be null for non-weighted sum
    bool normalize_by_lengths,
    float* out,
    bool is_weight_positional,
    bool use_offsets,
    int64_t output_stride /*=-1*/,
    int64_t input_stride /*=-1*/) {
  bool is8bit = is_same<InType, uint8_t>::value;
  if (output_stride == -1) {
    output_stride = block_size;
  }

  if (is8bit) {
    // block_size is the number of elements and fused_block_size is the size of
    // an entire row, including scale and bias.
    if (input_stride == -1) {
      const auto scale_bias_offset = 2 * sizeof(float);
      input_stride = block_size + scale_bias_offset;
    }
    int64_t current = 0;
    for (int m = 0; m < output_size; ++m) {
      memset(out, 0, sizeof(float) * block_size);
      int len = use_offsets ? offsets_or_lengths[m + 1] - offsets_or_lengths[m]
                            : offsets_or_lengths[m];
      if (current + len > index_size) {
        return false;
      }
      for (int i = 0; i < len; ++i) {
        int64_t idx = indices[current];
        if (idx < 0 || idx >= data_size) {
          return false;
        }

        const float* scale_bias = reinterpret_cast<const float*>(
            input + input_stride * idx + block_size);

        float weight = 1.0f;
        if (weights) {
          weight = weights[is_weight_positional ? i : current];
        }
        const float scale = weight * scale_bias[0];
        const float bias = weight * scale_bias[1];

        for (int j = 0; j < block_size; ++j) {
          out[j] =
              std::fma(scale, input[input_stride * idx + j], out[j] + bias);
        }

        ++current;
      }
      if (normalize_by_lengths && len) {
        float scale = 1.f / len;
        for (int j = 0; j < block_size; ++j) {
          out[j] *= scale;
        }
      }
      out += output_stride;
    }
    return current == index_size;
  } else {
    if (input_stride == -1) {
      input_stride = block_size;
    }

    // Reference implementation of FP32 SLS
    int64_t current = 0;
    for (int m = 0; m < output_size; ++m) {
      memset(out, 0, sizeof(float) * block_size);
      int len = use_offsets ? offsets_or_lengths[m + 1] - offsets_or_lengths[m]
                            : offsets_or_lengths[m];
      if (current + len > index_size) {
        return false;
      }
      for (int i = 0; i < len; ++i) {
        int64_t idx = indices[current];
        if (idx < 0 || idx >= data_size) {
          return false;
        }

        float w = 1.f;
        if (weights) {
          w = weights[is_weight_positional ? i : current];
        }

        for (int j = 0; j < block_size; ++j) {
          const InType* inptr = input + input_stride * idx + j;
          out[j] = std::fma(
              w,
              is_same<InType, float16>::value ? cpu_half2float(*inptr) : *inptr,
              out[j]);
        }

        ++current;
      }
      if (normalize_by_lengths && len) {
        float scale = 1.f / len;
        for (int j = 0; j < block_size; ++j) {
          out[j] *= scale;
        }
      }
      out += output_stride;
    }
    return current == index_size;
  }
}

template <typename IndexType, typename OffsetType>
bool EmbeddingSpMDMNBit_ref(
    int bit_rate,
    const int64_t block_size,
    const int64_t output_size,
    const int64_t index_size,
    const int64_t data_size,
    const uint8_t* input,
    const IndexType* indices,
    const OffsetType* offsets_or_lengths,
    const float* weights, // optional, can be null for non-weighted sum
    bool normalize_by_lengths,
    float* out,
    bool is_weight_positional,
    bool use_offsets) {
  assert((bit_rate == 2 || bit_rate == 4) && "bit_rate must be 2 or 4");
  int num_elem_per_byte = 8 / bit_rate;

  // block_size is the number of elements and fused_block_size is the size of
  // an entire row, including scale and bias.
  const auto scale_bias_offset = 2 * sizeof(float16);
  const int64_t fused_block_size =
      (block_size + num_elem_per_byte - 1) / num_elem_per_byte +
      scale_bias_offset;
  int64_t current = 0;
  for (int m = 0; m < output_size; ++m) {
    memset(out, 0, sizeof(float) * block_size);
    int len = use_offsets ? offsets_or_lengths[m + 1] - offsets_or_lengths[m]
                          : offsets_or_lengths[m];
    if (current + len > index_size) {
      return false;
    }
    for (int i = 0; i < len; ++i) {
      int64_t idx = indices[current];
      if (idx < 0 || idx >= data_size) {
        return false;
      }

      const float16* scale_bias = reinterpret_cast<const float16*>(
          input + fused_block_size * idx +
          (block_size + num_elem_per_byte - 1) / num_elem_per_byte);

      float weight = 1.0f;
      if (weights) {
        weight = weights[is_weight_positional ? i : current];
      }
      const float scale = weight * cpu_half2float(scale_bias[0]);
      const float bias = weight * cpu_half2float(scale_bias[1]);

      for (int j = 0; j < block_size; ++j) {
        uint8_t quantized =
            input[fused_block_size * idx + j / num_elem_per_byte];
        quantized >>= (j % num_elem_per_byte) * bit_rate;
        quantized &= (1 << bit_rate) - 1;

        out[j] = std::fma(scale, quantized, out[j] + bias);
      }

      ++current;
    }
    if (normalize_by_lengths && len) {
      float scale = 1.f / len;
      for (int j = 0; j < block_size; ++j) {
        out[j] *= scale;
      }
    }
    out += block_size;
  }
  return current == index_size;
}

template <typename InType, typename IndexType, typename OffsetType>
bool EmbeddingSpMDMRowWiseSparse_ref(
    const int64_t block_size,
    const int64_t output_size,
    const int64_t index_size,
    const int64_t uncompressed_data_size,
    // const int64_t compressed_data_size,
    const InType* input,
    const IndexType* indices,
    const int32_t* compressed_indices_table,
    const OffsetType* offsets_or_lengths,
    const float* weights, // optional, can be null for non-weighted sum
    bool normalize_by_lengths,
    float* out,
    bool is_weight_positional,
    bool use_offsets) {
  bool is8bit = is_same<InType, uint8_t>::value;

  if (is8bit) {
    // block_size is the number of elements and fused_block_size is the size of
    // an entire row, including scale and bias.
    const auto scale_bias_offset = 2 * sizeof(float);
    const int64_t fused_block_size = block_size + scale_bias_offset;
    int64_t current = 0;
    for (int m = 0; m < output_size; ++m) {
      memset(out, 0, sizeof(float) * block_size);
      int len = use_offsets ? offsets_or_lengths[m + 1] - offsets_or_lengths[m]
                            : offsets_or_lengths[m];
      if (current + len > index_size) {
        return false;
      }
      for (int i = 0; i < len; ++i) {
        IndexType uncompressed_idx = indices[current];
        if (uncompressed_idx < 0 ||
            uncompressed_idx >= uncompressed_data_size) {
          return false;
        }
        IndexType idx = compressed_indices_table[uncompressed_idx];
        if (idx == -1) {
          ++current;
          continue;
        }
        // if (idx < 0 || idx >= compressed_data_size) {
        //   return false;
        // }

        const float* scale_bias = reinterpret_cast<const float*>(
            input + fused_block_size * idx + block_size);

        float weight = 1.0f;
        if (weights) {
          weight = weights[is_weight_positional ? i : current];
        }
        const float scale = weight * scale_bias[0];
        const float bias = weight * scale_bias[1];

        for (int j = 0; j < block_size; ++j) {
          out[j] =
              std::fma(scale, input[fused_block_size * idx + j], out[j] + bias);
        }

        ++current;
      }
      if (normalize_by_lengths && len) {
        float scale = 1.f / len;
        for (int j = 0; j < block_size; ++j) {
          out[j] *= scale;
        }
      }
      out += block_size;
    }
    return current == index_size;
  } else {
    // Reference implementation of FP32 SLS
    int64_t current = 0;
    for (int m = 0; m < output_size; ++m) {
      memset(out, 0, sizeof(float) * block_size);
      int len = use_offsets ? offsets_or_lengths[m + 1] - offsets_or_lengths[m]
                            : offsets_or_lengths[m];
      if (current + len > index_size) {
        return false;
      }
      for (int i = 0; i < len; ++i) {
        IndexType uncompressed_idx = indices[current];
        if (uncompressed_idx < 0 ||
            uncompressed_idx >= uncompressed_data_size) {
          return false;
        }
        IndexType idx = compressed_indices_table[uncompressed_idx];
        if (idx == -1) {
          ++current;
          continue;
        }
        // if (idx < 0 || idx >= compressed_data_size) {
        //   return false;
        // }

        float w = 1.f;
        if (weights) {
          w = weights[is_weight_positional ? i : current];
        }

        for (int j = 0; j < block_size; ++j) {
          const InType* inptr = input + block_size * idx + j;
          out[j] = std::fma(
              w,
              is_same<InType, float16>::value ? cpu_half2float(*inptr) : *inptr,
              out[j]);
        }

        ++current;
      }
      if (normalize_by_lengths && len) {
        float scale = 1.f / len;
        for (int j = 0; j < block_size; ++j) {
          out[j] *= scale;
        }
      }
      out += block_size;
    }
    return current == index_size;
  }
}

template <typename IndexType, typename OffsetType>
bool EmbeddingSpMDMNBitRowWiseSparse_ref(
    int bit_rate,
    const int64_t block_size,
    const int64_t output_size,
    const int64_t index_size,
    const int64_t uncompressed_data_size,
    // const int64_t compressed_data_size,
    const uint8_t* input,
    const IndexType* indices,
    const int32_t* compressed_indices_table,
    const OffsetType* offsets_or_lengths,
    const float* weights, // optional, can be null for non-weighted sum
    bool normalize_by_lengths,
    float* out,
    bool is_weight_positional,
    bool use_offsets) {
  assert((bit_rate == 2 || bit_rate == 4) && "bit_rate must be 2 or 4");
  int num_elem_per_byte = 8 / bit_rate;

  // block_size is the number of elements and fused_block_size is the size of
  // an entire row, including scale and bias.
  const auto scale_bias_offset = 2 * sizeof(float16);
  const int64_t fused_block_size =
      (block_size + num_elem_per_byte - 1) / num_elem_per_byte +
      scale_bias_offset;
  int64_t current = 0;
  for (int m = 0; m < output_size; ++m) {
    memset(out, 0, sizeof(float) * block_size);
    int len = use_offsets ? offsets_or_lengths[m + 1] - offsets_or_lengths[m]
                          : offsets_or_lengths[m];
    if (current + len > index_size) {
      return false;
    }
    for (int i = 0; i < len; ++i, ++current) {
      IndexType uncompressed_idx = indices[current];
      if (uncompressed_idx < 0 || uncompressed_idx >= uncompressed_data_size) {
        return false;
      }
      IndexType idx = compressed_indices_table[uncompressed_idx];
      if (idx == -1) {
        continue;
      }
      // if (idx < 0 || idx >= compressed_data_size) {
      //   return false;
      // }

      const float16* scale_bias = reinterpret_cast<const float16*>(
          input + fused_block_size * idx +
          (block_size + num_elem_per_byte - 1) / num_elem_per_byte);

      float weight = 1.0f;
      if (weights) {
        weight = weights[is_weight_positional ? i : current];
      }
      const float scale = weight * cpu_half2float(scale_bias[0]);
      const float bias = weight * cpu_half2float(scale_bias[1]);

      for (int j = 0; j < block_size; ++j) {
        uint8_t quantized =
            input[fused_block_size * idx + j / num_elem_per_byte];
        quantized >>= (j % num_elem_per_byte) * bit_rate;
        quantized &= (1 << bit_rate) - 1;

        out[j] = std::fma(scale, quantized, out[j] + bias);
      }
    }
    if (normalize_by_lengths && len) {
      float scale = 1.f / len;
      for (int j = 0; j < block_size; ++j) {
        out[j] *= scale;
      }
    }
    out += block_size;
  }
  return current == index_size;
}

template <typename IndexType>
int sparse_adagrad_ref(
    int num_rows, // number of rows reading
    int block_size, // number of parameters per rows
    uint64_t param_size, // total number of parameters
    float* w, // input parameters
    const float* g, // input gradients
    float* h, // input momentums
    const IndexType* indices, // indices of each row
    float epsilon,
    float lr,
    float weight_decay,
    const double* counter,
    const int64_t counter_halflife) {
  for (auto i = 0; i < num_rows; ++i) {
    uint64_t idx = indices[i];
    auto offsetI = i * block_size;
    auto offsetIdx = idx * block_size;

    if (block_size + offsetIdx > param_size) {
      return i;
    }

    float freq =
        (counter && counter[idx] > 0) ? counter_halflife / counter[idx] : 1.0;

    const float* g_;
    const float* h_;
    const float* w_;
    float* nh_;
    float* nw_;

    g_ = g + offsetI;
    h_ = h + offsetIdx;
    w_ = w + offsetIdx;
    nh_ = h + offsetIdx;
    nw_ = w + offsetIdx;

    for (auto j = 0; j < block_size; ++j) {
      float gj = std::fma(weight_decay * freq, w_[j], g_[j]);
      float hj = h_[j] + gj * gj;
      nh_[j] = hj;
      nw_[j] = w_[j] + lr * gj / (std::sqrt(hj) + epsilon);
    }
  }
  return num_rows;
}

template <typename IndexType>
int rowwise_sparse_adagrad_ref(
    int num_rows, // number of rows reading
    int block_size, // number of parameters per rows
    uint64_t param_size, // total number of parameters
    float* w, // input parameters
    const float* g, // input gradients
    float* h, // input momentums
    const IndexType* indices, // indices of each row
    float epsilon,
    float lr,
    float weight_decay,
    const double* counter,
    const int64_t counter_halflife) {
  for (auto i = 0; i < num_rows; ++i) {
    uint64_t idx = indices[i];
    auto offsetI = i * block_size;
    auto offsetIdx = idx * block_size;

    if (block_size + offsetIdx > param_size) {
      return i;
    }

    float freq =
        (counter && counter[idx] > 0) ? counter_halflife / counter[idx] : 1.0;

    const float* g_;
    float* h_;
    float* w_;

    g_ = g + offsetI;
    h_ = h + idx; // This is different from sparse adagrad
    w_ = w + offsetIdx;

    float final_sum = 0.0f;
    // Note the following code assumes fbgemm will generate AVX2 code for
    // horizontal reduction, which is OK for now because fbgemm always uses AVX2
    // for SparseAdagrad due to its performance is bounded by memory bandwidth
    // hence no speedup from AVX512.
    // Non-vectorized version would be just
    // for (auto j = 0; j < block_size; ++j) {
    //   float gj = g_[j];
    //   final_sum += gj * gj;
    // }
    constexpr int VLEN = 8;
    array<float, VLEN> partial_sum = {0.0f};
    for (auto j = 0; j < block_size; ++j) {
      float gj = std::fma(weight_decay * freq, w_[j], g_[j]);
      partial_sum[j % VLEN] += gj * gj;
    }
    final_sum = ((partial_sum[0] + partial_sum[1]) +
                 (partial_sum[2] + partial_sum[3])) +
        ((partial_sum[4] + partial_sum[5]) + (partial_sum[6] + partial_sum[7]));
    final_sum /= block_size;
    float hi = *h_ = *h_ + final_sum;
    float float_step = lr / (std::sqrt(hi) + epsilon);

    for (auto j = 0; j < block_size; ++j) {
      float gj = std::fma(weight_decay * freq, w_[j], g_[j]);
      w_[j] += gj * float_step;
    }
  }
  return num_rows;
}

template <typename DataType, typename IndexType, typename OffsetType>
int rowwise_sparse_adagrad_fused_ref(
    int64_t block_size,
    int64_t output_size,
    int64_t index_size,
    int64_t data_size,
    DataType* w,
    const float* g,
    float* h,
    const IndexType* indices,
    const OffsetType* offsets_or_lengths,
    float epsilon,
    float lr,
    bool use_offsets,
    bool use_stochastic_rounding,
    int emu_vector_size,
    int64_t grad_stride) {
  if (grad_stride == -1) {
    grad_stride = block_size;
  }

  constexpr bool isFloat16w = std::is_same<float16, DataType>::value;
  // Local random buffer to emulate SIMD vector
  // R: generated 32bit base random numbers
  // r: extracted 8-bit for rounding
  constexpr int VLEN_MAX = 16;
  uint32_t R[VLEN_MAX], r[VLEN_MAX];
  int vlen = emu_vector_size;
  if (vlen != 8 && vlen != 16) {
    // Raise error as it may cause buffer overflow
    cerr << "Not supported emu_vector_size: " << emu_vector_size << endl;
    return 0;
  }

  int64_t current = 0;
  for (int m = 0; m < output_size; ++m) {
    int len = use_offsets ? offsets_or_lengths[m + 1] - offsets_or_lengths[m]
                          : offsets_or_lengths[m];
    if (current + len > index_size) {
      return false;
    }
    const float* g_ = g + m * grad_stride;
    // Note the following code assumes fbgemm will generate AVX2 code for
    // horizontal reduction, which is OK for now because fbgemm always uses AVX2
    // for SparseAdagrad due to its performance is bounded by memory bandwidth
    // hence no speedup from AVX512.
    // Non-vectorized version would be just
    // for (auto j = 0; j < block_size; ++j) {
    //   float gj = g_[j];
    //   final_sum += gj * gj;
    // }
    constexpr int VLEN_AVX2 = 8;
    array<float, VLEN_AVX2> partial_sum = {0.0f};
    for (auto j = 0; j < block_size; ++j) {
      float gj = g_[j];
      partial_sum[j % VLEN_AVX2] += gj * gj;
    }
    float final_sum = ((partial_sum[0] + partial_sum[1]) +
                       (partial_sum[2] + partial_sum[3])) +
        ((partial_sum[4] + partial_sum[5]) + (partial_sum[6] + partial_sum[7]));
    final_sum /= block_size;

    for (int i = 0; i < len; ++i, ++current) {
      int64_t idx = indices[current];
      if (idx < 0 || idx >= data_size) {
        return false;
      }

      float* h_ = h + idx;
      DataType* w_ = w + idx * block_size;

      float hi = *h_ = *h_ + final_sum;
      float float_step = lr / (std::sqrt(hi) + epsilon);

      int nvec = (block_size + vlen - 1) / vlen;
      int rem = (block_size % vlen) ? (block_size % vlen) : vlen;

      // Emulate JIT behavior of stochastic rounding with vector-length
      //
      // Generate R buffer every 4 steps of nvec loop. Each 8-bit in R
      // (uint32_t) will be used once. It is shifted to bits[5..13] then
      // added to FP32 weights before FP16 conversion.
      //
      // The shifted 8 bit region
      // +-------+--------+--------+--------+
      // |       |        |   xxxxx|xxx     |
      //  31      23       15       7      0
      //
      // Half float has 10 bits of mantissa, and float has 23, we are shifting
      // the bits to cover the region where half floats can't represent data.
      // This is bit 13-23 of the mantissa of fp32.
      // This will be effectively adding a random variable of [0,1]

      for (int n = 0; n < nvec; ++n) {
        int cur_vlen = (n == nvec - 1) ? rem : vlen;
        int sr_idx = n % 4;

        if (isFloat16w && use_stochastic_rounding) {
          if (sr_idx == 0) {
            for (int v = 0; v < vlen; ++v) {
              R[v] = rnd128_next(v, vlen);
              r[v] = (R[v] & 0xFFU) << 5;
            }
          } else if (sr_idx == 1) {
            for (int v = 0; v < vlen; ++v) {
              r[v] = ((R[v] & 0xFF00U) >> 8) << 5;
            }
          } else if (sr_idx == 2) {
            for (int v = 0; v < vlen; ++v) {
              r[v] = ((R[v] & 0xFF0000U) >> 16) << 5;
            }
          } else { // 3
            for (int v = 0; v < vlen; ++v) {
              r[v] = ((R[v] & 0xFF000000U) >> 24) << 5;
            }
          }
        }

        for (int v = 0; v < cur_vlen; ++v) {
          int j = n * vlen + v;
          if (isFloat16w) {
            union {
              float w_f32;
              uint32_t w_i32;
            };
            w_f32 = cpu_half2float(w_[j]);
            w_f32 = std::fma(float_step, g_[j], w_f32);
            if (use_stochastic_rounding) {
              w_i32 += r[v];
            }
            // Use truncate rounding to 'counterwork' the random added part
            w_[j] = cpu_float2half_rz(w_f32);
          } else { // float
            w_[j] += g_[j] * float_step;
          }
        }
      }
    }
  }

  return current == index_size;
}

template FBGEMM_API void transposeConvWeights(
    const conv_param_t<1>& conv_p,
    const std::int8_t* src,
    std::int8_t* dest);

template FBGEMM_API void transposeConvWeights(
    const conv_param_t<2>& conv_p,
    const std::int8_t* src,
    std::int8_t* dest);

template FBGEMM_API void transposeConvWeights(
    const conv_param_t<3>& conv_p,
    const std::int8_t* src,
    std::int8_t* dest);

#define INSTANTIATE_SPMDM_BASE(IN_TYPE, INDEX_TYPE, OFFSET_TYPE) \
  template FBGEMM_API bool EmbeddingSpMDM_ref(                   \
      const int64_t block_size,                                  \
      const int64_t output_size,                                 \
      const int64_t index_size,                                  \
      const int64_t data_size,                                   \
      const IN_TYPE* input,                                      \
      const INDEX_TYPE* indices,                                 \
      const OFFSET_TYPE* offsets_or_lengths,                     \
      const float* weights,                                      \
      bool normalize_by_lengths,                                 \
      float* out,                                                \
      bool is_weight_positional,                                 \
      bool use_offsets,                                          \
      int64_t input_stride,                                      \
      int64_t output_stride);                                    \
  template FBGEMM_API bool EmbeddingSpMDMRowWiseSparse_ref(      \
      const int64_t block_size,                                  \
      const int64_t output_size,                                 \
      const int64_t index_size,                                  \
      const int64_t uncompressed_data_size,                      \
      const IN_TYPE* input,                                      \
      const INDEX_TYPE* indices,                                 \
      const int32_t* compressed_indices_table,                   \
      const OFFSET_TYPE* offsets_or_lengths,                     \
      const float* weights,                                      \
      bool normalize_by_lengths,                                 \
      float* out,                                                \
      bool is_weight_positional,                                 \
      bool use_offsets);

#define INSTANTIATE_SPMDM_OFFSET_T(IN_TYPE, INDEX_TYPE)     \
  INSTANTIATE_SPMDM_BASE(IN_TYPE, INDEX_TYPE, std::int32_t) \
  INSTANTIATE_SPMDM_BASE(IN_TYPE, INDEX_TYPE, std::int64_t)

#define INSTANTIATE_SPMDM_INDEX_T(IN_TYPE)          \
  INSTANTIATE_SPMDM_OFFSET_T(IN_TYPE, std::int32_t) \
  INSTANTIATE_SPMDM_OFFSET_T(IN_TYPE, std::int64_t)

INSTANTIATE_SPMDM_INDEX_T(float)
INSTANTIATE_SPMDM_INDEX_T(float16)
INSTANTIATE_SPMDM_INDEX_T(std::uint8_t)

#undef INSTANTIATE_SPMDM_INDEX_T
#undef INSTANTIATE_SPMDM_OFFSET_T
#undef INSTANTIATE_SPMDM_BASE

#define INSTANTIATE_SPMDM_BASE(INDEX_TYPE, OFFSET_TYPE)         \
  template FBGEMM_API bool EmbeddingSpMDMNBit_ref(              \
      int bit_rate,                                             \
      const int64_t block_size,                                 \
      const int64_t output_size,                                \
      const int64_t index_size,                                 \
      const int64_t data_size,                                  \
      const uint8_t* input,                                     \
      const INDEX_TYPE* indices,                                \
      const OFFSET_TYPE* offsets_or_lengths,                    \
      const float* weights,                                     \
      bool normalize_by_lengths,                                \
      float* out,                                               \
      bool is_weight_positional,                                \
      bool use_offsets);                                        \
  template FBGEMM_API bool EmbeddingSpMDMNBitRowWiseSparse_ref( \
      int bit_rate,                                             \
      const int64_t block_size,                                 \
      const int64_t output_size,                                \
      const int64_t index_size,                                 \
      const int64_t uncompressed_data_size,                     \
      const uint8_t* input,                                     \
      const INDEX_TYPE* indices,                                \
      const int32_t* compressed_indices_table,                  \
      const OFFSET_TYPE* offsets_or_lengths,                    \
      const float* weights,                                     \
      bool normalize_by_lengths,                                \
      float* out,                                               \
      bool is_weight_positional,                                \
      bool use_offsets);

#define INSTANTIATE_SPMDM_OFFSET_T(INDEX_TYPE) \
  INSTANTIATE_SPMDM_BASE(INDEX_TYPE, int32_t)  \
  INSTANTIATE_SPMDM_BASE(INDEX_TYPE, int64_t)

INSTANTIATE_SPMDM_OFFSET_T(int32_t)
INSTANTIATE_SPMDM_OFFSET_T(int64_t)

#undef INSTANTIATE_SPMDM_OFFSET_T
#undef INSTANTIATE_SPMDM_BASE

template FBGEMM_API int sparse_adagrad_ref(
    int num_rows, // number of rows reading
    int block_size, // number of parameters per rows
    std::uint64_t param_size, // total number of parameters
    float* w, // input parameters
    const float* g, // input gradients
    float* h, // input momentums
    const std::int64_t* indices, // indices of each row
    float epsilon,
    float lr,
    float weight_decay,
    const double* counter,
    const int64_t counter_halflife);

template FBGEMM_API int sparse_adagrad_ref(
    int num_rows, // number of rows reading
    int block_size, // number of parameters per rows
    std::uint64_t param_size, // total number of parameters
    float* w, // input parameters
    const float* g, // input gradients
    float* h, // input momentums
    const std::int32_t* indices, // indices of each row
    float epsilon,
    float lr,
    float weight_decay,
    const double* counter,
    const int64_t counter_halflife);

template FBGEMM_API int rowwise_sparse_adagrad_ref(
    int num_rows, // number of rows reading
    int block_size, // number of parameters per rows
    std::uint64_t param_size, // total number of parameters
    float* w, // input parameters
    const float* g, // input gradients
    float* h, // input momentums
    const std::int64_t* indices, // indices of each row
    float epsilon,
    float lr,
    float weight_decay,
    const double* counter,
    const int64_t counter_halflife);

template FBGEMM_API int rowwise_sparse_adagrad_ref(
    int num_rows, // number of rows reading
    int block_size, // number of parameters per rows
    std::uint64_t param_size, // total number of parameters
    float* w, // input parameters
    const float* g, // input gradients
    float* h, // input momentums
    const std::int32_t* indices, // indices of each row
    float epsilon,
    float lr,
    float weight_decay,
    const double* counter,
    const int64_t counter_halflife);

#define INSTANTIATE_SPMDM_BASE(DATA_TYPE, INDEX_TYPE, OFFSET_TYPE) \
  template FBGEMM_API int rowwise_sparse_adagrad_fused_ref(        \
      int64_t block_size,                                          \
      int64_t output_size,                                         \
      int64_t index_size,                                          \
      int64_t data_size,                                           \
      DATA_TYPE* w,                                                \
      const float* g,                                              \
      float* h,                                                    \
      const INDEX_TYPE* indices,                                   \
      const OFFSET_TYPE* offsets_or_lengths,                       \
      float epsilon,                                               \
      float lr,                                                    \
      bool use_offsets,                                            \
      bool use_stochastic_rounding,                                \
      int emu_vector_size,                                         \
      int64_t grad_stride);

#define INSTANTIATE_SPMDM_OFFSET_T(DATA_TYPE, INDEX_TYPE) \
  INSTANTIATE_SPMDM_BASE(DATA_TYPE, INDEX_TYPE, int32_t)  \
  INSTANTIATE_SPMDM_BASE(DATA_TYPE, INDEX_TYPE, int64_t)

#define INSTANTIATE_SPMDM_INDEX_T(DATA_TYPE)     \
  INSTANTIATE_SPMDM_OFFSET_T(DATA_TYPE, int32_t) \
  INSTANTIATE_SPMDM_OFFSET_T(DATA_TYPE, int64_t)

INSTANTIATE_SPMDM_INDEX_T(float)
INSTANTIATE_SPMDM_INDEX_T(float16)

#undef INSTANTIATE_SPMDM_OFFSET_T
#undef INSTANTIATE_SPMDM_BASE

template <typename IndexType>
FBGEMM_API void compressed_indices_remap_ref(
    std::int32_t offsets_numel,
    const IndexType* indices,
    const int32_t* compressed_indices_mapping,
    const IndexType* offsets,
    const float* weights, // optional, can be null,
    IndexType* out_indices,
    IndexType* out_offsets,
    float* out_weights) {
  bool has_per_sample_weights = (weights != nullptr);
  out_offsets[0] = offsets[0];
  IndexType j = 0;
  for (int i = 1; i < offsets_numel; i++) {
    for (int32_t k = offsets[i - 1]; k < offsets[i]; k++) {
      if (compressed_indices_mapping[indices[k]] != -1) {
        out_indices[j] = compressed_indices_mapping[indices[k]];
        if (has_per_sample_weights) {
          out_weights[j] = weights[k];
        }
        j++;
      }
    }
    out_offsets[i] = j;
  }
}

#define INSTANTIATE_REMAP_BASE(INDEX_TYPE)               \
  template FBGEMM_API void compressed_indices_remap_ref( \
      std::int32_t offsets_numel,                        \
      const INDEX_TYPE* indices,                         \
      const int32_t* compressed_indices_mapping,         \
      const INDEX_TYPE* offsets,                         \
      const float* weights,                              \
      INDEX_TYPE* out_indices,                           \
      INDEX_TYPE* out_offsets,                           \
      float* out_weights);

INSTANTIATE_REMAP_BASE(int32_t)
INSTANTIATE_REMAP_BASE(int64_t)

#undef INSTANTIATE_REMAP_BASE

} // namespace fbgemm
