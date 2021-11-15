/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#define FBGEMM_EXPORTS
#include "fbgemm/FbgemmI8DepthwiseAvx2.h"

#include <stdexcept> // for logic_error
#include <string>

#include "./FbgemmI8DepthwiseAvx2-inl.h"
#include "./GenerateI8Depthwise.h"
#include "./MaskAvx2.h"
#include "fbgemm/Utils.h"
#include "fbgemm/UtilsAvx2.h"

using namespace std;

namespace fbgemm {

template <
    bool FUSE_RELU,
    bool HAS_BIAS,
    bool A_SYMMETRIC,
    bool B_SYMMETRIC,
    QuantizationGranularity Q_GRAN,
    typename BIAS_TYPE>
static ALWAYS_INLINE void depthwise_3d_kernel_(
    int T,
    int H,
    int W,
    int IC,
    int OC,
    int t,
    int h,
    int w,
    array<int, 3> F,
    int stride_t,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    const int32_t* B_zero_point,
    const int8_t* Bp,
    const float* C_multiplier,
    int32_t C_zero_point,
    int32_t* C_int32,
    uint8_t* C_uint8,
    int32_t* row_offsets,
    const int32_t* col_offsets,
    const BIAS_TYPE* bias,
    const float* act_times_w_scale,
    GenI8Depthwise::jit_kernel_signature* pregenerated_kernel = nullptr) {
  int R = F[1], S = F[2];
  int PAD_P = (F[0] - 1) / 2, PAD_T = (F[1] - 1) / 2, PAD_B = PAD_T,
      PAD_L = (F[2] - 1) / 2, PAD_R = PAD_L;
  int H_OUT = (H + PAD_T + PAD_B - R) / stride_h + 1;
  int W_OUT = (W + PAD_L + PAD_R - S) / stride_w + 1;
  int t_in = -PAD_P + t * stride_t;
  int h_in = -PAD_T + h * stride_h;
  int w_in = -PAD_L + w * stride_w;

  int remainder = OC % 32;
  if (remainder == 0) {
    remainder = 32;
  }

  GenI8Depthwise::jit_kernel_signature kernel = pregenerated_kernel
      ? *pregenerated_kernel
      : GenI8Depthwise().getOrCreate(
            /*D=*/3,
            F,
            OC / IC,
            /*compute_a_sum=*/!B_SYMMETRIC,
            remainder,
            /*prev_skip=*/std::max(-t_in, 0),
            /*next_skip=*/std::max(t_in + F[0] - T, 0),
            /*top_skip=*/std::max(-h_in, 0),
            /*bottom_skip=*/std::max(h_in + F[1] - H, 0),
            /*left_skip=*/std::max(-w_in, 0),
            /*right_skip=*/std::max(w_in + F[2] - W, 0));
  kernel(
      A + ((t_in * H + h_in) * W + w_in) * IC,
      Bp,
      C_int32,
      B_SYMMETRIC ? nullptr : row_offsets,
      H,
      W,
      IC,
      internal::avx2_ps_or_epi32_combined_mask,
      A_zero_point);

  if (OC == IC) {
    requantize_<FUSE_RELU, HAS_BIAS, Q_GRAN, A_SYMMETRIC, B_SYMMETRIC, 1>(
        A_zero_point,
        B_zero_point,
        C_multiplier,
        C_zero_point,
        C_int32,
        C_uint8 + ((t * H_OUT + h) * W_OUT + w) * OC,
        OC,
        row_offsets,
        col_offsets,
        bias,
        act_times_w_scale);
  } else {
    assert(OC / IC == 2);
    requantize_<FUSE_RELU, HAS_BIAS, Q_GRAN, A_SYMMETRIC, B_SYMMETRIC, 2>(
        A_zero_point,
        B_zero_point,
        C_multiplier,
        C_zero_point,
        C_int32,
        C_uint8 + ((t * H_OUT + h) * W_OUT + w) * OC,
        OC,
        row_offsets,
        col_offsets,
        bias,
        act_times_w_scale);
  }
}

template <
    bool FUSE_RELU,
    bool HAS_BIAS,
    bool A_SYMMETRIC,
    bool B_SYMMETRIC,
    typename BIAS_TYPE,
    QuantizationGranularity Q_GRAN>
static ALWAYS_INLINE void depthwise_3d_same_pad_(
    const conv_param_t<3>& conv_p,
    int32_t A_zero_point,
    const uint8_t* A,
    const int32_t* B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    const float* C_multiplier,
    int32_t C_zero_point,
    int32_t* C_int32,
    uint8_t* C_uint8,
    const int32_t* col_offsets,
    const BIAS_TYPE* bias,
    const float* act_times_w_scale,
    int thread_id,
    int num_threads) {
  int N = conv_p.MB;
  int T = conv_p.IN_DIM[0];
  int H = conv_p.IN_DIM[1];
  int W = conv_p.IN_DIM[2];
  int IC = conv_p.IC;
  int OC = conv_p.OC;
  array<int, 3> F = conv_p.K;
  int stride_t = conv_p.stride[0];
  int stride_h = conv_p.stride[1];
  int stride_w = conv_p.stride[2];

  assert(IC % 8 == 0);

  int K_T = F[0], K_H = F[1], K_W = F[2];
  int PAD_P = (F[0] - 1) / 2, PAD_N = PAD_P, PAD_T = (F[1] - 1) / 2,
      PAD_B = PAD_T, PAD_L = (F[2] - 1) / 2, PAD_R = PAD_L;
  int T_OUT = (T + PAD_P + PAD_N - K_T) / stride_t + 1;
  int H_OUT = (H + PAD_T + PAD_B - K_H) / stride_h + 1;
  int W_OUT = (W + PAD_L + PAD_R - K_W) / stride_w + 1;
  const int8_t* Bp = B.PackedMat();

  int32_t* row_offsets = static_cast<int32_t*>(
      fbgemmAlignedAlloc(64, (IC + 31) / 32 * 32 * sizeof(int32_t)));

  int n_begin, n_end, t_begin, t_end, h_begin, h_end;
  // Reuse the 3-dim partition scheme for parallelization in matrix
  // multiplication.
  thread_type_t th_info =
      fbgemmGetThreadPartition(N, T_OUT, H_OUT, thread_id, num_threads);
  // Calculate the begin and end index along the batch (N) dimension
  fbgemmPartition1D(
      th_info.g_thread_id, th_info.g_num_threads, N, n_begin, n_end);
  // Calculate the begin and end index along the T dimension
  fbgemmPartition1D(
      th_info.m_thread_id, th_info.m_num_threads, T_OUT, t_begin, t_end);
  // Calculate the begin and end index along the H dimension
  fbgemmPartition1D(
      th_info.n_thread_id, th_info.n_num_threads, H_OUT, h_begin, h_end);

  GenI8Depthwise::jit_kernel_signature middle_kernel;

  for (int n = n_begin; n < n_end; ++n) {
    const uint8_t* A_base = A + n * T * H * W * IC;
    uint8_t* C_uint8_base = C_uint8 + n * T_OUT * H_OUT * W_OUT * OC;

    int t;
    for (t = t_begin; t < PAD_P; ++t) {
      int h;
      for (h = h_begin; h < PAD_T; ++h) {
        for (int w = 0; w < W_OUT; ++w) {
          depthwise_3d_kernel_<
              FUSE_RELU,
              HAS_BIAS,
              A_SYMMETRIC,
              B_SYMMETRIC,
              Q_GRAN>(
              T,
              H,
              W,
              IC,
              OC,
              t,
              h,
              w,
              F,
              stride_t,
              stride_h,
              stride_w,
              A_zero_point,
              A_base,
              B_zero_point,
              Bp,
              C_multiplier,
              C_zero_point,
              C_int32,
              C_uint8_base,
              row_offsets,
              col_offsets,
              bias,
              act_times_w_scale);
        } // w
      } // h

      for (; h < std::min(H_OUT - PAD_B - stride_h + 1, h_end); ++h) {
        int w;
        for (w = 0; w < PAD_L; ++w) {
          depthwise_3d_kernel_<
              FUSE_RELU,
              HAS_BIAS,
              A_SYMMETRIC,
              B_SYMMETRIC,
              Q_GRAN>(
              T,
              H,
              W,
              IC,
              OC,
              t,
              h,
              w,
              F,
              stride_t,
              stride_h,
              stride_w,
              A_zero_point,
              A_base,
              B_zero_point,
              Bp,
              C_multiplier,
              C_zero_point,
              C_int32,
              C_uint8_base,
              row_offsets,
              col_offsets,
              bias,
              act_times_w_scale);
        } // w

        GenI8Depthwise::jit_kernel_signature kernel;
        for (; w < W_OUT - PAD_R - stride_w + 1; ++w) {
          if (w == PAD_L) {
            int remainder = OC % 32;
            if (remainder == 0) {
              remainder = 32;
            }
            int t_in = -PAD_P + t * stride_t;
            kernel = GenI8Depthwise().getOrCreate(
                /*D=*/3,
                F,
                OC / IC,
                /*compute_a_sum=*/!B_SYMMETRIC,
                remainder,
                /*prev_skip=*/std::max(-t_in, 0),
                /*next_skip=*/std::max(t_in + F[0] - T, 0),
                0,
                0,
                0,
                0);
          }
          depthwise_3d_kernel_<
              FUSE_RELU,
              HAS_BIAS,
              A_SYMMETRIC,
              B_SYMMETRIC,
              Q_GRAN>(
              T,
              H,
              W,
              IC,
              OC,
              t,
              h,
              w,
              F,
              stride_t,
              stride_h,
              stride_w,
              A_zero_point,
              A_base,
              B_zero_point,
              Bp,
              C_multiplier,
              C_zero_point,
              C_int32,
              C_uint8_base,
              row_offsets,
              col_offsets,
              bias,
              act_times_w_scale,
              &kernel);
        } // w

        for (; w < W_OUT; ++w) {
          depthwise_3d_kernel_<
              FUSE_RELU,
              HAS_BIAS,
              A_SYMMETRIC,
              B_SYMMETRIC,
              Q_GRAN>(
              T,
              H,
              W,
              IC,
              OC,
              t,
              h,
              w,
              F,
              stride_t,
              stride_h,
              stride_w,
              A_zero_point,
              A_base,
              B_zero_point,
              Bp,
              C_multiplier,
              C_zero_point,
              C_int32,
              C_uint8_base,
              row_offsets,
              col_offsets,
              bias,
              act_times_w_scale);
        } // w
      } // h

      for (; h < h_end; ++h) {
        for (int w = 0; w < W_OUT; ++w) {
          depthwise_3d_kernel_<
              FUSE_RELU,
              HAS_BIAS,
              A_SYMMETRIC,
              B_SYMMETRIC,
              Q_GRAN>(
              T,
              H,
              W,
              IC,
              OC,
              t,
              h,
              w,
              F,
              stride_t,
              stride_h,
              stride_w,
              A_zero_point,
              A_base,
              B_zero_point,
              Bp,
              C_multiplier,
              C_zero_point,
              C_int32,
              C_uint8_base,
              row_offsets,
              col_offsets,
              bias,
              act_times_w_scale);
        } // w
      } // h
    } // t

    for (; t < std::min(T_OUT - PAD_N - stride_t + 1, t_end); ++t) {
      int h;
      for (h = h_begin; h < PAD_T; ++h) {
        for (int w = 0; w < W_OUT; ++w) {
          depthwise_3d_kernel_<
              FUSE_RELU,
              HAS_BIAS,
              A_SYMMETRIC,
              B_SYMMETRIC,
              Q_GRAN>(
              T,
              H,
              W,
              IC,
              OC,
              t,
              h,
              w,
              F,
              stride_t,
              stride_h,
              stride_w,
              A_zero_point,
              A_base,
              B_zero_point,
              Bp,
              C_multiplier,
              C_zero_point,
              C_int32,
              C_uint8_base,
              row_offsets,
              col_offsets,
              bias,
              act_times_w_scale);
        } // w
      } // h

      for (; h < std::min(H_OUT - PAD_B - stride_h + 1, h_end); ++h) {
        int w;
        for (w = 0; w < PAD_L; ++w) {
          depthwise_3d_kernel_<
              FUSE_RELU,
              HAS_BIAS,
              A_SYMMETRIC,
              B_SYMMETRIC,
              Q_GRAN>(
              T,
              H,
              W,
              IC,
              OC,
              t,
              h,
              w,
              F,
              stride_t,
              stride_h,
              stride_w,
              A_zero_point,
              A_base,
              B_zero_point,
              Bp,
              C_multiplier,
              C_zero_point,
              C_int32,
              C_uint8_base,
              row_offsets,
              col_offsets,
              bias,
              act_times_w_scale);
        } // w

        for (; w < W_OUT - PAD_R - stride_w + 1; ++w) {
          if (n == n_begin && w == PAD_L) {
            int remainder = OC % 32;
            if (remainder == 0) {
              remainder = 32;
            }
            middle_kernel = GenI8Depthwise().getOrCreate(
                /*D=*/3,
                F,
                OC / IC,
                /*compute_a_sum=*/!B_SYMMETRIC,
                remainder,
                0,
                0,
                0,
                0,
                0,
                0);
          }
          depthwise_3d_kernel_<
              FUSE_RELU,
              HAS_BIAS,
              A_SYMMETRIC,
              B_SYMMETRIC,
              Q_GRAN>(
              T,
              H,
              W,
              IC,
              OC,
              t,
              h,
              w,
              F,
              stride_t,
              stride_h,
              stride_w,
              A_zero_point,
              A_base,
              B_zero_point,
              Bp,
              C_multiplier,
              C_zero_point,
              C_int32,
              C_uint8_base,
              row_offsets,
              col_offsets,
              bias,
              act_times_w_scale,
              &middle_kernel);
        }

        for (; w < W_OUT; ++w) {
          depthwise_3d_kernel_<
              FUSE_RELU,
              HAS_BIAS,
              A_SYMMETRIC,
              B_SYMMETRIC,
              Q_GRAN>(
              T,
              H,
              W,
              IC,
              OC,
              t,
              h,
              w,
              F,
              stride_t,
              stride_h,
              stride_w,
              A_zero_point,
              A_base,
              B_zero_point,
              Bp,
              C_multiplier,
              C_zero_point,
              C_int32,
              C_uint8_base,
              row_offsets,
              col_offsets,
              bias,
              act_times_w_scale);
        }
      } // h

      for (; h < h_end; ++h) {
        for (int w = 0; w < W_OUT; ++w) {
          depthwise_3d_kernel_<
              FUSE_RELU,
              HAS_BIAS,
              A_SYMMETRIC,
              B_SYMMETRIC,
              Q_GRAN>(
              T,
              H,
              W,
              IC,
              OC,
              t,
              h,
              w,
              F,
              stride_t,
              stride_h,
              stride_w,
              A_zero_point,
              A_base,
              B_zero_point,
              Bp,
              C_multiplier,
              C_zero_point,
              C_int32,
              C_uint8_base,
              row_offsets,
              col_offsets,
              bias,
              act_times_w_scale);
        } // w
      } // h
    } // t

    for (; t < t_end; ++t) {
      int h;
      for (h = h_begin; h < PAD_T; ++h) {
        for (int w = 0; w < W_OUT; ++w) {
          depthwise_3d_kernel_<
              FUSE_RELU,
              HAS_BIAS,
              A_SYMMETRIC,
              B_SYMMETRIC,
              Q_GRAN>(
              T,
              H,
              W,
              IC,
              OC,
              t,
              h,
              w,
              F,
              stride_t,
              stride_h,
              stride_w,
              A_zero_point,
              A_base,
              B_zero_point,
              Bp,
              C_multiplier,
              C_zero_point,
              C_int32,
              C_uint8_base,
              row_offsets,
              col_offsets,
              bias,
              act_times_w_scale);
        } // w
      } // h

      for (; h < std::min(H_OUT - PAD_B - stride_h + 1, h_end); ++h) {
        int w;
        for (w = 0; w < PAD_L; ++w) {
          depthwise_3d_kernel_<
              FUSE_RELU,
              HAS_BIAS,
              A_SYMMETRIC,
              B_SYMMETRIC,
              Q_GRAN>(
              T,
              H,
              W,
              IC,
              OC,
              t,
              h,
              w,
              F,
              stride_t,
              stride_h,
              stride_w,
              A_zero_point,
              A_base,
              B_zero_point,
              Bp,
              C_multiplier,
              C_zero_point,
              C_int32,
              C_uint8_base,
              row_offsets,
              col_offsets,
              bias,
              act_times_w_scale);
        } // w

        GenI8Depthwise::jit_kernel_signature kernel;
        for (; w < W_OUT - PAD_R - stride_w + 1; ++w) {
          if (w == PAD_L) {
            int remainder = OC % 32;
            if (remainder == 0) {
              remainder = 32;
            }
            int t_in = -PAD_P + t * stride_t;
            kernel = GenI8Depthwise().getOrCreate(
                /*D=*/3,
                F,
                OC / IC,
                /*compute_a_sum=*/!B_SYMMETRIC,
                remainder,
                /*prev_skip=*/std::max(-t_in, 0),
                /*next_skip=*/std::max(t_in + F[0] - T, 0),
                0,
                0,
                0,
                0);
          }
          depthwise_3d_kernel_<
              FUSE_RELU,
              HAS_BIAS,
              A_SYMMETRIC,
              B_SYMMETRIC,
              Q_GRAN>(
              T,
              H,
              W,
              IC,
              OC,
              t,
              h,
              w,
              F,
              stride_t,
              stride_h,
              stride_w,
              A_zero_point,
              A_base,
              B_zero_point,
              Bp,
              C_multiplier,
              C_zero_point,
              C_int32,
              C_uint8_base,
              row_offsets,
              col_offsets,
              bias,
              act_times_w_scale,
              &kernel);
        } // w

        for (; w < W_OUT; ++w) {
          depthwise_3d_kernel_<
              FUSE_RELU,
              HAS_BIAS,
              A_SYMMETRIC,
              B_SYMMETRIC,
              Q_GRAN>(
              T,
              H,
              W,
              IC,
              OC,
              t,
              h,
              w,
              F,
              stride_t,
              stride_h,
              stride_w,
              A_zero_point,
              A_base,
              B_zero_point,
              Bp,
              C_multiplier,
              C_zero_point,
              C_int32,
              C_uint8_base,
              row_offsets,
              col_offsets,
              bias,
              act_times_w_scale);
        } // w
      } // h

      for (; h < h_end; ++h) {
        for (int w = 0; w < W_OUT; ++w) {
          depthwise_3d_kernel_<
              FUSE_RELU,
              HAS_BIAS,
              A_SYMMETRIC,
              B_SYMMETRIC,
              Q_GRAN>(
              T,
              H,
              W,
              IC,
              OC,
              t,
              h,
              w,
              F,
              stride_t,
              stride_h,
              stride_w,
              A_zero_point,
              A_base,
              B_zero_point,
              Bp,
              C_multiplier,
              C_zero_point,
              C_int32,
              C_uint8_base,
              row_offsets,
              col_offsets,
              bias,
              act_times_w_scale);
        } // w
      } // h
    } // t
  } // for each n
  fbgemmAlignedFree(row_offsets);
}

// Dispatch A_SYMMETRIC and B_SYMMETRIC
template <
    bool FUSE_RELU,
    bool HAS_BIAS,
    typename BIAS_TYPE,
    QuantizationGranularity Q_GRAN>
static void depthwise_3d_same_pad_(
    const conv_param_t<3>& conv_p,
    int32_t A_zero_point,
    const uint8_t* A,
    const int32_t* B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    const float* C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const BIAS_TYPE* bias,
    const float* act_times_w_scale,
    int thread_id,
    int num_threads) {
  int32_t* C_int32_temp = static_cast<int32_t*>(
      fbgemmAlignedAlloc(64, (conv_p.OC + 31) / 32 * 32 * sizeof(int32_t)));
  if (A_zero_point == 0 || col_offsets == nullptr) {
    if (Q_GRAN == QuantizationGranularity::TENSOR && B_zero_point[0] == 0) {
      depthwise_3d_same_pad_<
          FUSE_RELU,
          HAS_BIAS,
          true /*A_symmetric*/,
          true /*B_symmetric*/,
          BIAS_TYPE,
          Q_GRAN>(
          conv_p,
          A_zero_point,
          A,
          B_zero_point,
          B,
          C_multiplier,
          C_zero_point,
          C_int32_temp,
          C,
          col_offsets,
          bias,
          act_times_w_scale,
          thread_id,
          num_threads);
    } else {
      depthwise_3d_same_pad_<
          FUSE_RELU,
          HAS_BIAS,
          true /*A_symmetric*/,
          false /*B_symmetric*/,
          BIAS_TYPE,
          Q_GRAN>(
          conv_p,
          A_zero_point,
          A,
          B_zero_point,
          B,
          C_multiplier,
          C_zero_point,
          C_int32_temp,
          C,
          col_offsets,
          bias,
          act_times_w_scale,
          thread_id,
          num_threads);
    }
  } else {
    if (Q_GRAN == QuantizationGranularity::TENSOR && B_zero_point[0] == 0) {
      depthwise_3d_same_pad_<
          FUSE_RELU,
          HAS_BIAS,
          false /*A_symmetric*/,
          true /*B_symmetric*/,
          BIAS_TYPE,
          Q_GRAN>(
          conv_p,
          A_zero_point,
          A,
          B_zero_point,
          B,
          C_multiplier,
          C_zero_point,
          C_int32_temp,
          C,
          col_offsets,
          bias,
          act_times_w_scale,
          thread_id,
          num_threads);
    } else {
      depthwise_3d_same_pad_<
          FUSE_RELU,
          HAS_BIAS,
          false /*A_symmetric*/,
          false /*B_symmetric*/,
          BIAS_TYPE,
          Q_GRAN>(
          conv_p,
          A_zero_point,
          A,
          B_zero_point,
          B,
          C_multiplier,
          C_zero_point,
          C_int32_temp,
          C,
          col_offsets,
          bias,
          act_times_w_scale,
          thread_id,
          num_threads);
    }
  }
  fbgemmAlignedFree(C_int32_temp);
}

// Dispatch HAS_BIAS
template <bool FUSE_RELU, typename BIAS_TYPE, QuantizationGranularity Q_GRAN>
static void depthwise_3d_same_pad_(
    const conv_param_t<3>& conv_p,
    int32_t A_zero_point,
    const uint8_t* A,
    const int32_t* B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    const float* C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const BIAS_TYPE* bias,
    const float* act_times_w_scale,
    int thread_id,
    int num_threads) {
  if (bias) {
    depthwise_3d_same_pad_<FUSE_RELU, true /*HAS_BIAS*/, BIAS_TYPE, Q_GRAN>(
        conv_p,
        A_zero_point,
        A,
        B_zero_point,
        B,
        C_multiplier,
        C_zero_point,
        C,
        col_offsets,
        bias,
        act_times_w_scale,
        thread_id,
        num_threads);
  } else {
    depthwise_3d_same_pad_<FUSE_RELU, false /*HAS_BIAS*/, BIAS_TYPE, Q_GRAN>(
        conv_p,
        A_zero_point,
        A,
        B_zero_point,
        B,
        C_multiplier,
        C_zero_point,
        C,
        col_offsets,
        bias,
        act_times_w_scale,
        thread_id,
        num_threads);
  }
}

// Dispatch FUSE_RELU
template <QuantizationGranularity Q_GRAN, typename BIAS_TYPE>
void depthwise_3d_same_pad(
    const conv_param_t<3>& conv_p,
    int32_t A_zero_point,
    const uint8_t* A,
    const int32_t* B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    const float* C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const BIAS_TYPE* bias,
    bool fuse_relu,
    const float* act_times_w_scale,
    int thread_id,
    int num_threads) {
  if (B.GetKernelProduct() != conv_p.K[0] * conv_p.K[1] * conv_p.K[2]) {
    string msg =
        "[FBGEMM_CONV_ERROR] Packed weight is expected to have kernel_prod " +
        to_string(conv_p.K[0] * conv_p.K[1] * conv_p.K[2]) + " but has " +
        to_string(B.GetKernelProduct());
    throw logic_error(msg);
  }
  if (conv_p.stride[0] == 0 || conv_p.stride[1] == 0 || conv_p.stride[2] == 0 ||
      num_threads == 0) {
    assert(
        0 &&
        "stride_t == 0 || stride_h == 0 || stride_w == 0 || num_threads == 0");
    return;
  }
  if (conv_p.MB == 0) {
    // In C2, batch size 0 is allowed, so we should just early return.
    return;
  }
  if (fuse_relu) {
    depthwise_3d_same_pad_<true /*FUSE_RELU*/, BIAS_TYPE, Q_GRAN>(
        conv_p,
        A_zero_point,
        A,
        B_zero_point,
        B,
        C_multiplier,
        C_zero_point,
        C,
        col_offsets,
        bias,
        act_times_w_scale,
        thread_id,
        num_threads);
  } else {
    depthwise_3d_same_pad_<false /*FUSE_RELU*/, BIAS_TYPE, Q_GRAN>(
        conv_p,
        A_zero_point,
        A,
        B_zero_point,
        B,
        C_multiplier,
        C_zero_point,
        C,
        col_offsets,
        bias,
        act_times_w_scale,
        thread_id,
        num_threads);
  }
}

template FBGEMM_API void depthwise_3d_same_pad<QuantizationGranularity::TENSOR>(
    const conv_param_t<3>& conv_p,
    int32_t A_zero_point,
    const uint8_t* A,
    const int32_t* B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    const float* C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const int32_t* bias,
    bool fuse_relu,
    const float* act_times_w_scale,
    int thread_id,
    int num_threads);

template FBGEMM_API void depthwise_3d_same_pad<QuantizationGranularity::TENSOR>(
    const conv_param_t<3>& conv_p,
    int32_t A_zero_point,
    const uint8_t* A,
    const int32_t* B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    const float* C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const float* bias,
    bool fuse_relu,
    const float* act_times_w_scale,
    int thread_id,
    int num_threads);

template FBGEMM_API void depthwise_3d_same_pad<QuantizationGranularity::GROUP>(
    const conv_param_t<3>& conv_p,
    int32_t A_zero_point,
    const uint8_t* A,
    const int32_t* B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    const float* C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const int32_t* bias,
    bool fuse_relu,
    const float* act_times_w_scale,
    int thread_id,
    int num_threads);

template FBGEMM_API void depthwise_3d_same_pad<QuantizationGranularity::GROUP>(
    const conv_param_t<3>& conv_p,
    int32_t A_zero_point,
    const uint8_t* A,
    const int32_t* B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    const float* C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const float* bias,
    bool fuse_relu,
    const float* act_times_w_scale,
    int thread_id,
    int num_threads);

template FBGEMM_API void
depthwise_3d_same_pad<QuantizationGranularity::OUT_CHANNEL>(
    const conv_param_t<3>& conv_p,
    int32_t A_zero_point,
    const uint8_t* A,
    const int32_t* B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    const float* C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const int32_t* bias,
    bool fuse_relu,
    const float* act_times_w_scale,
    int thread_id,
    int num_threads);

template FBGEMM_API void
depthwise_3d_same_pad<QuantizationGranularity::OUT_CHANNEL>(
    const conv_param_t<3>& conv_p,
    int32_t A_zero_point,
    const uint8_t* A,
    const int32_t* B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    const float* C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const float* bias,
    bool fuse_relu,
    const float* act_times_w_scale,
    int thread_id,
    int num_threads);

template <typename BIAS_TYPE>
void depthwise_3d_per_channel_quantization_same_pad(
    const conv_param_t<3>& conv_p,
    int32_t A_zero_point,
    const uint8_t* A,
    const int32_t* B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    const float* C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const BIAS_TYPE* bias,
    bool fuse_relu,
    const float* act_times_w_scale,
    int thread_id,
    int num_threads) {
  depthwise_3d_same_pad<QuantizationGranularity::OUT_CHANNEL>(
      conv_p,
      A_zero_point,
      A,
      B_zero_point,
      B,
      C_multiplier,
      C_zero_point,
      C,
      col_offsets,
      bias,
      fuse_relu,
      act_times_w_scale,
      thread_id,
      num_threads);
}

template FBGEMM_API void depthwise_3d_per_channel_quantization_same_pad(
    const conv_param_t<3>& conv_p,
    int32_t A_zero_point,
    const uint8_t* A,
    const int32_t* B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    const float* C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const int32_t* bias,
    bool fuse_relu,
    const float* act_times_w_scale,
    int thread_id,
    int num_threads);

template FBGEMM_API void depthwise_3d_per_channel_quantization_same_pad(
    const conv_param_t<3>& conv_p,
    int32_t A_zero_point,
    const uint8_t* A,
    const int32_t* B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    const float* C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const float* bias,
    bool fuse_relu,
    const float* act_times_w_scale,
    int thread_id,
    int num_threads);

template <QuantizationGranularity Q_GRAN, typename BIAS_TYPE>
FBGEMM_API void depthwise_3x3x3_pad_1(
    int N,
    int T,
    int H,
    int W,
    int IC,
    int OC,
    int stride_t,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    const int32_t* B_zero_point,
    const PackedDepthWiseConvMatrix& Bp,
    const float* C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const BIAS_TYPE* bias,
    bool fuse_relu,
    const float* act_times_w_scale,
    int thread_id,
    int num_threads) {
  conv_param_t<3> conv_p(
      N,
      IC,
      OC,
      {T, H, W},
      IC,
      {3, 3, 3},
      {stride_t, stride_h, stride_w},
      {1, 1, 1, 1, 1, 1});

  depthwise_3d_same_pad<Q_GRAN, BIAS_TYPE>(
      conv_p,
      A_zero_point,
      A,
      B_zero_point,
      Bp,
      C_multiplier,
      C_zero_point,
      C,
      col_offsets,
      bias,
      fuse_relu,
      act_times_w_scale,
      thread_id,
      num_threads);
}

template FBGEMM_API void depthwise_3x3x3_pad_1<QuantizationGranularity::TENSOR>(
    int N,
    int T,
    int H,
    int W,
    int IC,
    int OC,
    int stride_t,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    const int32_t* B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    const float* C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const int32_t* bias,
    bool fuse_relu,
    const float* act_times_w_scale,
    int thread_id,
    int num_threads);

template FBGEMM_API void depthwise_3x3x3_pad_1<QuantizationGranularity::TENSOR>(
    int N,
    int T,
    int H,
    int W,
    int IC,
    int OC,
    int stride_t,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    const int32_t* B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    const float* C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const float* bias,
    bool fuse_relu,
    const float* act_times_w_scale,
    int thread_id,
    int num_threads);

template FBGEMM_API void depthwise_3x3x3_pad_1<QuantizationGranularity::GROUP>(
    int N,
    int T,
    int H,
    int W,
    int IC,
    int OC,
    int stride_t,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    const int32_t* B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    const float* C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const int32_t* bias,
    bool fuse_relu,
    const float* act_times_w_scale,
    int thread_id,
    int num_threads);

template FBGEMM_API void depthwise_3x3x3_pad_1<QuantizationGranularity::GROUP>(
    int N,
    int T,
    int H,
    int W,
    int IC,
    int OC,
    int stride_t,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    const int32_t* B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    const float* C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const float* bias,
    bool fuse_relu,
    const float* act_times_w_scale,
    int thread_id,
    int num_threads);

template FBGEMM_API void
depthwise_3x3x3_pad_1<QuantizationGranularity::OUT_CHANNEL>(
    int N,
    int T,
    int H,
    int W,
    int IC,
    int OC,
    int stride_t,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    const int32_t* B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    const float* C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const int32_t* bias,
    bool fuse_relu,
    const float* act_times_w_scale,
    int thread_id,
    int num_threads);

template FBGEMM_API void
depthwise_3x3x3_pad_1<QuantizationGranularity::OUT_CHANNEL>(
    int N,
    int T,
    int H,
    int W,
    int IC,
    int OC,
    int stride_t,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    const int32_t* B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    const float* C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const float* bias,
    bool fuse_relu,
    const float* act_times_w_scale,
    int thread_id,
    int num_threads);

template <typename BIAS_TYPE>
FBGEMM_API void depthwise_3x3x3_pad_1(
    int N,
    int T,
    int H,
    int W,
    int IC_OC,
    int stride_t,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    int32_t B_zero_point,
    const PackedDepthWiseConvMatrix& Bp,
    float C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const BIAS_TYPE* bias,
    bool fuse_relu,
    float act_times_w_scale,
    int thread_id,
    int num_threads) {
  depthwise_3x3x3_pad_1<QuantizationGranularity::TENSOR>(
      N,
      T,
      H,
      W,
      IC_OC,
      IC_OC,
      stride_t,
      stride_h,
      stride_w,
      A_zero_point,
      A,
      &B_zero_point,
      Bp,
      &C_multiplier,
      C_zero_point,
      C,
      col_offsets,
      bias,
      fuse_relu,
      &act_times_w_scale,
      thread_id,
      num_threads);
}

template FBGEMM_API void depthwise_3x3x3_pad_1(
    int N,
    int T,
    int H,
    int W,
    int IC_OC,
    int stride_t,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    int32_t B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    float C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const int32_t* bias,
    bool fuse_relu,
    float act_times_w_scale,
    int thread_id,
    int num_threads);

template FBGEMM_API void depthwise_3x3x3_pad_1(
    int N,
    int T,
    int H,
    int W,
    int IC_OC,
    int stride_t,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    int32_t B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    float C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const float* bias,
    bool fuse_relu,
    float act_times_w_scale,
    int thread_id,
    int num_threads);

template <typename BIAS_TYPE>
FBGEMM_API void depthwise_3x3x3_per_channel_quantization_pad_1(
    int N,
    int T,
    int H,
    int W,
    int IC_OC,
    int stride_t,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    const int32_t* B_zero_point,
    const PackedDepthWiseConvMatrix& Bp,
    const float* C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const BIAS_TYPE* bias,
    bool fuse_relu,
    const float* act_times_w_scale,
    int thread_id,
    int num_threads) {
  depthwise_3x3x3_pad_1<QuantizationGranularity::OUT_CHANNEL>(
      N,
      T,
      H,
      W,
      IC_OC,
      IC_OC,
      stride_t,
      stride_h,
      stride_w,
      A_zero_point,
      A,
      B_zero_point,
      Bp,
      C_multiplier,
      C_zero_point,
      C,
      col_offsets,
      bias,
      fuse_relu,
      act_times_w_scale,
      thread_id,
      num_threads);
}

template FBGEMM_API void depthwise_3x3x3_per_channel_quantization_pad_1(
    int N,
    int T,
    int H,
    int W,
    int IC_OC,
    int stride_t,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    const int32_t* B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    const float* C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const int32_t* bias,
    bool fuse_relu,
    const float* act_times_w_scale,
    int thread_id,
    int num_threads);

template FBGEMM_API void depthwise_3x3x3_per_channel_quantization_pad_1(
    int N,
    int T,
    int H,
    int W,
    int IC_OC,
    int stride_t,
    int stride_h,
    int stride_w,
    int32_t A_zero_point,
    const uint8_t* A,
    const int32_t* B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    const float* C_multiplier,
    int32_t C_zero_point,
    uint8_t* C,
    const int32_t* col_offsets,
    const float* bias,
    bool fuse_relu,
    const float* act_times_w_scale,
    int thread_id,
    int num_threads);

} // namespace fbgemm
