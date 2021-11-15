/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include "./FbgemmI8DepthwiseAvx2-inl.h"
#include "./GenerateI8Depthwise.h"
#include "./MaskAvx2.h"
#include "fbgemm/Utils.h"
#include "fbgemm/UtilsAvx2.h"

namespace fbgemm {

template <
    int S,
    bool FUSE_RELU,
    bool HAS_BIAS,
    bool A_SYMMETRIC,
    bool B_SYMMETRIC,
    QuantizationGranularity Q_GRAN,
    typename BIAS_TYPE>
static ALWAYS_INLINE void depthwise_2d_kernel_(
    int H,
    int W,
    int IC,
    int OC,
    int h,
    int w,
    int stride_h,
    int stride_w,
    std::int32_t A_zero_point,
    const std::uint8_t* A,
    const std::int32_t* B_zero_point,
    const std::int8_t* Bp,
    const float* C_multiplier,
    std::int32_t C_zero_point,
    std::int32_t* C_int32,
    std::uint8_t* C_uint8,
    std::int32_t* row_offsets,
    const std::int32_t* col_offsets,
    const BIAS_TYPE* bias,
    const float* act_times_w_scale,
    GenI8Depthwise::jit_kernel_signature* pregenerated_kernel = nullptr) {
  constexpr int PAD_T = (S - 1) / 2, PAD_L = (S - 1) / 2, PAD_R = (S - 1) / 2;
  int W_OUT = (W + PAD_L + PAD_R - S) / stride_w + 1;
  int h_in = -PAD_T + h * stride_h;
  int w_in = -PAD_L + w * stride_w;

  int remainder = OC % 32;
  if (remainder == 0) {
    remainder = 32;
  }

  GenI8Depthwise::jit_kernel_signature kernel = pregenerated_kernel
      ? *pregenerated_kernel
      : GenI8Depthwise().getOrCreate(
            /*D=*/2,
            {1, S, S},
            OC / IC,
            /*compute_a_sum=*/!B_SYMMETRIC,
            remainder,
            0,
            0,
            /*top_skip=*/std::max(-h_in, 0),
            /*bottom_skip=*/std::max(h_in + S - H, 0),
            /*left_skip=*/std::max(-w_in, 0),
            /*right_skip=*/std::max(w_in + S - W, 0));

  kernel(
      A + (h_in * W + w_in) * IC,
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
        C_uint8 + (h * W_OUT + w) * OC,
        OC,
        row_offsets,
        col_offsets,
        bias,
        act_times_w_scale);
  } else {
    requantize_<FUSE_RELU, HAS_BIAS, Q_GRAN, A_SYMMETRIC, B_SYMMETRIC, 2>(
        A_zero_point,
        B_zero_point,
        C_multiplier,
        C_zero_point,
        C_int32,
        C_uint8 + (h * W_OUT + w) * OC,
        OC,
        row_offsets,
        col_offsets,
        bias,
        act_times_w_scale);
  }
}

// TODO: short-circuit when B_zero_point is 0 or A_zero_point is 0
// This implemntation should be general enough to handle not just 3x3 but other
// filter shapes by parameterizing with R and S but restricting it to just 3x3
// for now.
template <
    int S,
    bool FUSE_RELU,
    bool HAS_BIAS,
    bool A_SYMMETRIC,
    bool B_SYMMETRIC,
    typename BIAS_TYPE,
    QuantizationGranularity Q_GRAN>
static ALWAYS_INLINE void depthwise_2d_(
    int N,
    int H,
    int W,
    int IC,
    int OC,
    int stride_h,
    int stride_w,
    std::int32_t A_zero_point,
    const std::uint8_t* A,
    const std::int32_t* B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    const float* C_multiplier,
    std::int32_t C_zero_point,
    std::int32_t* C_int32,
    std::uint8_t* C_uint8,
    const std::int32_t* col_offsets,
    const BIAS_TYPE* bias,
    const float* act_times_w_scale,
    int thread_id,
    int num_threads) {
  assert(IC % 8 == 0);
  constexpr int R = S;
  constexpr int PAD_T = (R - 1) / 2, PAD_B = PAD_T, PAD_L = (S - 1) / 2,
                PAD_R = PAD_L;
  int H_OUT = (H + PAD_T + PAD_B - R) / stride_h + 1;
  int W_OUT = (W + PAD_L + PAD_R - S) / stride_w + 1;
  const std::int8_t* Bp = B.PackedMat();

  int32_t* row_offsets = static_cast<int32_t*>(
      fbgemmAlignedAlloc(64, (IC + 31) / 32 * 32 * sizeof(int32_t)));

  int n_begin, n_end, h_begin, h_end, w_begin, w_end;
  // Reuse the 3-dim partition scheme for parallelization in matrix
  // multiplication.
  thread_type_t th_info =
      fbgemmGetThreadPartition(N, H_OUT, W_OUT, thread_id, num_threads);
  // Calculate the begin and end index along the batch (N) dimension
  fbgemmPartition1D(
      th_info.g_thread_id, th_info.g_num_threads, N, n_begin, n_end);
  // Calculate the begin and end index along the H dimension
  fbgemmPartition1D(
      th_info.m_thread_id, th_info.m_num_threads, H_OUT, h_begin, h_end);
  // Calculate the begin and end index along the W dimension
  fbgemmPartition1D(
      th_info.n_thread_id, th_info.n_num_threads, W_OUT, w_begin, w_end);

  GenI8Depthwise::jit_kernel_signature middle_kernel;

  for (int n = n_begin; n < n_end; ++n) {
    const std::uint8_t* A_base = A + n * H * W * IC;
    std::uint8_t* C_uint8_base = C_uint8 + n * H_OUT * W_OUT * OC;

    int h = 0;
    int w = 0;

    for (h = h_begin; h < std::min(PAD_T, h_end); ++h) {
      for (w = w_begin; w < std::min(PAD_L, w_end); ++w) {
        depthwise_2d_kernel_<
            S,
            FUSE_RELU,
            HAS_BIAS,
            A_SYMMETRIC,
            B_SYMMETRIC,
            Q_GRAN,
            BIAS_TYPE>(
            H,
            W,
            IC,
            OC,
            h,
            w,
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

      for (; w < std::min(W_OUT - PAD_R - stride_w + 1, w_end); ++w) {
        depthwise_2d_kernel_<
            S,
            FUSE_RELU,
            HAS_BIAS,
            A_SYMMETRIC,
            B_SYMMETRIC,
            Q_GRAN,
            BIAS_TYPE>(
            H,
            W,
            IC,
            OC,
            h,
            w,
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

      for (; w < w_end; ++w) {
        depthwise_2d_kernel_<
            S,
            FUSE_RELU,
            HAS_BIAS,
            A_SYMMETRIC,
            B_SYMMETRIC,
            Q_GRAN,
            BIAS_TYPE>(
            H,
            W,
            IC,
            OC,
            h,
            w,
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
    }

    // h <= H_OUT - PAD_B - stride_h
    // h <= (H + PAD_T + PAD_B - S) / stride_h + 1 - PAD_B - stride_h
    // h_in <= -PAD_T +
    // ((H + PAD_T + PAD_B - S) / stride_h + 1 - PAD_B - stride_h) * stride_h
    // Case 1) For stride_h == 1,
    // h_in <= -PAD_T + H + PAD_T + PAD_B - S + 1 - PAD_B - 1
    // h_in + S - H <= 0
    // Case 2) For stride_h == 2,
    // h_in <= -PAD_L +
    // H + PAD_T + PAD_B - S + 1 + (1 - PAD_B - stride_h) * stride_h
    // h_in + S - H <= PAD_B * (1 - stride_h) + 1 + (1 - stride_h) * stride_h
    //              <= -PAD_B + 1 - stride_h <= 0
    for (; h < std::min(H_OUT - PAD_B - stride_h + 1, h_end); ++h) {
      for (w = w_begin; w < std::min(PAD_L, w_end); ++w) {
        depthwise_2d_kernel_<
            S,
            FUSE_RELU,
            HAS_BIAS,
            A_SYMMETRIC,
            B_SYMMETRIC,
            Q_GRAN,
            BIAS_TYPE>(
            H,
            W,
            IC,
            OC,
            h,
            w,
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

      for (; w < std::min(W_OUT - PAD_R - stride_w + 1, w_end); ++w) {
        if (n == n_begin && w == std::max(PAD_L, w_begin)) {
          int remainder = OC % 32;
          if (remainder == 0) {
            remainder = 32;
          }
          middle_kernel = GenI8Depthwise().getOrCreate(
              /*D=*/2,
              {1, S, S},
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
        depthwise_2d_kernel_<
            S,
            FUSE_RELU,
            HAS_BIAS,
            A_SYMMETRIC,
            B_SYMMETRIC,
            Q_GRAN,
            BIAS_TYPE>(
            H,
            W,
            IC,
            OC,
            h,
            w,
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

      for (; w < w_end; ++w) {
        depthwise_2d_kernel_<
            S,
            FUSE_RELU,
            HAS_BIAS,
            A_SYMMETRIC,
            B_SYMMETRIC,
            Q_GRAN,
            BIAS_TYPE>(
            H,
            W,
            IC,
            OC,
            h,
            w,
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
    }

    for (; h < h_end; ++h) {
      for (w = w_begin; w < std::min(PAD_L, w_end); ++w) {
        depthwise_2d_kernel_<
            S,
            FUSE_RELU,
            HAS_BIAS,
            A_SYMMETRIC,
            B_SYMMETRIC,
            Q_GRAN,
            BIAS_TYPE>(
            H,
            W,
            IC,
            OC,
            h,
            w,
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

      for (; w < std::min(W_OUT - PAD_R - stride_w + 1, w_end); ++w) {
        depthwise_2d_kernel_<
            S,
            FUSE_RELU,
            HAS_BIAS,
            A_SYMMETRIC,
            B_SYMMETRIC,
            Q_GRAN,
            BIAS_TYPE>(
            H,
            W,
            IC,
            OC,
            h,
            w,
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

      for (; w < w_end; ++w) {
        depthwise_2d_kernel_<
            S,
            FUSE_RELU,
            HAS_BIAS,
            A_SYMMETRIC,
            B_SYMMETRIC,
            Q_GRAN,
            BIAS_TYPE>(
            H,
            W,
            IC,
            OC,
            h,
            w,
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
    }
  } // for each n

  fbgemmAlignedFree(row_offsets);
}

// Dispatch A_SYMMETRIC and B_SYMMETRIC
template <
    int S,
    bool FUSE_RELU,
    bool HAS_BIAS,
    typename BIAS_TYPE,
    QuantizationGranularity Q_GRAN>
static void depthwise_2d_(
    int N,
    int H,
    int W,
    int IC,
    int OC,
    int stride_h,
    int stride_w,
    std::int32_t A_zero_point,
    const std::uint8_t* A,
    const std::int32_t* B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    const float* C_multiplier,
    std::int32_t C_zero_point,
    std::uint8_t* C,
    const std::int32_t* col_offsets,
    const BIAS_TYPE* bias,
    const float* act_times_w_scale,
    int thread_id,
    int num_threads) {
  int32_t* C_int32_temp = static_cast<int32_t*>(
      fbgemmAlignedAlloc(64, (OC + 31) / 32 * 32 * sizeof(int32_t)));
  if (A_zero_point == 0 || col_offsets == nullptr) {
    if (Q_GRAN == QuantizationGranularity::TENSOR && B_zero_point[0] == 0) {
      depthwise_2d_<
          S,
          FUSE_RELU,
          HAS_BIAS,
          true /*A_symmetric*/,
          true /*B_symmetric*/,
          BIAS_TYPE,
          Q_GRAN>(
          N,
          H,
          W,
          IC,
          OC,
          stride_h,
          stride_w,
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
      depthwise_2d_<
          S,
          FUSE_RELU,
          HAS_BIAS,
          true /*A_symmetric*/,
          false /*B_symmetric*/,
          BIAS_TYPE,
          Q_GRAN>(
          N,
          H,
          W,
          IC,
          OC,
          stride_h,
          stride_w,
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
      depthwise_2d_<
          S,
          FUSE_RELU,
          HAS_BIAS,
          false /*A_symmetric*/,
          true /*B_symmetric*/,
          BIAS_TYPE,
          Q_GRAN>(
          N,
          H,
          W,
          IC,
          OC,
          stride_h,
          stride_w,
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
      depthwise_2d_<
          S,
          FUSE_RELU,
          HAS_BIAS,
          false /*A_symmetric*/,
          false /*B_symmetric*/,
          BIAS_TYPE,
          Q_GRAN>(
          N,
          H,
          W,
          IC,
          OC,
          stride_h,
          stride_w,
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
template <
    int S,
    bool FUSE_RELU,
    typename BIAS_TYPE,
    QuantizationGranularity Q_GRAN>
static void depthwise_2d_(
    int N,
    int H,
    int W,
    int IC,
    int OC,
    int stride_h,
    int stride_w,
    std::int32_t A_zero_point,
    const std::uint8_t* A,
    const std::int32_t* B_zero_point,
    const PackedDepthWiseConvMatrix& B,
    const float* C_multiplier,
    std::int32_t C_zero_point,
    std::uint8_t* C,
    const std::int32_t* col_offsets,
    const BIAS_TYPE* bias,
    const float* act_times_w_scale,
    int thread_id,
    int num_threads) {
  if (bias) {
    depthwise_2d_<S, FUSE_RELU, true /*HAS_BIAS*/, BIAS_TYPE, Q_GRAN>(
        N,
        H,
        W,
        IC,
        OC,
        stride_h,
        stride_w,
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
    depthwise_2d_<S, FUSE_RELU, false /*HAS_BIAS*/, BIAS_TYPE, Q_GRAN>(
        N,
        H,
        W,
        IC,
        OC,
        stride_h,
        stride_w,
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

} // namespace fbgemm
