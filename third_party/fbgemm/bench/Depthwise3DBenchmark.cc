/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "./AlignedVec.h"
#include "./BenchUtils.h"
#include "fbgemm/FbgemmI8DepthwiseAvx2.h"
#include "fbgemm/Utils.h"
#include "src/RefImplementations.h"

using namespace std;
using namespace fbgemm;

int main() {
#ifdef _OPENMP
  // Use 1 thread unless OMP_NUM_THREADS is explicit set.
  const char* val = getenv("OMP_NUM_THREADS");
  if (val == nullptr || !*val) {
    omp_set_num_threads(1);
  }
#endif

  // From ResNeXt-3D-101
  // clang-format off
  vector<vector<int>> shapes_3d = {
    // NOTE: clang-format wants to use a different formatting but the current
    // formatting should be easier to read.
    // N, K, T_in, H_in, W_in, stride
    {   1,  64,   32,  56, 56, 1, },
    {   1, 128,   16,  28, 28, 1, },
    {   1, 256,    8,  14, 14, 1, },
    {   1, 512,    4,   7,  7, 1, },

    {   1, 128,   32,  56, 56, 2, },
    {   1, 256,   16,  28, 28, 2, },
    {   1, 512,    8,  14, 14, 2, },

    {   5,  64,   32,  56, 56, 1, },
    {   5, 128,   16,  28, 28, 1, },
    {   5, 256,    8,  14, 14, 1, },
    {   5, 512,    4,   7,  7, 1, },

    {   5, 128,   32,  56, 56, 2, },
    {   5, 256,   16,  28, 28, 2, },
    {   5, 512,    8,  14, 14, 2, },

    {  32,  24,    4,  56, 56, 1, },
    {  32,  24,    2,  28, 28, 1, },
    {  32,  48,    4,  56, 56, 1, },
    {  32,  48,    2,  28, 28, 1, },
    {  32,  48,    1,  14, 14, 1, },

    {   1,   8,    4,   4,  4, 1, },
  };
  // clang-format on

  // Depthwise is memory BW bound so we want to flush LLC.
  bool flush = true;
  std::vector<char> llc;
  if (flush) {
    llc.resize(128 * 1024 * 1024, 1.0);
  }

  constexpr int NWARMUP = 4;
  constexpr int NITER = 16;

  for (auto shape : shapes_3d) {
    int N = shape[0];
    int K = shape[1];
    int T = shape[2];
    int H = shape[3];
    int W = shape[4];
    int stride_t = shape[5];
    int stride_h = stride_t;
    int stride_w = stride_t;
    constexpr int K_T = 3, K_H = 3, K_W = 3;
    constexpr int PAD_P = 1, PAD_N = 1, PAD_T = 1, PAD_B = 1, PAD_L = 1,
                  PAD_R = 1;

    conv_param_t<3> conv_p(
        N,
        K,
        K,
        {T, H, W},
        K,
        {K_T, K_H, K_W},
        {stride_t, stride_h, stride_w},
        {PAD_P, PAD_T, PAD_L, PAD_N, PAD_B, PAD_R});
    int T_OUT = conv_p.OUT_DIM[0];
    int H_OUT = conv_p.OUT_DIM[1];
    int W_OUT = conv_p.OUT_DIM[2];

    int MDim = N * T_OUT * H_OUT * W_OUT;
    int KDim = K_T * K_H * K_W * K;
    int KDimPerGroup = KDim / conv_p.G;

    aligned_vector<uint8_t> A(N * T * H * W * K);
    aligned_vector<int8_t> B(KDim);
    aligned_vector<int32_t> C_ref(MDim * K), C(C_ref.size());
    aligned_vector<uint8_t> C_uint8_ref(C_ref.size()), C_uint8(C_ref.size());

    randFill<uint8_t>(A, 0, 86);
    int32_t A_zero_point = 43;

    randFill<int8_t>(B, -16, 16);
    int32_t B_zero_point = 5;

    aligned_vector<float> C_multiplier(1);
    randFill(C_multiplier, 0.001234f / 2, 0.001234f * 3 / 2);
    int32_t C_zero_point = 5;

    vector<int32_t> row_offsets(MDim);
    // im2col to compute row offset later
    vector<uint8_t> A_im2col(MDim * KDim);
    im2col_ref(conv_p, A.data(), A_zero_point, A_im2col.data());

    aligned_vector<int32_t> col_offsets(K);
    aligned_vector<int32_t> bias(K);
    randFill(col_offsets, -100, 100);
    randFill(bias, -40, 40);

    conv_ref(conv_p, A.data(), A_zero_point, B.data(), C_ref.data());

    for (int g = 0; g < conv_p.G; ++g) {
      // Compute row offset
      row_offsets_u8acc32_ref(
          MDim,
          KDimPerGroup,
          KDim,
          A_im2col.data() + g * KDimPerGroup,
          row_offsets.data());

      // Requantization
      requantize_u8acc32_ref(
          MDim,
          1,
          conv_p.G,
          C_ref.data() + g,
          C_uint8_ref.data() + g,
          C_multiplier.data(),
          C_zero_point,
          A_zero_point,
          &B_zero_point,
          row_offsets.data(),
          col_offsets.data() + g,
          bias.data() + g,
          K);
    }

    PackedDepthWiseConvMatrix Bp(K, 3 * 3 * 3, B.data());

    double bytes =
        (K *
         (N * (2.0 * sizeof(int32_t) * T_OUT * H_OUT * W_OUT + T * H * W) +
          K_T * K_H * K_W));
    double ops = 2.0 * N * T_OUT * H_OUT * W_OUT * K * K_T * K_H * K_W;

    double ttot = measureWithWarmup(
        [&]() {
          int num_threads = fbgemm_get_num_threads();
          int tid = fbgemm_get_thread_num();
          depthwise_3d_same_pad<QuantizationGranularity::TENSOR>(
              conv_p,
              A_zero_point,
              A.data(),
              &B_zero_point,
              Bp,
              C_multiplier.data(),
              C_zero_point,
              C_uint8.data(),
              col_offsets.data(),
              bias.data(),
              false, /* fuse_relu */
              nullptr, /* act_scale * w_scale */
              tid,
              num_threads);
        },
        NWARMUP,
        NITER,
        [&]() {
          if (flush) {
            llc_flush(llc);
          }
        },
        true /*useOpenMP*/);

    // correctness check
    for (int n = 0; n < N; ++n) {
      for (int t = 0; t < T_OUT; ++t) {
        for (int h = 0; h < H_OUT; ++h) {
          for (int w = 0; w < W_OUT; ++w) {
            for (int g = 0; g < K; ++g) {
              uint8_t expected = C_uint8_ref
                  [(((n * T_OUT + t) * H_OUT + h) * W_OUT + w) * K + g];
              uint8_t actual =
                  C_uint8[(((n * T_OUT + t) * H_OUT + h) * W_OUT + w) * K + g];
              if (expected != actual) {
                cerr << "Depthwise 3x3x3 results differ at (" << n << ", " << t
                     << ", " << h << ", " << w << ", " << g << "). expected "
                     << (int)expected << " actual " << (int)actual << endl;
                return -1;
              }
              assert(expected == actual);
            }
          } // w
        } // h
      } // t
    } // n

    // Report performance
    printf(
        "N = %d K = %d T = %d H = %d W = %d stride = %d with requantization "
        "fused\n",
        N,
        K,
        T,
        H,
        W,
        stride_h);
    printf("GB/s = %f Gops/s = %f\n", bytes / ttot / 1e9, ops / ttot / 1e9);
  } // for each shape

  return 0;
}
