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

  // From Xray OCR
  // clang-format off
  vector<vector<int>> shapes = {
    // NOTE: clang-format wants to use a different formatting but the current
    // formatting should be easier to read.
    // N,  G, K_per_G, H_in, W_in, stride, kernel
    {   1,  272, 1,  47, 125, 1, 3, },
    {   1,  272, 1,  47, 125, 1, 5, },
    {   1,  272, 1,  64, 125, 1, 3, },
    {   1,  272, 1,  66, 125, 1, 3, },
    {   1,  272, 1,  67, 100, 1, 3, },
    {   1,  272, 1,  71, 125, 1, 3, },
    {   1,  272, 1,  74, 125, 1, 3, },
    {   1,  272, 1,  75,  75, 1, 3, },
    {   1,  272, 1,  75,  76, 1, 3, },
    {   1,  272, 1,  75,  79, 1, 3, },
    {   1,  272, 1,  75,  85, 1, 3, },
    {   1,  272, 1,  75, 100, 1, 3, },
    {   1,  272, 1,  75, 103, 1, 3, },
    {   1,  272, 1,  75, 111, 1, 3, },
    {   1,  272, 1,  75, 113, 1, 3, },
    {   1,  272, 1,  94,  75, 1, 3, },
    {   1,  272, 1, 109,  75, 1, 3, },
    {   1,  272, 1, 113,  75, 1, 3, },
    {   1,  272, 1, 117,  75, 1, 3, },
    {   1,  544, 1,  24,  63, 1, 3, },
    {   1,  544, 1,  32,  63, 1, 3, },
    {   1,  544, 1,  33,  63, 1, 3, },
    {   1,  544, 1,  34,  50, 1, 3, },
    {   1,  544, 1,  36,  63, 1, 3, },
    {   1,  544, 1,  37,  63, 1, 3, },
    {   1,  544, 1,  38,  38, 1, 3, },
    {   1,  544, 1,  38,  40, 1, 3, },
    {   1,  544, 1,  38,  43, 1, 3, },
    {   1,  544, 1,  38,  50, 1, 3, },
    {   1,  544, 1,  38,  52, 1, 3, },
    {   1,  544, 1,  38,  56, 1, 3, },
    {   1,  544, 1,  38,  57, 1, 3, },
    {   1,  544, 1,  47,  38, 1, 3, },
    {   1,  544, 1,  55,  38, 1, 3, },
    {   1,  544, 1,  57,  38, 1, 3, },
    {   1,  544, 1,  59,  38, 1, 3, },
    {   1, 1088, 1,   7,   7, 1, 3, },
    {  51, 1088, 1,   7,   7, 1, 3, },
    {  59, 1088, 1,   7,   7, 1, 3, },
    {  70, 1088, 1,   7,   7, 1, 3, },
    {  71, 1088, 1,   7,   7, 1, 3, },
    {  77, 1088, 1,   7,   7, 1, 3, },
    {  79, 1088, 1,   7,   7, 1, 3, },
    {  84, 1088, 1,   7,   7, 1, 3, },
    {  85, 1088, 1,   7,   7, 1, 3, },
    {  89, 1088, 1,   7,   7, 1, 3, },
    {  93, 1088, 1,   7,   7, 1, 3, },
    {  96, 1088, 1,   7,   7, 1, 3, },
    { 100, 1088, 1,   7,   7, 1, 3, },

    {   1,  248, 1,  93, 250, 2, 3, },
    {   1,  248, 1, 128, 250, 2, 3, },
    {   1,  248, 1, 132, 250, 2, 3, },
    {   1,  248, 1, 131, 250, 2, 3, },
    {   1,  248, 1, 133, 200, 2, 3, },
    {   1,  248, 1, 141, 250, 2, 3, },
    {   1,  248, 1, 148, 250, 2, 3, },
    {   1,  248, 1, 150, 150, 2, 3, },
    {   1,  248, 1, 150, 151, 2, 3, },
    {   1,  248, 1, 150, 158, 2, 3, },
    {   1,  248, 1, 150, 169, 2, 3, },
    {   1,  248, 1, 150, 200, 2, 3, },
    {   1,  248, 1, 150, 205, 2, 3, },
    {   1,  248, 1, 150, 221, 2, 3, },
    {   1,  248, 1, 150, 225, 2, 3, },
    {   1,  248, 1, 188, 150, 2, 3, },
    {   1,  248, 1, 218, 150, 2, 3, },
    {   1,  248, 1, 225, 150, 2, 3, },
    {   1,  248, 1, 234, 150, 2, 3, },
    {   1,  272, 1,  47, 125, 2, 3, },
    {   1,  272, 1,  64, 125, 2, 3, },
    {   1,  272, 1,  66, 125, 2, 3, },
    {   1,  272, 1,  67, 100, 2, 3, },
    {   1,  272, 1,  71, 125, 2, 3, },
    {   1,  272, 1,  74, 125, 2, 3, },
    {   1,  272, 1,  75,  75, 2, 3, },
    {   1,  272, 1,  75,  76, 2, 3, },
    {   1,  272, 1,  75,  79, 2, 3, },
    {   1,  272, 1,  75,  85, 2, 3, },
    {   1,  272, 1,  75, 100, 2, 3, },
    {   1,  272, 1,  75, 103, 2, 3, },
    {   1,  272, 1,  75, 111, 2, 3, },
    {   1,  272, 1,  75, 113, 2, 3, },
    {   1,  272, 1,  94,  75, 2, 3, },
    {   1,  272, 1, 109,  75, 2, 3, },
    {   1,  272, 1, 113,  75, 2, 3, },
    {   1,  272, 1, 117,  75, 2, 3, },
    {   1,  544, 1,  14,  14, 2, 3, },
    {  51,  544, 1,  14,  14, 2, 3, },
    {  59,  544, 1,  14,  14, 2, 3, },
    {  70,  544, 1,  14,  14, 2, 3, },
    {  71,  544, 1,  14,  14, 2, 3, },
    {  77,  544, 1,  14,  14, 2, 3, },
    {  79,  544, 1,  14,  14, 2, 3, },
    {  84,  544, 1,  14,  14, 2, 3, },
    {  85,  544, 1,  14,  14, 2, 3, },
    {  89,  544, 1,  14,  14, 2, 3, },
    {  93,  544, 1,  14,  14, 2, 3, },
    {  96,  544, 1,  14,  14, 2, 3, },
    { 100,  544, 1,  14,  14, 2, 3, },

    {   1,   16, 1, 112, 112, 1, 3, },
    {   1,   24, 1,  56,  56, 1, 3, },
    {   1,   96, 1, 112, 112, 2, 3, },
    {   1,  192, 1,  28,  28, 1, 3, },
    {   1,   96, 1,  28,  28, 1, 5, },
    {   1,  144, 1,  56,  56, 2, 5, },
    {   1,  192, 1,  28,  28, 1, 5, },
    {   1,  192, 1,  28,  28, 2, 5, },
    {   1,  192, 1,  14,  14, 1, 5, },
    {   1,  336, 1,  14,  14, 1, 5, },
    {   1,  384, 1,  14,  14, 1, 5, },
    {   1,  672, 1,  14,  14, 1, 5, },
    {   1,  672, 1,  14,  14, 2, 5, },
    {   1, 1104, 1,   7,   7, 1, 5, },

    {   1,   32, 1, 112, 112, 1, 3, },
    {   1,  144, 1,  56,  56, 1, 3, },
    {   1,  240, 1,  28,  28, 2, 3, },
    {   1,  480, 1,  14,  14, 1, 3, },
    {   1, 1152, 1,   7,   7, 1, 3, },
    {   1,  240, 1,  28,  28, 1, 5, },
    {   1,  480, 1,  14,  14, 1, 5, },
    {   1,  576, 1,  14,  14, 1, 5, },
    {   1,  768, 1,  14,  14, 2, 5, },
    {   1, 1104, 1,   7,   7, 1, 3, },
    {   1, 1152, 1,   7,   7, 1, 5, },

    {   1,   32, 1, 400, 400, 1, 3, },
    {   1,   96, 1, 400, 400, 2, 3, },
    {   1,  144, 1, 200, 200, 1, 3, },
    {   1,  144, 1, 200, 200, 2, 3, },
    {   1,  192, 1, 100, 100, 1, 3, },
    {   1,  192, 1, 100, 100, 2, 3, },
    {   1,  384, 1,  50,  50, 1, 3, },
    {   1,  576, 1,  50,  50, 1, 3, },

    // 2 output channels per group
    {   1,  128, 2,  32, 100, 1, 3, },
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

  for (auto shape : shapes) {
    int N = shape[0];
    int G = shape[1];
    int OC_PER_G = shape[2];
    int H = shape[3];
    int W = shape[4];
    int stride_h = shape[5];
    int stride_w = stride_h;
    int R = shape[6];
    int S = R;
    int PAD_T = (R - 1) / 2, PAD_B = (R - 1) / 2, PAD_L = (S - 1) / 2,
        PAD_R = (S - 1) / 2;
    int OC = G * OC_PER_G;

    conv_param_t<2> conv_p(
        N,
        G,
        OC,
        {H, W},
        G,
        {R, S},
        {stride_h, stride_w},
        {PAD_T, PAD_L, PAD_B, PAD_R});
    int H_OUT = conv_p.OUT_DIM[0];
    int W_OUT = conv_p.OUT_DIM[1];

    int MDim = N * H_OUT * W_OUT;
    int KDim = R * S * G;
    int KDimPerGroup = KDim / conv_p.G;

    aligned_vector<uint8_t> A(N * H * W * G);
    aligned_vector<int8_t> B(KDim * OC_PER_G);
    aligned_vector<int8_t> B_tr(B.size());
    aligned_vector<int32_t> C_ref(MDim * OC), C(C_ref.size());
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

    aligned_vector<int32_t> col_offsets(OC);
    aligned_vector<int32_t> bias(OC);
    randFill(col_offsets, -100, 100);
    randFill(bias, -40, 40);

    // reference implementation conv_ref expects weights to be in G (R S C/G)
    // K/G
    transposeConvWeights(conv_p, B.data(), B_tr.data());
    conv_ref(conv_p, A.data(), A_zero_point, B_tr.data(), C_ref.data());

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
          OC_PER_G,
          OC,
          C_ref.data() + g * OC_PER_G,
          C_uint8_ref.data() + g * OC_PER_G,
          C_multiplier.data(),
          C_zero_point,
          A_zero_point,
          &B_zero_point,
          row_offsets.data(),
          col_offsets.data() + g * OC_PER_G,
          bias.data() + g * OC_PER_G,
          OC);
    }

    PackedDepthWiseConvMatrix Bp(OC, R * S, B.data());

    double bytes = G *
        (N * (2. * sizeof(int32_t) * H_OUT * W_OUT * OC_PER_G + H * W) +
         R * S * OC_PER_G);
    double ops = 2.0 * N * H_OUT * W_OUT * OC * R * S;

    double ttot = measureWithWarmup(
        [&]() {
          int num_threads = fbgemm_get_num_threads();
          int tid = fbgemm_get_thread_num();

          depthwise_2d_same_pad<QuantizationGranularity::TENSOR>(
              N,
              H,
              W,
              G,
              OC,
              stride_h,
              stride_w,
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
      for (int h = 0; h < H_OUT; ++h) {
        for (int w = 0; w < W_OUT; ++w) {
          for (int k = 0; k < OC; ++k) {
            uint8_t expected =
                C_uint8_ref[((n * H_OUT + h) * W_OUT + w) * OC + k];
            uint8_t actual = C_uint8[((n * H_OUT + h) * W_OUT + w) * OC + k];
            if (expected != actual) {
              cerr << "Depthwise 3x3 results differ at (" << n << ", " << h
                   << ", " << w << ", " << k << "). expected " << (int)expected
                   << " actual " << (int)actual << endl;
              return -1;
            }
            assert(expected == actual);
          }
        }
      }
    }

    // Report performance
    printf(
        "N = %d G = %d OC = %d H = %d W = %d stride = %d R = %d\n",
        N,
        G,
        OC,
        H,
        W,
        stride_h,
        R);
    printf("GB/s = %f Gops/s = %f\n", bytes / ttot / 1e9, ops / ttot / 1e9);
  } // for each shape

  return 0;
}
