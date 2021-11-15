/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <gtest/gtest.h>

#include "./TestUtils.h"
#include "bench/AlignedVec.h"
#include "bench/BenchUtils.h"
#include "fbgemm/Fbgemm.h"
#include "src/RefImplementations.h"

using namespace std;
using namespace fbgemm;

vector<QuantizationGranularity> qGranularityVals{
    QuantizationGranularity::TENSOR,
    QuantizationGranularity::GROUP,
    QuantizationGranularity::OUT_CHANNEL};

namespace {
class fbgemmIm2colTest
    : public testing::TestWithParam<tuple<QuantizationGranularity, bool>> {};
}; // namespace

INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    fbgemmIm2colTest,
    ::testing::Combine(
        ::testing::ValuesIn(qGranularityVals),
        ::testing::Bool()));

// clang-format off
// From Faster-RCNN with ShuffleNet
static vector<conv_param_t<>> shapes = {
  // MB, IC, OC, IH, IW, G, KH, KW, stride_h, stride_w, pad_h, pad_w
  conv_param_t<>(1, 32, 32, {14, 14}, 1, {3, 3}, {1, 1}, {0, 0, 0, 0}),
  conv_param_t<>(1, 32, 32, {14, 14}, 1, {3, 3}, {1, 1}, {1, 1, 1, 1}),
  conv_param_t<>(2, 32, 32, {14, 14}, 1, {3, 3}, {1, 1}, {0, 0, 0, 0}),
  conv_param_t<>(2, 32, 32, {28, 14}, 1, {3, 3}, {1, 1}, {1, 1, 0, 0}),
  conv_param_t<>(1, 32, 16, {12, 14}, 4, {3, 3}, {1, 1}, {0, 0, 0, 0}),
  conv_param_t<>(2, 32, 16, {16, 14}, 4, {3, 3}, {1, 1}, {0, 0, 0, 0}),
  conv_param_t<>(1, 544, 544, {14, 14}, 1, {3, 3}, {2, 2}, {1, 1, 1, 1}),
  conv_param_t<>(1, 8, 8, {4, 4}, 1, {3, 3}, {1, 1}, {1, 1, 0, 0}),
  // first layer of resnet50
  conv_param_t<>(1, 3, 64, {224, 224}, 1, {7, 7}, {2, 2}, {3, 3, 3, 3}),
};
// clang-format on

template <typename ACC_T, QuantizationGranularity Q_GRAN>
static void Im2colTest(bool b_symmetric) {
  for (auto conv_p : shapes) {
    for (int groups : {1, 4}) {
      if (conv_p.IC % groups != 0 || conv_p.OC % groups != 0) {
        continue;
      }
      conv_p.G = groups;
      aligned_vector<uint8_t> Aint8(
          conv_p.MB * conv_p.IN_DIM[0] * conv_p.IN_DIM[1] * conv_p.IC);
      aligned_vector<int8_t> Bint8(
          conv_p.K[0] * conv_p.K[1] * conv_p.IC * conv_p.OC);
      aligned_vector<int32_t> Cint32_ref(
          conv_p.MB * conv_p.OUT_DIM[0] * conv_p.OUT_DIM[1] * conv_p.OC);
      aligned_vector<uint8_t> Cint8_ref(Cint32_ref.size());
      aligned_vector<int32_t> Cint32_fb(Cint32_ref.size());
      aligned_vector<uint8_t> Cint8_fb(Cint32_ref.size());

      int ncols_per_quant_group = conv_p.OC;
      if (Q_GRAN == QuantizationGranularity::GROUP) {
        ncols_per_quant_group = conv_p.OC / conv_p.G;
      } else if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
        ncols_per_quant_group = 1;
      }
      int32_t Aint8_zero_point;
      aligned_vector<int32_t> Bint8_zero_point(
          conv_p.OC / ncols_per_quant_group);
      if (is_same<ACC_T, int32_t>::value) {
        randFill<uint8_t>(Aint8, 0, 80);
        Aint8_zero_point = 43;
        randFill<int8_t>(Bint8, -16, 16);
        randFill(Bint8_zero_point, -50, -10);
      } else {
        randFill<uint8_t>(Aint8, 0, 5);
        Aint8_zero_point = 4;
        randFill<int8_t>(Bint8, -4, 4);
        randFill(Bint8_zero_point, -3, -1);
      }
      if (b_symmetric) {
        randFill(Bint8_zero_point, 0, 0);
      }

      aligned_vector<float> C_multiplier(Bint8_zero_point.size());
      randFill(C_multiplier, 0.001234f / 2, 0.001234f * 3 / 2);
      int32_t C_zero_pt = 5;

      int MDim = conv_p.MB * conv_p.OUT_DIM[0] * conv_p.OUT_DIM[1];
      int NDim = conv_p.OC / conv_p.G;
      int KDim = conv_p.K[0] * conv_p.K[1] * conv_p.IC;
      int KDimPerGroup = KDim / conv_p.G;

      // computing row offset
      vector<int32_t> row_offsets(MDim);
      vector<uint8_t> Aint8_im2col(MDim * KDim);
      im2col_ref(conv_p, Aint8.data(), Aint8_zero_point, Aint8_im2col.data());

      // computing column offset
      vector<int32_t> col_offsets(conv_p.G * NDim);
      for (int g = 0; g < conv_p.G; ++g) {
        col_offsets_with_zero_pt_s8acc32_ref(
            KDimPerGroup,
            NDim,
            NDim,
            Bint8.data() + g * KDimPerGroup * NDim,
            Bint8_zero_point.data() + g * NDim / ncols_per_quant_group,
            col_offsets.data() + g * NDim,
            ncols_per_quant_group);
      }

      conv_ref(
          conv_p,
          Aint8.data(),
          Aint8_zero_point,
          Bint8.data(),
          Cint32_ref.data());

      for (int g = 0; g < conv_p.G; ++g) {
        row_offsets_u8acc32_ref(
            MDim,
            KDimPerGroup,
            KDim,
            Aint8_im2col.data() + g * KDimPerGroup,
            row_offsets.data());

        requantize_u8acc32_ref(
            MDim,
            NDim,
            conv_p.G * NDim,
            Cint32_ref.data() + g * NDim,
            Cint8_ref.data() + g * NDim,
            C_multiplier.data() + g * NDim / ncols_per_quant_group,
            C_zero_pt,
            Aint8_zero_point,
            Bint8_zero_point.data() + g * NDim / ncols_per_quant_group,
            row_offsets.data(),
            col_offsets.data() + g * NDim,
            nullptr,
            ncols_per_quant_group);
      }

      PackBMatrix<int8_t, ACC_T> packedB(
          matrix_op_t::NoTranspose,
          KDim,
          NDim,
          Bint8.data(),
          NDim,
          nullptr,
          conv_p.G);

#ifdef _OPENMP
#pragma omp parallel
#endif
      {
        vector<int32_t> row_offset_buf(
            PackAWithIm2Col<uint8_t, ACC_T>::rowOffsetBufferSize());

        PackAWithIm2Col<uint8_t, ACC_T> packA(
            conv_p,
            Aint8.data(),
            nullptr,
            Aint8_zero_point,
            row_offset_buf.data(),
            b_symmetric);

        DoNothing<> doNothingObj{};
        ReQuantizeOutput<false, Q_GRAN> outputProcObj(
            doNothingObj,
            C_multiplier.data(),
            C_zero_pt,
            Aint8_zero_point,
            Bint8_zero_point.data(),
            packA.getRowOffsetBuffer(),
            col_offsets.data(),
            nullptr,
            conv_p.G * NDim,
            conv_p.G);

        int num_threads = fbgemm_get_num_threads();
        int tid = fbgemm_get_thread_num();

        fbgemmPacked(
            packA,
            packedB,
            Cint8_fb.data(),
            Cint32_fb.data(),
            conv_p.G * NDim,
            outputProcObj,
            tid,
            num_threads);
      } // omp parallel

      // correctness check
      for (int n = 0; n < conv_p.MB; ++n) {
        for (int h = 0; h < conv_p.OUT_DIM[0]; ++h) {
          for (int w = 0; w < conv_p.OUT_DIM[1]; ++w) {
            for (int k = 0; k < conv_p.OC; ++k) {
              int32_t expected = Cint8_ref
                  [((n * conv_p.OUT_DIM[0] + h) * conv_p.OUT_DIM[1] + w) *
                       conv_p.OC +
                   k];
              int32_t actual = Cint8_fb
                  [((n * conv_p.OUT_DIM[0] + h) * conv_p.OUT_DIM[1] + w) *
                       conv_p.OC +
                   k];
              EXPECT_EQ(actual, expected)
                  << "Im2Col fused results differ at (" << n << ", " << h
                  << ", " << w << ", " << k << ").";
            }
          }
        }
      }
    } // for each groups
  } // for each shape
}

TEST_P(fbgemmIm2colTest, Acc32Test) {
  QuantizationGranularity q_granularity;
  bool b_symmetric;
  tie(q_granularity, b_symmetric) = GetParam();
  if (q_granularity == QuantizationGranularity::TENSOR) {
    Im2colTest<int32_t, QuantizationGranularity::TENSOR>(b_symmetric);
  } else if (q_granularity == QuantizationGranularity::GROUP) {
    Im2colTest<int32_t, QuantizationGranularity::GROUP>(b_symmetric);
  } else {
    Im2colTest<int32_t, QuantizationGranularity::OUT_CHANNEL>(b_symmetric);
  }
}

TEST_P(fbgemmIm2colTest, Acc16Test) {
  QuantizationGranularity q_granularity;
  bool b_symmetric;
  tie(q_granularity, b_symmetric) = GetParam();
  if (q_granularity == QuantizationGranularity::TENSOR) {
    Im2colTest<int16_t, QuantizationGranularity::TENSOR>(b_symmetric);
  } else if (q_granularity == QuantizationGranularity::GROUP) {
    Im2colTest<int16_t, QuantizationGranularity::GROUP>(b_symmetric);
  } else {
    Im2colTest<int16_t, QuantizationGranularity::OUT_CHANNEL>(b_symmetric);
  }
}

template <QuantizationGranularity Q_GRAN>
void SConvTest() {
  for (auto conv_p : shapes) {
    for (int groups : {1, 4}) {
      if (conv_p.IC % groups != 0 || conv_p.OC % groups != 0) {
        continue;
      }
      conv_p.G = groups;
      aligned_vector<uint8_t> Aint8(
          conv_p.MB * conv_p.IN_DIM[0] * conv_p.IN_DIM[1] * conv_p.IC);
      aligned_vector<int8_t> Bint8(
          conv_p.K[0] * conv_p.K[1] * conv_p.IC * conv_p.OC);
      aligned_vector<int32_t> Cint32_ref(
          conv_p.MB * conv_p.OUT_DIM[0] * conv_p.OUT_DIM[1] * conv_p.OC);
      aligned_vector<uint8_t> Cint8_ref(Cint32_ref.size());
      aligned_vector<int32_t> Cint32_fb(Cint32_ref.size());
      aligned_vector<uint8_t> Cint8_fb(Cint32_ref.size());

      int ncols_per_quant_group = conv_p.OC;
      if (Q_GRAN == QuantizationGranularity::GROUP) {
        ncols_per_quant_group = conv_p.OC / conv_p.G;
      } else if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
        ncols_per_quant_group = 1;
      }
      int32_t Aint8_zero_point;
      aligned_vector<int32_t> Bint8_zero_point(
          conv_p.OC / ncols_per_quant_group);
      randFill<uint8_t>(Aint8, 0, 5);
      Aint8_zero_point = 4;
      randFill<int8_t>(Bint8, -4, 4);
      randFill(Bint8_zero_point, -3, -1);

      aligned_vector<float> C_multiplier(Bint8_zero_point.size());
      randFill(C_multiplier, 0.001234f / 2, 0.001234f * 3 / 2);
      int32_t C_zero_pt = 5;

      int MDim = conv_p.MB * conv_p.OUT_DIM[0] * conv_p.OUT_DIM[1];
      int NDim = conv_p.OC / conv_p.G;
      int KDim = conv_p.K[0] * conv_p.K[1] * conv_p.IC;
      int KDimPerGroup = KDim / conv_p.G;

      // computing row offset
      vector<int32_t> row_offsets(MDim);
      vector<uint8_t> Aint8_im2col(MDim * KDim);
      im2col_ref(conv_p, Aint8.data(), Aint8_zero_point, Aint8_im2col.data());

      // computing column offset
      vector<int32_t> col_offsets(conv_p.G * NDim);
      for (int g = 0; g < conv_p.G; ++g) {
        col_offsets_with_zero_pt_s8acc32_ref(
            KDimPerGroup,
            NDim,
            NDim,
            Bint8.data() + g * KDimPerGroup * NDim,
            Bint8_zero_point.data() + g * NDim / ncols_per_quant_group,
            col_offsets.data() + g * NDim,
            ncols_per_quant_group);
      }

      conv_ref(
          conv_p,
          Aint8.data(),
          Aint8_zero_point,
          Bint8.data(),
          Cint32_ref.data());

      for (int g = 0; g < conv_p.G; ++g) {
        row_offsets_u8acc32_ref(
            MDim,
            KDimPerGroup,
            KDim,
            Aint8_im2col.data() + g * KDimPerGroup,
            row_offsets.data());

        requantize_u8acc32_ref(
            MDim,
            NDim,
            conv_p.G * NDim,
            Cint32_ref.data() + g * NDim,
            Cint8_ref.data() + g * NDim,
            C_multiplier.data() + g * NDim / ncols_per_quant_group,
            C_zero_pt,
            Aint8_zero_point,
            Bint8_zero_point.data() + g * NDim / ncols_per_quant_group,
            row_offsets.data(),
            col_offsets.data() + g * NDim,
            nullptr,
            ncols_per_quant_group);
      }

      float density = 0.0001f;
      CompressedSparseColumn B_csc(KDimPerGroup, conv_p.G * NDim);
      default_random_engine eng;
      binomial_distribution<> per_col_nnz_dist(KDimPerGroup, density);

      // TODO: refactor CSC construction as a reusable function
      vector<int> row_indices(KDimPerGroup);
      int total_nnz = 0;
      int ic_per_group = conv_p.IC / conv_p.G;
      for (int g = 0; g < conv_p.G; ++g) {
        for (int j = 0; j < NDim; ++j) {
          B_csc.ColPtr()[g * NDim + j] = total_nnz;

          int nnz_of_j = per_col_nnz_dist(eng);
          total_nnz += nnz_of_j;

          iota(row_indices.begin(), row_indices.end(), 0);
          shuffle(row_indices.begin(), row_indices.end(), eng);
          sort(row_indices.begin(), row_indices.begin() + nnz_of_j);

          for (int kidx = 0; kidx < nnz_of_j; ++kidx) {
            int rowidx = row_indices[kidx];
            int ic = g * ic_per_group + rowidx % ic_per_group;
            int kw = rowidx / ic_per_group % conv_p.K[1];
            int kh = rowidx / ic_per_group / conv_p.K[1];
            assert(kh < conv_p.K[0]);

            B_csc.KHs().push_back(kh);
            B_csc.KWs().push_back(kw);
            B_csc.ICs().push_back(ic);

            int8_t* bptr = &Bint8[(g * KDimPerGroup + rowidx) * NDim + j];
            B_csc.Values().push_back(*bptr);
            *bptr = 0;
          }
        }
      }
      B_csc.ColPtr()[conv_p.G * NDim] = total_nnz;

      PackBMatrix<int8_t, int16_t> packedB(
          matrix_op_t::NoTranspose,
          KDim,
          NDim,
          Bint8.data(),
          NDim,
          nullptr,
          conv_p.G);

#ifdef _OPENMP
#pragma omp parallel
#endif
      {
        vector<int32_t> row_offset_buf(
            PackAWithIm2Col<uint8_t, int16_t>::rowOffsetBufferSize());

        PackAWithIm2Col<uint8_t, int16_t> packA(
            conv_p,
            Aint8.data(),
            nullptr,
            Aint8_zero_point,
            row_offset_buf.data());

        DoNothing<> doNothingObj{};
        ReQuantizeOutput<false, Q_GRAN> reqObj(
            doNothingObj,
            C_multiplier.data(),
            C_zero_pt,
            Aint8_zero_point,
            Bint8_zero_point.data(),
            packA.getRowOffsetBuffer(),
            col_offsets.data(),
            nullptr,
            conv_p.G * NDim,
            conv_p.G);
        DoSConvOnInpBuffer<
            ReQuantizeOutput<false>::outType,
            int32_t,
            ReQuantizeOutput<false, Q_GRAN>>
            sconvObj(reqObj, Aint8.data(), conv_p, Aint8_zero_point, B_csc);

        int num_threads = fbgemm_get_num_threads();
        int tid = fbgemm_get_thread_num();

        fbgemmPacked(
            packA,
            packedB,
            Cint8_fb.data(),
            Cint32_fb.data(),
            conv_p.G * NDim,
            sconvObj,
            tid,
            num_threads);
      } // omp parallel

      // correctness check
      for (int n = 0; n < conv_p.MB; ++n) {
        for (int h = 0; h < conv_p.OUT_DIM[0]; ++h) {
          for (int w = 0; w < conv_p.OUT_DIM[1]; ++w) {
            for (int k = 0; k < conv_p.OC; ++k) {
              int32_t expected = Cint8_ref
                  [((n * conv_p.OUT_DIM[0] + h) * conv_p.OUT_DIM[1] + w) *
                       conv_p.OC +
                   k];
              int32_t actual = Cint8_fb
                  [((n * conv_p.OUT_DIM[0] + h) * conv_p.OUT_DIM[1] + w) *
                       conv_p.OC +
                   k];
              EXPECT_EQ(actual, expected)
                  << "Im2Col fused results differ at (" << n << ", " << h
                  << ", " << w << ", " << k << ").";
            }
          }
        }
      }
    } // for each groups
  } // for each shape
}

TEST_P(fbgemmIm2colTest, SConvTest) {
  QuantizationGranularity q_granularity;
  bool b_symmetric;
  tie(q_granularity, b_symmetric) = GetParam();
  // b_symmetric ignored for now
  if (q_granularity == QuantizationGranularity::TENSOR) {
    SConvTest<QuantizationGranularity::TENSOR>();
  } else if (q_granularity == QuantizationGranularity::GROUP) {
    SConvTest<QuantizationGranularity::GROUP>();
  } else {
    SConvTest<QuantizationGranularity::OUT_CHANNEL>();
  }
}

static vector<conv_param_t<3>> shapes_3d = {
    // MB, IC, OC, IT, IH, IW, G, KT, KH, KW, stride_t, stride_h, stride_w,
    // pad_t, pad_h, pad_w
    // conv_param_t<
    //     3>(1, 3, 64, {32, 112, 112}, 1, {3, 7, 7}, {1, 2, 2}, {1, 3, 3, 1, 3,
    //     3}),
    // conv_param_t<
    //     3>(1, 64, 64, {32, 56, 56}, 1, {1, 1, 1}, {1, 1, 1}, {0, 0, 0, 0, 0,
    //     0}),
    // conv_param_t<
    //     3>(1, 64, 256, {32, 56, 56}, 1, {1, 1, 1}, {1, 1, 1}, {0, 0, 0, 0, 0,
    //     0}),
    // conv_param_t<
    //     3>(1, 256, 64, {32, 56, 56}, 1, {1, 1, 1}, {1, 1, 1}, {0, 0, 0, 0, 0,
    //     0}),
    // conv_param_t<
    //     3>(1, 256, 128, {32, 56, 56}, 1, {1, 1, 1}, {1, 1, 1}, {0, 0, 0, 0,
    //     0, 0}),
    // conv_param_t<
    //     3>(1, 256, 512, {32, 56, 56}, 1, {1, 1, 1}, {2, 2, 2}, {0, 0, 0, 0,
    //     0, 0}),
    // conv_param_t<
    //     3>(1, 128, 512, {16, 28, 28}, 1, {1, 1, 1}, {1, 1, 1}, {0, 0, 0, 0,
    //     0, 0}),
    // conv_param_t<
    //     3>(1, 512, 128, {16, 28, 28}, 1, {1, 1, 1}, {1, 1, 1}, {0, 0, 0, 0,
    //     0, 0}),
    // conv_param_t<
    //     3>(1, 512, 256, {16, 28, 28}, 1, {1, 1, 1}, {1, 1, 1}, {0, 0, 0, 0,
    //     0, 0}),
    // conv_param_t<
    //     3>(1, 512, 1024, {16, 28, 28}, 1, {1, 1, 1}, {2, 2, 2}, {0, 0, 0, 0,
    //     0, 0}),
    // conv_param_t<
    //     3>(1, 256, 1024, {8, 14, 14}, 1, {1, 1, 1}, {1, 1, 1}, {0, 0, 0, 0,
    //     0, 0}),
    // conv_param_t<
    //     3>(1, 1024, 256, {8, 14, 14}, 1, {1, 1, 1}, {1, 1, 1}, {0, 0, 0, 0,
    //     0, 0}),
    // conv_param_t<
    //     3>(1, 1024, 512, {8, 14, 14}, 1, {1, 1, 1}, {1, 1, 1}, {0, 0, 0, 0,
    //     0, 0}),
    // conv_param_t<
    //     3>(1, 1024, 2048, {8, 14, 14}, 1, {1, 1, 1}, {2, 2, 2}, {0, 0, 0, 0,
    //     0, 0}),
    // conv_param_t<
    //     3>(1, 2048, 512, {8, 14, 14}, 1, {1, 1, 1}, {1, 1, 1}, {0, 0, 0, 0,
    //     0, 0}),
    // conv_param_t<
    //     3>(1, 512, 2048, {4, 7, 7}, 1, {1, 1, 1}, {1, 1, 1}, {0, 0, 0, 0, 0,
    //     0}),
    conv_param_t<3>(
        1,
        3,
        4,
        {32, 112, 112},
        1,
        {3, 7, 7},
        {1, 2, 2},
        {1, 3, 3, 1, 3, 3}),
    conv_param_t<3>(
        1,
        3,
        4,
        {32, 112, 112},
        1,
        {3, 7, 7},
        {1, 2, 2},
        {1, 3, 3, 1, 1, 0}),
    conv_param_t<
        3>(1, 8, 16, {4, 7, 7}, 1, {1, 1, 1}, {1, 1, 1}, {0, 0, 0, 0, 0, 0}),
    conv_param_t<
        3>(1, 8, 16, {8, 14, 14}, 1, {1, 1, 1}, {2, 2, 2}, {0, 0, 0, 0, 0, 0}),
};

template <typename ACC_T, QuantizationGranularity Q_GRAN>
static void Im2col3DTest(bool b_symmetric) {
  for (auto conv_p : shapes_3d) {
    for (int groups : {1, 4}) {
      if (conv_p.IC % groups != 0 || conv_p.OC % groups != 0) {
        continue;
      }
      conv_p.G = groups;
      aligned_vector<uint8_t> Aint8(
          conv_p.MB * conv_p.IN_DIM[0] * conv_p.IN_DIM[1] * conv_p.IN_DIM[2] *
          conv_p.IC);
      aligned_vector<int8_t> Bint8(
          conv_p.K[0] * conv_p.K[1] * conv_p.K[2] * conv_p.IC * conv_p.OC);
      aligned_vector<int32_t> Cint32_ref(
          conv_p.MB * conv_p.OUT_DIM[0] * conv_p.OUT_DIM[1] *
          conv_p.OUT_DIM[2] * conv_p.OC);
      aligned_vector<uint8_t> Cint8_ref(Cint32_ref.size());
      aligned_vector<int32_t> Cint32_fb(Cint32_ref.size());
      aligned_vector<uint8_t> Cint8_fb(Cint32_ref.size());

      int ncols_per_quant_group = conv_p.OC;
      if (Q_GRAN == QuantizationGranularity::GROUP) {
        ncols_per_quant_group = conv_p.OC / conv_p.G;
      } else if (Q_GRAN == QuantizationGranularity::OUT_CHANNEL) {
        ncols_per_quant_group = 1;
      }
      int32_t Aint8_zero_point;
      aligned_vector<int32_t> Bint8_zero_point(
          conv_p.OC / ncols_per_quant_group);
      if (is_same<ACC_T, int32_t>::value) {
        randFill<uint8_t>(Aint8, 0, 80);
        Aint8_zero_point = 43;
        randFill<int8_t>(Bint8, -16, 16);
        randFill(Bint8_zero_point, -50, -10);
      } else {
        randFill<uint8_t>(Aint8, 0, 5);
        Aint8_zero_point = 4;
        randFill<int8_t>(Bint8, -4, 4);
        randFill(Bint8_zero_point, -3, -1);
      }
      if (b_symmetric) {
        randFill(Bint8_zero_point, 0, 0);
      }

      aligned_vector<float> C_multiplier(Bint8_zero_point.size());
      randFill(C_multiplier, 0.001234f / 2, 0.001234f * 3 / 2);
      int32_t C_zero_pt = 5;

      int MDim =
          conv_p.MB * conv_p.OUT_DIM[0] * conv_p.OUT_DIM[1] * conv_p.OUT_DIM[2];
      int NDim = conv_p.OC / conv_p.G;
      int KDim = conv_p.K[0] * conv_p.K[1] * conv_p.K[2] * conv_p.IC;
      int KDimPerGroup = KDim / conv_p.G;

      // computing row offset
      vector<int32_t> row_offsets(MDim);
      vector<uint8_t> Aint8_im2col(MDim * KDim);
      im2col_ref(conv_p, Aint8.data(), Aint8_zero_point, Aint8_im2col.data());

      // computing column offset
      vector<int32_t> col_offsets(conv_p.G * NDim);
      for (int g = 0; g < conv_p.G; ++g) {
        col_offsets_with_zero_pt_s8acc32_ref(
            KDimPerGroup,
            NDim,
            NDim,
            Bint8.data() + g * KDimPerGroup * NDim,
            Bint8_zero_point.data() + g * NDim / ncols_per_quant_group,
            col_offsets.data() + g * NDim,
            ncols_per_quant_group);
      }

      conv_ref(
          conv_p,
          Aint8.data(),
          Aint8_zero_point,
          Bint8.data(),
          Cint32_ref.data());

      for (int g = 0; g < conv_p.G; ++g) {
        row_offsets_u8acc32_ref(
            MDim,
            KDimPerGroup,
            KDim,
            Aint8_im2col.data() + g * KDimPerGroup,
            row_offsets.data());

        requantize_u8acc32_ref(
            MDim,
            NDim,
            conv_p.G * NDim,
            Cint32_ref.data() + g * NDim,
            Cint8_ref.data() + g * NDim,
            C_multiplier.data() + g * NDim / ncols_per_quant_group,
            C_zero_pt,
            Aint8_zero_point,
            Bint8_zero_point.data() + g * NDim / ncols_per_quant_group,
            row_offsets.data(),
            col_offsets.data() + g * NDim,
            nullptr,
            ncols_per_quant_group);
      }

      PackBMatrix<int8_t, ACC_T> packedB(
          matrix_op_t::NoTranspose,
          KDim,
          NDim,
          Bint8.data(),
          NDim,
          nullptr,
          conv_p.G);

#ifdef _OPENMP
#pragma omp parallel
#endif
      {
        vector<int32_t> row_offset_buf(
            PackAWithIm2Col<uint8_t, ACC_T, 3>::rowOffsetBufferSize());

        PackAWithIm2Col<uint8_t, ACC_T, 3> packA(
            conv_p,
            Aint8.data(),
            nullptr,
            Aint8_zero_point,
            row_offset_buf.data(),
            b_symmetric);

        DoNothing<> doNothingObj{};
        ReQuantizeOutput<false, Q_GRAN> outputProcObj(
            doNothingObj,
            C_multiplier.data(),
            C_zero_pt,
            Aint8_zero_point,
            Bint8_zero_point.data(),
            packA.getRowOffsetBuffer(),
            col_offsets.data(),
            nullptr,
            conv_p.G * NDim,
            conv_p.G);

        int num_threads = fbgemm_get_num_threads();
        int tid = fbgemm_get_thread_num();

        fbgemmPacked(
            packA,
            packedB,
            Cint8_fb.data(),
            Cint32_fb.data(),
            conv_p.G * NDim,
            outputProcObj,
            tid,
            num_threads);
      } // omp parallel

      // correctness check
      for (int n = 0; n < conv_p.MB; ++n) {
        for (int t = 0; t < conv_p.OUT_DIM[0]; ++t) {
          for (int h = 0; h < conv_p.OUT_DIM[1]; ++h) {
            for (int w = 0; w < conv_p.OUT_DIM[2]; ++w) {
              for (int k = 0; k < conv_p.OC; ++k) {
                int32_t expected = Cint8_ref
                    [(((n * conv_p.OUT_DIM[0] + t) * conv_p.OUT_DIM[1] + h) *
                          conv_p.OUT_DIM[2] +
                      w) *
                         conv_p.OC +
                     k];
                int32_t actual = Cint8_fb
                    [(((n * conv_p.OUT_DIM[0] + t) * conv_p.OUT_DIM[1] + h) *
                          conv_p.OUT_DIM[2] +
                      w) *
                         conv_p.OC +
                     k];
                EXPECT_EQ(actual, expected)
                    << "Im2Col fused results differ at (" << n << ", " << t
                    << ", " << h << ", " << w << ", " << k << ").";
              }
            }
          }
        }
      }
    } // for each groups
  } // for each shape
}

TEST_P(fbgemmIm2colTest, 3DAcc32Test) {
  QuantizationGranularity q_granularity;
  bool b_symmetric;
  tie(q_granularity, b_symmetric) = GetParam();
  if (q_granularity == QuantizationGranularity::TENSOR) {
    Im2col3DTest<int32_t, QuantizationGranularity::TENSOR>(b_symmetric);
  } else if (q_granularity == QuantizationGranularity::GROUP) {
    Im2col3DTest<int32_t, QuantizationGranularity::GROUP>(b_symmetric);
  } else {
    Im2col3DTest<int32_t, QuantizationGranularity::OUT_CHANNEL>(b_symmetric);
  }
}

TEST_P(fbgemmIm2colTest, 3DAcc16Test) {
  QuantizationGranularity q_granularity;
  bool b_symmetric;
  tie(q_granularity, b_symmetric) = GetParam();
  if (q_granularity == QuantizationGranularity::TENSOR) {
    Im2col3DTest<int16_t, QuantizationGranularity::TENSOR>(b_symmetric);
  } else if (q_granularity == QuantizationGranularity::GROUP) {
    Im2col3DTest<int16_t, QuantizationGranularity::GROUP>(b_symmetric);
  } else {
    Im2col3DTest<int16_t, QuantizationGranularity::OUT_CHANNEL>(b_symmetric);
  }
}
