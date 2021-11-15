/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <gtest/gtest.h>

#include "./QuantizationHelpers.h"
#include "./TestUtils.h"
#include "bench/BenchUtils.h"
#include "fbgemm/Fbgemm.h"
#include "src/RefImplementations.h"

using namespace std;
using namespace fbgemm;

vector<matrix_op_t> transposeVals{matrix_op_t::NoTranspose,
                                  matrix_op_t::Transpose};

vector<QuantizationGranularity> qGranularityVals{
    QuantizationGranularity::TENSOR,
    QuantizationGranularity::GROUP,
    QuantizationGranularity::OUT_CHANNEL};

namespace {
class fbgemmu8s8acc32WithQuantGranularityTest
    : public testing::TestWithParam<
          tuple<matrix_op_t, matrix_op_t, bool, QuantizationGranularity>> {};
class fbgemmu8s8acc32Test
    : public testing::TestWithParam<tuple<matrix_op_t, matrix_op_t, bool>> {};
class fbgemmPackUnpackAcc32Test
    : public testing::TestWithParam<tuple<matrix_op_t, bool>> {};
}; // namespace

INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    fbgemmu8s8acc32WithQuantGranularityTest,
    ::testing::Combine(
        ::testing::ValuesIn(transposeVals),
        ::testing::ValuesIn(transposeVals),
        ::testing::Bool(),
        ::testing::ValuesIn(qGranularityVals)));

INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    fbgemmu8s8acc32Test,
    ::testing::Combine(
        ::testing::ValuesIn(transposeVals),
        ::testing::ValuesIn(transposeVals),
        ::testing::Bool()));

INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    fbgemmPackUnpackAcc32Test,
    ::testing::Combine(::testing::ValuesIn(transposeVals), ::testing::Bool()));

/**
 * @brief Shapes for unit test.
 */
static vector<vector<int>> GetShapes_() {
  // NMT
  // clang-format off
  vector<vector<int>> shapes = {
    // {M,    N,    K}
    {1, 128, 512},
    // {1, 1024, 256},
    // {1, 2048, 512},
    {1, 2048, 513},
    // {1, 2048, 514},

    {6, 512, 512},
    // {6, 2048, 512},
    // {6, 256, 1024},
    // {6, 1024, 256},
    {6, 2048, 256},
    {6, 2048, 257},
    // {6, 2048, 258},

    // {102, 1024, 512},
    {102, 2323, 256},
    // {102, 512, 256},
    {102, 512, 257},
    // {102, 512, 258},

    // {1024, 512, 258},

    {120, 4, 288},
  };
  // clang-format on
  return shapes;
}

/**
 * @brief Unit test for uint8 matrix A, int8 matrix B, and 32-bit
 * accumulation. Output processing: requantization -> nothing
 */
TEST_P(fbgemmu8s8acc32WithQuantGranularityTest, Test) {
  vector<vector<int>> shapes(GetShapes_());
  matrix_op_t atrans, btrans;
  bool test_ld;
  QuantizationGranularity q_granularity;
  tie(atrans, btrans, test_ld, q_granularity) = GetParam();

  for (auto shape : shapes) {
    for (int groups : {1, 3, 4}) {
      for (bool test_bias : {false, true}) {
        int m = shape[0];
        int n = shape[1];
        int k = shape[2];
        if (k % groups != 0) {
          continue;
        }
        int k_per_group = k / groups;

        // mxk matrix
        aligned_vector<uint8_t> Aint8(m * k);

        // kxn matrix
        aligned_vector<int8_t> Bint8_ref(k * n);

        aligned_vector<int32_t> Cint32_ref(m * n * groups);
        aligned_vector<uint8_t> Cint8_ref(Cint32_ref.size());
        aligned_vector<int32_t> Cint32_fb(Cint32_ref.size());
        aligned_vector<uint8_t> Cint8_fb(Cint32_ref.size());
        aligned_vector<int32_t> Cint32_buffer(Cint32_ref.size());

        randFill<uint8_t>(Aint8, 0, 255);
        int32_t Aint8_zero_point = 43;

        randFill<int8_t>(Bint8_ref, -128, 127);
        for (int g = 0; g < groups; ++g) {
          avoidOverflow(
              m,
              n,
              k_per_group,
              Aint8.data() + g * k_per_group,
              k,
              Bint8_ref.data() + g * k_per_group * n,
              n);
        }

        aligned_vector<int8_t> Bint8(Bint8_ref);

        // initialize bias
        aligned_vector<int32_t> bias_int32(groups * n);
        int32_t* bias = nullptr;
        if (test_bias) {
          randFill(bias_int32, -128, 127);
          bias = bias_int32.data();
        }

        if (btrans == matrix_op_t::Transpose) {
          aligned_vector<int8_t> Bint8_temp(Bint8.size());
          for (int g = 0; g < groups; ++g) {
            transpose_matrix(
                k_per_group,
                n,
                Bint8.data() + g * k_per_group * n,
                n,
                Bint8_temp.data() + g * k_per_group * n,
                k_per_group);
          }
          Bint8 = Bint8_temp;
        }

        // To test lda != k , we just reduce k by half and use the original k
        // as lda.
        int n_adjusted = n;
        if (test_ld) {
          if (btrans == matrix_op_t::NoTranspose) {
            n_adjusted = std::max(n / 2, 1);
          }
        }

        int ncols_per_quant_group = groups * n_adjusted;
        if (q_granularity == QuantizationGranularity::GROUP) {
          ncols_per_quant_group = n_adjusted;
        } else if (q_granularity == QuantizationGranularity::OUT_CHANNEL) {
          ncols_per_quant_group = 1;
        }
        aligned_vector<int32_t> Bint8_zero_point(
            groups * n_adjusted / ncols_per_quant_group);
        randFill(Bint8_zero_point, -50, -10);

        // computing column offset
        vector<int32_t> col_offsets(groups * n_adjusted);
        for (int g = 0; g < groups; ++g) {
          col_offsets_with_zero_pt_s8acc32_ref(
              k_per_group,
              n_adjusted,
              n,
              Bint8_ref.data() + g * k_per_group * n,
              Bint8_zero_point.data() + g * n_adjusted / ncols_per_quant_group,
              col_offsets.data() + g * n_adjusted,
              ncols_per_quant_group);
        }

        vector<int32_t> row_offsets(m);

        aligned_vector<float> C_multiplier(Bint8_zero_point.size());
        randFill(C_multiplier, 0.001234f / 2, 0.001234f * 3 / 2);
        int32_t C_zero_pt = 5;

        for (int g = 0; g < groups; ++g) {
          matmul_u8i8acc32_ref(
              m,
              n_adjusted,
              k_per_group,
              k,
              n,
              groups * n,
              Aint8.data() + g * k_per_group,
              Bint8_ref.data() + g * k_per_group * n,
              Cint32_ref.data() + g * n_adjusted);

          row_offsets_u8acc32_ref(
              m,
              k_per_group,
              k,
              Aint8.data() + g * k_per_group,
              row_offsets.data());

          requantize_u8acc32_ref(
              m,
              n_adjusted,
              groups * n,
              Cint32_ref.data() + g * n_adjusted,
              Cint8_ref.data() + g * n_adjusted,
              C_multiplier.data() + g * n_adjusted / ncols_per_quant_group,
              C_zero_pt,
              Aint8_zero_point,
              Bint8_zero_point.data() + g * n_adjusted / ncols_per_quant_group,
              row_offsets.data(),
              col_offsets.data() + g * n_adjusted,
              bias ? (bias + g * n_adjusted) : nullptr,
              ncols_per_quant_group);
        }

        if (atrans == matrix_op_t::Transpose) {
          aligned_vector<uint8_t> Aint8_temp(Aint8.size());
          transpose_matrix(m, k, Aint8.data(), k, Aint8_temp.data(), m);
          Aint8 = Aint8_temp;
        }

        PackBMatrix<int8_t> packedBN(
            btrans,
            k,
            n_adjusted,
            Bint8.data(),
            (btrans == matrix_op_t::Transpose) ? k_per_group : n,
            nullptr,
            groups);

#ifdef _OPENMP
#pragma omp parallel
#endif
        {
          vector<int32_t> row_offset_buf(
              PackAWithRowOffset<uint8_t>::rowOffsetBufferSize());

          PackAWithRowOffset<uint8_t> packAN(
              atrans,
              m,
              k,
              Aint8.data(),
              (atrans == matrix_op_t::Transpose) ? m : k,
              nullptr,
              groups,
              row_offset_buf.data());

          int num_threads = fbgemm_get_num_threads();
          int tid = fbgemm_get_thread_num();

          DoNothing<> doNothingObj{};

          if (q_granularity == QuantizationGranularity::TENSOR) {
            ReQuantizeOutput<false> outputProcObj(
                doNothingObj,
                C_multiplier.data(),
                C_zero_pt,
                Aint8_zero_point,
                Bint8_zero_point.data(),
                packAN.getRowOffsetBuffer(),
                col_offsets.data(),
                bias,
                groups * n_adjusted,
                groups);

            fbgemmPacked(
                packAN,
                packedBN,
                Cint8_fb.data(),
                Cint32_buffer.data(),
                groups * n,
                outputProcObj,
                tid,
                num_threads);
          } else if (q_granularity == QuantizationGranularity::GROUP) {
            ReQuantizeOutput<false, QuantizationGranularity::GROUP>
                outputProcObj(
                    doNothingObj,
                    C_multiplier.data(),
                    C_zero_pt,
                    Aint8_zero_point,
                    Bint8_zero_point.data(),
                    packAN.getRowOffsetBuffer(),
                    col_offsets.data(),
                    bias,
                    groups * n_adjusted,
                    groups);

            fbgemmPacked(
                packAN,
                packedBN,
                Cint8_fb.data(),
                Cint32_buffer.data(),
                groups * n,
                outputProcObj,
                tid,
                num_threads);
          } else {
            ReQuantizeOutput<false, QuantizationGranularity::OUT_CHANNEL>
                outputProcObj(
                    doNothingObj,
                    C_multiplier.data(),
                    C_zero_pt,
                    Aint8_zero_point,
                    Bint8_zero_point.data(),
                    packAN.getRowOffsetBuffer(),
                    col_offsets.data(),
                    bias,
                    groups * n_adjusted,
                    groups);

            fbgemmPacked(
                packAN,
                packedBN,
                Cint8_fb.data(),
                Cint32_buffer.data(),
                groups * n,
                outputProcObj,
                tid,
                num_threads);
          }
        }
        // printMatrix(matrix_op_t::NoTranspose, Cint32_local.data(),
        // m, n_adjusted, n, "C local");
        compare_validate_buffers(
            Cint8_ref.data(),
            Cint8_fb.data(),
            m,
            groups * n_adjusted,
            groups * n,
            static_cast<uint8_t>(0));
      } // test_bias
    } // for each groups
  } // for each shape
}

/**
 * @brief Unit test for uint8 matrix A, int8 matrix B, and 32-bit
 * accumulation. Directly output fp32 matrix C. Output processing:
 * requantization -> nothing
 */
TEST_P(fbgemmu8s8acc32WithQuantGranularityTest, TestFloatInputOutput) {
  vector<vector<int>> shapes(GetShapes_());
  matrix_op_t atrans, btrans;
  bool test_ld;
  QuantizationGranularity q_granularity;
  tie(atrans, btrans, test_ld, q_granularity) = GetParam();

  for (auto shape : shapes) {
    for (int groups : {1, 3, 4}) {
      int m = shape[0];
      int n = shape[1];
      int k = shape[2];
      if (k % groups != 0) {
        continue;
      }
      int k_per_group = k / groups;

      aligned_vector<float> Afp32(m * k);
      aligned_vector<uint8_t> Aint8(Afp32.size());

      aligned_vector<float> Bfp32(k * n);
      aligned_vector<int8_t> Bint8(Bfp32.size());

      aligned_vector<float> Cfp32_ref(m * n * groups);
      aligned_vector<float> Cfp32_fb(Cfp32_ref.size());

      aligned_vector<uint8_t> Cint8_fb(Cfp32_ref.size());
      aligned_vector<int32_t> Cint32_buffer(Cfp32_ref.size());

      randFill<uint8_t>(Aint8, 0, 255);
      int32_t Aint8_zero_point = 43;
      float Aint8_scale = 0.11;
      for (auto i = 0; i < Afp32.size(); ++i) {
        Afp32[i] = Aint8_scale * (Aint8[i] - Aint8_zero_point);
      }

      randFill<int8_t>(Bint8, -128, 127);
      for (int g = 0; g < groups; ++g) {
        avoidOverflow(
            m,
            n,
            k_per_group,
            Aint8.data() + g * k_per_group,
            k,
            Bint8.data() + g * k_per_group * n,
            n);
      }

      // To test lda != k , we just reduce k by half and use the original k
      // as lda.
      int n_adjusted = n;
      if (test_ld) {
        if (btrans == matrix_op_t::NoTranspose) {
          n_adjusted = std::max(n / 2, 1);
        }
      }

      int ncols_per_quant_group = groups * n_adjusted;
      if (q_granularity == QuantizationGranularity::GROUP) {
        ncols_per_quant_group = n_adjusted;
      } else if (q_granularity == QuantizationGranularity::OUT_CHANNEL) {
        ncols_per_quant_group = 1;
      }
      aligned_vector<int32_t> Bint8_zero_point(
          groups * n_adjusted / ncols_per_quant_group);
      randFill(Bint8_zero_point, -50, -10);
      aligned_vector<float> Bint8_scale(Bint8_zero_point.size());
      randFill(Bint8_scale, 0.49f / 2, 0.49f * 3 / 2);
      for (int i = 0; i < k; ++i) {
        int g = i / k_per_group;
        for (int j = 0; j < n_adjusted; ++j) {
          int quant_group = (g * n_adjusted + j) / ncols_per_quant_group;
          Bfp32[i * n + j] = Bint8_scale[quant_group] *
              (Bint8[i * n + j] - Bint8_zero_point[quant_group]);
        }
      }

      // computing column offset
      vector<int32_t> col_offsets(groups * n_adjusted);
      for (int g = 0; g < groups; ++g) {
        col_offsets_with_zero_pt_s8acc32_ref(
            k_per_group,
            n_adjusted,
            n,
            Bint8.data() + g * k_per_group * n,
            Bint8_zero_point.data() + g * n_adjusted / ncols_per_quant_group,
            col_offsets.data() + g * n_adjusted,
            ncols_per_quant_group);
      }

      if (btrans == matrix_op_t::Transpose) {
        aligned_vector<int8_t> Bint8_temp(Bint8.size());
        for (int g = 0; g < groups; ++g) {
          transpose_matrix(
              k_per_group,
              n,
              Bint8.data() + g * k_per_group * n,
              n,
              Bint8_temp.data() + g * k_per_group * n,
              k_per_group);
        }
        Bint8 = Bint8_temp;
      }

      for (int g = 0; g < groups; ++g) {
        cblas_sgemm_ref(
            matrix_op_t::NoTranspose,
            matrix_op_t::NoTranspose,
            m,
            n_adjusted,
            k_per_group,
            1.0f,
            Afp32.data() + g * k_per_group,
            k,
            Bfp32.data() + g * k_per_group * n,
            n,
            0.0f,
            Cfp32_ref.data() + g * n_adjusted,
            groups * n);
      }

      if (atrans == matrix_op_t::Transpose) {
        aligned_vector<float> Afp32_temp(Afp32.size());
        transpose_matrix(m, k, Afp32.data(), k, Afp32_temp.data(), m);
        Afp32 = Afp32_temp;
      }

      PackBMatrix<int8_t> packedBN(
          btrans,
          k,
          n_adjusted,
          Bint8.data(),
          (btrans == matrix_op_t::Transpose) ? k_per_group : n,
          nullptr,
          groups);

#ifdef _OPENMP
#pragma omp parallel
#endif
      {
        vector<int32_t> row_offset_buf(
            PackAWithQuantRowOffset<uint8_t>::rowOffsetBufferSize());

        PackAWithQuantRowOffset<uint8_t> packAN(
            atrans,
            m,
            k,
            Afp32.data(),
            (atrans == matrix_op_t::Transpose) ? m : k,
            nullptr, /*buffer for packed matrix*/
            Aint8_scale,
            Aint8_zero_point,
            groups,
            // This is just to test row_offset_buf = nullptr with at least
            // one configuration.
            groups == 3 ? nullptr : row_offset_buf.data());

        int num_threads = fbgemm_get_num_threads();
        int tid = fbgemm_get_thread_num();

        DoNothing<float, float> doNothingObj{};

        if (q_granularity == QuantizationGranularity::TENSOR) {
          ReQuantizeForFloat<false> outputProcObj(
              doNothingObj,
              Aint8_scale,
              Bint8_scale.data(),
              Aint8_zero_point,
              Bint8_zero_point.data(),
              packAN.getRowOffsetBuffer(),
              col_offsets.data(),
              nullptr,
              groups * n_adjusted,
              groups);

          fbgemmPacked(
              packAN,
              packedBN,
              Cfp32_fb.data(),
              reinterpret_cast<int32_t*>(Cfp32_fb.data()),
              groups * n,
              outputProcObj,
              tid,
              num_threads);
        } else if (q_granularity == QuantizationGranularity::GROUP) {
          ReQuantizeForFloat<false, QuantizationGranularity::GROUP>
              outputProcObj(
                  doNothingObj,
                  Aint8_scale,
                  Bint8_scale.data(),
                  Aint8_zero_point,
                  Bint8_zero_point.data(),
                  packAN.getRowOffsetBuffer(),
                  col_offsets.data(),
                  nullptr,
                  groups * n_adjusted,
                  groups);

          fbgemmPacked(
              packAN,
              packedBN,
              Cfp32_fb.data(),
              reinterpret_cast<int32_t*>(Cfp32_fb.data()),
              groups * n,
              outputProcObj,
              tid,
              num_threads);
        } else {
          ReQuantizeForFloat<false, QuantizationGranularity::OUT_CHANNEL>
              outputProcObj(
                  doNothingObj,
                  Aint8_scale,
                  Bint8_scale.data(),
                  Aint8_zero_point,
                  Bint8_zero_point.data(),
                  packAN.getRowOffsetBuffer(),
                  col_offsets.data(),
                  nullptr,
                  groups * n_adjusted,
                  groups);

          fbgemmPacked(
              packAN,
              packedBN,
              Cfp32_fb.data(),
              reinterpret_cast<int32_t*>(Cfp32_fb.data()),
              groups * n,
              outputProcObj,
              tid,
              num_threads);
        }
      }

      float maximum = 0;
      for (int i = 0; i < m; ++i) {
        for (int j = 0; j < groups * n_adjusted; ++j) {
          float c = Cfp32_ref[i * groups * n + j];
          maximum = std::max(maximum, std::abs(c));
        }
      }
      compare_validate_buffers(
          Cfp32_ref.data(),
          Cfp32_fb.data(),
          m,
          groups * n_adjusted,
          groups * n,
          maximum * 1e-5f);
    } // for each groups
  } // for each shape
}

/**
 * @brief Unit test for uint8 matrix A, int8 matrix B, and 32-bit
 * accumulation. Output processing: requantization -> nothing. Symmetric: the
 * zero point is 0.
 */
TEST_P(fbgemmu8s8acc32Test, TestSymmetricQuantizedInputOutput) {
  vector<vector<int>> shapes(GetShapes_());
  matrix_op_t atrans, btrans;
  bool test_ld;
  tie(atrans, btrans, test_ld) = GetParam();

  for (auto shape : shapes) {
    for (int groups : {1, 3, 4}) {
      int m = shape[0];
      int n = shape[1];
      int k = shape[2];
      if (k % groups != 0) {
        continue;
      }
      int k_per_group = k / groups;

      aligned_vector<uint8_t> Aint8(m * k);
      aligned_vector<int8_t> Bint8(k * n);

      aligned_vector<float> Cfp32_ref(m * n * groups);
      aligned_vector<int32_t> Cint32_fb(Cfp32_ref.size());

      randFill<uint8_t>(Aint8, 0, 255);
      aligned_vector<float> Afp32(Aint8.begin(), Aint8.end());

      // initialize B matrix
      randFill<int8_t>(Bint8, -128, 127);
      for (int g = 0; g < groups; ++g) {
        avoidOverflow(
            m,
            n,
            k_per_group,
            Aint8.data() + g * k_per_group,
            k,
            Bint8.data() + g * k_per_group * n,
            n);
      }

      aligned_vector<float> Bfp32(Bint8.begin(), Bint8.end());

      // To test lda != k , we just reduce k by half and use the original k
      // as lda.
      int n_adjusted = n;
      if (test_ld) {
        if (btrans == matrix_op_t::NoTranspose) {
          n_adjusted = std::max(n / 2, 1);
        }
      }

      if (atrans == matrix_op_t::Transpose) {
        aligned_vector<uint8_t> Aint8_temp(Aint8.size());
        transpose_matrix(m, k, Aint8.data(), k, Aint8_temp.data(), m);
        Aint8 = Aint8_temp;
      }

      if (btrans == matrix_op_t::Transpose) {
        aligned_vector<int8_t> Bint8_temp(Bint8.size());
        for (int g = 0; g < groups; ++g) {
          transpose_matrix(
              k_per_group,
              n,
              Bint8.data() + g * k_per_group * n,
              n,
              Bint8_temp.data() + g * k_per_group * n,
              k_per_group);
        }
        Bint8 = Bint8_temp;
      }

      for (int g = 0; g < groups; ++g) {
        cblas_sgemm_ref(
            matrix_op_t::NoTranspose,
            matrix_op_t::NoTranspose,
            m,
            n_adjusted,
            k_per_group,
            1.0f,
            Afp32.data() + g * k_per_group,
            k,
            Bfp32.data() + g * k_per_group * n,
            n,
            0.0f,
            Cfp32_ref.data() + g * n_adjusted,
            groups * n);
      }

      // B zero point defaults to 0
      PackBMatrix<int8_t> packedBN(
          btrans,
          k,
          n_adjusted,
          Bint8.data(),
          (btrans == matrix_op_t::Transpose) ? k_per_group : n,
          nullptr,
          groups);

#ifdef _OPENMP
#pragma omp parallel
#endif
      {
        // A zero point and row offset not required
        PackAMatrix<uint8_t> packAN(
            atrans,
            m,
            k,
            Aint8.data(),
            (atrans == matrix_op_t::Transpose) ? m : k,
            nullptr,
            groups);

        DoNothing<int32_t, int32_t> doNothingObj{};
        memCopy<> outputProcObj(doNothingObj);

        int num_threads = fbgemm_get_num_threads();
        int tid = fbgemm_get_thread_num();

        fbgemmPacked(
            packAN,
            packedBN,
            Cint32_fb.data(),
            Cint32_fb.data(),
            groups * n,
            outputProcObj,
            tid,
            num_threads);
      }

      // correctness check
      for (int i = 0; i < m; ++i) {
        for (int j = 0; j < groups * n_adjusted; ++j) {
          float expected = Cfp32_ref[i * groups * n + j];
          int32_t actual = Cint32_fb[i * groups * n + j];
          EXPECT_EQ(actual, expected)
              << "GEMM results differ at (" << i << ", " << j << "). ref "
              << expected << " FBGemm " << actual;
        }
      }
    } // for each groups
  } // for each shape
}

/**
 * @brief Unit test for packing and unpacking the weight tensor.
 */
TEST_P(fbgemmPackUnpackAcc32Test, TestPackUnpack) {
  vector<vector<int>> shapes(GetShapes_());
  matrix_op_t btrans;
  bool test_ld;
  tie(btrans, test_ld) = GetParam();

  BlockingFactors params;
  params.MCB = 48;
  params.NCB = 16;
  params.KCB = 256;
  params.MR = 1;
  params.NR = 16;
  params.ROW_INTERLEAVE = 4;
  params.NR_MIN = 16;
  vector<BlockingFactors*> vec_params_ptr = {&params, nullptr};

  for (auto shape : shapes) {
    for (int groups : {1, 3, 4}) {
      for (auto params_ptr : vec_params_ptr) {
        int n = shape[1];
        int k = shape[2];

        if (k % groups != 0) {
          continue;
        }
        int k_per_group = k / groups;

        // kxn matrix
        aligned_vector<int8_t> Bint8(k * n);
        randFill<int8_t>(Bint8, -128, 127);

        // To test lda != k , we just reduce k by half and use the original k
        // as lda.
        int n_adjusted = n;
        if (test_ld) {
          if (btrans == matrix_op_t::NoTranspose) {
            n_adjusted = std::max(n / 2, 1);
          }
        }

        // Note that packing for weight is performed during the constructor
        // stage.
        PackBMatrix<int8_t> packedWeights(
            btrans,
            k,
            n_adjusted,
            Bint8.data(),
            (btrans == matrix_op_t::Transpose) ? k_per_group : n,
            nullptr,
            groups,
            params_ptr);

        // Setup a buffer to get pack -> unpacked results
        aligned_vector<int8_t> unpack_buf(k * n, 0);

        // Perform unpacking
        packedWeights.unpack(unpack_buf.data(), params_ptr);

        // Sanity check
        for (int i = 0; i < k; i++) {
          for (int j = 0; j < n_adjusted; j++) {
            EXPECT_EQ(unpack_buf.data()[i * n + j], Bint8.data()[i * n + j])
                << "Pack/Unpack results differ at index (" << i << ", " << j
                << ", Reference: " << static_cast<int>(Bint8.data()[i * n + j])
                << ", Pack-Unpacked: "
                << static_cast<int>(unpack_buf.data()[i * n + j]);
          }
        }
      }
    }
  }
}
