/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <cpuinfo.h>
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
class fbgemmu8s8acc16WithQuantGranularityTest
    : public testing::TestWithParam<
          tuple<matrix_op_t, matrix_op_t, bool, QuantizationGranularity>> {};
class fbgemmu8s8acc16Test
    : public testing::TestWithParam<tuple<matrix_op_t, matrix_op_t, bool>> {};
class fbgemmPackUnpackAcc16Test
    : public testing::TestWithParam<tuple<matrix_op_t, bool>> {};
}; // namespace

INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    fbgemmu8s8acc16WithQuantGranularityTest,
    ::testing::Combine(
        ::testing::Values(matrix_op_t::NoTranspose),
        ::testing::ValuesIn(transposeVals),
        ::testing::Bool(),
        ::testing::ValuesIn(qGranularityVals)));

INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    fbgemmu8s8acc16Test,
    ::testing::Combine(
        ::testing::Values(matrix_op_t::NoTranspose),
        ::testing::ValuesIn(transposeVals),
        ::testing::Bool()));

INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    fbgemmPackUnpackAcc16Test,
    ::testing::Combine(::testing::ValuesIn(transposeVals), ::testing::Bool()));

/**
 * @brief Shapes for unit test.
 */
static vector<vector<int>> GetShapes_() {
  // clang-format off
  // NMT
  vector<vector<int>> shapes = {
    // {M,    N,    K}
    {1, 128, 512},
    {1, 1024, 256},
    {1, 2048, 512},
    {1, 2048, 513},
    {1, 2048, 514},

    {6, 512, 512},
    {6, 2048, 512},
    {6, 256, 1024},
    {6, 1024, 256},
    {6, 2048, 256},
    {6, 2048, 257},
    {6, 2048, 258},

    {102, 1024, 512},
    {102, 2323, 256},
    {102, 512, 256},
    {102, 512, 257},
    {102, 512, 258},

    {1024, 512, 258},

    {120, 4, 288},
  };
  // clang-format on
  return shapes;
}

/**
 * @brief Unit test for uint8 matrix A, int8 matrix B, and 16-bit
 * accumulation. Output processing: requantization -> nothing
 */
TEST_P(fbgemmu8s8acc16WithQuantGranularityTest, Test) {
  cpuinfo_initialize();
  if (fbgemmHasAvx512VnniSupport()) {
    // No need to use acc16 if VNNI is available
    return;
  }

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

      aligned_vector<uint8_t> Aint8(m * k);

      aligned_vector<int8_t> Bint8_ref(k * n);

      aligned_vector<int32_t> Cint32_ref(m * n * groups);
      aligned_vector<uint8_t> Cint8_ref(Cint32_ref.size());
      aligned_vector<int32_t> Cint32_fb(Cint32_ref.size());
      aligned_vector<uint8_t> Cint8_fb(Cint32_ref.size());
      aligned_vector<int32_t> Cint32_buffer(Cint32_ref.size());

      randFill<uint8_t>(Aint8, 0, 255);
      int32_t Aint8_zero_point = 43;

      randFill<int8_t>(Bint8_ref, -128, 127);
      aligned_vector<int8_t> Bint8(Bint8_ref);

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
        assert(
            atrans == matrix_op_t::NoTranspose &&
            "This case is not handled yet");
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
      randFill(Bint8_zero_point, -60, 0);

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

      int brow = 256;
      for (int g = 0; g < groups; ++g) {
        matmul_u8i8acc16_ref(
            m,
            n_adjusted,
            k_per_group,
            k,
            n,
            groups * n,
            brow,
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
            nullptr,
            ncols_per_quant_group);
      }

      PackBMatrix<int8_t, int16_t> packedBN(
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
            PackAWithRowOffset<uint8_t, int16_t>::rowOffsetBufferSize());

        PackAWithRowOffset<uint8_t, int16_t> packAN(
            matrix_op_t::NoTranspose,
            m,
            k,
            Aint8.data(),
            k,
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
              nullptr,
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
          ReQuantizeOutput<false, QuantizationGranularity::GROUP> outputProcObj(
              doNothingObj,
              C_multiplier.data(),
              C_zero_pt,
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
                  nullptr,
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
      } // omp parallel

      compare_validate_buffers(
          Cint8_ref.data(),
          Cint8_fb.data(),
          m,
          groups * n_adjusted,
          groups * n,
          static_cast<uint8_t>(0));
    } // for each groups
  } // for each shape
}

/**
 * @brief Unit test for uint8 matrix A, int8 matrix B, and 16-bit
 * accumulation. Output processing: spmdm -> requantization -> nothing
 */
TEST_P(fbgemmu8s8acc16WithQuantGranularityTest, SpMDMTest) {
  cpuinfo_initialize();
  if (fbgemmHasAvx512VnniSupport()) {
    // No need to use acc16 if VNNI is available
    return;
  }

  vector<vector<int>> shapes(GetShapes_());
  matrix_op_t atrans, btrans;
  bool test_ld;
  QuantizationGranularity q_granularity;
  tie(atrans, btrans, test_ld, q_granularity) = GetParam();

  for (auto shape : shapes) {
    for (int groups : {1, 3, 4}) {
      // very small density to test hyper sparsity case
      // moderate density to test the implementation using transpose
      for (float density : {0.0001f, 0.1f}) {
        int m = shape[0];
        int n = shape[1];
        int k = shape[2];
        if (k % groups != 0) {
          continue;
        }
        int k_per_group = k / groups;

        aligned_vector<uint8_t> Aint8(m * k);
        aligned_vector<int8_t> Bint8(k * n);

        aligned_vector<int32_t> Cint32_ref(m * n * groups);
        aligned_vector<uint8_t> Cint8_ref(Cint32_ref.size());
        aligned_vector<int32_t> Cint32_fb(Cint32_ref.size());
        aligned_vector<uint8_t> Cint8_fb(Cint32_ref.size());
        aligned_vector<int32_t> Cint32_buffer(Cint32_ref.size());

        randFill<uint8_t>(Aint8, 0, 255);
        int32_t Aint8_zero_point = 43;

        randFill<int8_t>(Bint8, -128, 127);

        // To test lda != k , we just reduce k by half and use the original k
        // as lda.
        int n_adjusted = n;
        if (test_ld) {
          assert(
              atrans == matrix_op_t::NoTranspose &&
              "This case is not handled yet");
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
              Bint8.data() + g * k_per_group * n,
              Bint8_zero_point.data() + g * n_adjusted / ncols_per_quant_group,
              col_offsets.data() + g * n_adjusted,
              ncols_per_quant_group);
        }

        CompressedSparseColumn B_csc(k_per_group, groups * n_adjusted);
        // Make sure density is big enough. Otherwise, we're not really testing
        // spmdm.
        // deterministic random number
        default_random_engine eng;
        binomial_distribution<> per_col_nnz_dist(k_per_group, density);

        vector<int> row_indices(k_per_group);
        int total_nnz = 0;
        for (int g = 0; g < groups; ++g) {
          for (int j = 0; j < n_adjusted; ++j) {
            B_csc.ColPtr()[g * n_adjusted + j] = total_nnz;

            int nnz_of_j = per_col_nnz_dist(eng);
            total_nnz += nnz_of_j;

            iota(row_indices.begin(), row_indices.end(), 0);
            shuffle(row_indices.begin(), row_indices.end(), eng);
            sort(row_indices.begin(), row_indices.begin() + nnz_of_j);

            for (int kidx = 0; kidx < nnz_of_j; ++kidx) {
              int rowidx = row_indices[kidx];
              B_csc.RowIdx().push_back(rowidx);
              int8_t* bptr = &Bint8[(g * k_per_group + rowidx) * n + j];
              int b_remainder = 0;
              if (kidx % 2 == 1) {
                // Make sure abs(b_prev + *bptr - b_remainder) <= 128
                int b_prev = B_csc.Values().back();
                b_remainder = std::max(b_prev + *bptr - 128, b_remainder);
                b_remainder = std::min(b_prev + *bptr + 128, b_remainder);
              }
              // put a portion of current B value that won't saturate
              // _mm256_maddubs_epi16 .
              B_csc.Values().push_back(*bptr - b_remainder);
              // put the remainder
              *bptr = b_remainder;
            }
          }
        }
        B_csc.ColPtr()[groups * n_adjusted] = total_nnz;

        aligned_vector<int8_t> Bint8_ref(Bint8);

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

        vector<int32_t> row_offsets(m);

        aligned_vector<float> C_multiplier(Bint8_zero_point.size());
        randFill(C_multiplier, 0.001234f / 2, 0.001234f * 3 / 2);
        int32_t C_zero_pt = 5;

        int brow = 256;
        for (int g = 0; g < groups; ++g) {
          matmul_u8i8acc16_ref(
              m,
              n_adjusted,
              k_per_group,
              k,
              n,
              groups * n,
              brow,
              Aint8.data() + g * k_per_group,
              Bint8_ref.data() + g * k_per_group * n,
              Cint32_ref.data() + g * n_adjusted);
        }

        bool accumulation = true;
        spmdm_ref(
            m,
            Aint8.data(),
            k,
            B_csc,
            accumulation,
            Cint32_ref.data(),
            groups * n,
            groups);

        for (int g = 0; g < groups; ++g) {
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
              nullptr,
              ncols_per_quant_group);
        }

        PackBMatrix<int8_t, int16_t> packedB(
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
              PackAWithRowOffset<uint8_t, int16_t>::rowOffsetBufferSize());

          PackAWithRowOffset<uint8_t, int16_t> packAN(
              matrix_op_t::NoTranspose,
              m,
              k,
              Aint8.data(),
              k,
              nullptr,
              groups,
              row_offset_buf.data());

          int num_threads = fbgemm_get_num_threads();
          int tid = fbgemm_get_thread_num();

          // spmdm -> requantization -> nothing
          // construct an output processing pipeline in reverse order
          // i.e. last output operation first
          // Last operation should always be DoNothing with
          // correct input and output type.
          DoNothing<> doNothingObj{};

          if (q_granularity == QuantizationGranularity::TENSOR) {
            // The second last operation is requantization back
            // to int8
            ReQuantizeOutput<false> reqObj(
                doNothingObj,
                C_multiplier.data(),
                C_zero_pt,
                Aint8_zero_point,
                Bint8_zero_point.data(),
                packAN.getRowOffsetBuffer(),
                col_offsets.data(),
                nullptr,
                groups * n_adjusted,
                groups);
            // the top most (first) operation in the output processing
            // pipeline is spmdm
            // outType = final output type after fullly processing through
            // pipeline inType = initial input type at the first call to the
            // whole pipeline
            DoSpmdmOnInpBuffer<
                ReQuantizeOutput<false>::outType,
                int32_t,
                ReQuantizeOutput<false>>
                spmdmObj(reqObj, Aint8.data(), k, B_csc, groups);

            fbgemmPacked(
                packAN,
                packedB,
                Cint8_fb.data(),
                Cint32_fb.data(),
                groups * n,
                spmdmObj,
                tid,
                num_threads);
          } else if (q_granularity == QuantizationGranularity::GROUP) {
            ReQuantizeOutput<false, QuantizationGranularity::GROUP> reqObj(
                doNothingObj,
                C_multiplier.data(),
                C_zero_pt,
                Aint8_zero_point,
                Bint8_zero_point.data(),
                packAN.getRowOffsetBuffer(),
                col_offsets.data(),
                nullptr,
                groups * n_adjusted,
                groups);
            DoSpmdmOnInpBuffer<
                ReQuantizeOutput<false>::outType,
                int32_t,
                ReQuantizeOutput<false, QuantizationGranularity::GROUP>>
                spmdmObj(reqObj, Aint8.data(), k, B_csc, groups);

            fbgemmPacked(
                packAN,
                packedB,
                Cint8_fb.data(),
                Cint32_fb.data(),
                groups * n,
                spmdmObj,
                tid,
                num_threads);
          } else {
            ReQuantizeOutput<false, QuantizationGranularity::OUT_CHANNEL>
                reqObj(
                    doNothingObj,
                    C_multiplier.data(),
                    C_zero_pt,
                    Aint8_zero_point,
                    Bint8_zero_point.data(),
                    packAN.getRowOffsetBuffer(),
                    col_offsets.data(),
                    nullptr,
                    groups * n_adjusted,
                    groups);
            DoSpmdmOnInpBuffer<
                ReQuantizeOutput<false>::outType,
                int32_t,
                ReQuantizeOutput<false, QuantizationGranularity::OUT_CHANNEL>>
                spmdmObj(reqObj, Aint8.data(), k, B_csc, groups);

            fbgemmPacked(
                packAN,
                packedB,
                Cint8_fb.data(),
                Cint32_fb.data(),
                groups * n,
                spmdmObj,
                tid,
                num_threads);
          }
        }

        compare_validate_buffers(
            Cint8_ref.data(),
            Cint8_fb.data(),
            m,
            groups * n_adjusted,
            groups * n,
            static_cast<uint8_t>(0));
      } // for each density
    } // for each groups
  } // for each shape
}

/**
 * @brief Unit test for uint8 matrix A, int8 matrix B, and 16-bit
 * accumulation. Output processing: nothing
 */
TEST_P(fbgemmu8s8acc16Test, NoRequantizeTest) {
  cpuinfo_initialize();
  if (fbgemmHasAvx512VnniSupport()) {
    // No need to use acc16 if VNNI is available
    return;
  }

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

      aligned_vector<int8_t> Bint8_ref(k * n);

      aligned_vector<int32_t> Cint32_ref(m * n * groups);
      aligned_vector<int32_t> Cint32_fb(Cint32_ref.size());
      aligned_vector<int32_t> Cint32_buffer(Cint32_ref.size());

      randFill<uint8_t>(Aint8, 0, 255);
      int32_t Aint8_zero_point = 43;

      randFill<int8_t>(Bint8_ref, -128, 127);
      aligned_vector<int8_t> Bint8(Bint8_ref);

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

      int32_t Bint8_zero_point = -30;
      // To test lda != k , we just reduce k by half and use the original k
      // as lda.
      int n_adjusted = n;
      if (test_ld) {
        assert(
            atrans == matrix_op_t::NoTranspose &&
            "This case is not handled yet");
        if (btrans == matrix_op_t::NoTranspose) {
          n_adjusted = std::max(n / 2, 1);
        }
      }

      // computing column offset
      vector<int32_t> col_offsets(groups * n_adjusted);
      for (int g = 0; g < groups; ++g) {
        col_offsets_with_zero_pt_s8acc32_ref(
            k_per_group,
            n_adjusted,
            n,
            Bint8_ref.data() + g * k_per_group * n,
            &Bint8_zero_point,
            col_offsets.data() + g * n_adjusted,
            n_adjusted);
      }

      vector<int32_t> row_offsets(m);

      int brow = 256;
      for (int g = 0; g < groups; ++g) {
        matmul_u8i8acc16_ref(
            m,
            n_adjusted,
            k_per_group,
            k,
            n,
            groups * n,
            brow,
            Aint8.data() + g * k_per_group,
            Bint8_ref.data() + g * k_per_group * n,
            Cint32_ref.data() + g * n_adjusted);

        row_offsets_u8acc32_ref(
            m,
            k_per_group,
            k,
            Aint8.data() + g * k_per_group,
            row_offsets.data());
      }

      PackBMatrix<int8_t, int16_t> packedBN(
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
            PackAWithRowOffset<uint8_t, int16_t>::rowOffsetBufferSize());

        PackAWithRowOffset<uint8_t, int16_t> packAN(
            matrix_op_t::NoTranspose,
            m,
            k,
            Aint8.data(),
            k,
            nullptr,
            groups,
            row_offset_buf.data());

        // DoNothing<> doNothingObj{};
        DoNothing<int32_t, int32_t> doNothingObj{};
        memCopy<> outputProcObj(doNothingObj);

        int num_threads = fbgemm_get_num_threads();
        int tid = fbgemm_get_thread_num();

        fbgemmPacked(
            packAN,
            packedBN,
            Cint32_fb.data(),
            Cint32_buffer.data(),
            groups * n,
            outputProcObj,
            tid,
            num_threads);
      }

      compare_validate_buffers(
          Cint32_ref.data(),
          Cint32_fb.data(),
          m,
          groups * n_adjusted,
          groups * n,
          static_cast<int32_t>(0));
    } // for each groups
  } // for each shape
}

/**
 * @brief Unit test for packing and unpacking the weight tensor.
 */
TEST_P(fbgemmPackUnpackAcc16Test, TestPackUnpack) {
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
        PackBMatrix<int8_t, int16_t> packedWeights(
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
