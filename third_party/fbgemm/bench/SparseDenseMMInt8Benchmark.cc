/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "bench/BenchUtils.h"
#include "fbgemm/FbgemmSparse.h"
#include "fbgemm/Utils.h"
#include "fbgemm/spmmUtils.h"
#include "src/RefImplementations.h"

#include <iomanip>
#include <iostream>

using namespace std;
using namespace fbgemm;

int main(int, char**) {
  vector<vector<int>> shapes = getSparseMatrixShapes();

  // C is MxN -> CT is NxM
  // A is MxK -> AT is KxM
  // B is KxN -> BT is NxK

  cout << setw(7) << "index" << setw(7) << "m" << setw(7) << "n" << setw(7)
       << "k" << setw(7) << "fnz" << setw(15) << "eff_GFLOPS" << setw(15)
       << "real_GFLOPS" << endl;

  int index = 0;
  for (auto const& s : shapes) {
    int m = s[0];
    int n = s[1];
    int k = s[2];

    for (float fnz = 0.20; fnz >= 0.20; fnz -= 0.01) {
      auto aData = getRandomBlockSparseMatrix<uint8_t>(
          m, k, 1.0, 1 /* rowBlockSize */, 1 /* colBlockSize */);
      auto bData = getRandomBlockSparseMatrix<int8_t>(k, n, fnz);
      auto cData = getRandomBlockSparseMatrix<int32_t>(
          m, n, 1.0, 1 /* rowBlockSize */, 1 /* colBlockSize */);

      aligned_vector<uint8_t> atData(k * m);
      aligned_vector<int8_t> btData(n * k);
      aligned_vector<int32_t> ctData(n * m);
      aligned_vector<int32_t> ctDataRef(n * m);
      aligned_vector<uint8_t> ctDataRef_u8(n * m);
      aligned_vector<int32_t> ctDataIntrin_i32(n * m);
      aligned_vector<uint8_t> ctDataIntrin_u8(n * m);

      transpose_matrix(m, k, aData.data(), k, atData.data(), m);
      transpose_matrix(k, n, bData.data(), n, btData.data(), k);

      unique_ptr<BCSRMatrix<>> bcsr = fbgemmDenseToBCSR(n, k, btData.data());

      // output scale and zero point
      float scale = 32.0f;
      int32_t zero_point = 2;

      int32_t act_zero_point = 2;

      // symmetric quant for weights
      aligned_vector<int32_t> weight_zero_point(n);
      randFill<int32_t>(weight_zero_point, 0, 0);

      // Each row of weight matrix has it's own scale
      // The following is a multiplication activation scale with
      // weight scales.
      aligned_vector<float> act_times_w_scale(n);
      randFill<float>(act_times_w_scale, -8.0f, 8.0f);

      trRequantizationParams_t reqParams = {act_zero_point,
                                            weight_zero_point.data(),
                                            zero_point,
                                            scale,
                                            bcsr->row_offsets.data(),
                                            nullptr,
                                            nullptr,
                                            act_times_w_scale.data()};

      // printMatrix(matrix_op_t::NoTranspose, btData.data(), n, k, k,
      // "btData"); printMatrix( matrix_op_t::NoTranspose, bcsr->rowBPtr.data(),
      // 1,
      // bcsr->rowBPtr.size(),
      // bcsr->rowBPtr.size(),
      // "rowBPtr");
      // printMatrix(
      // matrix_op_t::NoTranspose,
      // bcsr->colBIdx.data(),
      // 1,
      // bcsr->colBIdx.size(),
      // bcsr->colBIdx.size(),
      // "colBIdx");
      // printMatrix(
      // matrix_op_t::NoTranspose,
      // bcsr->values.data(),
      // 1,
      // bcsr->values.size(),
      // bcsr->values.size(),
      // "values");

      // We calculate C^T = B^T x A^T
      int ldat = m;
      int ldct = m;

      double effective_flop = m * n * k * 2;

      constexpr int NWARMUP = 20;
      constexpr int NITER = 100;

      auto secs_intrin = measureWithWarmup(
          [&]() {
            fbgemmSparseDenseInt8MM<false, QuantizationGranularity::TENSOR>(
                m,
                bcsr,
                atData.data(),
                ldat,
                ctDataIntrin_i32.data(),
                ctDataIntrin_u8.data(),
                ldct,
                reqParams);
          },
          NWARMUP,
          NITER,
          [&]() {
            cache_evict(atData);
            cache_evict(bcsr->rowBPtr);
            cache_evict(bcsr->colBIdx);
            cache_evict(bcsr->values);
            cache_evict(ctDataIntrin_i32);
            cache_evict(ctDataIntrin_u8);
          });

      // printMatrix(matrix_op_t::NoTranspose, btData.data(), n, k, k,
      // "btData");
      // printMatrix(matrix_op_t::NoTranspose, atData.data(), k, m, m,
      // "atData");
      // printMatrix(matrix_op_t::NoTranspose, ctData.data(), n, m, m,
      // "ctData");
      matmul_u8i8acc32_ref(
          m,
          n,
          k,
          k, // lda
          n, // ldb
          n, // ldc
          aData.data(),
          bData.data(),
          cData.data());
      transpose_matrix(m, n, cData.data(), n, ctDataRef.data(), m);

      // ctDataRef is nxm
      block_type_t block{0, n, 0, m};
      trRequantizeRef<false, QuantizationGranularity::TENSOR>(
          ctDataRef_u8.data(), ctDataRef.data(), block, m, m, reqParams);
      // printMatrix(matrix_op_t::NoTranspose, ctDataRef_u8.data(), n, m, m,
      // "ctDataRef_u8");
      // printMatrix(matrix_op_t::NoTranspose, ctDataIntrin_u8.data(), n, m, m,
      // "ctDataIntrin_u8");
      //
      // Compare results
      for (auto i = 0; i < ctDataRef.size(); i++) {
        if (std::abs(ctDataRef_u8[i] - ctDataIntrin_u8[i]) > 0) {
          fprintf(
              stderr,
              "Error: Results differ ref %d and test %d at %d\n",
              ctDataRef_u8[i],
              ctDataIntrin_u8[i],
              i);
          return 1;
        }
      }

      double effective_gflops_intrin = effective_flop / secs_intrin / 1e9;
      cout << "[" << setw(5) << index << "]" << setw(7) << m << setw(7) << n
           << setw(7) << k << fixed << setw(7) << setprecision(2) << fnz
           << setw(15) << setprecision(5) << effective_gflops_intrin << setw(15)
           << setprecision(5) << fnz * effective_gflops_intrin << endl;
      ++index;
    }
  }
}
