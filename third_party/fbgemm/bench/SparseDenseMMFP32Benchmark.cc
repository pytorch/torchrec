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

#include <iostream>
#include <iomanip>

using namespace std;
using namespace fbgemm;

int main(int, char**) {
  vector<vector<int>> shapes = getSparseMatrixShapes();

  // C is MxN -> CT is NxM
  // A is MxK -> AT is KxM
  // B is KxN -> BT is NxK

  cout << setw(7) << "index"
    << setw(7) << "m" << setw(7) << "n" << setw(7) << "k"
    << setw(7) << "fnz" << setw(15) << "eff_GFLOPS"
    << setw(15) << "real_GFLOPS" << endl;

  int index = 0;
  // for (int s = 64; s <= 128; s *= 2)
  for (auto const& s : shapes) {
    int m = s[0];
    int n = s[1];
    int k = s[2];

    for (float fnz = 0.20; fnz >= 0.20; fnz -= 0.01) {
      auto aData = getRandomSparseVector(m * k);
      auto bData = getRandomSparseVector(k * n, fnz);
      auto cData = getRandomSparseVector(m * n);

      aligned_vector<float> atData(k * m);
      aligned_vector<float> btData(n * k);
      aligned_vector<float> ctData(n * m);
      aligned_vector<float> ctDataRef(n * m);
      aligned_vector<float> ctDataIntrin(n * m);

      transpose_matrix(m, k, aData.data(), k, atData.data(), m);
      transpose_matrix(k, n, bData.data(), n, btData.data(), k);

      unique_ptr<CSRMatrix<float>> csr = fbgemmDenseToCSR(n, k, btData.data());

      // We calculate C^T = B^T x A^T
      int ldat = m;
      // int ldbt = k;
      int ldct = m;

      double effective_flop = m * n * k * 2;

      constexpr int NWARMUP = 20;
      constexpr int NITER = 100;

      auto secs_intrin = measureWithWarmup(
          [&]() {
            SparseDenseMM(
                n,
                m,
                csr->rowPtr.data(),
                csr->colIdx.data(),
                csr->values.data(),
                atData.data(),
                ldat,
                ctDataIntrin.data(),
                ldct);
          },
          NWARMUP,
          NITER,
          [&]() {
            cache_evict(atData);
            cache_evict(csr->rowPtr);
            cache_evict(csr->colIdx);
            cache_evict(csr->values);
            cache_evict(ctDataIntrin);
          });

      // printMatrix(matrix_op_t::NoTranspose, btData.data(), n, k, k,
      // "btData");
      // printMatrix(matrix_op_t::NoTranspose, atData.data(), k, m, m,
      // "atData");
      // printMatrix(matrix_op_t::NoTranspose, ctData.data(), n, m, m,
      // "ctData");
      cblas_sgemm_ref(
          matrix_op_t::NoTranspose,
          matrix_op_t::NoTranspose,
          m,
          n,
          k,
          1.0f,
          aData.data(),
          k,
          bData.data(),
          n,
          0.0f,
          cData.data(),
          n);
      transpose_matrix(m, n, cData.data(), n, ctDataRef.data(), m);
      // printMatrix(matrix_op_t::NoTranspose, ctDataRef.data(), n, m, m,
      // "ctData_Ref");
      //
      // Compare results
      for (auto i = 0; i < ctDataRef.size(); i++) {
        if (std::abs(ctDataRef[i] - ctDataIntrin[i]) > 1e-3) {
          fprintf(
              stderr,
              "Error: Results differ ref %f and test %f at %d\n",
              ctDataRef[i],
              ctDataIntrin[i],
              i);
          return 1;
        }
      }

      double effective_gflops_intrin = effective_flop / secs_intrin / 1e9;
      cout << "[" << setw(5) << index << "]"
        << setw(7) << m << setw(7) << n << setw(7) << k
        << fixed << setw(7) << setprecision(2) << fnz
        << setw(15) << setprecision(5) << effective_gflops_intrin
        << setw(15) << setprecision(5) << fnz * effective_gflops_intrin
        << endl;
      ++index;
    }
  }
}
