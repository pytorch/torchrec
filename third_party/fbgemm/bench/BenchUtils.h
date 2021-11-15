/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <chrono>
#include <functional>
#include <vector>

#include <immintrin.h>

#ifdef USE_BLAS
#if __APPLE__
// not sure whether need to differentiate TARGET_OS_MAC or TARGET_OS_IPHONE,
// etc.
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_MKL
#include <mkl.h>
#endif

#include "./AlignedVec.h"
#include "fbgemm/FbgemmBuild.h"
#include "fbgemm/FbgemmPackMatrixB.h"
#include "src/RefImplementations.h"

namespace fbgemm {

template <typename T>
void randFill(aligned_vector<T>& vec, T low, T high);

void llc_flush(std::vector<char>& llc);

// Same as omp_get_max_threads() when OpenMP is available, otherwise 1
int fbgemm_get_max_threads();
// Same as omp_get_num_threads() when OpenMP is available, otherwise 1
int fbgemm_get_num_threads();
// Same as omp_get_thread_num() when OpenMP is available, otherwise 0
int fbgemm_get_thread_num();

template <typename T>
NOINLINE float cache_evict(const T& vec) {
  auto const size = vec.size();
  auto const elemSize = sizeof(typename T::value_type);
  auto const dataSize = size * elemSize;

  const char* data = reinterpret_cast<const char*>(vec.data());
  constexpr int CACHE_LINE_SIZE = 64;
  // Not having this dummy computation significantly slows down the computation
  // that follows.
  float dummy = 0.0f;
  for (std::size_t i = 0; i < dataSize; i += CACHE_LINE_SIZE) {
    dummy += data[i] * 1.0f;
    _mm_mfence();
#ifndef _MSC_VER
    asm volatile("" ::: "memory");
#endif
    _mm_clflush(&data[i]);
  }

  return dummy;
}

/**
 * Parse application command line arguments
 *
 */
int parseArgumentInt(
    int argc,
    const char* argv[],
    const char* arg,
    int non_exist_val,
    int def_val);
bool parseArgumentBool(
    int argc,
    const char* argv[],
    const char* arg,
    bool def_val);

namespace {
struct empty_flush {
  void operator()() const {}
};
} // namespace

/**
 * @param Fn functor to execute
 * @param Fe data eviction functor
 */
template <class Fn, class Fe = std::function<void()>>
double measureWithWarmup(
    Fn&& fn,
    int warmupIterations,
    int measuredIterations,
    const Fe& fe = empty_flush(),
    bool useOpenMP = false) {
  for (int i = 0; i < warmupIterations; ++i) {
    // Evict data first
    fe();
    fn();
  }

  double ttot = 0.0;

#ifdef _OPENMP
#pragma omp parallel if (useOpenMP)
  {
#endif
    for (int i = 0; i < measuredIterations; ++i) {
      int thread_id = 0;
      std::chrono::time_point<std::chrono::high_resolution_clock> start, end;

#ifdef _OPENMP
      if (useOpenMP) {
        thread_id = omp_get_thread_num();
      }
#endif

      if (thread_id == 0) {
        fe();
      }

#ifdef _OPENMP
      if (useOpenMP) {
#pragma omp barrier
      }
#endif
      start = std::chrono::high_resolution_clock::now();

      fn();

#ifdef _OPENMP
      if (useOpenMP) {
#pragma omp barrier
      }
#endif

      end = std::chrono::high_resolution_clock::now();
      auto dur =
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

      if (thread_id == 0) {
        // TODO: measure load imbalance
        ttot += dur.count();
      }
    }

#ifdef _OPENMP
  }
#endif
  return ttot / 1e9 / measuredIterations;
}

/*
 * @brief Out-of-place transposition for M*N matrix ref.
 * @param M number of rows in input
 * @param K number of columns in input
 */
template <typename T>
void transpose_matrix(
    int M,
    int N,
    const T* src,
    int ld_src,
    T* dst,
    int ld_dst) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      dst[i * ld_dst + j] = src[i + j * ld_src];
    }
  } // for each output row
}

/*
 * @brief In-place transposition for nxk matrix ref.
 * @param n number of rows in input (number of columns in output)
 * @param k number of columns in input (number of rows in output)
 */
template <typename T>
void transpose_matrix(T* ref, int n, int k) {
  std::vector<T> local(n * k);
  transpose_matrix(n, k, ref, k, local.data(), n);
  memcpy(ref, local.data(), n * k * sizeof(T));
}

#if defined(USE_MKL)
void test_xerbla(char* srname, const int* info, int);
#endif

#define dataset 1

template <typename btype>
void performance_test(
    int num_instances,
    bool flush,
    int repetitions,
    bool is_mkl) {
#if defined(USE_MKL)
  mkl_set_xerbla((XerblaEntry)test_xerbla);
#endif

  float alpha = 1.f, beta = 1.f;
  matrix_op_t btran = matrix_op_t::Transpose;

#if dataset == 1
  const int NITER = (flush) ? 10 : 100;
  std::vector<std::vector<int>> shapes;
  for (auto m = 1; m < 120; m++) {
    // shapes.push_back({m, 128, 512});
    shapes.push_back({m, 512, 512});
  }

#elif dataset == 2
  const int NITER = (flush) ? 10 : 100;
#include "shapes_dataset.h"

#else
  flush = false;
  constexpr int NITER = 1;
  std::vector<std::vector<int>> shapes;
  std::random_device r;
  std::default_random_engine generator(r());
  std::uniform_int_distribution<int> dm(1, 100);
  std::uniform_int_distribution<int> dnk(1, 1024);
  for (int i = 0; i < 1000; i++) {
    int m = dm(generator);
    int n = dnk(generator);
    int k = dnk(generator);
    shapes.push_back({m, n, k});
  }
#endif

  std::string type;
  double gflops, gbs, ttot;
  for (auto s : shapes) {
    int m = s[0];
    int n = s[1];
    int k = s[2];

    // initialize with small numbers
    aligned_vector<int> Aint(m * k);
    randFill(Aint, 0, 4);
    std::vector<aligned_vector<float>> A;
    for (int i = 0; i < num_instances; ++i) {
      A.push_back(aligned_vector<float>(Aint.begin(), Aint.end()));
    }

    aligned_vector<int> Bint(k * n);
    randFill(Bint, 0, 4);
    aligned_vector<float> B(Bint.begin(), Bint.end());
    std::vector<std::unique_ptr<PackedGemmMatrixB<btype>>> Bp;
    for (int i = 0; i < num_instances; ++i) {
      Bp.emplace_back(std::unique_ptr<PackedGemmMatrixB<btype>>(
          new PackedGemmMatrixB<btype>(btran, k, n, alpha, B.data())));
    }
    auto kAligned = ((k * sizeof(float) + 64) & ~63) / sizeof(float);
    auto nAligned = ((n * sizeof(float) + 64) & ~63) / sizeof(float);
    std::vector<aligned_vector<float>> Bt(num_instances);
    auto& Bt_ref = Bt[0];

    if (btran == matrix_op_t::Transpose) {
      Bt_ref.resize(k * nAligned);
      for (auto row = 0; row < k; ++row) {
        for (auto col = 0; col < n; ++col) {
          Bt_ref[row * nAligned + col] = alpha * B[col * k + row];
        }
      }
    } else {
      Bt_ref.resize(kAligned * n);
      for (auto row = 0; row < k; ++row) {
        for (auto col = 0; col < n; ++col) {
          Bt_ref[col * kAligned + row] = alpha * B[col * k + row];
        }
      }
    }

    for (auto i = 1; i < num_instances; ++i) {
      Bt[i] = Bt_ref;
    }

    std::vector<aligned_vector<float>> C_ref;
    std::vector<aligned_vector<float>> C_fb;
    if (beta != 0.0f) {
      aligned_vector<int> Cint(m * n);
      randFill(Cint, 0, 4);
      for (int i = 0; i < num_instances; ++i) {
        C_ref.push_back(aligned_vector<float>(Cint.begin(), Cint.end()));
        C_fb.push_back(aligned_vector<float>(Cint.begin(), Cint.end()));
      }
    } else {
      for (int i = 0; i < num_instances; ++i) {
        C_ref.push_back(aligned_vector<float>(m * n, 1.f));
        C_fb.push_back(aligned_vector<float>(m * n, NAN));
      }
    }

    double nflops = 2.0 * m * n * k;
    double nbytes = 4.0 * m * k + sizeof(btype) * 1.0 * k * n + 4.0 * m * n;

    // warm up MKL and fbgemm
    // check correctness at the same time
    for (auto w = 0; w < 3; w++) {
#if defined(USE_MKL) || defined(USE_BLAS)
      cblas_sgemm(
          CblasRowMajor,
          CblasNoTrans,
          CblasNoTrans, // B is pretransposed, if required by operation
          m,
          n,
          k,
          1.0, // Mutliplication by Alpha is done during transpose of B
          A[0].data(),
          k,
          Bt[0].data(),
          btran == matrix_op_t::NoTranspose ? kAligned : nAligned,
          beta,
          C_ref[0].data(),
          n);
#else
      cblas_sgemm_ref(
          matrix_op_t::NoTranspose,
          matrix_op_t::NoTranspose,
          m,
          n,
          k,
          1.0,
          A[0].data(),
          k,
          Bt[0].data(),
          (btran == matrix_op_t::NoTranspose) ? kAligned : nAligned,
          beta,
          C_ref[0].data(),
          n);
#endif
#ifdef _OPENMP
#pragma omp parallel if (num_instances == 1)
#endif
      {
        int num_threads = num_instances == 1 ? fbgemm_get_num_threads() : 1;
        int tid = num_instances == 1 ? fbgemm_get_thread_num() : 0;
        cblas_gemm_compute(
            matrix_op_t::NoTranspose,
            m,
            A[0].data(),
            *Bp[0],
            beta,
            C_fb[0].data(),
            tid,
            num_threads);
      }

#if defined(USE_MKL) || defined(USE_BLAS)
      // Compare results
      for (auto i = 0; i < C_ref[0].size(); i++) {
        if (std::abs(C_ref[0][i] - C_fb[0][i]) > 1e-3) {
          fprintf(
              stderr,
              "Error: too high diff between fp32 ref %f and fp16 %f at %d\n",
              C_ref[0][i],
              C_fb[0][i],
              i);
          return;
        }
      }
#endif
    }

#if defined(USE_MKL)
    if (is_mkl) {
      // Gold via MKL sgemm
      type = "MKL_FP32";
#elif defined(USE_BLAS)
    type = "BLAS_FP32";
#else
    type = "REF_FP32";
#endif

      ttot = measureWithWarmup(
          [&]() {
            int copy = num_instances == 1 ? 0 : fbgemm_get_thread_num();
            for (int i = 0; i < repetitions; ++i) {
#if defined(USE_MKL) || defined(USE_BLAS)
              cblas_sgemm(
                  CblasRowMajor,
                  CblasNoTrans,
                  CblasNoTrans,
                  m,
                  n,
                  k,
                  1.0,
                  A[copy].data(),
                  k,
                  Bt[copy].data(),
                  btran == matrix_op_t::NoTranspose ? kAligned : nAligned,
                  beta,
                  C_ref[copy].data(),
                  n);
#else
            cblas_sgemm_ref(
                matrix_op_t::NoTranspose,
                matrix_op_t::NoTranspose,
                m,
                n,
                k,
                1.0,
                A[copy].data(),
                k,
                Bt[copy].data(),
                (btran == matrix_op_t::NoTranspose) ? kAligned : nAligned,
                beta,
                C_ref[copy].data(),
                n);
#endif
            }
          },
          3,
          NITER,
          [&]() {
            if (flush) {
              int copy = num_instances == 1 ? 0 : fbgemm_get_thread_num();
              cache_evict(A[copy]);
              cache_evict(Bt[copy]);
              cache_evict(C_ref[copy]);
            }
          },
          // Use OpenMP if num instances > 1
          num_instances > 1);

      gflops = nflops / ttot / 1e9;
      gbs = nbytes / ttot / 1e9;
      printf(
          "\n%30s m = %5d n = %5d k = %5d Gflops = %8.4lf GBytes = %8.4lf\n",
          type.c_str(),
          m,
          n,
          k,
          gflops * repetitions,
          gbs * repetitions);
#ifdef USE_MKL
    }
#endif
    type = "FBP_" + std::string(typeid(btype).name());

    ttot = measureWithWarmup(
        [&]() {
          // When executing in data decomposition (single-instance) mode
          // Different threads will access different regions of the same
          // matrices. Thus, copy to be used is always 0. The numbers of
          // threads would be the as number of threads in the parallel
          // region.
          // When running in functional decomposition (multi-instance) mode
          // different matrices are used. The copy to be used selected by
          // thread_id (thread_num), and the number of threads performance
          // the compute of the same instance is 1.
          int copy = num_instances == 1 ? 0 : fbgemm_get_thread_num();
          int num_threads = num_instances == 1 ? fbgemm_get_num_threads() : 1;
          int tid = num_instances == 1 ? fbgemm_get_thread_num() : 0;

          for (int i = 0; i < repetitions; ++i) {
            cblas_gemm_compute(
                matrix_op_t::NoTranspose,
                m,
                A[copy].data(),
                *Bp[copy],
                beta,
                C_fb[copy].data(),
                tid,
                num_threads);
          }
        },
        3,
        NITER,
        [&]() {
          if (flush) {
            int copy = num_instances == 1 ? 0 : fbgemm_get_thread_num();
            cache_evict(A[copy]);
            cache_evict(*Bp[copy]);
            cache_evict(C_fb[copy]);
          }
        },
        true /*useOpenMP*/);

    gflops = nflops / ttot / 1e9;
    gbs = nbytes / ttot / 1e9;
    printf(
        "%30s m = %5d n = %5d k = %5d Gflops = %8.4lf GBytes = %8.4lf\n",
        type.c_str(),
        m,
        n,
        k,
        gflops * repetitions,
        gbs * repetitions);
  }
}

aligned_vector<float> getRandomSparseVector(
    unsigned size,
    float fractionNonZeros = 1.0);

template <typename T>
aligned_vector<T> getRandomBlockSparseMatrix(
    int Rows,
    int Cols,
    float fractionNonZerosBlocks = 1.0,
    int RowBlockSize = 4,
    int ColBlockSize = 1,
    T low = 0,
    T high = 9);

} // namespace fbgemm
