/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "torchrec/inference/Batching.h"
#include "torchrec/inference/Utils.h"

#include <ATen/ATen.h>
#include <iostream>
#include <memory>

#include <folly/Benchmark.h>
#include <folly/io/IOBuf.h>

DEFINE_int64(num_dense_requests, 100, "Number of dense requests.");
DEFINE_int64(dense_batch_size, 100, "Batch size for dense features.");
DEFINE_int64(num_dense_features, 100, "Number of dense features.");

DEFINE_int64(num_sparse_requests, 100, "Number of sparse requests.");
DEFINE_int64(sparse_batch_size, 100, "Batch size for sparse features.");
DEFINE_int64(num_sparse_features, 100, "Number of sparse features.");

DEFINE_int64(num_embedding_requests, 100, "Number of embedding requests.");
DEFINE_int64(embedding_batch_size, 100, "Batch size for embedding features.");
DEFINE_int64(num_embedding_features, 100, "Number of embedding features.");

namespace torchrec {

BENCHMARK(combineFloatIValue, iters) {
  at::Tensor tensor;
  std::shared_ptr<PredictionRequest> request;
  std::vector<std::shared_ptr<PredictionRequest>> requests;

  BENCHMARK_SUSPEND {
    tensor = at::rand(
        {FLAGS_dense_batch_size, FLAGS_num_dense_features},
        at::TensorOptions().dtype(c10::kFloat));
    request = createRequest(tensor);

    for (auto i = 0; i < FLAGS_num_dense_requests; ++i) {
      requests.push_back(request);
    }
  }
  while (iters--) {
    combineFloat("ivalue", requests);
  }
}

BENCHMARK(combineFloatIOBuf, iters) {
  at::Tensor tensor;
  std::shared_ptr<PredictionRequest> request;
  std::vector<std::shared_ptr<PredictionRequest>> requests;

  BENCHMARK_SUSPEND {
    tensor = at::rand(
        {FLAGS_dense_batch_size, FLAGS_num_dense_features},
        at::TensorOptions().dtype(c10::kFloat));
    request = createRequest(tensor);

    for (auto i = 0; i < FLAGS_num_dense_requests; ++i) {
      requests.push_back(request);
    }
  }
  while (iters--) {
    combineFloat("io_buf", requests);
  }
}

BENCHMARK_DRAW_LINE();

BENCHMARK(combineSparse, iters) {
  std::vector<std::vector<int32_t>> input;
  JaggedTensor jagged;
  std::shared_ptr<PredictionRequest> request;
  std::vector<std::shared_ptr<PredictionRequest>> requests;

  BENCHMARK_SUSPEND {
    for (auto i = 0; i < FLAGS_num_sparse_features * FLAGS_sparse_batch_size;
         ++i) {
      input.push_back({1});
    }

    jagged = createJaggedTensor(input);
    request = createRequest(
        FLAGS_sparse_batch_size, FLAGS_num_sparse_features, jagged);

    for (auto i = 0; i < FLAGS_num_sparse_requests; ++i) {
      requests.push_back(request);
    }
  }

  while (iters--) {
    combineSparse("id_score_list_features", requests, true);
  }
}

BENCHMARK_DRAW_LINE();

BENCHMARK(combineEmbedding, iters) {
  std::vector<std::vector<int32_t>> input;
  at::Tensor embedding;
  std::shared_ptr<PredictionRequest> request;
  std::vector<std::shared_ptr<PredictionRequest>> requests;

  BENCHMARK_SUSPEND {
    for (auto i = 0; i < FLAGS_embedding_batch_size; ++i) {
      std::vector<int32_t> row;
      for (auto j = 0; j < FLAGS_num_embedding_features; ++j) {
        row.push_back(1);
      }
      input.push_back(row);
    }

    embedding = createEmbeddingTensor(input);
    request = createRequest(
        FLAGS_embedding_batch_size, FLAGS_num_embedding_features, embedding);

    for (auto i = 0; i < FLAGS_num_embedding_requests; ++i) {
      requests.push_back(request);
    }
  }

  while (iters--) {
    combineEmbedding("embedding_features", requests);
  }
}

} // namespace torchrec

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  folly::runBenchmarks();
  return 0;
}
