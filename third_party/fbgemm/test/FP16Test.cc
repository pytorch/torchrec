/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <gtest/gtest.h>

#include "fbgemm/FbgemmFP16.h"
#include "./FBGemmFPTest.h"

using FBGemmFP16Test = fbgemm::FBGemmFPTest<fbgemm::float16>;

INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    FBGemmFP16Test,
    ::testing::Values(
      std::pair<fbgemm::matrix_op_t, fbgemm::matrix_op_t>(
          fbgemm::matrix_op_t::NoTranspose, fbgemm::matrix_op_t::NoTranspose),
      std::pair<fbgemm::matrix_op_t, fbgemm::matrix_op_t>(
          fbgemm::matrix_op_t::NoTranspose, fbgemm::matrix_op_t::Transpose)/*,
      pair<matrix_op_t, matrix_op_t>(
          matrix_op_t::Transpose, matrix_op_t::NoTranspose),
      pair<matrix_op_t, matrix_op_t>(
          matrix_op_t::Transpose, matrix_op_t::Transpose)*/));

TEST_P(FBGemmFP16Test, Test) {
  TestRun();
}

TEST_P(FBGemmFP16Test, Unpack) {
  UnpackTestRun();
}
