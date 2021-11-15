/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <climits>
#include <limits>
#include <random>
#include <sstream>
#include <type_traits>

#include <gtest/gtest.h>

#include "fbgemm/QuantUtils.h"
#include "fbgemm/Types.h"
#include "fbgemm/Utils.h"

using namespace std;
using namespace fbgemm;

// tuple represents K, C, X, G, layout_t
// layout_t can be KCX or KXC
class QuantizeGroupwiseTest
    : public testing::TestWithParam<tuple<int, int, int, int, layout_t>> {};

class QuantizeTest : public testing::TestWithParam<int> {};
class FusedQuantizeDequantizeTest : public testing::TestWithParam<int> {};

// Parameter are bit_rate (i.e., the number of bits in quantized values),
// input rows, and input columns
class EmbeddingQuantizeTest
    : public testing::TestWithParam<tuple<int, int, int>> {};

// Parameter are input rows and input columns
// Scale and Bias are of type float (SBFloat)
class EmbeddingQuantizeSBFloatTest
    : public testing::TestWithParam<tuple<int, int>> {};

INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    QuantizeGroupwiseTest,
    ::testing::Combine(
        ::testing::ValuesIn({4, 12, 64}), // K
        ::testing::ValuesIn({12, 16, 32}), // C
        ::testing::ValuesIn({1, 10, 15, 30}), // X
        ::testing::ValuesIn({1, 4}), // G
        ::testing::ValuesIn({layout_t::KCX, layout_t::KXC})));

INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    QuantizeTest,
    ::testing::Values(1, 2, 5, 8, 9, 16, 20, 28, 32, 33));

INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    FusedQuantizeDequantizeTest,
    ::testing::Values(1, 2, 5, 8, 9, 16, 20, 28, 32, 33));

INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    EmbeddingQuantizeTest,
    ::testing::Combine(
        ::testing::ValuesIn({2, 4, 8}),
        ::testing::ValuesIn({1, 2, 3}),
        ::testing::ValuesIn({4, 8, 16, 20, 28, 32, 64, 84})));

INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    EmbeddingQuantizeSBFloatTest,
    ::testing::Combine(
        ::testing::ValuesIn({1, 2, 3}),
        ::testing::ValuesIn({1, 2, 5, 8, 9, 16, 20, 28, 32, 33, 64, 65})));

template <typename T, layout_t LT>
void ref_impl(
    const vector<float>& src,
    int K,
    int C,
    int X,
    int G,
    const vector<float>& scales,
    const vector<int>& zero_points,
    vector<T>& dst) {
  int C_per_G = C / G;
  for (int i = 0; i < K; ++i) {
    for (int g = 0; g < G; ++g) {
      for (int c = 0; c < C / G; ++c) {
        for (int x = 0; x < X; ++x) {
          float num;
          if (LT == layout_t::KCX) {
            num = src[(i * C + g * C_per_G + c) * X + x];
          } else {
            num = src[(i * X + x) * C + g * C_per_G + c];
          }
          int res = nearbyint(zero_points[g] + num / scales[g]);
          T final_res = min<T>(
              max<T>(res, numeric_limits<T>::min()), numeric_limits<T>::max());
          if (LT == layout_t::KCX) {
            dst[(i * C + g * C_per_G + c) * X + x] = final_res;
          } else {
            dst[(i * X + x) * C + g * C_per_G + c] = final_res;
          }
        }
      }
    }
  }
}

template <typename T, layout_t LT>
void runTests(
    const vector<float>& src,
    int K,
    int C,
    int X,
    int G,
    const vector<float>& scales,
    const vector<int>& zero_points,
    vector<T>& dst,
    vector<T>& dst_ref) {
  QuantizeGroupwise<T, LT>(
      src.data(), K, C, X, G, scales.data(), zero_points.data(), dst.data());

  ref_impl<T, LT>(src, K, C, X, G, scales, zero_points, dst_ref);
}

/**
 * There can be off-by-one error in quantized values due to how the mid-point
 * cases are rounded-off in vectorized vs scalar codes and due to adding of
 * zero_point before rounding vs after rounding. We ignore such differences
 * while comparing results.
 */
template <typename T>
::testing::AssertionResult isNear(
    const vector<T>& res,
    const vector<T>& res_ref) {
  bool match = true;
  if (res.size() == res_ref.size()) {
    for (int i = 0; i < res.size(); ++i) {
      if (!(res[i] == res_ref[i] || res[i] == res_ref[i] + 1 ||
            res[i] == res_ref[i] - 1)) {
        match = false;
        break;
      }
    }
  }
  if (match)
    return ::testing::AssertionSuccess();
  else
    return ::testing::AssertionFailure() << " Quantized results do not match";
}

// atol: absolute tolerance. <=0 means do not consider atol.
// rtol: relative tolerance. <=0 means do not consider rtol.
::testing::AssertionResult floatCloseAll(
    vector<float>& a,
    vector<float>& b,
    float atol = std::numeric_limits<float>::epsilon(),
    float rtol = 0) {
  std::stringstream ss;
  bool match = true;
  if (a.size() != b.size()) {
    ss << " size mismatch ";
    match = false;
  }
  if (match) {
    for (int i = 0; i < a.size(); i++) {
      const bool consider_absDiff = atol > 0;
      const bool consider_relDiff = rtol > 0 &&
          fabs(a[i]) > std::numeric_limits<float>::epsilon() &&
          fabs(b[i]) > std::numeric_limits<float>::epsilon();

      const float absDiff = fabs(a[i] - b[i]);
      const float relDiff = absDiff / fabs(a[i]);

      if (consider_absDiff && consider_relDiff) {
        if (absDiff > atol && relDiff > rtol) {
          ss << " mismatch at (" << i << ") " << endl;
          ss << "\t  ref: " << a[i] << " test: " << b[i] << endl;
          ss << "\t absolute diff: " << absDiff << " > " << atol << endl;
          ss << "\t relative diff: " << relDiff << " > " << rtol << endl;
          match = false;
        }
      } else if (consider_absDiff) {
        if (absDiff > atol) {
          ss << " mismatch at (" << i << ") " << endl;
          ss << "\t  ref: " << a[i] << " test: " << b[i] << endl;
          ss << "\t absolute diff: " << absDiff << " > " << atol << endl;
          match = false;
        }
      } else if (consider_relDiff) {
        if (relDiff > rtol) {
          ss << " mismatch at (" << i << ") " << endl;
          ss << "\t  ref: " << a[i] << " test: " << b[i] << endl;
          ss << "\t relative diff: " << relDiff << " > " << rtol << endl;
          match = false;
        }
      }
    }
  }
  if (match)
    return ::testing::AssertionSuccess();
  else
    return ::testing::AssertionFailure()
        << " results do not match. " << ss.str();
}

::testing::AssertionResult floatCloseAll(
    vector<float>& a,
    vector<float16>& b,
    float atol = std::numeric_limits<float>::epsilon(),
    float rtol = 0) {
  vector<float> b_float(b.size());
  const auto transform = [](float16 input) { return cpu_half2float(input); };
  std::transform(b.begin(), b.end(), b_float.begin(), transform);
  return floatCloseAll(a, b_float, atol, rtol);
}

::testing::AssertionResult floatCloseAll(
    vector<float16>& a,
    vector<float16>& b,
    float atol = std::numeric_limits<float>::epsilon(),
    float rtol = 0) {
  vector<float> a_float(a.size());
  vector<float> b_float(b.size());
  const auto transform = [](float16 input) { return cpu_half2float(input); };
  std::transform(a.begin(), a.end(), a_float.begin(), transform);
  std::transform(b.begin(), b.end(), b_float.begin(), transform);
  return floatCloseAll(a_float, b_float, atol, rtol);
}

template <typename T>
::testing::AssertionResult isQEmbeddingClose(
    const vector<uint8_t>& res_ref,
    const vector<uint8_t>& res,
    int out_rows,
    int out_emb_cols) {
  bool match = true;
  std::stringstream ss;
  int ld = out_emb_cols + 2 * sizeof(T);

  if (res.size() == res_ref.size()) {
    for (int i = 0; i < out_rows; ++i) {
      if (!match) {
        break;
      }
      // compare embedding values
      for (int j = 0; j < out_emb_cols; ++j) {
        if (res[i * ld + j] != res_ref[i * ld + j]) {
          match = false;
          ss << " mismatch at (" << i << ", " << j << ") ";
          ss << "ref: " << static_cast<uint32_t>(res_ref[i * ld + j])
             << ", test: " << static_cast<uint32_t>(res[i * ld + j]) << "\n";
          break;
        }
      }
      // compare scale/bias
      float scaleTest, scaleRef, biasTest, biasRef;
      if (is_same<T, float16>::value) {
        // half scale and bias
        scaleTest = cpu_half2float(reinterpret_cast<const float16*>(
            res.data() + i * ld + out_emb_cols)[0]);
        biasTest = cpu_half2float(reinterpret_cast<const float16*>(
            res.data() + i * ld + out_emb_cols)[1]);
        scaleRef = cpu_half2float(reinterpret_cast<const float16*>(
            res_ref.data() + i * ld + out_emb_cols)[0]);
        biasRef = cpu_half2float(reinterpret_cast<const float16*>(
            res_ref.data() + i * ld + out_emb_cols)[1]);
      } else {
        // float scale and bias
        scaleTest = reinterpret_cast<const float*>(
            res.data() + i * ld + out_emb_cols)[0];
        biasTest = reinterpret_cast<const float*>(
            res.data() + i * ld + out_emb_cols)[1];
        scaleRef = reinterpret_cast<const float*>(
            res_ref.data() + i * ld + out_emb_cols)[0];
        biasRef = reinterpret_cast<const float*>(
            res_ref.data() + i * ld + out_emb_cols)[1];
      }
      if (fabs(scaleTest - scaleRef) > std::numeric_limits<float>::epsilon()) {
        ss << " scale mismatch for row:" << i;
        ss << " ref: " << scaleRef << ", test: " << scaleTest << "\n";
        match = false;
      }
      if (fabs(biasTest - biasRef) > std::numeric_limits<float>::epsilon()) {
        ss << " bias mismatch for row:" << i;
        ss << " ref: " << biasRef << ", test: " << biasTest << "\n";
        match = false;
      }
    }
  } else {
    ss << " size mismatch ";
    match = false;
  }

  if (match)
    return ::testing::AssertionSuccess();
  else
    return ::testing::AssertionFailure()
        << " Quantized Embeddings do not match." << ss.str();
}

/**
 * Test for QuantizeGroupwise
 */
TEST_P(QuantizeGroupwiseTest, quantizeGTest) {
  int K, C, X, G;
  layout_t layout;
  tie(K, C, X, G, layout) = GetParam();

  random_device rd;
  mt19937 gen(rd());

  uniform_real_distribution<float> disFP(0.1, 1.1);

  vector<float> inp(K * C * X);
  generate(inp.begin(), inp.end(), [&, disFP]() mutable { return disFP(gen); });

  vector<float> scales(G);
  generate(scales.begin(), scales.end(), [&, disFP]() mutable {
    return disFP(gen);
  });

  uniform_int_distribution<> disUInt8(0, 8);
  vector<int> zero_points_uint8(G);
  generate(
      zero_points_uint8.begin(),
      zero_points_uint8.end(),
      [&, disUInt8]() mutable { return disUInt8(gen); });

  uniform_int_distribution<> disInt8(-64, 63);
  vector<int> zero_points_int8(G);
  generate(
      zero_points_int8.begin(), zero_points_int8.end(), [&, disInt8]() mutable {
        return disInt8(gen);
      });

  uniform_int_distribution<> disInt32(-512, 512);
  vector<int> zero_points_int32(G);
  generate(
      zero_points_int32.begin(),
      zero_points_int32.end(),
      [&, disInt32]() mutable { return disInt32(gen); });

  vector<uint8_t> dstuint8(K * C * X);
  vector<uint8_t> dstuint8_ref(K * C * X);

  vector<int8_t> dstint8(K * C * X);
  vector<int8_t> dstint8_ref(K * C * X);

  vector<int32_t> dstint32(K * C * X);
  vector<int32_t> dstint32_ref(K * C * X);

  if (layout == layout_t::KCX) {
    runTests<uint8_t, layout_t::KCX>(
        inp, K, C, X, G, scales, zero_points_uint8, dstuint8, dstuint8_ref);
    runTests<int8_t, layout_t::KCX>(
        inp, K, C, X, G, scales, zero_points_int8, dstint8, dstint8_ref);
    runTests<int32_t, layout_t::KCX>(
        inp, K, C, X, G, scales, zero_points_int32, dstint32, dstint32_ref);
  } else {
    runTests<uint8_t, layout_t::KXC>(
        inp, K, C, X, G, scales, zero_points_uint8, dstuint8, dstuint8_ref);
    runTests<int8_t, layout_t::KXC>(
        inp, K, C, X, G, scales, zero_points_int8, dstint8, dstint8_ref);
    runTests<int32_t, layout_t::KXC>(
        inp, K, C, X, G, scales, zero_points_int32, dstint32, dstint32_ref);
  }

  EXPECT_TRUE(isNear(dstuint8, dstuint8_ref));
  EXPECT_TRUE(isNear(dstint8, dstint8_ref));
  EXPECT_TRUE(isNear(dstint32, dstint32_ref));
}

template <typename T>
void runQuantizeTests(
    const vector<float>& src,
    float scale,
    int zero_point,
    vector<T>& dst,
    vector<T>& dst_ref) {
  // reference
  for (int i = 0; i < src.size(); ++i) {
    dst_ref[i] = Quantize<T>(src[i], zero_point, scale, CHAR_BIT * sizeof(T));
  }

  TensorQuantizationParams qparams;
  qparams.scale = scale;
  qparams.zero_point = zero_point;
  qparams.precision = CHAR_BIT * sizeof(T);

  Quantize<T>(src.data(), dst.data(), src.size(), qparams);
}

/**
 * Test for QuantizeGroupwise
 */
TEST_P(QuantizeTest, quantizeTest) {
  int len;
  len = GetParam();

  random_device rd;
  mt19937 gen(rd());

  uniform_real_distribution<float> disFP(-1.0e6, 1.0e6);

  vector<float> inp(len);
  generate(inp.begin(), inp.end(), [&, disFP]() mutable { return disFP(gen); });

  float scale = disFP(gen);

  // Generate a number between [0, 255] both inclusive
  uniform_int_distribution<> disUInt8(0, 255);
  int zero_point_uint8 = disUInt8(gen);

  uniform_int_distribution<> disInt8(-128, 127);
  int zero_point_int8 = disInt8(gen);

  vector<uint8_t> dstuint8(len);
  vector<uint8_t> dstuint8_ref(len);

  vector<int8_t> dstint8(len);
  vector<int8_t> dstint8_ref(len);

  runQuantizeTests<uint8_t>(
      inp, scale, zero_point_uint8, dstuint8, dstuint8_ref);
  runQuantizeTests<int8_t>(inp, scale, zero_point_int8, dstint8, dstint8_ref);

  EXPECT_TRUE(isNear(dstuint8, dstuint8_ref));
  EXPECT_TRUE(isNear(dstint8, dstint8_ref));
}

// vector and scalar code should have the same behavior
TEST(QuantizeTestSingle, vectorScalar) {
  // This length will exercise both the vector and scalar path
  int len = 33;
  vector<float> src(len);
  vector<uint8_t> dst(len);

  for (int i = 0; i < len; ++i) {
    src[i] = -2.9483526e-05;
  }
  float scale = 2.3124334356729307e-07;
  int zero_point = 128;

  TensorQuantizationParams qparams;
  qparams.scale = scale;
  qparams.zero_point = zero_point;
  qparams.precision = CHAR_BIT * sizeof(uint8_t);

  Quantize<uint8_t>(src.data(), dst.data(), len, qparams);

  // Check if all elements are equal
  EXPECT_TRUE(
      adjacent_find(dst.begin(), dst.end(), not_equal_to<int>()) == dst.end());
}

TEST(QuantizeTest, cornerCases) {
  TensorQuantizationParams qparams;
  qparams.scale = 1.19209e-07;
  qparams.zero_point = 0;
  qparams.precision = 8;
  std::vector<float> src1 = {3.40282e+38, -2.16845e+38};

  std::vector<int8_t> dst_int8(src1.size());
  Quantize<int8_t>(src1.data(), dst_int8.data(), dst_int8.size(), qparams);
  EXPECT_EQ(dst_int8[0], 127);
  EXPECT_EQ(dst_int8[1], -128);

  // Tests vectorized and remainder paths
  std::vector<float> src2 = {
      3.40282e+38,
      -2.16845e+38,
      3.40282e+38,
      -2.16845e+38,
      3.40282e+38,
      -2.16845e+38,
      3.40282e+38,
      -2.16845e+38,
      3.40282e+38};
  std::vector<uint8_t> dst_uint8(src2.size());
  Quantize<uint8_t>(src2.data(), dst_uint8.data(), dst_uint8.size(), qparams);
  EXPECT_EQ(dst_uint8[0], 255);
  EXPECT_EQ(dst_uint8[1], 0);
  EXPECT_EQ(dst_uint8[8], 255);

  qparams.precision = 16;
  std::vector<int16_t> dst_int16(src2.size());
  Quantize<int16_t>(src2.data(), dst_int16.data(), dst_int16.size(), qparams);
  EXPECT_EQ(dst_int16[0], 32767);
  EXPECT_EQ(dst_int16[1], -32768);
}

TEST(QuantizeTestQParams, chooseQParamsSymmetric) {
  // Test that symmetric quantization of weights set zero point exactly to 0.
  float min = -1.6165;
  float max = 0.5685;
  int32_t qmin = -128;
  int32_t qmax = 127;

  bool preserve_sparsity = true;

  TensorQuantizationParams result =
      ChooseQuantizationParams(min, max, qmin, qmax, preserve_sparsity);
  EXPECT_FLOAT_EQ(result.scale, 0.012628906);
  EXPECT_EQ(result.zero_point, 0);
}

template <typename T>
void runFusedQuantizeDequantizeTests(
    const vector<float>& src,
    float scale,
    int zero_point,
    vector<float>& dst,
    vector<float>& dst_ref) {
  TensorQuantizationParams qparams;
  qparams.scale = scale;
  qparams.zero_point = zero_point;
  qparams.precision = CHAR_BIT * sizeof(T);
  // reference
  for (int i = 0; i < src.size(); ++i) {
    dst_ref[i] = FusedQuantizeDequantize<T>(src[i], qparams);
  }
  FusedQuantizeDequantize<T>(src.data(), dst.data(), src.size(), qparams);
}

TEST_P(FusedQuantizeDequantizeTest, fusedQuantizeDequantizeTest) {
  int len;
  len = GetParam();

  random_device rd;
  mt19937 gen(rd());

  uniform_real_distribution<float> disFP(-1.0e6, 1.0e6);

  vector<float> inp(len);
  generate(inp.begin(), inp.end(), [&, disFP]() mutable { return disFP(gen); });

  float scale = disFP(gen);

  // Generate a number between [0, 255] both inclusive
  uniform_int_distribution<> disUInt8(0, 255);
  int zero_point_uint8 = disUInt8(gen);

  uniform_int_distribution<> disInt8(-128, 127);
  int zero_point_int8 = disInt8(gen);

  vector<float> dstfloat(len);
  vector<float> dstfloat_ref(len);

  runFusedQuantizeDequantizeTests<uint8_t>(
      inp, scale, zero_point_uint8, dstfloat, dstfloat_ref);
  EXPECT_TRUE(floatCloseAll(dstfloat, dstfloat_ref));

  runFusedQuantizeDequantizeTests<int8_t>(
      inp, scale, zero_point_int8, dstfloat, dstfloat_ref);
  EXPECT_TRUE(floatCloseAll(dstfloat, dstfloat_ref));
}

// vector and scalar code should have the same behavior
TEST(FusedQuantizeDequantizeTestSingle, vectorScalar) {
  // This length will exercise both the vector and scalar path
  int len = 33;
  vector<float> src(len);
  vector<float> dst(len);

  for (int i = 0; i < len; ++i) {
    src[i] = -2.9483526e-05;
  }
  float scale = 2.3124334356729307e-07;
  int zero_point = 128;

  TensorQuantizationParams qparams;
  qparams.scale = scale;
  qparams.zero_point = zero_point;
  qparams.precision = CHAR_BIT * sizeof(uint8_t);

  FusedQuantizeDequantize<uint8_t>(src.data(), dst.data(), src.size(), qparams);
  // Check if all elements are equal
  EXPECT_TRUE(
      adjacent_find(dst.begin(), dst.end(), not_equal_to<float>()) ==
      dst.end());
}

TEST(FusedQuantizeDequantizeTest, cornerCases) {
  TensorQuantizationParams qparams;
  qparams.scale = 1.19209e-07;
  qparams.zero_point = 0;
  qparams.precision = 8;
  vector<float> src1 = {3.40282e+38, -2.16845e+38};
  vector<float> ref = {1.5139543e-05, -1.5258752e-05};

  vector<float> dst_int8(src1.size());

  FusedQuantizeDequantize<int8_t>(
      src1.data(), dst_int8.data(), src1.size(), qparams);
  EXPECT_TRUE(floatCloseAll(dst_int8, ref));

  // Tests vectorized and remainder paths
  vector<float> src2 = {
      3.40282e+38,
      -2.16845e+38,
      3.40282e+38,
      -2.16845e+38,
      3.40282e+38,
      -2.16845e+38,
      3.40282e+38,
      -2.16845e+38,
      3.40282e+38};

  vector<float> ref2 = {
      3.0398295e-05,
      0,
      3.0398295e-05,
      0,
      3.0398295e-05,
      0,
      3.0398295e-05,
      0,
      3.0398295e-05};

  std::vector<float> dst_uint8(src2.size(), 0);

  FusedQuantizeDequantize<uint8_t>(
      src2.data(), dst_uint8.data(), src2.size(), qparams);

  EXPECT_TRUE(floatCloseAll(dst_uint8, ref2));
}

// Parameter are bit_rate (i.e., the number of bits in quantized values).
class EmbeddingQuantizeFixedNumberTest : public testing::TestWithParam<int> {
 protected:
  // clang-format off
  EmbeddingQuantizeFixedNumberTest() {
    float_test_input = {
      1, 1, 1, 1,               // All the same. Range: 0, min: 1
      -64, -2.75, 61.625, 191,  // Range: 255, min: -64. Picking 61.625 because it differs under FP16 (will become 61.5).
    };
    assert(float_test_input.size() == row * col);

    float16_test_input.reserve(float_test_input.size());
    std::transform(
        float_test_input.begin(),
        float_test_input.end(),
        float16_test_input.begin(),
        [](float input) { return cpu_float2half_rn(input); });

    // Results are hand calculated.
    expected_output_half[8] = {
      0, 0, 0, 0,       0x00, 0x3c, 0x00, 0x3c,   // Scale: 1, bias: 1
      0, 61, 126, 255,  0x00, 0x3c, 0x00, 0xd4,   // Scale: 1, bias: -64
    };
    expected_output_half[4] = {
      0x00, 0x00, 0x00, 0x3c, 0x00, 0x3c,    // 0, 0, 0, 0, Scale: 1, bias: 1
      0x40, 0xf7, 0x40, 0x4c, 0x00, 0xd4,    // 0, 4, 7, 15, Scale: 17, bias: -64
      0, 0, 0, 0                             // Padding
    };
    expected_output_half[2] = {
      0b00000000, 0x00, 0x3c, 0x00, 0x3c,    // 0, 0, 0, 0, Scale: 1, bias: 1
      0b11010100, 0x50, 0x55, 0x00, 0xd4,    // 0, 1, 1, 3, Scale: 85, bias: -64
      0, 0, 0, 0, 0, 0                       // Padding
    };
    expected_output_float = {
      0, 0, 0, 0,       0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f,   // Scale: 0, bias: 1
      0, 61, 126, 255,  0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x80, 0xc2,   // Scale: 1, bias: -64
    };
  }
  // clang-format on

  const int row = 2;
  const int col = 4;
  const int out_cols_half = col + 2 * sizeof(float16);
  const int out_cols_float = col + 2 * sizeof(float);
  std::vector<float> float_test_input;
  std::vector<float16> float16_test_input;
  std::map</*bit_rate*/ int, /*output*/ std::vector<uint8_t>>
      expected_output_half;
  std::vector<uint8_t> expected_output_float;
};

INSTANTIATE_TEST_CASE_P(
    InstantiationName,
    EmbeddingQuantizeFixedNumberTest,
    ::testing::ValuesIn({2, 4, 8}));

TEST_P(EmbeddingQuantizeFixedNumberTest, embeddingFloatToQuantizedSBHalfTest) {
  const int bit_rate = GetParam();
  vector<uint8_t> outVectHalfTest(row * out_cols_half);

  FloatOrHalfToFusedNBitRowwiseQuantizedSBHalfRef<float>(
      bit_rate, float_test_input.data(), row, col, outVectHalfTest.data());
  EXPECT_TRUE(isQEmbeddingClose<float16>(
      expected_output_half[bit_rate], outVectHalfTest, row, col));
  FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf<float>(
      bit_rate, float_test_input.data(), row, col, outVectHalfTest.data());
  EXPECT_TRUE(isQEmbeddingClose<float16>(
      expected_output_half[bit_rate], outVectHalfTest, row, col));

  FloatOrHalfToFusedNBitRowwiseQuantizedSBHalfRef<float16>(
      bit_rate, float16_test_input.data(), row, col, outVectHalfTest.data());
  EXPECT_TRUE(isQEmbeddingClose<float16>(
      expected_output_half[bit_rate], outVectHalfTest, row, col));
  FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf<float16>(
      bit_rate, float16_test_input.data(), row, col, outVectHalfTest.data());
  EXPECT_TRUE(isQEmbeddingClose<float16>(
      expected_output_half[bit_rate], outVectHalfTest, row, col));

  vector<uint8_t> outVecFloatTest(row * out_cols_float);
  FloatOrHalfToFused8BitRowwiseQuantizedSBFloatRef<float>(
      float_test_input.data(), row, col, outVecFloatTest.data());
  EXPECT_TRUE(isQEmbeddingClose<float>(
      expected_output_float, outVecFloatTest, row, col));
  FloatOrHalfToFused8BitRowwiseQuantizedSBFloat<float>(
      float_test_input.data(), row, col, outVecFloatTest.data());
  EXPECT_TRUE(isQEmbeddingClose<float>(
      expected_output_float, outVecFloatTest, row, col));

  FloatOrHalfToFused8BitRowwiseQuantizedSBFloatRef<float16>(
      float16_test_input.data(), row, col, outVecFloatTest.data());
  EXPECT_TRUE(isQEmbeddingClose<float>(
      expected_output_float, outVecFloatTest, row, col));
  FloatOrHalfToFused8BitRowwiseQuantizedSBFloat<float16>(
      float16_test_input.data(), row, col, outVecFloatTest.data());
  EXPECT_TRUE(isQEmbeddingClose<float>(
      expected_output_float, outVecFloatTest, row, col));
}

// Scale and bias are of type float16
TEST_P(EmbeddingQuantizeTest, embeddingHalfTest) {
  int bit_rate, rows, cols;
  tie(bit_rate, rows, cols) = GetParam();

  random_device rd;
  mt19937 gen(rd());

  uniform_real_distribution<float> disFP(-10.0f, 10.0f);

  vector<float> inpVec(rows * cols);
  vector<float> dequantOutRef(rows * cols);
  vector<float> dequantOutTest(rows * cols);

  generate(inpVec.begin(), inpVec.end(), [&, disFP]() mutable {
    return disFP(gen);
  });

  int elements_per_byte = 8 / bit_rate;

  int out_emb_cols = (cols + elements_per_byte - 1) / elements_per_byte;
  int out_cols = out_emb_cols + 2 * sizeof(float16);
  int outVecSize = rows * out_cols;

  vector<uint8_t> outVecRef(outVecSize);
  vector<uint8_t> outVecTest(outVecSize);

  FloatOrHalfToFusedNBitRowwiseQuantizedSBHalfRef<float>(
      bit_rate, inpVec.data(), rows, cols, outVecRef.data());
  FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf<float>(
      bit_rate, inpVec.data(), rows, cols, outVecTest.data());
  EXPECT_TRUE(
      isQEmbeddingClose<float16>(outVecRef, outVecTest, rows, out_emb_cols));

  FusedNBitRowwiseQuantizedSBHalfToFloatOrHalfRef<float>(
      bit_rate, outVecTest.data(), rows, out_cols, dequantOutRef.data());
  FusedNBitRowwiseQuantizedSBHalfToFloatOrHalf<float>(
      bit_rate, outVecTest.data(), rows, out_cols, dequantOutTest.data());
  EXPECT_TRUE(floatCloseAll(dequantOutRef, dequantOutTest, 1e-3));

  generate(inpVec.begin(), inpVec.end(), [&, disFP]() mutable {
    return cpu_half2float(cpu_float2half_rn(disFP(gen)));
  });
  vector<float16> inpHalfVec(rows * cols);
  std::transform(
      inpVec.begin(), inpVec.end(), inpHalfVec.begin(), [](float input) {
        return cpu_float2half_rn(input);
      });
  vector<uint8_t> outVecRefFromHalf(outVecSize);
  vector<uint8_t> outVecTestFromHalf(outVecSize);
  FloatOrHalfToFusedNBitRowwiseQuantizedSBHalfRef<float>(
      bit_rate, inpVec.data(), rows, cols, outVecRef.data());
  FloatOrHalfToFusedNBitRowwiseQuantizedSBHalfRef<float16>(
      bit_rate, inpHalfVec.data(), rows, cols, outVecRefFromHalf.data());
  EXPECT_TRUE(isQEmbeddingClose<float16>(
      outVecRefFromHalf, outVecRef, rows, out_emb_cols));
  FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf<float16>(
      bit_rate, inpHalfVec.data(), rows, cols, outVecTestFromHalf.data());
  EXPECT_TRUE(isQEmbeddingClose<float16>(
      outVecRefFromHalf, outVecTestFromHalf, rows, out_emb_cols));

  vector<float16> dequantOutHalfRef(rows * cols);
  vector<float16> dequantOutHalfTest(rows * cols);
  FusedNBitRowwiseQuantizedSBHalfToFloatOrHalfRef<float>(
      bit_rate, outVecRef.data(), rows, out_cols, dequantOutRef.data());
  FusedNBitRowwiseQuantizedSBHalfToFloatOrHalfRef<float16>(
      bit_rate, outVecRef.data(), rows, out_cols, dequantOutHalfRef.data());
  constexpr int NumberOfFP16Matissa = 9;
  EXPECT_TRUE(floatCloseAll(
      dequantOutRef, dequantOutHalfRef, 1e-3, pow(2, NumberOfFP16Matissa)));
  FusedNBitRowwiseQuantizedSBHalfToFloatOrHalf<float16>(
      bit_rate, outVecRef.data(), rows, out_cols, dequantOutHalfTest.data());
  EXPECT_TRUE(floatCloseAll(
      dequantOutHalfRef,
      dequantOutHalfTest,
      1e-3,
      pow(2, NumberOfFP16Matissa)));
}

// Scale and bias are of type float
TEST_P(EmbeddingQuantizeSBFloatTest, embeddingFloatTest) {
  int rows, cols;
  tie(rows, cols) = GetParam();

  random_device rd;
  mt19937 gen(rd());

  uniform_real_distribution<float> disFP(-10.0f, 10.0f);

  vector<float> inpVec(rows * cols);
  vector<float> dequantOutTest(rows * cols);
  vector<float> dequantOutRef(rows * cols);

  generate(inpVec.begin(), inpVec.end(), [&, disFP]() mutable {
    return disFP(gen);
  });

  int out_cols = cols + 2 * sizeof(float);
  int outVecSize = rows * out_cols;

  vector<uint8_t> outVecRef(outVecSize);
  vector<uint8_t> outVecTest(outVecSize);

  FloatOrHalfToFused8BitRowwiseQuantizedSBFloatRef<float>(
      inpVec.data(), rows, cols, outVecRef.data());
  FloatOrHalfToFused8BitRowwiseQuantizedSBFloat<float>(
      inpVec.data(), rows, cols, outVecTest.data());

  // The number of input columns is the same as the number of output columns
  EXPECT_TRUE(isQEmbeddingClose<float>(outVecRef, outVecTest, rows, cols));

  Fused8BitRowwiseQuantizedSBFloatToFloatOrHalfRef<float>(
      outVecTest.data(), rows, out_cols, dequantOutRef.data());
  Fused8BitRowwiseQuantizedSBFloatToFloatOrHalf<float>(
      outVecTest.data(), rows, out_cols, dequantOutTest.data());
  EXPECT_TRUE(floatCloseAll(dequantOutRef, dequantOutTest, 1e-3));

  generate(inpVec.begin(), inpVec.end(), [&, disFP]() mutable {
    return cpu_half2float(cpu_float2half_rn(disFP(gen)));
  });
  vector<float16> inpHalfVec(rows * cols);
  std::transform(
      inpVec.begin(), inpVec.end(), inpHalfVec.begin(), [](float input) {
        return cpu_float2half_rn(input);
      });
  vector<uint8_t> outVecRefFromHalf(outVecSize);
  vector<uint8_t> outVecTestFromHalf(outVecSize);
  FloatOrHalfToFused8BitRowwiseQuantizedSBFloatRef<float>(
      inpVec.data(), rows, cols, outVecRef.data());
  FloatOrHalfToFused8BitRowwiseQuantizedSBFloatRef<float16>(
      inpHalfVec.data(), rows, cols, outVecRefFromHalf.data());
  EXPECT_TRUE(
      isQEmbeddingClose<float16>(outVecRefFromHalf, outVecRef, rows, cols));
  FloatOrHalfToFused8BitRowwiseQuantizedSBFloat<float16>(
      inpHalfVec.data(), rows, cols, outVecTestFromHalf.data());
  EXPECT_TRUE(isQEmbeddingClose<float16>(
      outVecRefFromHalf, outVecTestFromHalf, rows, cols));

  vector<float16> dequantOutHalfRef(rows * cols);
  vector<float16> dequantOutHalfTest(rows * cols);
  Fused8BitRowwiseQuantizedSBFloatToFloatOrHalfRef<float>(
      outVecRef.data(), rows, out_cols, dequantOutRef.data());
  Fused8BitRowwiseQuantizedSBFloatToFloatOrHalfRef<float16>(
      outVecRef.data(), rows, out_cols, dequantOutHalfRef.data());
  constexpr int NumberOfFP16Matissa = 9;
  EXPECT_TRUE(floatCloseAll(
      dequantOutRef, dequantOutHalfRef, 1e-3, pow(2, NumberOfFP16Matissa)));
  Fused8BitRowwiseQuantizedSBFloatToFloatOrHalf<float16>(
      outVecRef.data(), rows, out_cols, dequantOutHalfTest.data());
  EXPECT_TRUE(floatCloseAll(
      dequantOutHalfRef,
      dequantOutHalfTest,
      1e-3,
      pow(2, NumberOfFP16Matissa)));
}
