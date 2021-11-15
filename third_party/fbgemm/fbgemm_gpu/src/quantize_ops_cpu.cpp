/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ATen/ATen.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/library.h>
#include "fbgemm/QuantUtils.h"
#include "fbgemm_gpu/sparse_ops_utils.h"

namespace fbgemm {

at::Tensor& _float_to_fused8bitrowwise_cpu_out(
    at::Tensor& output,
    const at::Tensor& input) {
  TENSOR_ON_CPU(input);
  TORCH_CHECK(
      input.dim() >= 2,
      "Tensor 'input' must have >= 2 dimension(s). Found ",
      input.ndimension());

  const auto input_sizes = input.sizes();
  const auto last_dim = input_sizes.size() - 1;
  const int32_t nrows = c10::size_to_dim_(last_dim, input_sizes);
  const int32_t ncols = input_sizes[last_dim];
  const int32_t output_columns = ncols + 2 * sizeof(float);

  auto output_dims = input_sizes.vec();
  output_dims[last_dim] = output_columns;
  at::native::resize_(output, output_dims, c10::nullopt);

  fbgemm::FloatOrHalfToFused8BitRowwiseQuantizedSBFloat<float>(
      input.data_ptr<float>(), nrows, ncols, output.data_ptr<uint8_t>());

  return output;
}

at::Tensor& _fused8bitrowwise_to_float_cpu_out(
    at::Tensor& output,
    const at::Tensor& input) {
  TENSOR_ON_CPU(input);
  TORCH_CHECK(
      input.dim() >= 2,
      "Tensor 'input' must have >= 2 dimension(s). Found ",
      input.ndimension());

  const auto input_sizes = input.sizes();
  const auto last_dim = input_sizes.size() - 1;
  const int32_t nrows = c10::size_to_dim_(last_dim, input_sizes);
  const int32_t ncols = input_sizes[last_dim];
  const int32_t output_columns = ncols - 2 * sizeof(float);

  auto output_dims = input_sizes.vec();
  output_dims[last_dim] = output_columns;
  at::native::resize_(output, output_dims, c10::nullopt);

  fbgemm::Fused8BitRowwiseQuantizedSBFloatToFloatOrHalf<float>(
      input.data_ptr<uint8_t>(), nrows, ncols, output.data_ptr<float>());

  return output;
}


at::Tensor _float_to_fusednbitrowwise_cpu(
    const at::Tensor& input,
    const int64_t bit_rate) {
  TENSOR_ON_CPU(input);
  TENSOR_NDIM_EQUALS(input, 2);

  const auto input_sizes = input.sizes();
  const int32_t nrows = input_sizes[0];
  const int32_t ncols = input_sizes[1];
  const int32_t num_elem_per_byte = 8 / bit_rate;
  TORCH_CHECK(
      ncols % (2 * num_elem_per_byte) == 0,
      "ncols needs to be multiple of 2 Bytes (half type size) to make the address aligned");
  const int32_t output_columns =
      (ncols + num_elem_per_byte - 1) / num_elem_per_byte +
      2 * sizeof(at::Half);
  auto output = at::empty(
      {nrows, output_columns},
      input.options().dtype(at::kByte)); // at::kBytes for uint8_t

  fbgemm::FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf<float>(
      bit_rate,
      input.data_ptr<float>(),
      nrows,
      ncols,
      output.data_ptr<uint8_t>());

  return output;
}

at::Tensor _fusednbitrowwise_to_float_cpu(
    const at::Tensor& input,
    const int64_t bit_rate) {
  TENSOR_ON_CPU(input);
  TENSOR_NDIM_EQUALS(input, 2);

  const auto input_sizes = input.sizes();
  const int32_t nrows = input_sizes[0];
  const int32_t ncols = input_sizes[1];
  const int32_t num_elem_per_byte = 8 / bit_rate;
  const int32_t output_columns =
      (ncols - 2 * sizeof(at::Half)) * num_elem_per_byte;

  auto output = at::empty(
      {nrows, output_columns}, // 4 = sizeof(float)
      input.options().dtype(at::kFloat)); // at::kBytes for uint8_t

  fbgemm::FusedNBitRowwiseQuantizedSBHalfToFloatOrHalf<float>(
      bit_rate,
      input.data_ptr<uint8_t>(),
      nrows,
      ncols,
      output.data_ptr<float>());

  return output;
}

namespace {


at::Tensor _float_to_fused8bitrowwise_cpu(const at::Tensor& input) {
  auto output = at::empty(
      {0},
      input.options().dtype(at::kByte)); // at::kBytes for uint8_t
  return _float_to_fused8bitrowwise_cpu_out(output, input);
}

at::Tensor _fused8bitrowwise_to_float_cpu(const at::Tensor& input) {
  auto output = at::empty(
      {0},
      input.options().dtype(at::kFloat)); // at::kBytes for uint8_t
  return _fused8bitrowwise_to_float_cpu_out(output, input);
}


} // namespace
} // namespace fbgemm

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def("FloatToFused8BitRowwiseQuantized(Tensor t) -> Tensor");
  m.def(
      "FloatToFused8BitRowwiseQuantizedOut(Tensor output, Tensor input) -> Tensor");
  m.def("Fused8BitRowwiseQuantizedToFloat(Tensor input) -> Tensor");
  m.def(
      "Fused8BitRowwiseQuantizedToFloatOut(Tensor output, Tensor input) -> Tensor");
  m.def(
      "FloatToFusedNBitRowwiseQuantizedSBHalf(Tensor input, int bit_rate) -> Tensor");
  m.def(
      "FusedNBitRowwiseQuantizedSBHalfToFloat(Tensor input, int bit_rate) -> Tensor");
}

TORCH_LIBRARY_IMPL(fbgemm, CPU, m) {
  m.impl("FloatToFused8BitRowwiseQuantized", fbgemm::_float_to_fused8bitrowwise_cpu);
  m.impl(
      "FloatToFused8BitRowwiseQuantizedOut", fbgemm::_float_to_fused8bitrowwise_cpu_out);
  m.impl(
      "Fused8BitRowwiseQuantizedToFloat", fbgemm::_fused8bitrowwise_to_float_cpu);
  m.impl(
      "Fused8BitRowwiseQuantizedToFloatOut", fbgemm::_fused8bitrowwise_to_float_cpu_out);
  m.impl(
      "FloatToFusedNBitRowwiseQuantizedSBHalf", fbgemm::_float_to_fusednbitrowwise_cpu);
  m.impl(
      "FusedNBitRowwiseQuantizedSBHalfToFloat", fbgemm::_fusednbitrowwise_to_float_cpu);
}
