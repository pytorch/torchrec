/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <map>
#include <tuple>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>

#include "codegen/embedding_forward_split_cpu.h"
#include "fbgemm/FbgemmEmbedding.h"

using namespace at;

namespace {
template <typename scalar_t, typename grad_t>
void split_embedding_backward_approx_cpu_kernel(
    Tensor grad_output,
    Tensor host_weights,
    const TensorAccessor<int64_t, 1> weights_offsets_data,
    const TensorAccessor<int, 1> D_offsets_data,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    Tensor indice_weights,
    int T,
    int B,
    {% if "momentum1_offsets" in args.split_function_arg_names %}
    const TensorAccessor<int64_t, 1> momentum1_offsets_data,
    {% endif %}
    {% if "momentum2_offsets" in args.split_function_arg_names %}
    const TensorAccessor<int64_t, 1> momentum2_offsets_data,
    {% endif %}
    {{ args.split_cpu_kernel_args | join(", ") }}) {
  auto grad_output_data = grad_output.accessor<grad_t, 2>();
  auto host_weights_data = host_weights.accessor<scalar_t, 1>();
  const auto indices_data = indices.accessor<int64_t, 1>();
  const auto offsets_data = offsets.accessor<int64_t, 1>();
  // If indice_weights are not defined, then this accessor won't be used
  auto indice_weights_data = indice_weights.defined()
      ? indice_weights.accessor<grad_t, 1>()
      : TensorAccessor<grad_t, 1>(nullptr, nullptr, nullptr);

  for (int64_t t = 0; t < T; ++t) {
    int feature_begin = t; // to conform interface with exact
    const auto D_begin = D_offsets_data[t];
    const auto D = D_offsets_data[t + 1] - D_offsets_data[t];
    const auto table_begin = weights_offsets_data[t];
    at::parallel_for(0, B, 0, [&](int64_t b_begin, int64_t b_end) {
      for (int64_t b = b_begin; b < b_end; ++b) {
        const auto pool_begin = offsets_data[t * B + b];
        const auto pool_end = offsets_data[t * B + b + 1];
        const auto L = pool_end - pool_begin;
        const double scale_factor =
            // NOTE: MEAN pooling will not work with indice_weights!
            (pooling_mode == MEAN && !indice_weights.defined() && L > 0)
            ? 1.0 / L
            : 1.0;
        for (auto p = pool_begin; p < pool_end; ++p) {
          auto idx = indices_data[p];
          const int64_t embedding_begin = table_begin + idx * D;
          scalar_t grad_buffer[D];
          for (int64_t d = 0; d < D; ++d) {
            grad_buffer[d] = scale_factor *
                (indice_weights.defined()
                     ? grad_output_data[b][D_begin + d] * indice_weights_data[p]
                     : grad_output_data[b][D_begin + d]);
          }
          {{ split_weight_update_cpu }};
        } // for each p
      } // for each b
    }); // parallel for B
  } // for each t
}
} // namespace

// The template for approximate optimizers
{{ "void" if not dense else "Tensor" }}
split_embedding_backward_codegen_{{ optimizer }}_cpu(
    Tensor grad_output,
    Tensor host_weights,
    {% if not dense %}
    Tensor weights_placements,
    {% endif %}
    Tensor weights_offsets,
    Tensor D_offsets,
    int64_t max_D,
    Tensor hash_size_cumsum,
    int64_t total_hash_size_bits,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    Tensor indice_weights,
    {% if not dense %}
    bool stochastic_rounding,
    {% endif %}
    {{args.split_function_args | join(", ")}}
) {
  int64_t T = D_offsets.numel() - 1;
  TORCH_CHECK(T > 0);
  // offsets = [T x B  + 1]
  int64_t B = (offsets.size(0) - 1) / T;
  TORCH_CHECK(B >= 0);

  const auto weights_offsets_data = weights_offsets.accessor<int64_t, 1>();
  const auto D_offsets_data = D_offsets.accessor<int, 1>();
  {%if "momentum1_offsets" in args.split_function_arg_names %}
  const auto momentum1_offsets_data = momentum1_offsets.accessor<int64_t, 1>();
  {% endif %}
  {%if "momentum2_offsets" in args.split_function_arg_names %}
  const auto momentum2_offsets_data = momentum2_offsets.accessor<int64_t, 1>();
  {% endif %}

  TORCH_CHECK(host_weights.dim() == 1);

  {% if optimizer == "approx_rowwise_adagrad" %}

  // TODO: fp16 and weighted
  bool use_fbgemm =
      (host_weights.scalar_type() == ScalarType::Float/* ||
       host_weights.scalar_type() == ScalarType::Half*/) &&
      grad_output.scalar_type() == ScalarType::Float &&
      !indice_weights.defined() && pooling_mode == SUM;

  if (use_fbgemm) {
    auto grad_stride = grad_output.size(1);
    const float* grad_output_data = grad_output.data_ptr<float>();
    float* host_weights_data = host_weights.data_ptr<float>();
    const int64_t* indices_data = indices.data_ptr<int64_t>();
    const int64_t* offsets_data = offsets.data_ptr<int64_t>();
    const auto hash_size_cumsum_data = hash_size_cumsum.accessor<int64_t, 1>();
    float* momentum1_data = momentum1_host.data_ptr<float>();

    at::parallel_for(0, T * B, 0, [&](int64_t tb_begin, int64_t tb_end) {
      int t_begin = tb_begin / B;
      int t_end = (tb_end + B - 1) / B;
      for (int t = t_begin; t < t_end; ++t) {
        auto D_begin = D_offsets_data[t];
        auto D = D_offsets_data[t + 1] - D_offsets_data[t];
        auto table_begin = weights_offsets_data[t];
        auto momentum_begin = momentum1_offsets_data[t];

        int64_t hash_size;
        int t_temp = t + 1;
        do {
          hash_size = hash_size_cumsum_data[t_temp] - hash_size_cumsum_data[t];
          ++t_temp;
        } while (hash_size == 0);

        int b_begin = (t == t_begin) ? tb_begin % B : 0;
        int b_end = (t == t_end - 1 && tb_end % B != 0) ? tb_end % B : B;

        auto kernel =
            fbgemm::GenerateRowWiseSparseAdaGradFused<int64_t, int64_t, float>(
                D,
                /*prefetch=*/16,
                /*use_offsets=*/true,
                /*use_stochastic_round=*/true,
                /*grad_stride=*/grad_stride);
        auto offsets_begin_ptr = offsets_data + t * B + b_begin;
        auto index_size = offsets_data[t * B + b_end] - *offsets_begin_ptr;
        bool success = kernel(
            b_end - b_begin,
            index_size,
            hash_size,
            reinterpret_cast<float*>(host_weights_data + table_begin),
            reinterpret_cast<const float*>(
                grad_output_data + b_begin * grad_stride + D_begin),
            reinterpret_cast<float*>(momentum1_data + momentum_begin),
            indices_data + *offsets_begin_ptr,
            offsets_begin_ptr,
            eps,
            // fbgemm follows caffe2 convention of negative learning rate
            -learning_rate);
        // TODO: more friendly error msg.
        // See report_error_ in embedding_forward_split_cpu.cpp
        TORCH_CHECK(success);
      }
    }); // parallel_for
    return;
  } // use_fbgemm

  {% endif %}

  AT_DISPATCH_FLOATING_TYPES(
      grad_output.scalar_type(), "split_embedding_backward_cpu", [&]() {
        using grad_t = scalar_t;
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            host_weights.scalar_type(),
            "split_embedding_backward_cpu_inner",
            [&]() {
              split_embedding_backward_approx_cpu_kernel<scalar_t, grad_t>(
                  grad_output,
                  host_weights,
                  weights_offsets_data,
                  D_offsets_data,
                  indices,
                  offsets,
                  pooling_mode,
                  indice_weights,
                  T,
                  B,
                  {% if "momentum1_offsets" in args.split_function_arg_names %}
                  momentum1_offsets_data,
                  {% endif %}
                  {% if "momentum2_offsets" in args.split_function_arg_names %}
                  momentum2_offsets_data,
                  {% endif %}
                  {{ args.split_cpu_kernel_arg_constructors | join(", ") }});
            }); // dispatch host_weights.scalar_type()
      }); // dispatch grad_output.scalar_type()

  return;
}
