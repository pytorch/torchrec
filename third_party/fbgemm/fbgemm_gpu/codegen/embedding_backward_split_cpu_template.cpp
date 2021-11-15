/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <map>
#include <tuple>
#include <utility>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>

#include "codegen/embedding_forward_split_cpu.h"
#include "fbgemm/FbgemmEmbedding.h"
#include "fbgemm/Types.h"

using namespace at;

namespace internal {
template <typename T>
struct half2float16 {
  using type = T;
};

template <>
struct half2float16<at::Half> {
  using type = fbgemm::float16;
};
} // namespace internal

namespace {
template <typename scalar_t>
void split_embedding_backward_exact_cpu_kernel(
    Tensor grad_output,
    Tensor host_weights,
    const TensorAccessor<int64_t, 1> weights_offsets_data,
    const TensorAccessor<int, 1> D_offsets_data,
    Tensor hash_size_cumsum,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    Tensor indice_weights,
    int num_tables,
    int B,
    const int* table_to_feature_offset,
    {% if "momentum1_offsets" in args.split_function_arg_names %}
    const TensorAccessor<int64_t, 1> momentum1_offsets_data,
    {% endif %}
    {% if "momentum2_offsets" in args.split_function_arg_names %}
    const TensorAccessor<int64_t, 1> momentum2_offsets_data,
    {% endif %}
    {{ args.split_cpu_kernel_args | join(", ") }}) {
  using grad_t = acc_type<scalar_t, true>;

  // const auto grad_output_accessor = grad_output.accessor<grad_t, 2>();
  const grad_t* grad_output_data = grad_output.data_ptr<grad_t>();
  auto host_weights_data = host_weights.accessor<scalar_t, 1>();
  const auto hash_size_cumsum_data = hash_size_cumsum.accessor<int64_t, 1>();

  const bool has_weights = indice_weights.defined();
  auto grad_stride = grad_output.size(1);

  std::vector<::internal::BatchedHyperCompressedSparseColumn> batched_cscs(
      num_tables);

  auto get_hash_size = [&hash_size_cumsum_data](int feature_begin) {
    int64_t hash_size;
    int t_temp = feature_begin + 1;
    do {
      hash_size =
          hash_size_cumsum_data[t_temp] - hash_size_cumsum_data[feature_begin];
      ++t_temp;
    } while (hash_size == 0);
    TORCH_CHECK(
        hash_size < ((1L << 31) - 1),
        "CPU exact rowwise adagrad currently doesn't support embedding tables "
        "with more than 2B rows");
    return hash_size;
  };

  for (int t = 0; t < num_tables; ++t) {
    int feature_begin = table_to_feature_offset[t];
    int64_t hash_size = get_hash_size(feature_begin);

    ::internal::batched_csr2csc(
        batched_cscs[t],
        B,
        offsets.accessor<int64_t, 1>(),
        indices.accessor<int64_t, 1>(),
        indice_weights.defined()
            ? indice_weights.accessor<grad_t, 1>()
            : TensorAccessor<grad_t, 1>(nullptr, nullptr, nullptr),
        pooling_mode,
        table_to_feature_offset + t,
        hash_size);
  }
  // sort based csr2csc handles segment_ids differently
  bool is_csr2csc_sort = batched_cscs[0].weights == nullptr;

  for (int t = 0; t < num_tables; ++t) {
    int feature_begin = table_to_feature_offset[t];

    int c_begin = batched_cscs[t].table_ptr[0];
    int c_end = batched_cscs[t].table_ptr[1];
    int* col_segment_ptr = batched_cscs[t].column_segment_ptr;
    int* col_segment_indices = batched_cscs[t].column_segment_indices;

    auto hash_size = get_hash_size(feature_begin);

    const auto D_begin = D_offsets_data[feature_begin];
    const auto D =
        D_offsets_data[feature_begin + 1] - D_offsets_data[feature_begin];
    const auto table_begin = weights_offsets_data[feature_begin];
    bool is_shared_table =
        table_to_feature_offset[t + 1] > table_to_feature_offset[t] + 1;

    {% if optimizer == "rowwise_adagrad" %}
    constexpr bool use_fbgemm = std::is_same<scalar_t, float>::value;
    // || std::is_same<scalar_t, at::Half>::value;
    if (use_fbgemm && !is_shared_table) {
      // fbgemm handles common case of no shared table
      using fbgemm_weight_t = typename ::internal::half2float16<scalar_t>::type;
      auto spmdm_kernel = fbgemm::GenerateEmbeddingSpMDMWithStrides<
          fbgemm_weight_t,
          /*IndexType=*/int32_t,
          /*OffsetType=*/int32_t>(
          D,
          batched_cscs[t].weights != nullptr,
          /*normalize_by_lengths=*/false,
          /*prefetch=*/16,
          /*is_weight_positional=*/false,
          /*use_offsets=*/true,
          /*output_stride=*/-1,
          /*input_stride=*/grad_stride);
      auto rowwise_adagrad_kernel =
          fbgemm::GenerateSparseAdaGrad</*IndexType=*/int>(D, /*rowwise=*/true);

      constexpr int C_BLOCK = 64;
      at::parallel_for(c_begin, c_end, C_BLOCK, [&](int64_t c0, int64_t c1) {
        grad_t grad_blocked_buffer[C_BLOCK * D];
        for (int64_t c = c0; c < c1; c += C_BLOCK) {
          const int* offsets_begin_ptr = col_segment_ptr + c;
          int64_t c_block_end = std::min(c + C_BLOCK, c1);
          bool success = spmdm_kernel(
              c_block_end - c,
              col_segment_ptr[c_block_end] - *offsets_begin_ptr,
              B,
              reinterpret_cast<const fbgemm_weight_t*>(
                  grad_output_data + D_begin),
              batched_cscs[t].row_indices + *offsets_begin_ptr,
              offsets_begin_ptr,
              batched_cscs[t].weights == nullptr
                  ? nullptr
                  : batched_cscs[t].weights + *offsets_begin_ptr,
              reinterpret_cast<float*>(grad_blocked_buffer));
          // TODO: more friendly error msg.
          TORCH_CHECK(success);
          int num_rows_processed = rowwise_adagrad_kernel(
              c_block_end - c,
              hash_size * D,
              reinterpret_cast<float*>(&host_weights_data[table_begin]),
              reinterpret_cast<const float*>(grad_blocked_buffer),
              reinterpret_cast<float*>(
                  &momentum1_host[momentum1_offsets_data[feature_begin]]),
              col_segment_indices + c,
              eps,
              -learning_rate,
              /*weight_decay=*/0,
              /*counter=*/nullptr,
              /*counter_halflife=*/0);
          // TODO: more friendly error msg.
          TORCH_CHECK(num_rows_processed == c_block_end - c);
        } // for each c
      }); // parallel for
    } else
    {% endif %}
    {
      // no fbgemm
      // TODO: to parallelize, we should easily identify segments belong to
      // the same column.
      grad_t grad_buffer[D];
      for (int c = c_begin; c < c_end; ++c) {
        int64_t idx = col_segment_indices[c];
        if (c == c_begin || col_segment_indices[c - 1] != idx) {
          memset(grad_buffer, 0, D * sizeof(grad_t));
        }
        const int64_t embedding_begin = table_begin + idx * D;
        for (int r = col_segment_ptr[c]; r < col_segment_ptr[c + 1]; ++r) {
          int D_offset = D_begin;
          if (is_shared_table) {
            D_offset +=
                batched_cscs[t].column_segment_ids[is_csr2csc_sort ? r : c] * D;
          }
          int b = batched_cscs[t].row_indices[r];
          for (int64_t d = 0; d < D; ++d) {
            grad_buffer[d] += batched_cscs[t].weights != nullptr
                ? grad_output_data[b * grad_stride + D_offset + d] *
                    batched_cscs[t].weights[r]
                : grad_output_data[b * grad_stride + D_offset + d];
          }
        }
        if (c == c_end - 1 || col_segment_indices[c + 1] != idx) {
          {{ split_weight_update_cpu }}
        }
      } // for each c
    } // no fbgemm
  } // for each table
}

template <typename scalar_t>
void split_embedding_backward_exact_cpu_dense_kernel(
    Tensor grad,
    Tensor grad_output,
    const TensorAccessor<int64_t, 1> weights_offsets_data,
    const TensorAccessor<int, 1> D_offsets_data,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    Tensor indice_weights,
    int num_tables,
    int B,
    const int* table_to_feature_offset) {
  auto grad_data = grad.data_ptr<scalar_t>();

  auto grad_output_data = grad_output.accessor<scalar_t, 2>();

  const auto indices_data = indices.accessor<int64_t, 1>();
  const auto offsets_data = offsets.accessor<int64_t, 1>();
  const auto indice_weights_data = indice_weights.defined()
      ?
      // If indice_weights are not defined, then this accessor won't be
      // used
      indice_weights.accessor<scalar_t, 1>()
      : grad.accessor<scalar_t, 1>(); // this is just to make compiler
                                      // happy

  at::parallel_for(0, num_tables, 0, [&](int64_t t_begin, int64_t t_end) {
    for (int64_t t = table_to_feature_offset[t_begin];
         t < table_to_feature_offset[t_end];
         ++t) {
      const auto D_begin = D_offsets_data[t];
      const auto D = D_offsets_data[t + 1] - D_offsets_data[t];
      const auto table_begin = weights_offsets_data[t];
      for (int64_t b = 0; b < B; ++b) {
        const auto pool_begin = offsets_data[t * B + b];
        const auto pool_end = offsets_data[t * B + b + 1];
        const auto L = pool_end - pool_begin;
        const scalar_t scale_factor =
            // NOTE: MEAN pooling will not work with indice_weights!
            (pooling_mode == MEAN && !indice_weights.defined() && L > 0)
            ? 1.0 / L
            : 1.0;
        for (auto p = pool_begin; p < pool_end; ++p) {
          const int64_t embedding_begin = table_begin + indices_data[p] * D;
          const scalar_t v = indice_weights.defined()
              ? (indice_weights_data[p] * scale_factor)
              : scale_factor;
          for (int64_t d = 0; d < D; ++d) {
            grad_data[embedding_begin + d] +=
                grad_output_data[b][D_begin + d] * v;
          }
        }
      }
    }
  }); // parallel_for
}
} // namespace

// The template for exact optimizers
{{ "void" if not dense else "Tensor" }}  split_embedding_backward_codegen_{{ optimizer }}_cpu(
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
    {{ args.split_function_args | join(", ") }}
) {

  int64_t T = D_offsets.numel() - 1;
  TORCH_CHECK(T > 0);
  // offsets = [T x B  + 1]
  int64_t B = (offsets.size(0) - 1) / T;
  TORCH_CHECK(B >= 0);

  const auto weights_offsets_data = weights_offsets.accessor<int64_t, 1>();
  const auto D_offsets_data = D_offsets.accessor<int, 1>();

  const auto hash_size_cumsum_data = hash_size_cumsum.accessor<int64_t, 1>();

  int num_tables = 0; // # of physical tables
  int table_to_feature_offset[T + 1];
  table_to_feature_offset[0] = 0;
  for (int feature = 0; feature < T - 1; ++feature) {
    if (hash_size_cumsum_data[feature + 1] != hash_size_cumsum_data[feature]) {
      ++num_tables;
      table_to_feature_offset[num_tables] = feature + 1;
    }
  }
  ++num_tables;
  table_to_feature_offset[num_tables] = T;

  TORCH_CHECK(host_weights.dim() == 1);

  {% if not dense %}
  {% if "momentum1_offsets" in args.split_function_arg_names %}
  const auto momentum1_offsets_data = momentum1_offsets.accessor<int64_t, 1>();
  {% endif %}
  {% if "momentum2_offsets" in args.split_function_arg_names %}
  const auto momentum2_offsets_data = momentum2_offsets.accessor<int64_t, 1>();
  {% endif %}

  grad_output = grad_output.contiguous();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      host_weights.scalar_type(), "split_embedding_backward_exact_cpu", [&]() {
        split_embedding_backward_exact_cpu_kernel<scalar_t>(
            grad_output,
            host_weights,
            weights_offsets_data,
            D_offsets_data,
            hash_size_cumsum,
            indices,
            offsets,
            pooling_mode,
            indice_weights,
            num_tables,
            B,
            table_to_feature_offset,
            {% if "momentum1_offsets" in args.split_function_arg_names %}
            momentum1_offsets_data,
            {% endif %}
            {% if "momentum2_offsets" in args.split_function_arg_names %}
            momentum2_offsets_data,
            {% endif %}
            {{ args.split_cpu_kernel_arg_constructors | join(", ") }});
      });

  return;

  {% else %}

  // When input is dense enough, avoid sorting and just treat as dense.
  auto grad = zeros_like(host_weights, grad_output.dtype());
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.scalar_type(), "split_embedding_backward_exact_cpu", [&]() {

        split_embedding_backward_exact_cpu_dense_kernel<scalar_t>(
            grad,
            grad_output,
            weights_offsets_data,
            D_offsets_data,
            indices,
            offsets,
            pooling_mode,
            indice_weights,
            num_tables,
            B,
            table_to_feature_offset);
      }); // dispatch host_weights.scalar_type()

  return grad;
  {% endif %}
}
