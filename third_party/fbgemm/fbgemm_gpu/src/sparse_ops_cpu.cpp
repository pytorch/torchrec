/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ATen/ATen.h>
#include <ATen/TypeDefault.h>
#include <torch/library.h>
#include "ATen/Parallel.h"

#include "fbgemm_gpu/sparse_ops_utils.h"

namespace {

// To avoid multiple threads are touching the same cache line.
// Assume cache line size is 64B and element size is at least 4B like float or
// int32.
constexpr int FALSE_SHARING_PAD = 16;

} // namespace

namespace fbgemm {

at::Tensor native_empty_like(const at::Tensor& self) {
  return at::native::empty_like(
      self,
      optTypeMetaToScalarType(self.options().dtype_opt()),
      self.options().layout_opt(),
      self.options().device_opt(),
      self.options().pinned_memory_opt(),
      c10::nullopt);
}

template <typename T>
void prefix_sum(const int length, const T* const array, T* const presum) {
  presum[0] = 0;
  for (const auto i : c10::irange(length)) {
    presum[i + 1] = array[i] + presum[i];
  }
}

// NOTE : _permute_indices_weights_kernel_cpu and _permute_lengths_cpu_kernel
// have to use the same grain size for consistent partitioning across threads.
template <
    bool has_weight,
    typename offsets_t,
    typename indices_t,
    typename weights_t>
void _permute_indices_weights_kernel_cpu(
    const int32_t T,
    const int32_t B,
    const indices_t* const __restrict__ indices,
    const weights_t* const __restrict__ weights,
    const int32_t* const __restrict__ permute,
    const offsets_t* const __restrict__ input_offsets,
    const int64_t* const __restrict__ output_offsets_per_thread_cumsum,
    indices_t* const __restrict__ permuted_indices,
    weights_t* const __restrict__ permuted_weights,
    const offsets_t* const __restrict__ permuted_lengths) {
  at::parallel_for(
      0, T * B, FALSE_SHARING_PAD, [&](int64_t tb_begin, int64_t tb_end) {
        offsets_t output_start = output_offsets_per_thread_cumsum
            [at::get_thread_num() * FALSE_SHARING_PAD];
        int64_t t_begin = tb_begin / B;
        int64_t t_end = (tb_end + B - 1) / B;
        for (const auto t : c10::irange(t_begin, t_end)) {
          int64_t b_begin = (t == t_begin) ? tb_begin % B : 0;
          int64_t b_end = (t == t_end - 1 && tb_end % B != 0) ? tb_end % B : B;
          for (const auto b : c10::irange(b_begin, b_end)) {
            offsets_t permuted_length = permuted_lengths[t * B + b];
            const offsets_t input_start = input_offsets[permute[t] * B + b];
            for (const auto i : c10::irange(permuted_length)) {
              permuted_indices[output_start + i] = indices[input_start + i];
              if (has_weight) {
                permuted_weights[output_start + i] = weights[input_start + i];
              }
            }
            output_start += permuted_length;
          } // for each b
        } // for each t
      }); // parallel_for T * B
}

template <typename index_t>
void _permute_lengths_cpu_kernel(
    const int32_t T,
    const int32_t B,
    const index_t* const __restrict__ lengths,
    int64_t lengths_size,
    const int32_t* const __restrict__ permute,
    index_t* const __restrict__ permuted_lengths,
    index_t* const __restrict__ input_offsets,
    int64_t* const __restrict__ output_offsets_per_thread_cumsum) {
  int num_threads = at::get_num_threads();
  std::vector<int> input_offsets_per_thread_cumsum(
      (num_threads + 1) * FALSE_SHARING_PAD, 0);

  // First parallel for: populate permuted_lengths, and compute per-thread
  // summation of lengths (input_offsets_per_thread_cumsum) and permuted_lengths
  // (output_offsets_per_thread_cumsum)
  at::parallel_for(
      0, T * B, FALSE_SHARING_PAD, [&](int64_t tb_begin, int64_t tb_end) {
        index_t current_input_offset = 0;
        // Have a separate loop for summing up lengths because lengths_size
        // can be smaller than T * B.
        for (int tb = tb_begin; tb < std::min(tb_end, lengths_size); ++tb) {
          current_input_offset += lengths[tb];
        }

        index_t current_output_offset = 0;
        int64_t t_begin = tb_begin / B;
        int64_t t_end = (tb_end + B - 1) / B;
        for (const auto t : c10::irange(t_begin, t_end)) {
          int64_t b_begin = (t == t_begin) ? tb_begin % B : 0;
          int64_t b_end = (t == t_end - 1 && tb_end % B != 0) ? tb_end % B : B;
          for (const auto b : c10::irange(b_begin, b_end)) {
            auto permuted_length = lengths[permute[t] * B + b];
            permuted_lengths[t * B + b] = permuted_length;
            current_output_offset += permuted_length;
          }
        }
        input_offsets_per_thread_cumsum
            [(at::get_thread_num() + 1) * FALSE_SHARING_PAD] =
                current_input_offset;
        output_offsets_per_thread_cumsum
            [(at::get_thread_num() + 1) * FALSE_SHARING_PAD] =
                current_output_offset;
      });

  // Inter-thread reduction
  for (const auto t : c10::irange(1, num_threads)) {
    input_offsets_per_thread_cumsum[(t + 1) * FALSE_SHARING_PAD] +=
        input_offsets_per_thread_cumsum[t * FALSE_SHARING_PAD];
    output_offsets_per_thread_cumsum[(t + 1) * FALSE_SHARING_PAD] +=
        output_offsets_per_thread_cumsum[t * FALSE_SHARING_PAD];
  }

  // Second parallel for: populate input_offsets
  // NOTE: this works assuming the partitioning will be the same as the
  // first parallel_for.
  at::parallel_for(
      0, T * B, FALSE_SHARING_PAD, [&](int64_t tb_begin, int64_t tb_end) {
        index_t current_input_offset = input_offsets_per_thread_cumsum
            [at::get_thread_num() * FALSE_SHARING_PAD];
        if (tb_begin < lengths_size) {
          input_offsets[tb_begin] = current_input_offset;
        }
        for (const auto tb :
             c10::irange(tb_begin, std::min(tb_end - 1, lengths_size))) {
          current_input_offset += lengths[tb];
          input_offsets[tb + 1] = current_input_offset;
        }
      });
  if (lengths_size >= T * B) {
    input_offsets[T * B] =
        input_offsets_per_thread_cumsum[num_threads * FALSE_SHARING_PAD];
  }

  // Handle cases when lengths_size > T * B
  for (const auto i : c10::irange(T * B, lengths_size)) {
    input_offsets[i + 1] = lengths[i] + input_offsets[i];
  }
}

template <
    bool sequence,
    bool has_weight,
    typename offset_t,
    typename index_t,
    typename scalar_t>
void _block_bucketize_sparse_features_cpu(
    at::Tensor lengths,
    at::Tensor indices,
    c10::optional<at::Tensor> weights,
    bool bucketize_pos,
    at::Tensor block_sizes,
    int64_t my_size,
    at::Tensor new_lengths,
    at::Tensor new_indices,
    c10::optional<at::Tensor> new_weights,
    c10::optional<at::Tensor> new_pos,
    c10::optional<at::Tensor> unbucketize_permute) {
  // allocate tensors and buffers
  const auto lengths_size = lengths.numel();
  const auto new_lengths_size = lengths_size * my_size;
  const int32_t T = block_sizes.numel();
  const int32_t B = lengths_size / T;
  auto offsets = at::empty({lengths_size + 1}, lengths.options());
  auto new_offsets = at::empty({new_lengths_size + 1}, lengths.options());
  const offset_t* lengths_data = lengths.data_ptr<offset_t>();
  offset_t* offsets_data = offsets.data_ptr<offset_t>();
  const index_t* indices_data = indices.data_ptr<index_t>();
  scalar_t* weights_data;
  scalar_t* new_weights_data;
  index_t* new_pos_data;
  index_t* unbucketize_permute_data;
  offset_t* new_lengths_data = new_lengths.data_ptr<offset_t>();
  offset_t* new_offsets_data = new_offsets.data_ptr<offset_t>();
  index_t* new_indices_data = new_indices.data_ptr<index_t>();
  index_t* block_sizes_data = block_sizes.data_ptr<index_t>();
  using uindex_t = std::make_unsigned_t<index_t>;
  using uoffset_t = std::make_unsigned_t<offset_t>;

  if (sequence) {
    unbucketize_permute_data = unbucketize_permute.value().data_ptr<index_t>();
  }
  if (has_weight) {
    weights_data = weights.value().data_ptr<scalar_t>();
    new_weights_data = new_weights.value().data_ptr<scalar_t>();
  }
  if (bucketize_pos) {
    new_pos_data = new_pos.value().data_ptr<index_t>();
  }

  // count nonzeros
  prefix_sum(lengths_size, lengths_data, offsets_data);
  assert(offsets_data[lengths_size] == indices.numel());
  for (const auto t : c10::irange(T)) {
    auto blk_size = block_sizes_data[t];
    for (const auto b : c10::irange(B)) {
      const auto b_t = t * B + b;
      const offset_t rowstart = offsets_data[b_t];
      const offset_t rowend = offsets_data[b_t + 1];
      for (const auto i : c10::irange(rowstart, rowend)) {
        // We have use cases using none-hashed raw indices that can be either
        // negative or larger than embedding table hash_size (blk_size *
        // my_size). In cases of none-hashed indices we need to ensure
        // bucketization can distribute them into different ranks and within
        // range of blk_size, we expect the later embedding module to take care
        // of hashing indices calculation.
        const auto idx = static_cast<int64_t>(indices_data[i]);
        const auto p =
            idx < blk_size * my_size ? idx / blk_size : idx % my_size;
        new_lengths_data[p * lengths_size + b_t]++;
      }
    }
  }

  // bucketize nonzeros
  prefix_sum(new_lengths_size, new_lengths_data, new_offsets_data);
  assert(new_offsets_data[new_lengths_size] == new_indices.numel());
  for (const auto t : c10::irange(T)) {
    auto blk_size = block_sizes_data[t];
    for (const auto b : c10::irange(B)) {
      const auto b_t = t * B + b;
      const offset_t rowstart = offsets_data[b_t];
      const offset_t rowend = offsets_data[b_t + 1];
      for (const auto i : c10::irange(rowstart, rowend)) {
        // We have use cases using none-hashed raw indices that can be either
        // negative or larger than embedding table hash_size (blk_size *
        // my_size). In cases of none-hashed indices we need to ensure
        // bucketization can distribute them into different ranks and within
        // range of blk_size, we expect the later embedding module to take care
        // of hashing indices calculation.
        const auto idx = static_cast<int64_t>(indices_data[i]);
        const auto p =
            idx < blk_size * my_size ? idx / blk_size : idx % my_size;
        const uindex_t new_idx = idx % blk_size;
        const uoffset_t pos = new_offsets_data[p * lengths_size + b_t];
        new_indices_data[pos] = new_idx;
        if (sequence) {
          unbucketize_permute_data[i] = pos;
        }
        new_offsets_data[p * lengths_size + b_t]++;
        if (has_weight) {
          new_weights_data[pos] = weights_data[i];
        }
        if (bucketize_pos) {
          new_pos_data[pos] = i - rowstart;
        }
      }
    }
  }
}

std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>>
permute_sparse_data_cpu(
    const at::Tensor& permute,
    const at::Tensor& lengths,
    const at::Tensor& indices,
    const c10::optional<at::Tensor>& weights,
    const c10::optional<int64_t>& permuted_lengths_sum) {
  TENSOR_ON_CPU(permute);
  TENSOR_ON_CPU(lengths);
  TENSOR_ON_CPU(indices);
  TENSOR_ON_CPU(weights);

  const auto permute_contig = permute.expect_contiguous();
  const auto lengths_contig = lengths.expect_contiguous();
  const auto indices_contig = indices.expect_contiguous();
  // the data to permute over can be less or more with or without
  // repetitions
  const auto T = permute.numel();
  const auto B = lengths.view({lengths.sizes()[0], -1}).sizes()[1];

  at::Tensor permuted_lengths;
  at::Tensor permuted_indices;
  at::Tensor permuted_weights;

  permuted_lengths = at::empty({T, B}, lengths.options());

  const auto lengths_size = lengths.numel();
  auto input_offsets = at::empty({lengths_size + 1}, lengths.options());

  int num_threads = at::get_num_threads();
  std::vector<int64_t> output_offsets_per_thread_cumsum(
      (num_threads + 1) * FALSE_SHARING_PAD, 0);

  AT_DISPATCH_INDEX_TYPES(
      lengths.scalar_type(), "permute_lengths_cpu_kernel", ([&] {
        _permute_lengths_cpu_kernel(
            T,
            B,
            lengths_contig->data_ptr<index_t>(),
            lengths_size,
            permute.data_ptr<int32_t>(),
            permuted_lengths.data_ptr<index_t>(),
            input_offsets.data_ptr<index_t>(),
            output_offsets_per_thread_cumsum.data());
      })); // for each scalar_t

  int64_t permuted_indices_size = 0;
  if (permuted_lengths_sum.has_value()) {
    permuted_indices_size = permuted_lengths_sum.value();
  } else {
    permuted_indices_size =
        output_offsets_per_thread_cumsum[num_threads * FALSE_SHARING_PAD];
  }
  permuted_indices = at::empty(permuted_indices_size, indices.options());
  AT_DISPATCH_INDEX_TYPES(
      input_offsets.scalar_type(), "permute_indices_weights_kernel_1", ([&] {
        using offsets_t = index_t;
        AT_DISPATCH_ALL_TYPES(
            indices.scalar_type(), "permute_indices_weights_kernel_2", ([&] {
              using indices_t = scalar_t;
              AT_DISPATCH_FLOATING_TYPES(
                  weights.has_value() ? weights.value().scalar_type()
                                      : at::ScalarType::Float,
                  "permute_indices_weights_kernel_3",
                  ([&] {
                    using weights_t = scalar_t;
                    if (weights.has_value()) {
                      const auto weights_value_contig =
                          weights.value().expect_contiguous();
                      permuted_weights = at::empty(
                          permuted_indices_size, weights.value().options());
                      _permute_indices_weights_kernel_cpu<
                          true,
                          index_t,
                          indices_t,
                          weights_t>(
                          T,
                          B,
                          indices_contig->data_ptr<indices_t>(),
                          weights_value_contig->data_ptr<weights_t>(),
                          permute_contig->data_ptr<int32_t>(),
                          input_offsets.data_ptr<offsets_t>(),
                          output_offsets_per_thread_cumsum.data(),
                          permuted_indices.data_ptr<indices_t>(),
                          permuted_weights.data_ptr<weights_t>(),
                          permuted_lengths.data_ptr<offsets_t>());
                    } else {
                      _permute_indices_weights_kernel_cpu<
                          false,
                          index_t,
                          indices_t,
                          weights_t>(
                          T,
                          B,
                          indices_contig->data_ptr<indices_t>(),
                          nullptr,
                          permute_contig->data_ptr<int32_t>(),
                          input_offsets.data_ptr<offsets_t>(),
                          output_offsets_per_thread_cumsum.data(),
                          permuted_indices.data_ptr<indices_t>(),
                          nullptr,
                          permuted_lengths.data_ptr<offsets_t>());
                    }
                  })); // for each weights_t
            })); // for each indices_t
      })); // for each offsets_t
  return {permuted_lengths, permuted_indices, permuted_weights};
}

std::tuple<
    at::Tensor,
    at::Tensor,
    c10::optional<at::Tensor>,
    c10::optional<at::Tensor>,
    c10::optional<at::Tensor>>
block_bucketize_sparse_features_cpu(
    at::Tensor lengths,
    at::Tensor indices,
    bool bucketize_pos,
    bool sequence,
    at::Tensor block_sizes,
    int64_t my_size,
    c10::optional<at::Tensor> weights) {
  const auto lengths_size = lengths.numel();
  const auto new_lengths_size = lengths_size * my_size;
  auto new_lengths = at::zeros({new_lengths_size}, lengths.options());
  auto new_indices = native_empty_like(indices);
  at::Tensor new_weights;
  at::Tensor new_pos;
  at::Tensor unbucketize_permute;
  if (bucketize_pos) {
    new_pos = native_empty_like(indices);
  }
  if (weights.has_value()) {
    const auto lengths_sum = indices.numel();
    at::Tensor weights_value = weights.value();
    new_weights = native_empty_like(weights_value);
    if (sequence) {
      unbucketize_permute = at::empty({lengths_sum}, indices.options());
      AT_DISPATCH_INDEX_TYPES(
          lengths.scalar_type(),
          "block_bucketize_sparse_features_weights_cpu_1",
          ([&] {
            using offset_t = index_t;
            AT_DISPATCH_INDEX_TYPES(
                indices.scalar_type(),
                "block_bucketize_sparse_features_weights_cpu_2",
                ([&] {
                  AT_DISPATCH_FLOATING_TYPES(
                      weights_value.scalar_type(),
                      "bucketize_sparse_features_weights_cpu_3",
                      ([&] {
                        _block_bucketize_sparse_features_cpu<
                            true,
                            true,
                            offset_t,
                            index_t,
                            scalar_t>(
                            lengths,
                            indices,
                            weights,
                            bucketize_pos,
                            block_sizes,
                            my_size,
                            new_lengths,
                            new_indices,
                            new_weights,
                            new_pos,
                            unbucketize_permute);
                      }));
                }));
          }));
    } else {
      AT_DISPATCH_INDEX_TYPES(
          lengths.scalar_type(),
          "block_bucketize_sparse_features_weights_cpu_1",
          ([&] {
            using offset_t = index_t;
            AT_DISPATCH_INDEX_TYPES(
                indices.scalar_type(),
                "block_bucketize_sparse_features_weights_cpu_2",
                ([&] {
                  AT_DISPATCH_FLOATING_TYPES(
                      weights_value.scalar_type(),
                      "bucketize_sparse_features_weights_cpu_3",
                      ([&] {
                        _block_bucketize_sparse_features_cpu<
                            false,
                            true,
                            offset_t,
                            index_t,
                            scalar_t>(
                            lengths,
                            indices,
                            weights,
                            bucketize_pos,
                            block_sizes,
                            my_size,
                            new_lengths,
                            new_indices,
                            new_weights,
                            new_pos,
                            unbucketize_permute);
                      }));
                }));
          }));
    }
  } else {
    if (sequence) {
      const auto lengths_sum = indices.numel();
      unbucketize_permute = at::empty({lengths_sum}, indices.options());
      AT_DISPATCH_INDEX_TYPES(
          lengths.scalar_type(), "block_bucketize_sparse_features_cpu_1", ([&] {
            using offset_t = index_t;
            AT_DISPATCH_INDEX_TYPES(
                indices.scalar_type(),
                "block_bucketize_sparse_features_cpu_2",
                ([&] {
                  _block_bucketize_sparse_features_cpu<
                      true,
                      false,
                      offset_t,
                      index_t,
                      std::nullptr_t>(
                      lengths,
                      indices,
                      weights,
                      bucketize_pos,
                      block_sizes,
                      my_size,
                      new_lengths,
                      new_indices,
                      new_weights,
                      new_pos,
                      unbucketize_permute);
                }));
          }));
    } else {
      AT_DISPATCH_INDEX_TYPES(
          lengths.scalar_type(), "block_bucketize_sparse_features_cpu_1", ([&] {
            using offset_t = index_t;
            AT_DISPATCH_INDEX_TYPES(
                indices.scalar_type(),
                "block_bucketize_sparse_features_cpu_2",
                ([&] {
                  _block_bucketize_sparse_features_cpu<
                      false,
                      false,
                      offset_t,
                      index_t,
                      std::nullptr_t>(
                      lengths,
                      indices,
                      weights,
                      bucketize_pos,
                      block_sizes,
                      my_size,
                      new_lengths,
                      new_indices,
                      new_weights,
                      new_pos,
                      unbucketize_permute);
                }));
          }));
    }
  }
  return {new_lengths, new_indices, new_weights, new_pos, unbucketize_permute};
}

// 1D exclusive scan: output[i] = input[i-1] + input[i-2] + input[i-3]
// Used as a helper to several functions below.
template <class T, class U>
U exclusive_scan_ptrs_cpu(
    const int64_t N,
    const T* const input,
    U* const output) {
  U cumsum = 0;
  for (const auto i : c10::irange(N)) {
    output[i] = cumsum;
    cumsum += input[i];
  }
  return cumsum;
}

at::Tensor asynchronous_exclusive_cumsum_cpu(const at::Tensor& t_in) {
  TENSOR_ON_CPU(t_in);

  const auto t_in_contig = t_in.expect_contiguous();
  auto output = native_empty_like(*t_in_contig);
  AT_DISPATCH_ALL_TYPES(
      t_in_contig->type(), "asynchronous_exclusive_cumsum_cpu_kernel", ([&] {
        exclusive_scan_ptrs_cpu(
            t_in_contig->numel(),
            t_in_contig->data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>());
      }));
  return output;
}

at::Tensor asynchronous_inclusive_cumsum_cpu(const at::Tensor& t_in) {
  TENSOR_ON_CPU(t_in);

  const auto t_in_contig = t_in.expect_contiguous();
  auto output = native_empty_like(*t_in_contig);
  AT_DISPATCH_ALL_TYPES(
      t_in_contig->type(), "asynchronous_inclusive_cumsum_cpu_kernel", ([&] {
        scalar_t cumsum = 0;
        const auto* input_ptr = t_in_contig->data_ptr<scalar_t>();
        const auto N = t_in_contig->numel();
        auto* output_ptr = output.data_ptr<scalar_t>();

        for (const auto i : c10::irange(N)) {
          cumsum += input_ptr[i];
          output_ptr[i] = cumsum;
        }
      }));
  return output;
}

at::Tensor asynchronous_complete_cumsum_cpu(const at::Tensor& t_in) {
  TENSOR_ON_CPU(t_in);
  TORCH_CHECK(t_in.dim() == 1);

  const auto t_in_contig = t_in.expect_contiguous();
  auto output = at::zeros({t_in.numel() + 1}, t_in.options());
  AT_DISPATCH_ALL_TYPES(
      t_in_contig->type(), "asynchronous_complete_cumsum_cpu_kernel", ([&] {
        const auto N = t_in_contig->numel();
        const auto last_sum = exclusive_scan_ptrs_cpu(
            N, t_in_contig->data_ptr<scalar_t>(), output.data_ptr<scalar_t>());
        output.data_ptr<scalar_t>()[N] = last_sum;
      }));
  return output;
}

at::Tensor reorder_batched_ad_lengths_cpu(
    const at::Tensor& cat_ad_lengths,
    const at::Tensor& batch_offsets,
    const int64_t num_ads_in_batch) {
  const int64_t B = batch_offsets.numel() - 1;
  const int64_t T = cat_ad_lengths.numel() / num_ads_in_batch;
  at::Tensor reordered_cat_ad_lengths = at::empty_like(cat_ad_lengths);

  const auto* batch_offsets_data = batch_offsets.data_ptr<int32_t>();

  for (auto b = 0; b < B; b++) {
    const int32_t num_ads_b = batch_offsets_data[b + 1] - batch_offsets_data[b];
    for (auto t = 0; t < T; t++) {
      const int32_t input_segment_start =
          T * batch_offsets_data[b] + t * num_ads_b;
      const int32_t output_segment_start =
          t * num_ads_in_batch + batch_offsets_data[b];
      for (auto i = 0; i < num_ads_b; i++) {
        reordered_cat_ad_lengths[output_segment_start + i] =
            cat_ad_lengths[input_segment_start + i];
      }
    }
  }

  return reordered_cat_ad_lengths;
}

at::Tensor reorder_batched_ad_indices_cpu(
    const at::Tensor& cat_ad_offsets,
    const at::Tensor& cat_ad_indices,
    const at::Tensor& reordered_cat_ad_offsets,
    const at::Tensor& batch_offsets,
    const int64_t num_ads_in_batch) {
  const int64_t B = batch_offsets.numel() - 1;
  const int64_t T = (cat_ad_offsets.numel() - 1) / num_ads_in_batch;
  at::Tensor reordered_cat_ad_indices = at::empty_like(cat_ad_indices);

  const auto* batch_offsets_data = batch_offsets.data_ptr<int32_t>();
  const auto* cat_ad_offsets_data = cat_ad_offsets.data_ptr<int32_t>();
  const auto* reordered_cat_ad_offsets_data =
      reordered_cat_ad_offsets.data_ptr<int32_t>();
  const auto* cat_ad_indices_data = cat_ad_indices.data_ptr<int32_t>();

  for (auto b = 0; b < B; b++) {
    const int32_t num_ads_b = batch_offsets_data[b + 1] - batch_offsets_data[b];
    for (auto t = 0; t < T; t++) {
      const int32_t input_segment_offset_start =
          T * batch_offsets_data[b] + t * num_ads_b;
      const int32_t input_segment_offset_end =
          T * batch_offsets_data[b] + t * num_ads_b + num_ads_b;

      const auto input_segment_start =
          cat_ad_offsets_data[input_segment_offset_start];
      const auto input_segment_end =
          cat_ad_offsets_data[input_segment_offset_end];

      const auto output_segment_offset_start =
          t * num_ads_in_batch + batch_offsets_data[b];
      const auto output_segment_start =
          reordered_cat_ad_offsets_data[output_segment_offset_start];

      for (auto i = 0; i < input_segment_end - input_segment_start; i++) {
        reordered_cat_ad_indices[output_segment_start + i] =
            cat_ad_indices_data[input_segment_start + i];
      }
    }
  }

  return reordered_cat_ad_indices;
}

at::Tensor offsets_range_cpu(const at::Tensor& offsets, int64_t range_size) {
  TENSOR_ON_CPU(offsets);
  TENSOR_NDIM_EQUALS(offsets, 1);

  const auto offsets_arg = at::TensorArg(offsets, "offsets", 1);
  checkScalarTypes("_offsets_range_cpu", offsets_arg, {at::kLong, at::kInt});
  auto range = at::empty(range_size, offsets.options());
  if (range_size == 0) {
    return range;
  }
  const auto offsets_contig = offsets.expect_contiguous();
  const auto N = offsets_contig->numel();
  AT_DISPATCH_INDEX_TYPES(
      offsets_contig->scalar_type(), "offsets_range_kernel", [&]() {
        const index_t* offsets_data = offsets_contig->data_ptr<index_t>();
        index_t* range_data = range.data_ptr<index_t>();

        index_t last = range_size;
        for (int64_t i = N - 1; i >= 0; --i) {
          index_t first = offsets_data[i];
          std::iota(range_data + first, range_data + last, 0);
          last = first;
        }
      });

  return range;
}

/// CPU version of batched_unary_embeddings forward pass.
///
/// Sums up `weight` embeddings according to `offsets` and `indices`.
/// `table_offests` is a helper struct to quickly navigate through tables in
/// `weight` -- it is caller's responsibility to keep it in sync with `weight`.
/// Visualization of op semantics: https://fburl.com/9a4uktmb
///
/// This version is only for numerical verification so not optimized for
/// performance.
///
/// @param weight        - Weight for the embeddings.
/// @param table_offsets - Index offsets for each table entry in `weight`.
/// @param offsets       - Offsets for the starting point of each summation.
/// @param indices       - Indices for the embeddings to fetch (from `weight`).
/// @return The sumed embeddings.
at::Tensor batched_unary_embeddings_forward_cpu(
    const at::Tensor& weight,
    const at::Tensor& table_offsets,
    const at::Tensor& offsets,
    const at::Tensor& indices) {
  TENSOR_ON_CPU(weight);
  TENSOR_ON_CPU(table_offsets);
  TENSOR_ON_CPU(offsets);
  TENSOR_ON_CPU(indices);

  // N: number of tasks, T: number of tables, B: batch size
  const int32_t N = weight.sizes()[0];
  const int32_t T = table_offsets.numel() - 1;
  const int32_t B = (offsets.numel() - 1) / T;
  TORCH_CHECK(N > 0);
  TORCH_CHECK(T > 0);
  TORCH_CHECK(B > 0);

  // Only supporting limited data types for now.
  TORCH_CHECK(weight.scalar_type() == at::ScalarType::Float);

  // Make sure the index_t are consistent among table_offsets, offsets and
  // indices
  TORCH_CHECK(table_offsets.scalar_type() == offsets.scalar_type());
  TORCH_CHECK(table_offsets.scalar_type() == indices.scalar_type());

  auto output = at::empty({N, B, T}, weight.options());
  auto* output_data = output.data_ptr<float>();
  const auto* weight_data = weight.data_ptr<float>();

  AT_DISPATCH_INDEX_TYPES(
      table_offsets.scalar_type(), "unary_indices", ([&] {
        const index_t* table_offsets_data = table_offsets.data_ptr<index_t>();
        const index_t* offsets_data = offsets.data_ptr<index_t>();
        const index_t* indices_data = indices.data_ptr<index_t>();
        const index_t sum_E = table_offsets_data[T];

        for (const auto n : c10::irange(N)) {
          for (const auto b : c10::irange(B)) {
            for (const auto t : c10::irange(T)) {
              const index_t indices_start = offsets_data[t * B + b];
              const index_t indices_end = offsets_data[t * B + b + 1];
              float sum = 0;
              for (const auto l : c10::irange(indices_start, indices_end)) {
                const index_t idx =
                    n * sum_E + table_offsets_data[t] + indices_data[l];
                // Since we don't care about the performance of CPU impl, adding
                // the boundary check here. OOB will result in undefined
                // behavior for GPU impl.
                TORCH_CHECK(idx < weight.numel());
                sum += weight_data[idx];
              }
              output_data[(n * B + b) * T + t] = sum;
            }
          }
        }
      }));

  return output;
}

template <typename index_t, typename scalar_t>
void jagged_2d_to_dense_forward_kernel(
    int32_t B,
    int32_t max_L,
    int32_t D,
    const index_t* offsets,
    const scalar_t* embeddings_data,
    scalar_t* padded_embeddings_data) {
  const auto block_size = max_L * D;
  const auto embedding_byte_size = D * sizeof(scalar_t);
  for (auto b = 0; b < B; ++b) {
    auto start_idx = offsets[b];
    auto end_idx = offsets[b + 1];
    auto length = end_idx - start_idx;
    if (length > max_L) {
      length = max_L;
    }
    auto padding_length = max_L - length;
    memcpy(
        &padded_embeddings_data[b * block_size],
        &embeddings_data[start_idx * D],
        length * embedding_byte_size);
    memset(
        &padded_embeddings_data[b * block_size + length * D],
        0,
        padding_length * embedding_byte_size);
  }
}

at::Tensor jagged_2d_to_dense_forward_cpu(
    at::Tensor embeddings,
    at::Tensor offsets,
    int64_t max_L) {
  TORCH_CHECK(embeddings.dim() == 2);
  TORCH_CHECK(offsets.dim() == 1);
  TORCH_CHECK(max_L > 0);

  const auto B = offsets.numel() - 1;
  const auto D = embeddings.size(1);
  const auto embeddings_contig = embeddings.expect_contiguous();
  const auto offsets_contig = offsets.expect_contiguous();

  if (embeddings.size(0) == 0) {
    return at::zeros({B, max_L, D}, embeddings.options());
  }

  auto padded_embeddings = at::empty({B, max_L, D}, embeddings.options());
  AT_DISPATCH_INDEX_TYPES(
      offsets_contig->scalar_type(),
      "jagged_2d_to_dense_forward_by_offsets",
      ([&]() {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            embeddings_contig->scalar_type(),
            "jagged_2d_to_dense_forward_by_embeddings",
            ([&]() {
              jagged_2d_to_dense_forward_kernel(
                  B,
                  max_L,
                  D,
                  offsets_contig->data_ptr<index_t>(),
                  embeddings_contig->data_ptr<scalar_t>(),
                  padded_embeddings.data_ptr<scalar_t>());
            }));
      }));

  return padded_embeddings;
}

template <typename index_t, typename scalar_t>
void jagged_1d_to_dense_kernel(
    int64_t B,
    int64_t max_L,
    scalar_t padding_value,
    const index_t* offsets,
    const scalar_t* values_data,
    scalar_t* padded_values_data) {
  const auto block_size = max_L;
  const auto value_byte_size = sizeof(scalar_t);
  for (auto b = 0; b < B; ++b) {
    auto start_idx = offsets[b];
    auto end_idx = offsets[b + 1];
    auto length = end_idx - start_idx;
    if (length > max_L) {
      // Guard against the case that lengths[b] > max_L. This is
      // a valid use case.
      length = max_L;
    }
    auto padding_length = max_L - length;
    memcpy(
        &padded_values_data[b * block_size],
        &values_data[start_idx],
        length * value_byte_size);
    for (int l = 0, offset = b * block_size + length;
         l < padding_length; ++l, ++offset) {
      padded_values_data[offset] = padding_value;
    }
  }
}

at::Tensor jagged_1d_to_dense_cpu(
    at::Tensor values,
    at::Tensor offsets,
    int64_t max_L,
    int64_t padding_value) {
  TORCH_CHECK(values.dim() == 1);
  TORCH_CHECK(offsets.dim() == 1);
  TORCH_CHECK(max_L > 0);

  const auto B = offsets.numel() - 1;
  const auto values_contig = values.expect_contiguous();
  const auto offsets_contig = offsets.expect_contiguous();

  if (values.size(0) == 0 && padding_value == 0) {
    return at::zeros({B, max_L}, values.options());
  }

  auto padded_values = at::empty({B, max_L}, values.options());
  AT_DISPATCH_INDEX_TYPES(
      offsets_contig->scalar_type(),
      "jagged_1d_to_dense_1",
      ([&]() {
        AT_DISPATCH_ALL_TYPES(
            values_contig->scalar_type(),
            "jagged_1d_to_dense_2",
            ([&]() {
              jagged_1d_to_dense_kernel<index_t, scalar_t>(
                  B,
                  max_L,
                  padding_value,
                  offsets_contig->data_ptr<index_t>(),
                  values_contig->data_ptr<scalar_t>(),
                  padded_values.data_ptr<scalar_t>());
            }));
      }));

  return padded_values;
}
} // namespace fbgemm

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def(
      "permute_sparse_data(Tensor permute, Tensor lengths, Tensor values, Tensor? weights=None, int? permuted_lengths_sum=None) -> (Tensor, Tensor, Tensor?)");
  m.def(
      "block_bucketize_sparse_features(Tensor lengths, Tensor indices, bool bucketize_pos, bool sequence, Tensor block_sizes, int my_size, Tensor? weights=None) -> (Tensor, Tensor, Tensor?, Tensor?, Tensor?)");
  m.def("asynchronous_exclusive_cumsum(Tensor t_in) -> Tensor");
  m.def("asynchronous_inclusive_cumsum(Tensor t_in) -> Tensor");
  m.def("asynchronous_complete_cumsum(Tensor t_in) -> Tensor");
  m.def(
      "reorder_batched_ad_lengths(Tensor cat_ad_lengths, Tensor batch_offsets, int num_ads_in_batch) -> Tensor");
  m.def(
      "reorder_batched_ad_indices(Tensor cat_ad_offsets, Tensor cat_ad_indices, Tensor reordered_cat_ad_offsets, Tensor batch_offsets, int num_ads_in_batch) -> Tensor");
  m.def("offsets_range(Tensor offsets, int range_size) -> Tensor");
  m.def(
      "batched_unary_embeddings(Tensor weight, Tensor table_offsets, Tensor offsets, Tensor indices) -> Tensor");
  m.def(
      "jagged_2d_to_dense(Tensor embeddings, Tensor offsets, int max_sequence_length) -> Tensor");
   m.def(
      "jagged_1d_to_dense(Tensor values, Tensor offsets, int max_sequence_length, int padding_value) -> Tensor");
}

TORCH_LIBRARY_IMPL(fbgemm, CPU, m) {
  m.impl("permute_sparse_data", fbgemm::permute_sparse_data_cpu);
  m.impl(
      "block_bucketize_sparse_features",
      fbgemm::block_bucketize_sparse_features_cpu);
  m.impl(
      "asynchronous_exclusive_cumsum",
      fbgemm::asynchronous_exclusive_cumsum_cpu);
  m.impl(
      "asynchronous_inclusive_cumsum",
      fbgemm::asynchronous_inclusive_cumsum_cpu);
  m.impl(
      "asynchronous_complete_cumsum", fbgemm::asynchronous_complete_cumsum_cpu);
  m.impl("reorder_batched_ad_lengths", fbgemm::reorder_batched_ad_lengths_cpu);
  m.impl("reorder_batched_ad_indices", fbgemm::reorder_batched_ad_indices_cpu);
  m.impl("offsets_range", fbgemm::offsets_range_cpu);
  m.impl(
      "batched_unary_embeddings", fbgemm::batched_unary_embeddings_forward_cpu);
  m.impl("jagged_2d_to_dense", fbgemm::jagged_2d_to_dense_forward_cpu);
  m.impl("jagged_1d_to_dense", fbgemm::jagged_1d_to_dense_cpu);
}
