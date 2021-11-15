/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
{% set wdesc = "weighted" if weighted else "unweighted" %}
#include "codegen/embedding_backward_template_helpers.cuh"

{% if not dense %}
constexpr int32_t kCacheLocationMissing = -1;
{% endif %}
enum {
  DEVICE = 0,
  MANAGED = 1,
  MANAGED_CACHING = 2,
};

constexpr size_t kBackwardMaxThreads = 512;

using namespace at;
using namespace fbgemm_gpu;

__global__ void
split_embedding_backward_codegen_{{ optimizer }}_{{ wdesc }}_find_long_segments(
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        sorted_linear_indices_num_runs,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        sorted_linear_indices_run_lengths,
    PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        long_run_ids,
    PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits>
        num_long_run_ids,
    int32_t max_segment_length_per_warp) {
  const int32_t num_runs = sorted_linear_indices_num_runs[0];
  for (auto run_id = blockIdx.x * blockDim.x + threadIdx.x; run_id < num_runs; run_id += blockDim.x * gridDim.x) {
    if (sorted_linear_indices_run_lengths[run_id] >= max_segment_length_per_warp) {
        auto long_run_idx = gpuAtomicIncrement(&num_long_run_ids[0]);
        long_run_ids[long_run_idx] = run_id;
    }
  }
}

template <
    typename emb_t,
    typename cache_t,
    size_t kMaxVecsPerThread>
__global__ void
__launch_bounds__(kMaxThreads)
split_embedding_backward_codegen_{{ optimizer }}_{{ wdesc }}_kernel_cta_per_row_1(
    const PackedTensorAccessor32<acc_type<cache_t, true>, 2, RestrictPtrTraits>
        grad_output,
    PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> dev_weights,
    {% if not dense %}
    PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> uvm_weights,
    PackedTensorAccessor64<cache_t, 2, RestrictPtrTraits> lxu_cache_weights,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        weights_placements,
    {% endif %}
    const PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> weights_offsets,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> D_offsets,
    const PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits>
        hash_size_cumsum,
    const PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits>
        sorted_linear_indices_run,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        sorted_linear_indices_cumulative_run_lengths,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        sorted_linear_indices_run_lengths,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        long_run_ids,
    const PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits>
        num_long_run_ids,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> sorted_infos,
    {% if not dense %}
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        sorted_lxu_cache_locations,
    {% endif %}
    {% if weighted %}
    const PackedTensorAccessor32<acc_type<cache_t, true>, 1, RestrictPtrTraits> sorted_indice_weights,
    {% endif %}
    {% if not dense %}
    bool stochastic_rounding,
    PhiloxCudaState stochastic_rounding_philox_args,
    {% else %}
    PackedTensorAccessor64<cache_t, 1, RestrictPtrTraits> grad_dev_weights,
    {% endif %}
    FixedDivisor fd,
    {{ args.split_kernel_args | join(", ") }}) {
  int32_t T = D_offsets.size(0) - 1;
  const int32_t B = grad_output.size(0);
  const int32_t num_long_runs = num_long_run_ids[0];
  for (int32_t long_run_id = blockIdx.x; long_run_id < num_long_runs; long_run_id += gridDim.x) {
        int32_t current_run_id = long_run_ids[long_run_id];
        const int64_t linear_index = sorted_linear_indices_run[current_run_id];
        const int32_t segment_start =
            sorted_linear_indices_cumulative_run_lengths[current_run_id];
        const int32_t segment_end =
            sorted_linear_indices_cumulative_run_lengths[current_run_id + 1];
        const int32_t SL = segment_end - segment_start;
        const int32_t warp_id = threadIdx.y;
        const int32_t lane_id = threadIdx.x;

        // Note that with shared embedding tables we can have multiple tables
        // (i.e. different values of `t` sharing the same segment).
        //
        const auto info_0 = sorted_infos[segment_start];
        int32_t t_0 = fd.Div(info_0); //info_0 / B;
        int64_t hash_size = hash_size_cumsum[t_0];
        int32_t D = D_offsets[t_0 + 1] - D_offsets[t_0];
        int64_t idx = linear_index - hash_size;

        const int32_t SL_per_warp = div_round_up(SL, blockDim.y);
        const int32_t sl_start = SL_per_warp * warp_id;
        const int32_t sl_end = min(SL_per_warp * (warp_id + 1), SL);
        Vec4T<acc_type<cache_t, true>> grad_sum[kMaxVecsPerThread];
        for (int32_t sl = sl_start; sl < sl_end; sl += kWarpSize) {
            int32_t sl_j = sl + threadIdx.x;
            int32_t b_t = sl_j < sl_end ? sorted_infos[segment_start + sl_j] : 0;
            int32_t b; //= b_t % B;
            int32_t t; //= b_t / B;
            fd.DivMod(b_t, &t, &b);
            int32_t D_start = sl_j < sl_end ? D_offsets[t] : 0;
            {% if weighted %}
            acc_type<cache_t, true> idx_weight = sl_j < sl_end ? sorted_indice_weights[segment_start + sl_j] : 0.0;
            {% endif %}

            for (int32_t j = 0; j < kWarpSize && sl + j < sl_end; ++j) {
                int32_t b_j = __shfl_sync(0xFFFFFFFF, b, j);
                int32_t D_start_j = __shfl_sync(0xFFFFFFFF, D_start, j);
                {% if weighted %}
                acc_type<cache_t, true> idx_weight_j = __shfl_sync(0xFFFFFFFF, idx_weight, j);
                {% endif %}

        #pragma unroll kMaxVecsPerThread
                for (int32_t i = 0;
                    i < kMaxVecsPerThread && 4 * kWarpSize * i + threadIdx.x * 4 < D;
                    ++i) {
                    int32_t d = 4 * kWarpSize * i + threadIdx.x * 4;
                    Vec4T<acc_type<cache_t, true>> grad_out_vec(
                        &grad_output[b_j][0] + D_start_j + d);
                    {% if weighted %}
                    grad_sum[i].fma_(grad_out_vec, idx_weight_j);
                    {% else %}
                    grad_sum[i].acc.x += grad_out_vec.acc.x;
                    grad_sum[i].acc.y += grad_out_vec.acc.y;
                    grad_sum[i].acc.z += grad_out_vec.acc.z;
                    grad_sum[i].acc.w += grad_out_vec.acc.w;
                    {% endif %}
                }
            }
        }
        // do shared memory reduction only if we used multiple blocks.
        if (SL > SL_per_warp) {
            struct SharedMemory<Vec4T<acc_type<cache_t, true>>> smem;
            Vec4T<acc_type<cache_t, true>>* shared_grad_sums = smem.getPointer();

    #pragma unroll kMaxVecsPerThread
            for (int32_t i = 0;
                i < kMaxVecsPerThread && 4 * kWarpSize * i + threadIdx.x * 4 < D;
                ++i) {
            shared_grad_sums
                [lane_id + i * kWarpSize +
                warp_id * kMaxVecsPerThread * kWarpSize] = grad_sum[i];
            }
            __syncthreads();
            if (blockDim.y >= 32) {
            if (warp_id < 16) {
    #pragma unroll kMaxVecsPerThread
                for (int32_t i = 0; i < kMaxVecsPerThread &&
                    4 * kWarpSize * i + threadIdx.x * 4 < D;
                    ++i) {
                shared_grad_sums
                    [lane_id + i * kWarpSize +
                    warp_id * kMaxVecsPerThread * kWarpSize] =
                        vec4_acc(
                            shared_grad_sums
                                [lane_id + i * kWarpSize +
                                warp_id * kMaxVecsPerThread * kWarpSize],
                            shared_grad_sums
                                [lane_id + i * kWarpSize +
                                (warp_id + 16) * kMaxVecsPerThread * kWarpSize]);
                }
            }
            __syncthreads();
            }
            if (blockDim.y >= 16) {
            if (warp_id < 8) {
    #pragma unroll kMaxVecsPerThread
                for (int32_t i = 0; i < kMaxVecsPerThread &&
                    4 * kWarpSize * i + threadIdx.x * 4 < D;
                    ++i) {
                shared_grad_sums
                    [lane_id + i * kWarpSize +
                    warp_id * kMaxVecsPerThread * kWarpSize] =
                        vec4_acc(
                            shared_grad_sums
                                [lane_id + i * kWarpSize +
                                warp_id * kMaxVecsPerThread * kWarpSize],
                            shared_grad_sums
                                [lane_id + i * kWarpSize +
                                (warp_id + 8) * kMaxVecsPerThread * kWarpSize]);
                }
            }
            __syncthreads();
            }
            if (blockDim.y >= 8) {
            if (warp_id < 4) {
    #pragma unroll kMaxVecsPerThread
                for (int32_t i = 0; i < kMaxVecsPerThread &&
                    4 * kWarpSize * i + threadIdx.x * 4 < D;
                    ++i) {
                shared_grad_sums
                    [lane_id + i * kWarpSize +
                    warp_id * kMaxVecsPerThread * kWarpSize] =
                        vec4_acc(
                            shared_grad_sums
                                [lane_id + i * kWarpSize +
                                warp_id * kMaxVecsPerThread * kWarpSize],
                            shared_grad_sums
                                [lane_id + i * kWarpSize +
                                (warp_id + 4) * kMaxVecsPerThread * kWarpSize]);
                }
            }
            __syncthreads();
            }
            if (blockDim.y >= 4) {
            if (warp_id < 2) {
    #pragma unroll kMaxVecsPerThread
                for (int32_t i = 0; i < kMaxVecsPerThread &&
                    4 * kWarpSize * i + threadIdx.x * 4 < D;
                    ++i) {
                shared_grad_sums
                    [lane_id + i * kWarpSize +
                    warp_id * kMaxVecsPerThread * kWarpSize] =
                        vec4_acc(
                            shared_grad_sums
                                [lane_id + i * kWarpSize +
                                warp_id * kMaxVecsPerThread * kWarpSize],
                            shared_grad_sums
                                [lane_id + i * kWarpSize +
                                (warp_id + 2) * kMaxVecsPerThread * kWarpSize]);
                }
            }
            __syncthreads();
            }
            if (warp_id == 0) {
    #pragma unroll kMaxVecsPerThread
            for (int32_t i = 0;
                i < kMaxVecsPerThread && 4 * kWarpSize * i + threadIdx.x * 4 < D;
                ++i) {
                grad_sum[i] = vec4_acc(
                    shared_grad_sums
                        [lane_id + i * kWarpSize +
                        warp_id * kMaxVecsPerThread * kWarpSize],
                    shared_grad_sums
                        [lane_id + i * kWarpSize +
                        (warp_id + 1) * kMaxVecsPerThread * kWarpSize]);
            }
            }
        }

        if (warp_id == 0) {
            int64_t weights_offset = weights_offsets[t_0];
            {% if not dense %}
            emb_t* __restrict__ weights{nullptr};
            cache_t* __restrict__ cache_weights{nullptr};
            int32_t D_emb = D;
            if (std::is_same<emb_t, uint8_t>::value) {
                D_emb += kINT8QparamsBytes;
            }
            const auto weights_placement = weights_placements[t_0];
            if (weights_placement == DEVICE) {
                weights = &dev_weights[weights_offset + idx * D_emb];
            } else {
                weights = &uvm_weights[weights_offset + idx * D_emb];
            }
            if (weights_placement == MANAGED_CACHING) {
                int32_t cache_idx = sorted_lxu_cache_locations[segment_start];
                if (cache_idx != kCacheLocationMissing) {
                    cache_weights = &lxu_cache_weights[cache_idx][0];
                }
            }
            {% for tensor in args.split_tensors %}
            acc_type<cache_t, true>* __restrict__ {{ tensor }};
            const auto {{ tensor }}_placement = {{ tensor }}_placements[t_0];
            int64_t {{ tensor }}_offset = {{ tensor }}_offsets[t_0];
            if ({{ tensor }}_placement == DEVICE) {
                {{ tensor }} = &{{ tensor }}_dev[{{ tensor }}_offset];
            } else {
                {{ tensor }} = &{{ tensor }}_uvm[{{ tensor }}_offset];
            }
            {% endfor %}


            struct SharedMemory<Vec4T<acc_type<cache_t, true>>> weight_update_buffer;
            Vec4T<acc_type<cache_t, true>>* shared_weight_update_row = weight_update_buffer.getPointer();

            auto weight_row_template = WeightRow<emb_t, cache_t, acc_type<cache_t, true>>(weights, cache_weights, D, nullptr);
            if (!std::is_same<emb_t, float>::value && stochastic_rounding) {
                StochasticRoundingRNGState state;
                // different for every *run* and every *thread*.
                auto stochastic_rounding_seeds =
                    at::cuda::philox::unpack(stochastic_rounding_philox_args);
                stochastic_rounding_init(
                    std::get<0>(stochastic_rounding_seeds) ^
                        std::get<1>(stochastic_rounding_seeds),
                    threadIdx.x + current_run_id * blockDim.x,
                    &state);
                weight_row_template.set_stoc_state(&state);
            }

            float2 qparams_template;
            if (std::is_same<emb_t, uint8_t>::value && !cache_weights) {
                qparams_template = weight_row_template.load_qparams();
            }

            {{ split_precomputation }}

            float2 qparams_new;
            #pragma unroll kMaxVecsPerThread
            for (int32_t i = 0;
                    i < kMaxVecsPerThread && 4 * kWarpSize * i + threadIdx.x * 4 < D;
                    ++i) {
                int32_t d = 4 * kWarpSize * i + threadIdx.x * 4;
                Vec4T<acc_type<cache_t, true>> weight_new = weight_row_template.load(d, qparams_template);
                auto& grad = grad_sum[i];
                {{ split_weight_update }}
                if (std::is_same<emb_t, uint8_t>::value && !cache_weights) {
                    shared_weight_update_row[lane_id + i * kWarpSize] = weight_new;
                } else {
                    weight_row_template.store(weight_new, d, qparams_new); // qparams_new not used if embedding is not int8
                }
            }
            if (std::is_same<emb_t, uint8_t>::value && !cache_weights) {
                // calculate qparams from updated weight row
                qparams_new = thrust_find_qparams<acc_type<cache_t, true>>(shared_weight_update_row, D);
                weight_row_template.store_qparams(qparams_new);

                #pragma unroll kMaxVecsPerThread
                for (int32_t i = 0;
                        i < kMaxVecsPerThread && 4 * kWarpSize * i + threadIdx.x * 4 < D;
                        ++i) {
                    int32_t d = 4 * kWarpSize * i + threadIdx.x * 4;
                    weight_row_template.store(shared_weight_update_row[lane_id + i * kWarpSize], d, qparams_new);
                }
            }
            {% else %}
        #pragma unroll kMaxVecsPerThread
            for (int32_t i = 0;
                i < kMaxVecsPerThread && 4 * kWarpSize * i + threadIdx.x * 4 < D;
                ++i) {
                int32_t d = 4 * kWarpSize * i + threadIdx.x * 4;
                auto& grad = grad_sum[i];
                grad.store(&grad_dev_weights[weights_offset + idx * D + d]);
            }
            {% endif %}
        }
    }
}


template <
    typename emb_t,
    typename cache_t,
    size_t kMaxVecsPerThread>
__global__
__launch_bounds__(kBackwardMaxThreads)
void
split_embedding_backward_codegen_{{ optimizer }}_{{ wdesc }}_kernel_warp_per_row_1(
    const PackedTensorAccessor32<acc_type<cache_t,true>, 2, RestrictPtrTraits>
        grad_output,
    PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> dev_weights,
    {% if not dense %}
    PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> uvm_weights,
    PackedTensorAccessor64<cache_t, 2, RestrictPtrTraits> lxu_cache_weights,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        weights_placements,
    {% endif %}
    const PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> weights_offsets,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> D_offsets,
    const PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits>
        hash_size_cumsum,
    const PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits>
        sorted_linear_indices_run,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        sorted_linear_indices_cumulative_run_lengths,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        sorted_linear_indices_run_lengths,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> sorted_infos,
    {% if not dense %}
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        sorted_lxu_cache_locations,
    {% endif %}
    {% if weighted %}
    const PackedTensorAccessor32<acc_type<cache_t, true>, 1, RestrictPtrTraits> sorted_indice_weights,
    {% endif %}
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        sorted_linear_indices_num_runs,
    int32_t max_segment_length_per_warp,
    {% if not dense %}
    bool stochastic_rounding,
    PhiloxCudaState stochastic_rounding_philox_args,
    {% else %}
    PackedTensorAccessor64<cache_t, 1, RestrictPtrTraits> grad_dev_weights,
    {% endif %}
    FixedDivisor fd,
    {{ args.split_kernel_args | join(", ") }}) {

    const int32_t T = D_offsets.size(0) - 1;
    const int32_t B = grad_output.size(0);
    const int32_t run_id = blockIdx.x * blockDim.y + threadIdx.y;

    if (run_id >= sorted_linear_indices_run.size(0)) {
        return;
    }
    if (run_id >= sorted_linear_indices_num_runs[0]) {
        return;
    }
    const int64_t linear_index = sorted_linear_indices_run[run_id];
    const int32_t segment_start =
        sorted_linear_indices_cumulative_run_lengths[run_id];
    const int32_t segment_end =
        sorted_linear_indices_cumulative_run_lengths[run_id + 1];
    const int32_t SL = segment_end - segment_start;

    if (SL >= max_segment_length_per_warp) {
        return;
    }

    // now, each segment corresponds to exactly one table `t` and row in
    // that table (`idx`). Thus, we can hoist out some of the book-keeping.
    const auto info_0 = sorted_infos[segment_start];
    int32_t t_0 = fd.Div(info_0); // info_0 / B;

    int64_t hash_size = hash_size_cumsum[t_0];
    int32_t D = D_offsets[t_0 + 1] - D_offsets[t_0];
    int64_t idx = linear_index - hash_size;

    const int32_t SL_per_warp = div_round_up(SL, blockDim.y);
    const int32_t sl_start = 0;
    const int32_t sl_end = SL;
    Vec4T<acc_type<cache_t, true>> grad_sum[kMaxVecsPerThread];
    for (int32_t sl = sl_start; sl < sl_end; sl += kWarpSize) {
        int32_t sl_j = sl + threadIdx.x;
        int32_t b_t = sl_j < sl_end ? sorted_infos[segment_start + sl_j] : 0;
        int32_t b; //= b_t % B;
        int32_t t; //= b_t / B;
        fd.DivMod(b_t, &t, &b);
        int32_t D_start = D_offsets[t];
        {% if weighted %}
        acc_type<cache_t, true> idx_weight = sl_j < sl_end ? sorted_indice_weights[segment_start + sl_j] : 0.0;
        {% endif %}

        for (int32_t j = 0; j < kWarpSize && sl + j < sl_end; ++j) {
            int32_t b_j = __shfl_sync(0xFFFFFFFF, b, j);
            int32_t D_start_j = __shfl_sync(0xFFFFFFFF, D_start, j);
            {% if weighted %}
            acc_type<cache_t, true> idx_weight_j = __shfl_sync(0xFFFFFFFF, idx_weight, j);
            {% endif %}

    #pragma unroll kMaxVecsPerThread
        for (int32_t i = 0;
            i < kMaxVecsPerThread && 4 * kWarpSize * i + threadIdx.x * 4 < D;
            ++i) {
            int32_t d = 4 * kWarpSize * i + threadIdx.x * 4;
            Vec4T<acc_type<cache_t, true>> grad_out_vec(
                &grad_output[b_j][0] + D_start_j + d);
                {% if weighted %}
                grad_sum[i].fma_(grad_out_vec, idx_weight_j);
                {% else %}
                grad_sum[i].acc.x += grad_out_vec.acc.x;
                grad_sum[i].acc.y += grad_out_vec.acc.y;
                grad_sum[i].acc.z += grad_out_vec.acc.z;
                grad_sum[i].acc.w += grad_out_vec.acc.w;
                {% endif %}
            }
        }
    }
    int64_t weights_offset = weights_offsets[t_0];
    {% if not dense %}
    emb_t* __restrict__ weights{nullptr};
    cache_t* __restrict__ cache_weights{nullptr};
    int32_t D_emb = D;
    if (std::is_same<emb_t, uint8_t>::value) {
        D_emb += kINT8QparamsBytes;
    }
    const auto weights_placement = weights_placements[t_0];
    if (weights_placement == DEVICE) {
        weights = &dev_weights[weights_offset + idx * D_emb];
    } else {
        weights = &uvm_weights[weights_offset + idx * D_emb];
    }
    if (weights_placement == MANAGED_CACHING) {
        int32_t cache_idx = sorted_lxu_cache_locations[segment_start];
        if (cache_idx != kCacheLocationMissing) {
            cache_weights = &lxu_cache_weights[cache_idx][0];
        }
    }
    {% for tensor in args.split_tensors %}
    acc_type<cache_t, true>* __restrict__ {{ tensor }};
    const auto {{ tensor }}_placement = {{ tensor }}_placements[t_0];
    int64_t {{ tensor }}_offset = {{ tensor }}_offsets[t_0];
    if ({{ tensor }}_placement == DEVICE) {
        {{ tensor }} = &{{ tensor }}_dev[{{ tensor }}_offset];
    } else {
        {{ tensor }} = &{{ tensor }}_uvm[{{ tensor }}_offset];
    }
    {% endfor %}

    struct SharedMemory<Vec4T<acc_type<cache_t, true>>> weight_update_buffer;
    Vec4T<acc_type<cache_t, true>>* shared_weight_update_row = weight_update_buffer.getPointer();
    auto weight_row_template = WeightRow<emb_t, cache_t, acc_type<cache_t, true>>(weights, cache_weights, D, nullptr);
    if (!std::is_same<emb_t, float>::value && stochastic_rounding) {
        StochasticRoundingRNGState state;
        // different for every *run* and every *thread*.
        auto stochastic_rounding_seeds =
            at::cuda::philox::unpack(stochastic_rounding_philox_args);
        stochastic_rounding_init(
            std::get<0>(stochastic_rounding_seeds) ^
                std::get<1>(stochastic_rounding_seeds),
            threadIdx.x + run_id * blockDim.x,
            &state);
        weight_row_template.set_stoc_state(&state);
    }
    float2 qparams_template;
    if (std::is_same<emb_t, uint8_t>::value && !cache_weights){
        qparams_template = weight_row_template.load_qparams();
    }

    {{ split_precomputation }}

    float2 qparams_new;
    #pragma unroll kMaxVecsPerThread
    for (int32_t i = 0;
            i < kMaxVecsPerThread && 4 * kWarpSize * i + threadIdx.x * 4 < D;
            ++i) {
        int32_t d = 4 * kWarpSize * i + threadIdx.x * 4;
        Vec4T<acc_type<cache_t, true>> weight_new = weight_row_template.load(d, qparams_template);
        auto& grad = grad_sum[i];
        {{ split_weight_update }}
        if (std::is_same<emb_t, uint8_t>::value && !cache_weights) {
            shared_weight_update_row[threadIdx.x + i * kWarpSize + threadIdx.y * kMaxVecsPerThread * kWarpSize] = weight_new;
        } else {
            weight_row_template.store(weight_new, d, qparams_new); // qparams_new not used if type is not int8
        }
    }

    if (std::is_same<emb_t, uint8_t>::value && !cache_weights) {
        // calculate new qparams after row update
        qparams_new = thrust_find_qparams<acc_type<cache_t, true>>(&shared_weight_update_row[threadIdx.y * kMaxVecsPerThread * kWarpSize], D);
        weight_row_template.store_qparams(qparams_new);

        // fetch cached updated row from shared mem and quantize on-the-fly when saving to lowp embedding
        #pragma unroll kMaxVecsPerThread
        for (int32_t i = 0;
                i < kMaxVecsPerThread && 4 * kWarpSize * i + threadIdx.x * 4 < D;
                ++i) {
            int32_t d = 4 * kWarpSize * i + threadIdx.x * 4;
            weight_row_template.store(shared_weight_update_row[threadIdx.x  + i * kWarpSize + threadIdx.y * kMaxVecsPerThread * kWarpSize], d, qparams_new);
        }
    }
    {% else %}
#pragma unroll kMaxVecsPerThread
    for (int32_t i = 0;
        i < kMaxVecsPerThread && 4 * kWarpSize * i + threadIdx.x * 4 < D;
        ++i) {
        int32_t d = 4 * kWarpSize * i + threadIdx.x * 4;
        auto& grad = grad_sum[i];
        grad.store(&grad_dev_weights[weights_offset + idx * D + d]);
    }
    {% endif %}
}

template <typename cache_t, typename emb_t>
__global__ void __launch_bounds__(kMaxThreads) grad_mean_kernel(
    const PackedTensorAccessor32<acc_type<cache_t, true>, 2, RestrictPtrTraits>
        grad_output,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> D_offsets,
    const PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> offsets,
    PackedTensorAccessor32<acc_type<cache_t, true>, 2, RestrictPtrTraits>
        grad_output_mean) {
  int32_t B = grad_output.size(0);
  int32_t T = D_offsets.size(0) - 1;
  int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
  int32_t b = b_t % B;
  int32_t t = b_t / B;

  if (b_t >= B * T) {
    return;
  }
  int32_t D_start = D_offsets[t];
  int32_t D_end = D_offsets[t + 1];
  int32_t D = D_end - D_start;
  int64_t indices_start = offsets[t * B + b];
  int64_t indices_end = offsets[t * B + b + 1];
  int32_t L = indices_end - indices_start;

  if (L != 0) {
    for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
      Vec4T<acc_type<cache_t, true>> grad_out_vec(&grad_output[b][D_start + d * 4]);
      grad_out_vec.acc.x /= L;
      grad_out_vec.acc.y /= L;
      grad_out_vec.acc.z /= L;
      grad_out_vec.acc.w /= L;
      grad_out_vec.store(&grad_output_mean[b][D_start + d * 4]);
    }
  } else {
    for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
      Vec4T<acc_type<cache_t, true>> grad_out_vec(&grad_output[b][D_start + d * 4]);
      grad_out_vec.store(&grad_output_mean[b][D_start + d * 4]);
    }
  }
}

{{ "void" if not dense else "Tensor" }} split_embedding_backward_codegen_{{ optimizer }}_{{ wdesc }}_exact_cuda(
    Tensor grad_output,
    Tensor dev_weights,
    {% if not dense %}
    Tensor uvm_weights,
    Tensor lxu_cache_weights,
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
    {% if weighted %}
    Tensor indice_weights,
    {% endif %}
    {% if not dense %}
    Tensor lxu_cache_locations,
    {% endif %}
    int64_t unused_,
    int64_t max_segment_length_per_warp,
    {% if not dense %}
    bool stochastic_rounding,
    {% endif %}
    {{ args.split_function_args | join(", ") }}) {
    at::cuda::OptionalCUDAGuard device_guard;
    device_guard.set_index(dev_weights.get_device());

    {% if dense %}
    auto grad_dev_weights = zeros_like(dev_weights);
    {% endif %}

    // short-circuit if there are zero indices.
    if (indices.numel() == 0) {
        return {{ "grad_dev_weights" if dense else "" }};
    }

    int32_t T = D_offsets.numel() - 1;
    TORCH_CHECK(T > 0);
    // offsets = [B x T  + 1]
    const auto B = (offsets.size(0) - 1) / T;
    TORCH_CHECK(B > 0);
    auto BT_block_size = kMaxThreads / kWarpSize;
    TORCH_CHECK(BT_block_size * kWarpSize <= kMaxThreads);
    TORCH_CHECK(max_D <= {{ max_embedding_dim }});

    // V100: 96 KB; A100: 160 KB.
    int max_shared_bytes = 0;
    cudaDeviceGetAttribute(&max_shared_bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev_weights.get_device());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    int shared_kb = max_shared_bytes >> 10;
    // V100: 64 KB; A100: 96 KB.
    // Use 2/3 of the available GPU shared mem; leave rooms for L1$.
    int used_shared_kb = round_down(shared_kb * 2 / 3, 16);
    TORCH_CHECK(used_shared_kb > 0);
    int used_shared_bytes = used_shared_kb << 10;

    auto infos = at::empty_like(indices, indices.options().dtype(kInt));
    auto infos_sorted = at::empty_like(infos);
    auto linear_indices = at::empty_like(indices);
    auto linear_indices_sorted = at::empty_like(indices);
    linearize_index_kernel<<<
        div_round_up(B * T, kMaxThreads),
        kMaxThreads,
        0,
        at::cuda::getCurrentCUDAStream()>>>(
        hash_size_cumsum.packed_accessor32<int64_t, 1, RestrictPtrTraits>(),
        indices.packed_accessor32<int64_t, 1, RestrictPtrTraits>(),
        offsets.packed_accessor32<int64_t, 1, RestrictPtrTraits>(),
        infos.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
        linear_indices.packed_accessor32<int64_t, 1, RestrictPtrTraits>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    {
        size_t temp_storage_bytes = 0;
        AT_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
            nullptr,
            temp_storage_bytes,
            linear_indices.data_ptr<int64_t>(),
            linear_indices_sorted.data_ptr<int64_t>(),
            infos.data_ptr<int32_t>(),
            infos_sorted.data_ptr<int32_t>(),
            linear_indices.numel(),
            0,
            total_hash_size_bits,
            at::cuda::getCurrentCUDAStream(),
            false));
        auto temp_storage = at::empty(
            {static_cast<int64_t>(temp_storage_bytes)},
            indices.options().dtype(kByte));
        AT_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
            temp_storage.data_ptr(),
            temp_storage_bytes,
            linear_indices.data_ptr<int64_t>(),
            linear_indices_sorted.data_ptr<int64_t>(),
            infos.data_ptr<int32_t>(),
            infos_sorted.data_ptr<int32_t>(),
            linear_indices.numel(),
            0,
            total_hash_size_bits,
            at::cuda::getCurrentCUDAStream(),
            false));
    }
    {% if not dense %}
    auto lxu_cache_locations_sorted = at::empty_like(lxu_cache_locations);
    if (lxu_cache_locations.size(0) > 0) {
        size_t temp_storage_bytes = 0;
        AT_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
            nullptr,
            temp_storage_bytes,
            linear_indices.data_ptr<int64_t>(),
            linear_indices_sorted.data_ptr<int64_t>(),
            lxu_cache_locations.data_ptr<int32_t>(),
            lxu_cache_locations_sorted.data_ptr<int32_t>(),
            linear_indices.numel(),
            0,
            total_hash_size_bits,
            at::cuda::getCurrentCUDAStream(),
            false));
        auto temp_storage = at::empty(
            {static_cast<int64_t>(temp_storage_bytes)},
            indices.options().dtype(kByte));
        AT_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
            temp_storage.data_ptr(),
            temp_storage_bytes,
            linear_indices.data_ptr<int64_t>(),
            linear_indices_sorted.data_ptr<int64_t>(),
            lxu_cache_locations.data_ptr<int32_t>(),
            lxu_cache_locations_sorted.data_ptr<int32_t>(),
            linear_indices.numel(),
            0,
            total_hash_size_bits,
            at::cuda::getCurrentCUDAStream(),
            false));
    }
    {% endif %}
    auto sorted_linear_indices_run = at::empty_like(indices);
    auto sorted_linear_indices_run_lengths =
        at::zeros_like(indices, indices.options().dtype(kInt));
    auto sorted_linear_indices_num_runs =
        at::zeros({1}, indices.options().dtype(kInt));

    {
        size_t temp_storage_bytes = 0;
        AT_CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(
            nullptr,
            temp_storage_bytes,
            linear_indices_sorted.data_ptr<int64_t>(),
            sorted_linear_indices_run.data_ptr<int64_t>(),
            sorted_linear_indices_run_lengths.data_ptr<int32_t>(),
            sorted_linear_indices_num_runs.data_ptr<int32_t>(),
            linear_indices_sorted.numel(),
            at::cuda::getCurrentCUDAStream()));
        // Allocate temporary storage
        auto temp_storage = at::empty(
            {static_cast<int64_t>(temp_storage_bytes)},
            indices.options().dtype(kByte));
        // Run encoding
        AT_CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(
            temp_storage.data_ptr(),
            temp_storage_bytes,
            linear_indices_sorted.data_ptr<int64_t>(),
            sorted_linear_indices_run.data_ptr<int64_t>(),
            sorted_linear_indices_run_lengths.data_ptr<int32_t>(),
            sorted_linear_indices_num_runs.data_ptr<int32_t>(),
            linear_indices_sorted.numel(),
            at::cuda::getCurrentCUDAStream()));
    }

    auto sorted_linear_indices_cumulative_run_lengths =
        asynchronous_complete_cumsum(sorted_linear_indices_run_lengths);

    {% if not dense %}
    DISPATCH_EMB_CACHE_TYPES(
    {% else %}
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    {% endif %}
        dev_weights.type(),
        {% if not dense %}
        lxu_cache_weights.type(),
        {% endif %}
        "split_embedding_backward_{{ optimizer }}_exact_kernel",
        ([&] {

            {% if weighted %}
            auto indice_weights_sorted = at::empty_like(indice_weights);
            {
            size_t temp_storage_bytes = 0;
            AT_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
                nullptr,
                temp_storage_bytes,
                linear_indices.data_ptr<int64_t>(),
                linear_indices_sorted.data_ptr<int64_t>(),
                {% if not dense %}
                indice_weights.data_ptr<acc_type<cache_t, true>>(),
                indice_weights_sorted.data_ptr<acc_type<cache_t, true>>(),
                {% else %}
                indice_weights.data_ptr<acc_type<scalar_t, true>>(),
                indice_weights_sorted.data_ptr<acc_type<scalar_t, true>>(),
                {% endif %}
                linear_indices.numel(),
                0,
                total_hash_size_bits,
                at::cuda::getCurrentCUDAStream(),
                false));
            auto temp_storage = at::empty(
                {static_cast<int64_t>(temp_storage_bytes)},
                indices.options().dtype(kByte));
            AT_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
                temp_storage.data_ptr(),
                temp_storage_bytes,
                linear_indices.data_ptr<int64_t>(),
                linear_indices_sorted.data_ptr<int64_t>(),
                {% if not dense %}
                indice_weights.data_ptr<acc_type<cache_t, true>>(),
                indice_weights_sorted.data_ptr<acc_type<cache_t, true>>(),
                {% else %}
                indice_weights.data_ptr<acc_type<scalar_t, true>>(),
                indice_weights_sorted.data_ptr<acc_type<scalar_t, true>>(),
                {% endif %}
                linear_indices.numel(),
                0,
                total_hash_size_bits,
                at::cuda::getCurrentCUDAStream(),
                false));
            }
            {% endif %}

            auto grad_output_accessor = grad_output.packed_accessor32<
                acc_type<{{ "scalar_t" if dense else "cache_t" }}, true>,
                2,
                RestrictPtrTraits>();
            Tensor grad_output_mean;
            if (pooling_mode == MEAN) {
              grad_output_mean = at::empty_like(grad_output);
              grad_mean_kernel<{{ "scalar_t, scalar_t" if dense else "cache_t, emb_t" }}>
                  <<<div_round_up((B * T), kMaxThreads / kWarpSize),
                     dim3(kWarpSize, kMaxThreads / kWarpSize),
                     0,
                     at::cuda::getCurrentCUDAStream()>>>(
                      grad_output_accessor,
                      D_offsets
                          .packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                      offsets
                          .packed_accessor32<int64_t, 1, RestrictPtrTraits>(),
                      grad_output_mean.packed_accessor32<
                          acc_type<{{ "scalar_t" if dense else "cache_t" }}, true>,
                          2,
                          RestrictPtrTraits>());
              C10_CUDA_KERNEL_LAUNCH_CHECK();
              grad_output_accessor = grad_output_mean.packed_accessor32<
                  acc_type<{{ "scalar_t" if dense else "cache_t" }}, true>,
                  2,
                  RestrictPtrTraits>();
            }

            {% if not dense %}
            PhiloxCudaState rng_engine_inputs;
            if (stochastic_rounding && !std::is_same<emb_t, float>::value) {
                auto gen = at::cuda::detail::getDefaultCUDAGenerator();
                std::lock_guard<std::mutex> lock(gen.mutex());
                rng_engine_inputs =
                    at::check_generator<at::CUDAGeneratorImpl>(gen)
                        ->philox_cuda_state(4);
            }
            {% endif %}
            {% for kMaxVecsPerThread in range(1, max_embedding_dim // 128 + 1) %}
            if (max_D <= {{ 128 * kMaxVecsPerThread }}) {
            // Stay under used_shared_kb of shared memory (V100: 64 KB; A100: 96 KB), BT_block_size must be a power of two.
            while (BT_block_size * sizeof(acc_type<{{ "scalar_t" if dense else "cache_t" }}, true>) * 4 * kWarpSize * {{ kMaxVecsPerThread }} >= used_shared_bytes) {
                BT_block_size /= 2;
            }
            TORCH_CHECK(BT_block_size >= 1);
            if (std::is_same<{{ "scalar_t" if dense else "emb_t" }}, double>::value) {
                // Otherwise we see CUDA kernel launch failures despite the above checks.
                BT_block_size = 1;
            }

            auto long_run_ids = at::empty_like(sorted_linear_indices_run_lengths);
            auto num_long_run_ids = at::zeros({1}, indices.options().dtype(kLong));
            split_embedding_backward_codegen_{{ optimizer }}_{{ wdesc }}_find_long_segments<<<
                div_round_up(sorted_linear_indices_run_lengths.numel(), kMaxThreads),
                kMaxThreads,
                0,
                at::cuda::getCurrentCUDAStream()
            >>>(
                sorted_linear_indices_num_runs.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                sorted_linear_indices_run_lengths.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                long_run_ids.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                num_long_run_ids.packed_accessor32<int64_t, 1, RestrictPtrTraits>(),
                max_segment_length_per_warp);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            // Check https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x
            // "Compute capability 7.x devices allow a single thread block to
            // address the full capacity of shared memory: 96 KB on Volta,
            // 64 KB on Turing. Kernels relying on shared memory allocations
            // over 48 KB per block are architecture-specific, as such they
            // must use dynamic shared memory (rather than statically sized
            // arrays) and require an explicit opt-in using cudaFuncSetAttribute()".
            cudaFuncSetAttribute(
                split_embedding_backward_codegen_{{ optimizer }}_{{ wdesc }}_kernel_cta_per_row_1<
                {% if not dense %}
                emb_t,
                cache_t,
                {% else %}
                scalar_t,
                scalar_t,
                {% endif %}
                {{ kMaxVecsPerThread }}>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                used_shared_bytes); // V100: 64 KB; A100: 96 KB.
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            split_embedding_backward_codegen_{{ optimizer }}_{{ wdesc }}_kernel_cta_per_row_1<
                {% if not dense %}
                emb_t,
                cache_t,
                {% else %}
                scalar_t,
                scalar_t,
                {% endif %}
                {{ kMaxVecsPerThread }}>
                <<<div_round_up(linear_indices.numel(), 32 * kWarpSize),
                    dim3(kWarpSize, BT_block_size),
                    BT_block_size * sizeof(acc_type<{{ "scalar_t" if dense else "cache_t" }}, true>) * 4 * kWarpSize *
                        {{ kMaxVecsPerThread }},
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output_accessor,
                    {% if not dense %}
                    dev_weights.packed_accessor64<emb_t, 1, RestrictPtrTraits>(),
                    uvm_weights.packed_accessor64<emb_t, 1, RestrictPtrTraits>(),
                    lxu_cache_weights.packed_accessor64<cache_t, 2, RestrictPtrTraits>(),
                    weights_placements.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                    {% else %}
                    dev_weights.packed_accessor64<scalar_t, 1, RestrictPtrTraits>(),
                    {% endif %}
                    weights_offsets.packed_accessor32<int64_t, 1, RestrictPtrTraits>(),
                    D_offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                    hash_size_cumsum.packed_accessor32<int64_t, 1, RestrictPtrTraits>(),
                    sorted_linear_indices_run
                        .packed_accessor32<int64_t, 1, RestrictPtrTraits>(),
                    sorted_linear_indices_cumulative_run_lengths
                        .packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                    sorted_linear_indices_run_lengths
                        .packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                    long_run_ids.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                    num_long_run_ids.packed_accessor32<int64_t, 1, RestrictPtrTraits>(),
                    infos_sorted.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                    {% if not dense %}
                    lxu_cache_locations_sorted.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                    {% endif %}
                    {% if weighted %}
                    indice_weights_sorted.packed_accessor32<acc_type<{{ "scalar_t" if dense else "cache_t" }}, true>, 1, RestrictPtrTraits>(),
                    {% endif %}
                    {% if not dense %}
                    stochastic_rounding,
                    rng_engine_inputs,
                    {% else %}
                    grad_dev_weights.packed_accessor64<scalar_t, 1, RestrictPtrTraits>(),
                    {% endif %}
                    FixedDivisor(B),
                    {{ args.split_kernel_arg_constructors | join(", ") }});
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            cudaFuncSetAttribute(
                split_embedding_backward_codegen_{{ optimizer }}_{{ wdesc }}_kernel_warp_per_row_1<
                {% if not dense %}
                emb_t,
                cache_t,
                {% else %}
                scalar_t,
                scalar_t,
                {% endif %}
                {{ kMaxVecsPerThread }}>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                used_shared_bytes); // V100: 64 KB; A100: 96 KB.
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            split_embedding_backward_codegen_{{ optimizer }}_{{ wdesc }}_kernel_warp_per_row_1<
                {% if not dense %}
                emb_t,
                cache_t,
                {% else %}
                scalar_t,
                scalar_t,
                {% endif %}
                {{ kMaxVecsPerThread }}>
                <<<div_round_up(linear_indices.numel(), kBackwardMaxThreads / kWarpSize),
                    dim3(kWarpSize, kBackwardMaxThreads / kWarpSize),
                    BT_block_size * sizeof(
                    acc_type<
                    {% if not dense %}
                    cache_t
                    {% else %}
                    scalar_t
                    {% endif %},
                    true>) * 4 * kWarpSize *
                        {{ kMaxVecsPerThread }},
                    at::cuda::getCurrentCUDAStream()>>>(
                    grad_output_accessor,
                    {% if not dense %}
                    dev_weights.packed_accessor64<emb_t, 1, RestrictPtrTraits>(),
                    uvm_weights.packed_accessor64<emb_t, 1, RestrictPtrTraits>(),
                    lxu_cache_weights.packed_accessor64<cache_t, 2, RestrictPtrTraits>(),
                    weights_placements.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                    {% else %}
                    dev_weights.packed_accessor64<scalar_t, 1, RestrictPtrTraits>(),
                    {% endif %}
                    weights_offsets.packed_accessor32<int64_t, 1, RestrictPtrTraits>(),
                    D_offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                    hash_size_cumsum.packed_accessor32<int64_t, 1, RestrictPtrTraits>(),
                    sorted_linear_indices_run
                        .packed_accessor32<int64_t, 1, RestrictPtrTraits>(),
                    sorted_linear_indices_cumulative_run_lengths
                        .packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                    sorted_linear_indices_run_lengths
                        .packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                    infos_sorted.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                    {% if not dense %}
                    lxu_cache_locations_sorted.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                    {% endif %}
                    {% if weighted %}
                    indice_weights_sorted.packed_accessor32<acc_type<{{ "scalar_t" if dense else "cache_t" }}, true>, 1, RestrictPtrTraits>(),
                    {% endif %}
                    sorted_linear_indices_num_runs
                        .packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                    max_segment_length_per_warp,
                    {% if not dense %}
                    stochastic_rounding,
                    rng_engine_inputs,
                    {% else %}
                    grad_dev_weights.packed_accessor64<scalar_t, 1, RestrictPtrTraits>(),
                    {% endif %}
                    FixedDivisor(B),
                    {{ args.split_kernel_arg_constructors | join(", ") }});
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return;
        }
        {% endfor %}
        }));

    return {{ "grad_dev_weights" if dense else "" }};
}
