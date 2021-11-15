/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
{% set wdesc =  "weighted" if weighted else "unweighted" %}
#include "codegen/embedding_forward_template_helpers.cuh"

{% if not dense %}
#include "codegen/embedding_common.h"
constexpr int32_t kCacheLocationMissing = -1;
{% endif %}
enum {
  DEVICE = 0,
  MANAGED = 1,
  MANAGED_CACHING = 2,
};

constexpr size_t kForwardMaxThreads = 512;

using namespace at;
using namespace fbgemm_gpu;

template <
    typename emb_t,
    typename cache_t,
    {% if not dense %}
    typename output_t,
    {% endif %}
    typename index_t,
    size_t kMaxVecsPerThread>
__launch_bounds__(kForwardMaxThreads)
__global__ void {{ "dense" if dense else "split" }}_embedding_codegen_forward_{{ wdesc }}_kernel(
    const PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> dev_weights,
    {% if not dense %}
    const PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> uvm_weights,
    const PackedTensorAccessor64<cache_t, 2, RestrictPtrTraits>
        lxu_cache_weights,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        weights_placements,
    {% endif %}
    const PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> weights_offsets,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> D_offsets,
    const PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> indices,
    const PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> offsets,
    int64_t pooling_mode,
    {% if weighted %}
    PackedTensorAccessor32<acc_type<cache_t, true>, 1, RestrictPtrTraits>
        indice_weights,
    {% endif %}
    {% if not dense %}
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        lxu_cache_locations,
    PackedTensorAccessor32<output_t, 2, RestrictPtrTraits>
        output // [B][total_D],
    {% else %}
    PackedTensorAccessor32<acc_type<cache_t,true>, 2, RestrictPtrTraits>
        output // [B][total_D],
    {% endif %}
    ) {
    int32_t B = output.size(0);
    int32_t T = D_offsets.size(0) - 1;
    int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
    int32_t t = b_t / B;
    int32_t b = b_t % B;

    if (b_t >= B * T) {
        return;
    }
    int64_t weights_offset = weights_offsets[t];
    int32_t D_start = D_offsets[t];
    int32_t D_end = D_offsets[t + 1];
    int32_t D = D_end - D_start;
    index_t indices_start = offsets[t * B + b];
    index_t indices_end = offsets[t * B + b + 1];
    int32_t L = indices_end - indices_start;
    const emb_t* __restrict__ weights;
    {% if not dense %}
    const auto placement = weights_placements[t];
    if (placement == DEVICE) {
        weights = &dev_weights[weights_offset];
    } else {
        weights = &uvm_weights[weights_offset];
    }
    {% else %}
    weights = &dev_weights[weights_offset];
    {% endif %}


    Vec4T<cache_t> accumulators[kMaxVecsPerThread];
    for (int32_t l_start = 0; l_start < L; l_start += kWarpSize) {
        int32_t l = l_start + threadIdx.x;
        int64_t idx = l < L ? indices[indices_start + l] : 0;
        {% if not dense %}
        int32_t cache_idx = (placement == MANAGED_CACHING && l < L) ? lxu_cache_locations[indices_start + l] : 0;
        {% endif %}
        {% if weighted %}
        acc_type<cache_t, true> idx_weight = l < L ? indice_weights[indices_start + l] : 0;
        {% endif %}
        for (auto j = 0; j < kWarpSize && l_start + j < L; ++j) {
            int64_t idx_j = __shfl_sync(0xFFFFFFFF, idx, j);

            {% if not dense %}
            int32_t cache_idx_j = __shfl_sync(0xFFFFFFFF, cache_idx, j);
            {% endif %}

            {% if weighted %}
            acc_type<cache_t, true> idx_weight_j = __shfl_sync(0xFFFFFFFF, idx_weight, j);
            {% endif %}

            int32_t D_emb = D;
            if (std::is_same<emb_t, uint8_t>::value) {
                D_emb += kINT8QparamsBytes;
            }
            {% if not dense %}
            auto weight_row_cache = WeightRow<emb_t, cache_t, cache_t>(
                const_cast<emb_t*>(&weights[idx_j * D_emb]),
                const_cast<cache_t*>(&lxu_cache_weights[cache_idx_j][0]),
                D,
                nullptr);
            float2 qparams_cache; // assume cache is fp16/fp32 which doesn't require qparams

            {% endif %}
            auto weight_row_emb = WeightRow<emb_t, cache_t, cache_t>(
                const_cast<emb_t*>(&weights[idx_j * D_emb]),
                nullptr,
                D,
                nullptr);
            float2 qparams_emb;
            if (std::is_same<emb_t, uint8_t>::value) {
                qparams_emb = weight_row_emb.load_qparams();
            }
            #pragma unroll kMaxVecsPerThread
            for (int32_t i = 0;
                i < kMaxVecsPerThread && 4 * kWarpSize * i + threadIdx.x * 4 < D;
                ++i) {
                int32_t d = 4 * kWarpSize * i + threadIdx.x * 4;
                {% if not dense %}
                if (placement == MANAGED_CACHING && cache_idx_j != kCacheLocationMissing) {
                    Vec4T<cache_t> weight = weight_row_cache.load(d, qparams_cache);
                    {% if weighted %}
                    accumulators[i].fma_(weight, idx_weight_j);
                    {% else %}
                    accumulators[i].acc.x += weight.acc.x;
                    accumulators[i].acc.y += weight.acc.y;
                    accumulators[i].acc.z += weight.acc.z;
                    accumulators[i].acc.w += weight.acc.w;
                    {% endif %}
                } else {
                    Vec4T<cache_t> weight = weight_row_emb.load(d, qparams_emb);
                    {% if weighted %}
                    accumulators[i].fma_(weight, idx_weight_j);
                    {% else %}
                    accumulators[i].acc.x += weight.acc.x;
                    accumulators[i].acc.y += weight.acc.y;
                    accumulators[i].acc.z += weight.acc.z;
                    accumulators[i].acc.w += weight.acc.w;
                    {% endif %}
                }
                {% else %}
                    Vec4T<cache_t> weight = weight_row_emb.load(d, qparams_emb);
                    {% if weighted %}
                    accumulators[i].fma_(weight, idx_weight_j);
                    {% else %}
                    accumulators[i].acc.x += weight.acc.x;
                    accumulators[i].acc.y += weight.acc.y;
                    accumulators[i].acc.z += weight.acc.z;
                    accumulators[i].acc.w += weight.acc.w;
                    {% endif %}
                {% endif %}
            }
        }
    }

    {% if not dense %}
    if (!std::is_same<output_t, uint8_t>::value) {
        #pragma unroll kMaxVecsPerThread
        for (int32_t i = 0;
        i < kMaxVecsPerThread && 4 * kWarpSize * i + threadIdx.x * 4 < D;
        ++i) {
            if (pooling_mode == MEAN && L != 0) {
                accumulators[i].acc.x /= L;
                accumulators[i].acc.y /= L;
                accumulators[i].acc.z /= L;
                accumulators[i].acc.w /= L;
            }
            int32_t d = 4 * kWarpSize * i + threadIdx.x * 4;
            accumulators[i].store(&output[b][D_start + d]);
        }
    } else {
        // apply per feature row-wise int8
        float thread_local_min = std::numeric_limits<float>::max();
        float thread_local_max = std::numeric_limits<float>::min();
        float2 qparams;

        #pragma unroll kMaxVecsPerThread
        for (int32_t i = 0;
            i < kMaxVecsPerThread && 4 * kWarpSize * i + threadIdx.x * 4 < D;
            ++i) {
            if (pooling_mode == MEAN && L != 0) {
                accumulators[i].acc.x /= L;
                accumulators[i].acc.y /= L;
                accumulators[i].acc.z /= L;
                accumulators[i].acc.w /= L;
            }
            thread_local_max = max(thread_local_max, vec4_max(accumulators[i]));
            thread_local_min = min(thread_local_max, vec4_min(accumulators[i]));
        }

        qparams = warp_find_qparams(thread_local_min, thread_local_max);
        int output_D_start = D_start + t * 8;
        int output_D_end = output_D_start + D;

        #pragma unroll kMaxVecsPerThread
        for (int32_t i = 0;
            i < kMaxVecsPerThread && 4 * kWarpSize * i + threadIdx.x * 4 < D;
            ++i) {
            int32_t d = 4 * kWarpSize * i + threadIdx.x * 4;
            nearest_rounding_vector<output_t, cache_t>(&output[b][output_D_start + d], accumulators[i], qparams);
        }
        if (threadIdx.x == 0) {
            store_qparams_to_row(&output[b][output_D_end], qparams);
        }

    }
    {% else %}
    // no pooled embedding quantization fusion for dense embeddings
    #pragma unroll kMaxVecsPerThread
    for (int32_t i = 0;
        i < kMaxVecsPerThread && 4 * kWarpSize * i + threadIdx.x * 4 < D;
        ++i) {
        int32_t d = 4 * kWarpSize * i + threadIdx.x * 4;
        if (pooling_mode == MEAN && L != 0) {
            accumulators[i].acc.x /= L;
            accumulators[i].acc.y /= L;
            accumulators[i].acc.z /= L;
            accumulators[i].acc.w /= L;
        }
        accumulators[i].store(&output[b][D_start + d]);
    }
    {% endif %}
}

Tensor {{ "dense" if dense else "split" }}_embedding_codegen_forward_{{ wdesc }}_cuda(
    Tensor dev_weights,
    {% if not dense %}
    Tensor uvm_weights,
    Tensor lxu_cache_weights,
    Tensor weights_placements,
    {% endif %}
    Tensor weights_offsets,
    Tensor D_offsets,
    int64_t total_D,
    int64_t max_D,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    {% if weighted %}
    Tensor indice_weights,
    {% endif %}
    {% if not dense %}
    Tensor lxu_cache_locations,
    int64_t output_dtype,
    {% endif %}
    int64_t unused
) {
    at::cuda::OptionalCUDAGuard device_guard;
    device_guard.set_index(dev_weights.get_device());

    int32_t T = D_offsets.numel() - 1;
    TORCH_CHECK(T > 0);
    // offsets = [B x T  + 1]
    int32_t B = (offsets.size(0) - 1) / T;
    TORCH_CHECK(B >= 0);
    TORCH_CHECK(total_D > 0);
    TORCH_CHECK(total_D % 4 == 0);
    TORCH_CHECK(max_D <= {{ max_embedding_dim }});
    at::Tensor output;

    {% if dense %}
    if (dev_weights.type().scalarType() == at::kHalf || dev_weights.type().scalarType() == at::kByte) {
        output = empty({B, total_D}, dev_weights.options().dtype(at::kFloat));
    } else {
        output = empty({B, total_D}, dev_weights.options());
    }
    {% else %}

    SparseType o_dtype = static_cast<SparseType>(output_dtype);
    TORCH_CHECK(o_dtype == SparseType::FP32 || o_dtype == SparseType::FP16 || o_dtype == SparseType::INT8);
    if (o_dtype == SparseType::FP32) {
        output = empty({B, total_D}, dev_weights.options().dtype(at::kFloat));
    } else if (o_dtype == SparseType::FP16) {
        output = empty({B, total_D}, dev_weights.options().dtype(at::kHalf));
    } else if (o_dtype == SparseType::INT8) {
        output = empty({B, int64_t(total_D + T * kINT8QparamsBytes)}, dev_weights.options().dtype(at::kByte));
    }

    {% endif %}
    if (B == 0) {
        return output;
    }

    {% if not dense %}
    DISPATCH_EMB_CACHE_OUTPUT_TYPES(
    {% else %}
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    {% endif %}
        dev_weights.type(),
        {% if not dense %}
        lxu_cache_weights.type(),
        output.type(),
        {% endif %}
        "batched_embedding_forward_kernel_2", ([&] {
        {% for kMaxVecsPerThread in range(1, max_embedding_dim // 128 + 1) %}
        if (max_D <= {{ 128 * kMaxVecsPerThread }}) {
            {% if not dense %}
            split_embedding_codegen_forward_{{ wdesc }}_kernel<emb_t, cache_t, output_t, int64_t, {{ kMaxVecsPerThread }}><<<
            {% else %}
            dense_embedding_codegen_forward_{{ wdesc }}_kernel<scalar_t, scalar_t, int64_t, {{ kMaxVecsPerThread }}><<<
            {% endif %}
                div_round_up((B * T), kForwardMaxThreads / kWarpSize),
                dim3(kWarpSize, kForwardMaxThreads / kWarpSize),
                0,
                at::cuda::getCurrentCUDAStream()>>>(
                dev_weights.packed_accessor64<{{ "scalar_t" if dense else "emb_t" }}, 1, RestrictPtrTraits>(),
                {% if not dense %}
                uvm_weights.packed_accessor64<emb_t, 1, RestrictPtrTraits>(),
                lxu_cache_weights.packed_accessor64<cache_t, 2, RestrictPtrTraits>(),
                weights_placements.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                {% endif %}
                weights_offsets.packed_accessor32<int64_t, 1, RestrictPtrTraits>(),
                D_offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                indices.packed_accessor32<int64_t, 1, RestrictPtrTraits>(),
                offsets.packed_accessor32<int64_t, 1, RestrictPtrTraits>(),
                pooling_mode,
                {% if weighted %}
                indice_weights.packed_accessor32<acc_type<{{ "scalar_t" if dense else "cache_t" }}, true>, 1, RestrictPtrTraits>(),
                {% endif %}
                {% if not dense %}
                lxu_cache_locations.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                output.packed_accessor32<
                    output_t,
                    2,
                    RestrictPtrTraits>()
                );
                {% else %}
                output.packed_accessor32<
                    acc_type<scalar_t, true>,
                    2,
                    RestrictPtrTraits>()
                );
                {% endif %}

            return;
        }
        {% endfor %}
        }));

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
