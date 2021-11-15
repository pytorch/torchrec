/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

{% set wdesc =  "weighted" if weighted else "unweighted" %}

#include <ATen/ATen.h>
#ifdef FBGEMM_GPU_WITH_CUDA
#include <ATen/cuda/CUDAContext.h>
#endif

#include "codegen/embedding_common.h"
#include "fbgemm_gpu/dispatch_macros.h"

#include <immintrin.h>
#include <emmintrin.h>

namespace {
enum PoolingMode { SUM = 0, MEAN = 1, NONE = 2 };

// Keep in sync with EmbeddingLocation in split_table_batched_embeddings_ops.py
enum {
  DEVICE = 0,
  MANAGED = 1,
  MANAGED_CACHING = 2,
  HOST = 3,
};

using namespace at;

// From https://stackoverflow.com/questions/55084047/intel-vector-instruction-to-zero-extend-8-4-bit-values-packed-in-a-32-bit-int-to
// TODO: dispatch at architecture time?
__attribute__((always_inline)) inline __m256i cvt_nib_epi32_HSW(uint32_t x) {
    __uint64_t x_b = _pdep_u64(x, 0x0F0F0F0F0F0F0F0F);
    __m128i x_v = _mm_cvtsi64_si128(x_b);
    return _mm256_cvtepu8_epi32(x_v);
}

__attribute__((always_inline)) inline __m256i cvt_nib_epi32_SKL(uint32_t x) {
    __m256i input = _mm256_set1_epi32(x);
    __m256i shifted = _mm256_srlv_epi32(input,_mm256_set_epi32(28,24,20,16,12,8,4,0));
    return _mm256_and_si256(shifted, _mm256_set1_epi32(0xF));
}

__attribute__((always_inline)) inline __m256i cvt_hnib_epi32_SKL(uint16_t x) {
    __m256i input = _mm256_set1_epi32(x);
    __m256i shifted = _mm256_srlv_epi32(input,_mm256_set_epi32(14,12,10,8,6,4,2,0));
    return _mm256_and_si256(shifted, _mm256_set1_epi32(0x3));
}

__attribute__((always_inline)) inline __m256i cvt_byte_SKL(uint64_t x) {
    return _mm256_cvtepu8_epi32(_mm_set1_epi64x(x));
}


inline int32_t unpadded_row_size_in_bytes(int32_t dim, SparseType weight_ty) {
    if (weight_ty == SparseType::FP32) { return dim * 4; }
    if (weight_ty == SparseType::FP16) { return dim * 2; }
    if (weight_ty == SparseType::INT8) { return dim + 4; }
    if (weight_ty == SparseType::INT4) { return dim / 2 + 4; }
    if (weight_ty == SparseType::INT2) { return dim / 4 + 4; }
    return 0;
}

uint32_t div_round_up(uint32_t a, uint32_t b) {
  return ((a + b - 1) / b);
}

uint32_t round_up(uint32_t a, uint32_t b) {
  return ((a + b - 1) / b) * b;
}

inline int32_t padded_row_size_in_bytes(int32_t dim, SparseType weight_ty) {
  auto r = unpadded_row_size_in_bytes(dim, weight_ty);
  return round_up(r, 16);
}


inline uint32_t pruned_hash_function(uint32_t h) {
    // MurmorHash3 32-bit mixing function.
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

}

void pruned_hashmap_insert_{{ wdesc }}_cpu(
    Tensor indices,
    Tensor dense_indices,
    Tensor offsets,
    Tensor hash_table,
    Tensor hash_table_offsets) {
    int32_t T = hash_table_offsets.size(0) - 1;
    int32_t B = (offsets.size(0) - 1) / T;
    TORCH_CHECK(B > 0);
    const auto* indices_acc = indices.data_ptr<int32_t>();
    const auto* dense_indices_acc = dense_indices.data_ptr<int32_t>();

    const auto* offsets_acc = offsets.data_ptr<int32_t>();
    auto hash_table_acc = hash_table.accessor<int32_t, 2>();
    const auto hash_table_offsets_acc = hash_table_offsets.accessor<int64_t, 1>();


    for (int32_t t = 0; t < T; ++t) {
        int64_t table_start = hash_table_offsets_acc[t];
        int64_t table_end = hash_table_offsets_acc[t + 1];
        if (table_start == table_end) {
            continue;
        }
        int64_t capacity = table_end - table_start;
        for (int32_t b = 0; b < B; ++b) {
            int32_t indices_start = offsets_acc[t * B + b];
            int32_t indices_end = offsets_acc[t * B + b + 1];
            int32_t L = indices_end - indices_start;
            for (int32_t l = 0; l < L; ++l) {
                int32_t idx = indices_acc[indices_start + l];
                int32_t dense_idx = dense_indices_acc[indices_start + l];
                if (dense_idx == -1) {
                    // -1 means this row has been pruned, do not insert it.
                    continue;
                }

                uint32_t slot = pruned_hash_function(static_cast<uint32_t>(idx)) % capacity;
                while (true) {
                    int32_t slot_sparse_idx = hash_table_acc[table_start + static_cast<int64_t>(slot)][0];
                    // empty slot
                    if (slot_sparse_idx == -1) {
                        hash_table_acc[table_start + static_cast<int64_t>(slot)][0] = idx;
                        hash_table_acc[table_start + static_cast<int64_t>(slot)][1] = dense_idx;
                        break;
                    }
                    // already exists (shouldn't happen in practice)
                    if (slot_sparse_idx == idx) {
                        hash_table_acc[table_start + static_cast<int64_t>(slot)][1] = dense_idx;
                        break;
                    }
                    // linear probe
                    slot = (slot + 1) % capacity;
                }
            }
        }
    }
    return;
}

template <typename output_t>
struct VecT {};

template <>
struct VecT<float> {
    void store_vec(at::Half* output_acc, __m256& acci) {
        // Error;
        TORCH_CHECK(false);
    }
    void store_vec(float* output_acc, __m256& acci) {
        _mm256_storeu_ps(output_acc, acci);
    }
};

template <>
struct VecT<at::Half> {
    void store_vec(at::Half* output_acc, __m256& acci) {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output_acc), _mm256_cvtps_ph(acci, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    }
    void store_vec(float* output_acc, __m256& acci) {
        // Error;
        TORCH_CHECK(false);
    }
};

// TODO: add uint8 store instructions

template <typename output_t>
void store_result(
    const int32_t D_vecs,
    const int32_t D_tail_elements,
    const std::vector<__m256>& acc,
    __m256& scale_vec,
    output_t* output_acc,
    const bool acc_scaling
) {
    if (D_tail_elements == 0) {
        for (auto i = 0; i < D_vecs; ++i) {
            auto acci = acc_scaling ? _mm256_mul_ps(acc[i], scale_vec) : acc[i];
            VecT<output_t> vec_st;
            vec_st.store_vec(&output_acc[8 * i], acci);
        }
    } else {
        for (auto i = 0; i < D_vecs - 1; ++i) {
            auto acci = acc_scaling ? _mm256_mul_ps(acc[i], scale_vec) : acc[i];
            VecT<output_t> vec_st;
            vec_st.store_vec(&output_acc[8 * i], acci);
        }
        if (std::is_same<output_t, float>::value) {
            std::array<float, 8> vs;
            auto acci = acc_scaling ? _mm256_mul_ps(acc[D_vecs - 1], scale_vec) : acc[D_vecs - 1];
            VecT<output_t> vec_st;
            vec_st.store_vec(vs.data(), acci);
            // To check D_tail_elements size
            std::copy(vs.data(), vs.data() + D_tail_elements, &output_acc[8 * (D_vecs - 1)]);
        } else if (std::is_same<output_t, at::Half>::value) {
            std::array<Half, 8> vs;
            auto acci = acc_scaling ? _mm256_mul_ps(acc[D_vecs - 1], scale_vec) : acc[D_vecs - 1];
            VecT<output_t> vec_st;
            vec_st.store_vec(vs.data(), acci);
            std::copy(vs.data(), vs.data() + D_tail_elements, &output_acc[8 * (D_vecs - 1)]);
        }
    }
}

Tensor int_nbit_split_embedding_codegen_forward_{{ wdesc }}_cpu(
    Tensor dev_weights,
    Tensor uvm_weights,
    Tensor weights_placements,
    Tensor weights_offsets,
    Tensor weights_tys,
    Tensor D_offsets,
    int64_t total_D,
    Tensor indices,
    Tensor offsets,
    int64_t pooling_mode,
    {% if weighted %}
    Tensor indice_weights,
    {% endif %}
    int64_t output_dtype,
    int64_t unused
) {
    int32_t T = D_offsets.numel() - 1;
    TORCH_CHECK(T > 0);
    // offsets = [B x T  + 1]
    int32_t B = (offsets.size(0) - 1) / T;
    TORCH_CHECK(B >= 0);
    TORCH_CHECK(total_D > 0);
    bool pinned_memory = false;
#ifdef FBGEMM_GPU_WITH_CUDA
    if (globalContext().hasCUDA() && ::at::cuda::is_available()) {
      pinned_memory = true;
    }
#endif

    at::Tensor output;
    SparseType o_dtype = static_cast<SparseType>(output_dtype);
    TORCH_CHECK(o_dtype == SparseType::FP32 || o_dtype == SparseType::FP16);
    if (o_dtype == SparseType::FP32) {
        output = at::empty({B, total_D}, dev_weights.options().dtype(at::kFloat).pinned_memory(pinned_memory));
    } else if (o_dtype == SparseType::FP16) {
        output = at::empty({B, total_D}, dev_weights.options().dtype(at::kHalf).pinned_memory(pinned_memory));
    }


    if (B == 0) {
        return output;
    }

    const int32_t* weights_placements_ptr = weights_placements.data_ptr<int32_t>();
    const uint8_t* weights_acc;

    const auto* weights_tys_acc = weights_tys.data_ptr<uint8_t>();

    DISPATCH_OUTPUT_TYPES(output.type(), "intn_split_embedding_codegen_forward_kernel", ([&] {
        auto* output_acc = output.data_ptr<output_t>();
        {% if weighted %}
        const float* indice_weights_acc = indice_weights.data_ptr<float>();
        {% endif %}
        // Empty array filled with zeros (thus accumulating to zero).
        // max-D = 1024, max-sizeof(T) = sizeof(float) = 4.
        alignas(32) static constexpr std::array<uint8_t, 1024 * 4> zero_row = {0};
        std::vector<__m256> acc; //, {{ kMaxVecsPerThread }} > acc;

        AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "int_nbit_split_embedding_codegen_forward_", [&] () {
            const auto* indices_acc = indices.data_ptr<index_t>();
            const auto* offsets_acc = offsets.data_ptr<index_t>();
            const auto* D_offsets_acc = D_offsets.data_ptr<int32_t>();
            const auto* weights_offsets_acc = weights_offsets.data_ptr<int64_t>();

            int32_t num_indices_m_1 = indices.numel() - 1;

            for (int32_t t = 0; t < T; ++t) {
                const int32_t D_start = D_offsets_acc[t];
                const int32_t D = D_offsets_acc[t+1] - D_offsets_acc[t];
                const auto placement = weights_placements_ptr[t];
                TORCH_CHECK(placement != DEVICE);
                if (placement == HOST) {
                    weights_acc = dev_weights.data_ptr<uint8_t>();
                } else {
                    weights_acc = uvm_weights.data_ptr<uint8_t>();
                }
                const uint8_t* weights = &weights_acc[weights_offsets_acc[t]];
                auto weight_ty = static_cast<SparseType>(weights_tys_acc[t]);
                const int32_t D_vecs = div_round_up(D, 8);
                const int32_t D_tail_elements = D % 8;
                const int32_t D_bytes = padded_row_size_in_bytes(D, weight_ty);

                if (weight_ty == SparseType::FP32) {
                    for (int32_t b = 0; b < B; ++b) {
                        int32_t indices_start = offsets_acc[t * B + b];
                        int32_t indices_end = offsets_acc[t * B + b + 1];
                        int32_t L = indices_end - indices_start;
                        acc.resize(D_vecs);
                        for (auto i = 0; i < D_vecs; ++i) {
                            acc[i] = _mm256_setzero_ps();
                        }
                        for (int32_t l = 0; l < L; ++l) {
                            int64_t idx = indices_acc[indices_start + l];
                            const __m256* row = idx == -1 ? reinterpret_cast<const __m256*>(zero_row.data()) : reinterpret_cast<const __m256*>(&weights[idx * D_bytes]);

                            int64_t prefetch_idx = indices_acc[std::min<int32_t>(indices_start + l + 1, num_indices_m_1)];
                            _mm_prefetch(&weights[prefetch_idx * D_bytes], _MM_HINT_T0);

                            {% if weighted %}
                            auto scale = _mm256_set1_ps(indice_weights_acc[indices_start + l]);
                            {% endif %}

                            for (auto i = 0; i < D_vecs; ++i) {
                                // Note that we don't guarantee 32-byte alignment for row starts (just 16-byte), and therefore use unaligned loads.
                                {% if weighted %}
                                acc[i] = _mm256_fmadd_ps(scale, _mm256_loadu_ps(reinterpret_cast<const float*>(&row[i])), acc[i]);
                                {% else %}
                                acc[i] = _mm256_add_ps(_mm256_loadu_ps(reinterpret_cast<const float*>(&row[i])), acc[i]);
                                {% endif %}
                            }
                        }

                        const bool acc_scaling = (pooling_mode == MEAN && L > 0);
                        const float acc_scale_factor = acc_scaling ? 1.0 / L : 1.0;
                        __m256 scale_vec = _mm256_set1_ps(acc_scale_factor);
                        store_result<output_t>(D_vecs, D_tail_elements, acc, scale_vec, output_acc + b * total_D + D_start, acc_scaling);
                    }
                } else if (weight_ty == SparseType::FP16) {
                    for (int32_t b = 0; b < B; ++b) {
                        int32_t indices_start = offsets_acc[t * B + b];
                        int32_t indices_end = offsets_acc[t * B + b + 1];
                        int32_t L = indices_end - indices_start;
                        acc.resize(D_vecs);
                        for (auto i = 0; i < D_vecs; ++i) {
                            acc[i] = _mm256_setzero_ps();
                        }
                        int32_t l = 0;
                        int32_t LUnroll = (L / 2) * 2;
                        for (; l < LUnroll; l += 2) {
                            int64_t idx0 = indices_acc[indices_start + l + 0];
                            int64_t idx1 = indices_acc[indices_start + l + 1];

                            const __m128i* row0 = idx0 == -1 ? reinterpret_cast<const __m128i*>(zero_row.data()) : reinterpret_cast<const __m128i*>(&weights[idx0 * D_bytes]);
                            const __m128i* row1 = idx1 == -1 ? reinterpret_cast<const __m128i*>(zero_row.data()) : reinterpret_cast<const __m128i*>(&weights[idx1 * D_bytes]);

                            int64_t prefetch_idx0 = indices_acc[std::min<int32_t>(indices_start + l + 2, num_indices_m_1)];
                            int64_t prefetch_idx1 = indices_acc[std::min<int32_t>(indices_start + l + 3, num_indices_m_1)];
                            _mm_prefetch(&weights[prefetch_idx0 * D_bytes], _MM_HINT_T0);
                            _mm_prefetch(&weights[prefetch_idx1 * D_bytes], _MM_HINT_T0);

                            {% if weighted %}
                            auto scale0 = _mm256_set1_ps(indice_weights_acc[indices_start + l + 0]);
                            auto scale1 = _mm256_set1_ps(indice_weights_acc[indices_start + l + 1]);
                            {% endif %}
                            for (auto i = 0; i < D_vecs; ++i) {
                                {% if weighted %}
                                acc[i] = _mm256_fmadd_ps(scale0, _mm256_cvtph_ps(row0[i]), acc[i]);
                                acc[i] = _mm256_fmadd_ps(scale1, _mm256_cvtph_ps(row1[i]), acc[i]);
                                {% else %}
                                acc[i] = _mm256_add_ps(_mm256_cvtph_ps(row0[i]), acc[i]);
                                acc[i] = _mm256_add_ps(_mm256_cvtph_ps(row1[i]), acc[i]);
                                {% endif %}
                            }
                        }
                        for (; l < L; ++l) {
                            int64_t idx = indices_acc[indices_start + l];
                            const __m128i* row = idx == -1 ? reinterpret_cast<const __m128i*>(zero_row.data()) : reinterpret_cast<const __m128i*>(&weights[idx * D_bytes]);

                            int64_t prefetch_idx = indices_acc[std::min<int32_t>(indices_start + l + 1, num_indices_m_1)];
                            _mm_prefetch(&weights[prefetch_idx * D_bytes], _MM_HINT_T0);

                            {% if weighted %}
                            auto scale = _mm256_set1_ps(indice_weights_acc[indices_start + l]);
                            {% endif %}

                            for (auto i = 0; i < D_vecs; ++i) {
                                {% if weighted %}
                                acc[i] = _mm256_fmadd_ps(scale, _mm256_cvtph_ps(row[i]), acc[i]);
                                {% else %}
                                acc[i] = _mm256_add_ps(_mm256_cvtph_ps(row[i]), acc[i]);
                                {% endif %}
                            }
                        }

                        const bool acc_scaling = (pooling_mode == MEAN && L > 0);
                        const float acc_scale_factor = acc_scaling ? 1.0 / L : 1.0;
                        __m256 scale_vec = _mm256_set1_ps(acc_scale_factor);
                        store_result<output_t>(D_vecs, D_tail_elements, acc, scale_vec, output_acc + b * total_D + D_start, acc_scaling);
                    }
                } else if (weight_ty == SparseType::INT8) {
                    for (int32_t b = 0; b < B; ++b) {
                        int32_t indices_start = offsets_acc[t * B + b];
                        int32_t indices_end = offsets_acc[t * B + b + 1];
                        int32_t L = indices_end - indices_start;
                        acc.resize(D_vecs);
                        for (auto i = 0; i < D_vecs; ++i) {
                            acc[i] = _mm256_setzero_ps();
                        }

                        int32_t l = 0;
                        int32_t LUnroll = (L / 2) * 2;
                        for (; l < LUnroll; l += 2) {
                            int64_t idx0 = indices_acc[indices_start + l + 0];
                            int64_t idx1 = indices_acc[indices_start + l + 1];

                            const uint32_t* row0 = idx0 == -1 ? reinterpret_cast<const uint32_t*>(zero_row.data()) : reinterpret_cast<const uint32_t*>(&weights[idx0 * D_bytes]);
                            const uint32_t* row1 = idx1 == -1 ? reinterpret_cast<const uint32_t*>(zero_row.data()) : reinterpret_cast<const uint32_t*>(&weights[idx1 * D_bytes]);

                            // note: unaligned accesses
                            const uint64_t* vrow0 = reinterpret_cast<const uint64_t*>(row0 + 1);
                            const uint64_t* vrow1 = reinterpret_cast<const uint64_t*>(row1 + 1);

                            uint32_t scale_shift0 = row0[0];
                            uint32_t scale_shift1 = row1[0];

                            int64_t prefetch_idx0 = indices_acc[std::min<int32_t>(indices_start + l + 2, num_indices_m_1)];
                            int64_t prefetch_idx1 = indices_acc[std::min<int32_t>(indices_start + l + 3, num_indices_m_1)];
                            _mm_prefetch(&weights[prefetch_idx0 * D_bytes], _MM_HINT_T0);
                            _mm_prefetch(&weights[prefetch_idx1 * D_bytes], _MM_HINT_T0);

                            auto scale0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<uint16_t>(scale_shift0 & 0xFFFF)));
                            auto scale1 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<uint16_t>(scale_shift1 & 0xFFFF)));
                            auto shift0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<uint16_t>((scale_shift0 >> 16) & 0xFFFF)));
                            auto shift1 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<uint16_t>((scale_shift1 >> 16) & 0xFFFF)));

                            {% if weighted %}
                            auto idx_weight0 = _mm256_set1_ps(indice_weights_acc[indices_start + l + 0]);
                            auto idx_weight1 = _mm256_set1_ps(indice_weights_acc[indices_start + l + 1]);
                            scale0 = _mm256_mul_ps(scale0, idx_weight0);
                            scale1 = _mm256_mul_ps(scale1, idx_weight1);

                            shift0 = _mm256_mul_ps(shift0, idx_weight0);
                            shift1 = _mm256_mul_ps(shift1, idx_weight1);
                            {% endif %}

                            for (auto i = 0; i < D_vecs; ++i) {
                                acc[i] = _mm256_fmadd_ps(scale0, _mm256_cvtepi32_ps(cvt_byte_SKL(vrow0[i])), _mm256_add_ps(acc[i], shift0));
                                acc[i] = _mm256_fmadd_ps(scale1, _mm256_cvtepi32_ps(cvt_byte_SKL(vrow1[i])), _mm256_add_ps(acc[i], shift1));
                            }
                        }
                        for (; l < L; ++l) {
                            int64_t idx = indices_acc[indices_start + l];
                            const uint32_t* row = idx == -1 ? reinterpret_cast<const uint32_t*>(zero_row.data()) : reinterpret_cast<const uint32_t*>(&weights[idx * D_bytes]);
                            const uint64_t* vrow = reinterpret_cast<const uint64_t*>(row + 1);
                            uint32_t scale_shift = row[0];

                            int64_t prefetch_idx = indices_acc[std::min<int32_t>(indices_start + l + 1, num_indices_m_1)];
                            _mm_prefetch(&weights[prefetch_idx * D_bytes], _MM_HINT_T0);

                            auto scale = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<uint16_t>(scale_shift & 0xFFFF)));
                            auto shift = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<uint16_t>((scale_shift >> 16) & 0xFFFF)));

                            {% if weighted %}
                            auto idx_weight = _mm256_set1_ps(indice_weights_acc[indices_start + l]);
                            scale = _mm256_mul_ps(scale, idx_weight);
                            shift = _mm256_mul_ps(shift, idx_weight);
                            {% endif %}

                            for (auto i = 0; i < D_vecs; ++i) {
                                acc[i] = _mm256_fmadd_ps(scale, _mm256_cvtepi32_ps(cvt_byte_SKL(vrow[i])), _mm256_add_ps(acc[i], shift));
                            }
                        }

                        const bool acc_scaling = (pooling_mode == MEAN && L > 0);
                        const float acc_scale_factor = acc_scaling ? 1.0 / L : 1.0;
                        __m256 scale_vec = _mm256_set1_ps(acc_scale_factor);
                        store_result<output_t>(D_vecs, D_tail_elements, acc, scale_vec, output_acc + b * total_D + D_start, acc_scaling);
                    }
                } else if (weight_ty == SparseType::INT4) {
                    for (int32_t b = 0; b < B; ++b) {
                        int32_t indices_start = offsets_acc[t * B + b];
                        int32_t indices_end = offsets_acc[t * B + b + 1];
                        int32_t L = indices_end - indices_start;
                        acc.resize(D_vecs);
                        for (auto i = 0; i < D_vecs; ++i) {
                            acc[i] = _mm256_setzero_ps();
                        }

                        int32_t l = 0;
                        int32_t LUnroll = (L / 2) * 2;
                        for (; l < LUnroll; l += 2) {
                            int64_t idx0 = indices_acc[indices_start + l + 0];
                            int64_t idx1 = indices_acc[indices_start + l + 1];

                            const uint32_t* row0 = idx0 == -1 ? reinterpret_cast<const uint32_t*>(zero_row.data()) : reinterpret_cast<const uint32_t*>(&weights[idx0 * D_bytes]);
                            const uint32_t* row1 = idx1 == -1 ? reinterpret_cast<const uint32_t*>(zero_row.data()) : reinterpret_cast<const uint32_t*>(&weights[idx1 * D_bytes]);
                            const uint32_t* vrow0 = reinterpret_cast<const uint32_t*>(row0 + 1);
                            const uint32_t* vrow1 = reinterpret_cast<const uint32_t*>(row1 + 1);

                            uint32_t scale_shift0 = row0[0];
                            uint32_t scale_shift1 = row1[0];

                            int64_t prefetch_idx0 = indices_acc[std::min<int32_t>(indices_start + l + 2, num_indices_m_1)];
                            int64_t prefetch_idx1 = indices_acc[std::min<int32_t>(indices_start + l + 3, num_indices_m_1)];
                            _mm_prefetch(&weights[prefetch_idx0 * D_bytes], _MM_HINT_T0);
                            _mm_prefetch(&weights[prefetch_idx1 * D_bytes], _MM_HINT_T0);

                            auto scale0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<uint16_t>(scale_shift0 & 0xFFFF)));
                            auto scale1 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<uint16_t>(scale_shift1 & 0xFFFF)));
                            auto shift0 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<uint16_t>((scale_shift0 >> 16) & 0xFFFF)));
                            auto shift1 = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<uint16_t>((scale_shift1 >> 16) & 0xFFFF)));

                            {% if weighted %}
                            auto idx_weight0 = _mm256_set1_ps(indice_weights_acc[indices_start + l + 0]);
                            auto idx_weight1 = _mm256_set1_ps(indice_weights_acc[indices_start + l + 1]);
                            scale0 = _mm256_mul_ps(scale0, idx_weight0);
                            scale1 = _mm256_mul_ps(scale1, idx_weight1);

                            shift0 = _mm256_mul_ps(shift0, idx_weight0);
                            shift1 = _mm256_mul_ps(shift1, idx_weight1);
                            {% endif %}

                            for (auto i = 0; i < D_vecs; ++i) {
                                acc[i] = _mm256_fmadd_ps(scale0, _mm256_cvtepi32_ps(cvt_nib_epi32_SKL(vrow0[i])), _mm256_add_ps(acc[i], shift0));
                                acc[i] = _mm256_fmadd_ps(scale1, _mm256_cvtepi32_ps(cvt_nib_epi32_SKL(vrow1[i])), _mm256_add_ps(acc[i], shift1));
                            }
                        }
                        for (; l < L; ++l) {
                            int64_t idx = indices_acc[indices_start + l];
                            const uint32_t* row = idx == -1 ? reinterpret_cast<const uint32_t*>(zero_row.data()) : reinterpret_cast<const uint32_t*>(&weights[idx * D_bytes]);
                            const uint32_t* vrow = reinterpret_cast<const uint32_t*>(row + 1);
                            uint32_t scale_shift = row[0];

                            int64_t prefetch_idx = indices_acc[std::min<int32_t>(indices_start + l + 1, num_indices_m_1)];
                            _mm_prefetch(&weights[prefetch_idx * D_bytes], _MM_HINT_T0);

                            auto scale = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<uint16_t>(scale_shift & 0xFFFF)));
                            auto shift = _mm256_cvtph_ps(_mm_set1_epi16(static_cast<uint16_t>((scale_shift >> 16) & 0xFFFF)));

                            {% if weighted %}
                            auto idx_weight = _mm256_set1_ps(indice_weights_acc[indices_start + l]);
                            scale = _mm256_mul_ps(scale, idx_weight);
                            shift = _mm256_mul_ps(shift, idx_weight);
                            {% endif %}

                            for (auto i = 0; i < D_vecs; ++i) {
                                acc[i] = _mm256_fmadd_ps(scale, _mm256_cvtepi32_ps(cvt_nib_epi32_SKL(vrow[i])), _mm256_add_ps(acc[i], shift));
                            }
                        }

                        const bool acc_scaling = (pooling_mode == MEAN && L > 0);
                        const float acc_scale_factor = acc_scaling ? 1.0 / L : 1.0;
                        __m256 scale_vec = _mm256_set1_ps(acc_scale_factor);
                        store_result<output_t>(D_vecs, D_tail_elements, acc, scale_vec, output_acc + b * total_D + D_start, acc_scaling);
                    }
                } else {
                    throw std::logic_error("Unsupported SparseType: " + std::to_string(static_cast<int>(weight_ty)));
                }
            }
            return;
        });
    }));
    return output;
}

Tensor pruned_hashmap_lookup_{{ wdesc }}_cpu(
    Tensor indices,
    Tensor offsets,
    Tensor hash_table,
    Tensor hash_table_offsets) {
    int32_t T = hash_table_offsets.size(0) - 1;
    int32_t B = (offsets.size(0) - 1) / T;
    TORCH_CHECK(B > 0);
    auto dense_indices = empty_like(indices);
    const auto* indices_acc = indices.data_ptr<int32_t>();
    auto* dense_indices_acc = dense_indices.data_ptr<int32_t>();

    const auto* offsets_acc = offsets.data_ptr<int32_t>();
    const auto hash_table_acc = hash_table.accessor<int32_t, 2>();
    const auto hash_table_offsets_acc = hash_table_offsets.accessor<int64_t, 1>();

    for (int32_t t = 0; t < T; ++t) {
        int64_t table_start = hash_table_offsets_acc[t];
        int64_t table_end = hash_table_offsets_acc[t + 1];
        int64_t capacity = table_end - table_start;

        for (int32_t b = 0; b < B; ++b) {
            int32_t indices_start = offsets_acc[t * B + b];
            int32_t indices_end = offsets_acc[t * B + b + 1];
            int32_t L = indices_end - indices_start;

            if (table_start == table_end) {
                for (int32_t l = 0; l < L; ++l) {
                    dense_indices_acc[indices_start + l] = indices_acc[indices_start + l];
                }
            } else {
                for (int32_t l = 0; l < L; ++l) {
                    int32_t idx = indices_acc[indices_start + l];
                    uint32_t slot = pruned_hash_function(static_cast<uint32_t>(idx)) % capacity;
                    while (true) {
                        int32_t slot_sparse_idx = hash_table_acc[table_start + static_cast<int64_t>(slot)][0];

                        // empty slot
                        if (slot_sparse_idx == -1) {
                            dense_indices_acc[indices_start + l] = -1;
                            break;
                        }
                        // already exists
                        if (slot_sparse_idx == idx) {
                            dense_indices_acc[indices_start + l] = hash_table_acc[table_start + static_cast<int64_t>(slot)][1];
                            break;
                        }
                        // linear probe
                        slot = (slot + 1) % capacity;
                    }
                }
            }
        }
    }
    return dense_indices;
}

{% if not weighted %}
Tensor pruned_array_lookup_cpu(
    Tensor indices,
    Tensor offsets,
    Tensor index_remappings,
    Tensor index_remappings_offsets) {
    int32_t T = index_remappings_offsets.size(0) - 1;
    int32_t B = (offsets.size(0) - 1) / T;
    TORCH_CHECK(B > 0);
    auto dense_indices = empty_like(indices);
    const auto* indices_acc = indices.data_ptr<int32_t>();
    auto* dense_indices_acc = dense_indices.data_ptr<int32_t>();
    const auto* offsets_acc = offsets.data_ptr<int32_t>();

    const auto index_remappings_acc = index_remappings.data_ptr<int32_t>();
    const auto index_remappings_offsets_acc = index_remappings_offsets.data_ptr<int64_t>();

    for (int32_t t = 0; t < T; ++t) {
        int64_t index_remappings_start = index_remappings_offsets_acc[t];
        int64_t index_remappings_end = index_remappings_offsets_acc[t + 1];
        int64_t capacity = index_remappings_end - index_remappings_start;
        int32_t indices_start = offsets_acc[t * B];
        int32_t indices_end = offsets_acc[(t + 1) * B];
        for (int32_t i = indices_start; i < indices_end; ++i) {
            int32_t idx = indices_acc[i];
            dense_indices[i] = capacity ? index_remappings_acc[index_remappings_start + idx] : idx;
        }
    }
    return dense_indices;
}
{% endif %}
