#!/usr/bin/env python3

# pyre-unsafe

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import unittest
from typing import Optional, Tuple

import fbgemm_gpu.split_table_batched_embeddings_ops as split_table_batched_embeddings_ops
import hypothesis.strategies as st
import numpy as np
import torch
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_embedding_inference_converter import SplitEmbInferenceConverter
from fbgemm_gpu.split_table_batched_embeddings_ops import OptimType
from fbgemm_gpu.test.test_utils import gpu_available
from hypothesis import Verbosity, given, settings
from torch import nn


EMB_WEIGHT_UNIFORM_INIT_BOUND = 0.000316
MAX_EXAMPLES = 40


def div_round_up(a: int, b: int) -> int:
    return int((a + b - 1) // b) * b


def to_device(t: torch.Tensor, use_cpu: bool) -> torch.Tensor:
    return t.cpu() if use_cpu else t.cuda()


def get_table_batched_offsets_from_dense(
    merged_indices: torch.Tensor, use_cpu: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    (T, B, L) = merged_indices.size()
    lengths = np.ones((T, B)) * L
    flat_lengths = lengths.flatten()
    return (
        to_device(merged_indices.contiguous().view(-1), use_cpu),
        to_device(
            torch.tensor(([0] + np.cumsum(flat_lengths).tolist())).long(),
            use_cpu,
        ),
    )


class SparseArch(nn.Module):
    """
    The testing module with split table batched embedding op
    """

    def __init__(
        self,
        emb_dim,
        num_tables,
        num_rows,
        use_cpu,
    ):
        super().__init__()
        pooling_mode = split_table_batched_embeddings_ops.PoolingMode.SUM
        Ds = [emb_dim] * num_tables
        Es = [num_rows] * num_tables

        device = (
            split_table_batched_embeddings_ops.ComputeDevice.CPU
            if use_cpu
            else split_table_batched_embeddings_ops.ComputeDevice.CUDA
        )
        loc = (
            split_table_batched_embeddings_ops.EmbeddingLocation.HOST
            if use_cpu
            else split_table_batched_embeddings_ops.EmbeddingLocation.DEVICE
        )

        self.emb_module = (
            split_table_batched_embeddings_ops.SplitTableBatchedEmbeddingBagsCodegen(
                embedding_specs=[
                    (
                        E,
                        D,
                        loc,
                        device,
                    )
                    for (E, D) in zip(Es, Ds)
                ],
                weights_precision=SparseType.FP32,
                optimizer=OptimType.EXACT_SGD,
                learning_rate=0.05,
                pooling_mode=pooling_mode,
            )
        )

        self.emb_module.init_embedding_weights_uniform(
            -EMB_WEIGHT_UNIFORM_INIT_BOUND, +EMB_WEIGHT_UNIFORM_INIT_BOUND
        )

    def forward(self, indices, offsets):
        return self.emb_module(indices, offsets)


class QuantizedSplitEmbeddingsTest(unittest.TestCase):
    @given(
        T=st.integers(min_value=1, max_value=10),
        D=st.integers(min_value=2, max_value=128),
        B=st.integers(min_value=1, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        L=st.integers(min_value=0, max_value=20),
        pooling_mode=st.sampled_from(
            [
                split_table_batched_embeddings_ops.PoolingMode.SUM,
                split_table_batched_embeddings_ops.PoolingMode.MEAN,
            ]
        ),
        quantize_type=st.sampled_from(
            [
                SparseType.INT8,
                SparseType.INT4,
                # TODO: support SparseType.INT2,
                SparseType.FP16,
            ]
        ),
        use_cpu=st.booleans() if gpu_available else st.just(True),
        pruning_ratio=st.sampled_from([None, 0.0]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_quantize_workflow(
        self,
        T: int,
        D: int,
        B: int,
        log_E: int,
        L: int,
        pooling_mode: split_table_batched_embeddings_ops.PoolingMode,
        quantize_type: SparseType,
        pruning_ratio: Optional[float],
        use_cpu: bool,
    ) -> None:
        E = int(10 ** log_E)
        Es = [E] * T
        D_alignment = 8 if not quantize_type == SparseType.INT2 else 16
        D = div_round_up(D, D_alignment)

        xs = [torch.randint(low=0, high=e, size=(B, L)) for e in Es]
        x = torch.cat([x.view(1, B, L) for x in xs], dim=0)
        # indices: T, B, L; offsets: T * B + 1
        (indices, offsets) = get_table_batched_offsets_from_dense(x, use_cpu=use_cpu)
        sparse_arch = SparseArch(emb_dim=D, num_tables=T, num_rows=E, use_cpu=use_cpu)

        # Fake quantize to make the original weight in FP32 all be exactly
        # representable by INT8 row-wise quantized values
        if quantize_type == quantize_type.INT8:
            for t in range(T):
                sparse_arch.emb_module.split_embedding_weights()[t].data.copy_(
                    torch.ops.fbgemm.Fused8BitRowwiseQuantizedToFloat(
                        torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(
                            sparse_arch.emb_module.split_embedding_weights()[t].data
                        )
                    )
                )
        elif quantize_type == quantize_type.INT4 or quantize_type == quantize_type.INT2:
            for t in range(T):
                sparse_arch.emb_module.split_embedding_weights()[t].data.copy_(
                    torch.ops.fbgemm.FusedNBitRowwiseQuantizedSBHalfToFloat(
                        torch.ops.fbgemm.FloatToFusedNBitRowwiseQuantizedSBHalf(
                            sparse_arch.emb_module.split_embedding_weights()[t].data,
                            bit_rate=quantize_type.bit_rate(),
                        ),
                        bit_rate=quantize_type.bit_rate(),
                    )
                )

        emb_out = sparse_arch(indices, offsets)  # B, T, D

        # Apply the quantization transformations on the model!
        split_emb_infer_converter = SplitEmbInferenceConverter(
            quantize_type=quantize_type,
            pruning_ratio=pruning_ratio,
        )
        split_emb_infer_converter.convert_model(sparse_arch)
        assert (
            type(sparse_arch.emb_module)
            == split_table_batched_embeddings_ops.IntNBitTableBatchedEmbeddingBagsCodegen
        )
        assert sparse_arch.emb_module.use_cpu == use_cpu
        quantized_emb_out = sparse_arch(indices.int(), offsets.int())  # B, T, D

        # Compare FP32 emb module vs. quantize_type (FP16, INT8, INT4, INT2) emb module
        torch.testing.assert_allclose(
            emb_out.float().cpu(),
            quantized_emb_out.float().cpu(),
            atol=1.0e-1,
            rtol=1.0e-1,
        )

    @given(
        use_cpu=st.booleans() if gpu_available else st.just(True),
        use_array_for_index_remapping=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_l2_norm_pruning_workflow(
        self,
        use_cpu: bool,
        use_array_for_index_remapping: bool,
    ) -> None:
        D = 128
        T = 2
        E = 5
        indices = torch.Tensor([3, 0, 2, 2, 3, 4, 2]).int()
        offsets = torch.Tensor([0, 1, 4, 6, 7]).int()
        weights = [
            (torch.Tensor([0.4, 0.1, -0.2, 0.2, 0.3]).float().view(E, 1))
            * (torch.Tensor([1.0] * E * D).view(E, D)),
            (torch.Tensor([-0.8, 0.2, 0.5, -0.1, 0.9]).float().view(E, 1))
            * (torch.Tensor([1.0] * E * D).view(E, D)),
        ]

        # Inputs for 3 test cases. Each row is used in one test case.
        pruning_ratios = [0.9, 0.5, 0.1]
        remapped_indices = [
            torch.Tensor([0, 4]).int(),
            torch.Tensor([3, 0, 2, 2, 4, 2]).int(),
            indices,
        ]
        remapped_offsets = [
            torch.Tensor([0, 0, 1, 2, 2]).int(),
            torch.Tensor([0, 1, 4, 5, 6]).int(),
            offsets,
        ]

        # Start to test.
        logging.info("use cpu = {}".format(use_cpu))
        for pruning_ratio, remapped_index, remapped_offset in zip(
            pruning_ratios, remapped_indices, remapped_offsets
        ):
            logging.info("pruning ratio = {}.".format(pruning_ratio))
            sparse_arch = SparseArch(
                emb_dim=D, num_tables=T, num_rows=E, use_cpu=use_cpu
            )
            for idx in range(T):
                sparse_arch.emb_module.split_embedding_weights()[idx].copy_(
                    weights[idx]
                )
            emb_out = sparse_arch(
                to_device(remapped_index, use_cpu), to_device(remapped_offset, use_cpu)
            )  # B, T, D

            # Apply pruning / quantization transformations on the model!
            split_emb_infer_converter = SplitEmbInferenceConverter(
                quantize_type=SparseType.FP16,
                pruning_ratio=pruning_ratio,
                use_array_for_index_remapping=use_array_for_index_remapping,
            )
            split_emb_infer_converter.convert_model(sparse_arch)
            assert (
                type(sparse_arch.emb_module)
                == split_table_batched_embeddings_ops.IntNBitTableBatchedEmbeddingBagsCodegen
            )
            assert sparse_arch.emb_module.use_cpu == use_cpu
            pruned_emb_out = sparse_arch(
                to_device(indices, use_cpu), to_device(offsets, use_cpu)
            )  # B, T, D

            # Compare FP32 emb module with remapped index vs. FP16 emb module with pruning
            torch.testing.assert_allclose(
                emb_out.float().cpu(),
                pruned_emb_out.float().cpu(),
                atol=1.0e-1,
                rtol=1.0e-1,
            )

    @given(
        T=st.integers(min_value=1, max_value=10),
        D=st.integers(min_value=2, max_value=128),
        log_E=st.integers(min_value=3, max_value=5),
        pruning_ratio=st.floats(min_value=0.0, max_value=1.0, exclude_max=True),
        use_cpu=st.booleans() if gpu_available else st.just(True),
        use_array_for_index_remapping=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=MAX_EXAMPLES, deadline=None)
    def test_pruning_workflow_large_scale(
        self,
        T: int,
        D: int,
        log_E: int,
        pruning_ratio: Optional[float],
        use_cpu: bool,
        use_array_for_index_remapping: bool,
    ) -> None:
        E = int(10 ** log_E)
        D_alignment = 8
        D = div_round_up(D, D_alignment)
        sparse_arch = SparseArch(emb_dim=D, num_tables=T, num_rows=E, use_cpu=use_cpu)

        # Make sure that each row has a unique L2 norm.
        embedding_weights_before = sparse_arch.emb_module.split_embedding_weights()
        for weights in embedding_weights_before:
            for i in range(weights.size()[0]):
                weights[i].uniform_(i * 0.01, (i + 1) * 0.01)

        # Collect #rows before pruning.
        num_rows_before = [weight.size()[0] for weight in embedding_weights_before]

        # Apply pruning / quantization transformations on the model!
        split_emb_infer_converter = SplitEmbInferenceConverter(
            quantize_type=SparseType.FP16,
            pruning_ratio=pruning_ratio,
            use_array_for_index_remapping=use_array_for_index_remapping,
        )
        split_emb_infer_converter.convert_model(sparse_arch)
        embedding_weights_after = sparse_arch.emb_module.split_embedding_weights()
        assert (
            type(sparse_arch.emb_module)
            == split_table_batched_embeddings_ops.IntNBitTableBatchedEmbeddingBagsCodegen
        )
        assert sparse_arch.emb_module.use_cpu == use_cpu

        # Collect #rows after pruning.
        embedding_weights_after = sparse_arch.emb_module.split_embedding_weights()
        num_rows_after = [weight[0].size()[0] for weight in embedding_weights_after]

        # Check #rows after pruning aligns with the specified pruning ratio.
        self.assertEqual(len(num_rows_before), len(num_rows_after))
        for before, after in zip(num_rows_before, num_rows_after):
            self.assertEqual(
                math.ceil(before * (1.0 - pruning_ratio)), # type: ignore
                after,
                msg="original_num_rows = {}, pruning ratio = {}".format(
                    before, pruning_ratio
                ),
            )


if __name__ == "__main__":
    unittest.main()
