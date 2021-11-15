#!/usr/bin/env python3

# pyre-unsafe

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from typing import Optional, Tuple

import fbgemm_gpu.split_table_batched_embeddings_ops as split_table_batched_embeddings_ops
import numpy as np
import torch
from fbgemm_gpu.split_embedding_configs import SparseType
from torch import Tensor, nn

# TODO: move torch.ops.fb.embedding_bag_rowwise_prune to OSS
torch.ops.load_library("//caffe2/torch/fb/sparsenn:sparsenn_operators")

# TODO: add per-feature based converter option (based on embedding_specs during inference)
# TODO: optimize embedding pruning and quantization latency.
class SplitEmbInferenceConverter:
    def __init__(
            self, quantize_type: SparseType, pruning_ratio: Optional[float],
            use_array_for_index_remapping: bool = True,
    ):
        self.quantize_type = quantize_type
        # TODO(yingz): Change the pruning ratio to per-table settings.
        self.pruning_ratio = pruning_ratio
        self.use_array_for_index_remapping = use_array_for_index_remapping

    def convert_model(self, model: torch.nn.Module) -> nn.Module:
        self._process_split_embs(model)
        return model

    def _prune_by_weights_l2_norm(self, new_num_rows, weights) -> Tuple[Tensor, float]:
        assert new_num_rows > 0
        from numpy.linalg import norm

        indicators = []
        for row in weights:
            indicators.append(norm(row.cpu().numpy(), ord=2))
        sorted_indicators = sorted(indicators, reverse=True)
        threshold = None
        for i in range(new_num_rows, len(sorted_indicators)):
            if sorted_indicators[i] < sorted_indicators[new_num_rows - 1]:
                threshold = sorted_indicators[i]
                break
        if threshold is None:
            threshold = sorted_indicators[-1] - 1
        return (torch.tensor(indicators), threshold)

    def _prune_embs(
        self,
        idx: int,
        num_rows: int,
        module: split_table_batched_embeddings_ops.SplitTableBatchedEmbeddingBagsCodegen,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # TODO(yingz): Avoid DtoH / HtoD overhead.
        weights = module.split_embedding_weights()[idx].cpu()
        if self.pruning_ratio is None:
            return (weights, None)
        new_num_rows = int(math.ceil(num_rows * (1.0 - self.pruning_ratio)))  # type: ignore
        if new_num_rows == num_rows:
            return (weights, None)

        (indicators, threshold) = self._prune_by_weights_l2_norm(new_num_rows, weights)

        return torch.ops.fb.embedding_bag_rowwise_prune(
            weights, indicators, threshold, torch.int32
        )

    def _quantize_embs(
        self, weight: Tensor, weight_ty: SparseType
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if weight_ty == SparseType.FP16:
            q_weight = weight.half()
            # FIXME: How to view the PyTorch Tensor as a different type (e.g., uint8)
            # Here it uses numpy and it will introduce DtoH/HtoD overhead.
            res_weight = torch.tensor(
                q_weight.cpu().numpy().view(np.uint8)
            ).contiguous()
            return (res_weight, None)

        elif weight_ty == SparseType.INT8:
            q_weight = torch.ops.fbgemm.FloatToFused8BitRowwiseQuantized(weight)
            res_weight = torch.tensor(q_weight[:, :-8].cpu().numpy().view(np.uint8))
            res_scale_shift = torch.tensor(
                q_weight[:, -8:]
                .contiguous()
                .cpu()
                .numpy()
                .view(np.float32)
                .astype(np.float16)
                .view(np.uint8)
            )  # [-4, -2]: scale; [-2:]: bias
            return (res_weight, res_scale_shift)

        elif weight_ty == SparseType.INT4 or weight_ty == SparseType.INT2:
            q_weight = torch.ops.fbgemm.FloatToFusedNBitRowwiseQuantizedSBHalf(
                weight,
                bit_rate=self.quantize_type.bit_rate(),
            )
            res_weight = torch.tensor(q_weight[:, :-4].cpu().numpy().view(np.uint8))
            res_scale_shift = torch.tensor(
                q_weight[:, -4:].contiguous().cpu().numpy().view(np.uint8)
            )  # [-4, -2]: scale; [-2:]: bias
            return (res_weight, res_scale_shift)

        elif weight_ty == SparseType.FP32:
            return (weight, None)

        else:
            raise RuntimeError("Unsupported SparseType: {}".format(weight_ty))

    def _process_split_embs(self, model: nn.Module) -> None:
        for name, child in model.named_children():
            if isinstance(
                child,
                split_table_batched_embeddings_ops.SplitTableBatchedEmbeddingBagsCodegen,
            ):
                embedding_specs = []
                use_cpu = (
                    child.embedding_specs[0][3]
                    == split_table_batched_embeddings_ops.ComputeDevice.CPU
                )
                for (E, D, _, _) in child.embedding_specs:
                    weights_ty = self.quantize_type
                    if D % weights_ty.align_size() != 0:
                        logging.warn(
                            f"Embedding dim {D} couldn't be divided by align size {weights_ty.align_size()}!"
                        )
                        assert D % 4 == 0
                        weights_ty = (
                            SparseType.FP16
                        )  # fall back to FP16 if dimension couldn't be aligned with the required size
                    embedding_specs.append(("", E, D, weights_ty))

                weight_lists = []
                new_embedding_specs = []
                index_remapping_list = []
                for t, (_, E, D, weight_ty) in enumerate(embedding_specs):
                    # Try to prune embeddings.
                    (pruned_weight, index_remapping) = self._prune_embs(t, E, child)
                    new_embedding_specs.append(
                        (
                            "",
                            pruned_weight.size()[0],
                            D,
                            weight_ty,
                            split_table_batched_embeddings_ops.EmbeddingLocation.HOST
                            if use_cpu
                            else split_table_batched_embeddings_ops.EmbeddingLocation.DEVICE,
                        )
                    )
                    index_remapping_list.append(index_remapping)

                    # Try to quantize embeddings.
                    weight_lists.append(self._quantize_embs(pruned_weight, weight_ty))

                q_child = split_table_batched_embeddings_ops.IntNBitTableBatchedEmbeddingBagsCodegen(
                    embedding_specs=new_embedding_specs,
                    index_remapping=index_remapping_list
                    if self.pruning_ratio is not None
                    else None,
                    pooling_mode=child.pooling_mode,
                    device="cpu" if use_cpu else torch.cuda.current_device(),
                    weight_lists=weight_lists,
                    use_array_for_index_remapping=self.use_array_for_index_remapping,
                )
                setattr(model, name, q_child)
            else:
                self._process_split_embs(child)
