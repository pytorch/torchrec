#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import logging
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    EmbeddingLocation,
    IntNBitTableBatchedEmbeddingBagsCodegen,
    PoolingMode,
    rounded_row_size_in_bytes,
)
from torchrec.distributed.batched_embedding_kernel import (
    BaseBatchedEmbedding,
    BaseBatchedEmbeddingBag,
    BatchedDenseEmbedding,
    BatchedDenseEmbeddingBag,
)
from torchrec.distributed.embedding_kernel import BaseEmbedding
from torchrec.distributed.embedding_types import (
    compute_kernel_to_embedding_location,
    GroupedEmbeddingConfig,
)
from torchrec.distributed.fused_params import (
    fused_param_bounds_check_mode,
    is_fused_param_quant_state_dict_split_scale_bias,
    is_fused_param_register_tbe,
    is_fused_param_weighted,
    tbe_fused_params,
    TBEToRegisterMixIn,
)
from torchrec.distributed.types import BoundsCheckMode
from torchrec.distributed.utils import append_prefix
from torchrec.modules.embedding_configs import (
    DATA_TYPE_NUM_BITS,
    data_type_to_sparse_type,
    DataType,
    dtype_to_data_type,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


logger: logging.Logger = logging.getLogger(__name__)


def _copy_config(
    original: GroupedEmbeddingConfig,
    data_type: DataType,
    sparse_type: SparseType,
    device: torch.device,
) -> GroupedEmbeddingConfig:
    # Adjust config to quantized version.
    # This obviously doesn't work for column-wise sharding.
    config = copy.deepcopy(original)
    config.data_type = data_type
    for table in config.embedding_tables:
        row_alignment = 16 if device.type == "cuda" else 1
        table.local_cols = rounded_row_size_in_bytes(
            table.local_cols, sparse_type, row_alignment
        )
        if table.local_metadata is not None:
            table.local_metadata.shard_sizes = [
                table.local_rows,
                table.local_cols,
            ]

        global_metadata = table.global_metadata
        if global_metadata is not None:
            for shard_meta in global_metadata.shards_metadata:
                if shard_meta != table.local_metadata:
                    shard_meta.shard_sizes = [
                        shard_meta.shard_sizes[0],
                        rounded_row_size_in_bytes(
                            shard_meta.shard_sizes[1], sparse_type, row_alignment
                        ),
                    ]
            global_metadata.size = torch.Size(
                [
                    global_metadata.size[0],
                    sum(
                        shard_meta.shard_sizes[1]
                        for shard_meta in global_metadata.shards_metadata
                    ),
                ]
            )
    return config


def _quantize_weight(
    state_dict: Dict[str, torch.Tensor],
    data_type: DataType,
) -> List[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
    quant_weight_list = []
    for weight in state_dict.values():
        if weight.dtype == torch.float or weight.dtype == torch.float16:
            quantized_weights = (
                torch.ops.fbgemm.FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf(
                    weight, DATA_TYPE_NUM_BITS[data_type]
                )
            )
        else:
            raise Exception("Unsupported dtype: {weight.dtype}")

        # weight and 4 byte scale shift (2xfp16)
        quant_weight = quantized_weights[:, :-4]
        scale_shift = quantized_weights[:, -4:]

        quant_weight_list.append((quant_weight, scale_shift))
    return quant_weight_list


def _get_runtime_device(
    device: Optional[torch.device], config: GroupedEmbeddingConfig
) -> torch.device:
    if device is not None and device.type != "meta":
        return device
    else:
        return (
            torch.device("cpu")
            if all(
                (
                    table.local_metadata is not None
                    and table.local_metadata.placement is not None
                    and table.local_metadata.placement.device().type == "cpu"
                )
                or (
                    table.global_metadata is not None
                    and len(table.global_metadata.shards_metadata)
                    and table.global_metadata.shards_metadata[0].placement is not None
                    # pyre-ignore: Undefined attribute [16]
                    and table.global_metadata.shards_metadata[0].placement.device().type
                    == "cpu"
                )
                for table in config.embedding_tables
            )
            else torch.device("cuda")
        )


@torch.fx.wrap
def _unwrap_kjt(
    features: KeyedJaggedTensor,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    # Here it should always follow cuda path, runtime device cannot be meta
    return (
        features.values().int(),
        features.offsets().int(),
        features.weights_or_none(),
    )


@torch.fx.wrap
def _unwrap_kjt_for_cpu(
    features: KeyedJaggedTensor,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    assert features.device().type == "cpu" or features.device().type == "meta"
    return features.values(), features.offsets(), features.weights_or_none()


class QuantBatchedEmbeddingBag(
    BaseBatchedEmbeddingBag[
        Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
    ],
    TBEToRegisterMixIn,
):
    def __init__(
        self,
        config: GroupedEmbeddingConfig,
        pg: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
        fused_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(config, pg, device)

        managed: List[EmbeddingLocation] = []
        for table in config.embedding_tables:
            if device is not None and device.type == "cuda":
                managed.append(
                    compute_kernel_to_embedding_location(table.compute_kernel)
                )
            else:
                managed.append(EmbeddingLocation.HOST)
        self._config: GroupedEmbeddingConfig = config
        self._emb_module_registered: bool = is_fused_param_register_tbe(fused_params)
        self._is_weighted: Optional[bool] = is_fused_param_weighted(fused_params)
        self._quant_state_dict_split_scale_bias: bool = (
            is_fused_param_quant_state_dict_split_scale_bias(fused_params)
        )
        bounds_check_mode: Optional[BoundsCheckMode] = fused_param_bounds_check_mode(
            fused_params
        )

        index_remapping = [
            table.pruning_indices_remapping for table in config.embedding_tables
        ]
        if all(v is None for v in index_remapping):
            index_remapping = None
        self._runtime_device: torch.device = _get_runtime_device(device, config)
        # 16 for CUDA, 1 for others like CPU and MTIA.
        self._tbe_row_alignment: int = 16 if self._runtime_device.type == "cuda" else 1
        self._emb_module: IntNBitTableBatchedEmbeddingBagsCodegen = (
            IntNBitTableBatchedEmbeddingBagsCodegen(
                embedding_specs=[
                    (
                        table.name,
                        local_rows,
                        (
                            local_cols
                            if self._quant_state_dict_split_scale_bias
                            else table.embedding_dim
                        ),
                        data_type_to_sparse_type(config.data_type),
                        location,
                    )
                    for local_rows, local_cols, table, location in zip(
                        self._local_rows,
                        self._local_cols,
                        config.embedding_tables,
                        managed,
                    )
                ],
                device=device,
                # pyre-ignore
                index_remapping=index_remapping,
                pooling_mode=self._pooling,
                feature_table_map=self._feature_table_map,
                row_alignment=self._tbe_row_alignment,
                uvm_host_mapped=True,  # Use cudaHostAlloc for UVM CACHING to fix imbalance numa memory issue
                bounds_check_mode=(
                    bounds_check_mode if bounds_check_mode else BoundsCheckMode.WARNING
                ),
                **(tbe_fused_params(fused_params) or {}),
            )
        )
        if device is not None:
            self._emb_module.initialize_weights()

    def init_parameters(self) -> None:
        pass

    @property
    def emb_module(
        self,
    ) -> IntNBitTableBatchedEmbeddingBagsCodegen:
        return self._emb_module

    def get_tbes_to_register(
        self,
    ) -> Dict[IntNBitTableBatchedEmbeddingBagsCodegen, GroupedEmbeddingConfig]:
        return {self._emb_module: self._config}

    def forward(self, features: KeyedJaggedTensor) -> torch.Tensor:
        if self._runtime_device.type == "cpu":
            indices, offsets, per_sample_weights = _unwrap_kjt_for_cpu(features)
        else:
            indices, offsets, per_sample_weights = _unwrap_kjt(features)
        if self._is_weighted:
            assert per_sample_weights is not None
        elif self._is_weighted is not None:
            per_sample_weights = None
        # Conditional call of .forward function for FX:
        # emb_module() can go through FX only if emb_module is registered in named_modules (FX node call_module)
        # emb_module.forward() does not require registering emb_module in named_modules (FX node call_function)
        # For some post processing that requires TBE emb_module copied in fx.GraphModule we need to be call_module, as it will copies this module inside fx.GraphModule unchanged.
        if self._emb_module_registered:
            return self.emb_module(
                indices=indices,
                offsets=offsets,
                per_sample_weights=per_sample_weights,
            )
        else:
            return self.emb_module.forward(
                indices=indices,
                offsets=offsets,
                per_sample_weights=per_sample_weights,
            )

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        assert (
            remove_duplicate
        ), "remove_duplicate=False not supported in QuantBatchedEmbeddingBag.named_split_embedding_weights"
        for config, (weight, weight_qscale, weight_qbias) in zip(
            self._config.embedding_tables,
            self.emb_module.split_embedding_weights_with_scale_bias(
                split_scale_bias_mode=(
                    2 if self._quant_state_dict_split_scale_bias else 0
                )
            ),
        ):
            yield append_prefix(prefix, f"{config.name}.weight"), weight
            if self._quant_state_dict_split_scale_bias:
                yield append_prefix(
                    prefix, f"{config.name}.weight_qscale"
                ), weight_qscale
                yield append_prefix(prefix, f"{config.name}.weight_qbias"), weight_qbias

    def split_embedding_weights(
        self,
    ) -> List[Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]]:
        return [
            (weight, qscale, qbias)
            for weight, qscale, qbias in self.emb_module.split_embedding_weights_with_scale_bias(
                split_scale_bias_mode=(
                    2 if self._quant_state_dict_split_scale_bias else 0
                )
            )
        ]

    @classmethod
    def from_float(cls, module: BaseEmbedding) -> "QuantBatchedEmbeddingBag":
        assert hasattr(
            module, "qconfig"
        ), "BaseEmbedding input float module must have qconfig defined"

        data_type = dtype_to_data_type(module.qconfig.weight().dtype)
        sparse_type = data_type_to_sparse_type(data_type)

        # TODO Can we simplify this with state_dict = module.state_dict()?
        state_dict = (
            dict(module.named_split_embedding_weights())
            if isinstance(module, BatchedDenseEmbeddingBag)
            else dict(module.named_parameters())
        )
        device = next(iter(state_dict.values())).device

        config = _copy_config(module.config, data_type, sparse_type, device)
        ret = QuantBatchedEmbeddingBag(config=config, device=device)

        # pyre-ignore
        quant_weight_list = _quantize_weight(state_dict, data_type)
        ret.emb_module.assign_embedding_weights(quant_weight_list)

        return ret


class QuantBatchedEmbedding(
    BaseBatchedEmbedding[
        Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
    ],
    TBEToRegisterMixIn,
):
    def __init__(
        self,
        config: GroupedEmbeddingConfig,
        pg: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
        fused_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(config, pg, device)

        managed: List[EmbeddingLocation] = []
        for table in config.embedding_tables:
            if device is not None and device.type == "cuda":
                managed.append(
                    compute_kernel_to_embedding_location(table.compute_kernel)
                )
            else:
                managed.append(EmbeddingLocation.HOST)
        self._config: GroupedEmbeddingConfig = config
        self._emb_module_registered: bool = is_fused_param_register_tbe(fused_params)
        self._quant_state_dict_split_scale_bias: bool = (
            is_fused_param_quant_state_dict_split_scale_bias(fused_params)
        )
        self._runtime_device: torch.device = _get_runtime_device(device, config)
        # 16 for CUDA, 1 for others like CPU and MTIA.
        self._tbe_row_alignment: int = 16 if self._runtime_device.type == "cuda" else 1
        self._emb_module: IntNBitTableBatchedEmbeddingBagsCodegen = (
            IntNBitTableBatchedEmbeddingBagsCodegen(
                embedding_specs=[
                    (
                        table.name,
                        local_rows,
                        (
                            local_cols
                            if self._quant_state_dict_split_scale_bias
                            else table.embedding_dim
                        ),
                        data_type_to_sparse_type(config.data_type),
                        location,
                    )
                    for local_rows, local_cols, table, location in zip(
                        self._local_rows,
                        self._local_cols,
                        config.embedding_tables,
                        managed,
                    )
                ],
                device=device,
                pooling_mode=PoolingMode.NONE,
                feature_table_map=self._feature_table_map,
                row_alignment=self._tbe_row_alignment,
                uvm_host_mapped=True,  # Use cudaHostAlloc for UVM CACHING to fix imbalance numa memory issue
                **(tbe_fused_params(fused_params) or {}),
            )
        )
        if device is not None:
            self._emb_module.initialize_weights()

    @property
    def emb_module(
        self,
    ) -> IntNBitTableBatchedEmbeddingBagsCodegen:
        return self._emb_module

    def get_tbes_to_register(
        self,
    ) -> Dict[IntNBitTableBatchedEmbeddingBagsCodegen, GroupedEmbeddingConfig]:
        return {self._emb_module: self._config}

    def split_embedding_weights(
        self,
    ) -> List[Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]]:
        return [
            (weight, qscale, qbias)
            for weight, qscale, qbias in self.emb_module.split_embedding_weights_with_scale_bias(
                split_scale_bias_mode=(
                    2 if self._quant_state_dict_split_scale_bias else 0
                )
            )
        ]

    def forward(self, features: KeyedJaggedTensor) -> torch.Tensor:
        if self._runtime_device.type == "cpu":
            # To distinguish with QEBC for fx tracing on CPU embedding.
            values, offsets, _ = _unwrap_kjt_for_cpu(features)
        else:
            values, offsets, _ = _unwrap_kjt(features)

        if self._emb_module_registered:
            return self.emb_module(
                indices=values,
                offsets=offsets,
            )
        else:
            return self.emb_module.forward(
                indices=values,
                offsets=offsets,
            )

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        for config, (weight, weight_qscale, weight_qbias) in zip(
            self._config.embedding_tables,
            self.emb_module.split_embedding_weights_with_scale_bias(
                split_scale_bias_mode=(
                    2 if self._quant_state_dict_split_scale_bias else 0
                )
            ),
        ):
            yield append_prefix(prefix, f"{config.name}.weight"), weight
            if self._quant_state_dict_split_scale_bias:
                yield append_prefix(
                    prefix, f"{config.name}.weight_qscale"
                ), weight_qscale
                yield append_prefix(prefix, f"{config.name}.weight_qbias"), weight_qbias

    @classmethod
    def from_float(cls, module: BaseEmbedding) -> "QuantBatchedEmbedding":
        assert hasattr(
            module, "qconfig"
        ), "BaseEmbedding input float module must have qconfig defined"

        data_type = dtype_to_data_type(module.qconfig.weight().dtype)
        sparse_type = data_type_to_sparse_type(data_type)

        # TODO Can we simplify this with state_dict = module.state_dict()?
        state_dict = (
            dict(module.named_split_embedding_weights())
            if isinstance(module, BatchedDenseEmbedding)
            else dict(module.named_parameters())
        )
        device = next(iter(state_dict.values())).device

        config = _copy_config(module.config, data_type, sparse_type, device)
        ret = QuantBatchedEmbedding(config=config, device=device)

        # pyre-ignore
        quant_weight_list = _quantize_weight(state_dict, data_type)
        ret.emb_module.assign_embedding_weights(quant_weight_list)

        return ret
