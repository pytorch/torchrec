#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Dict, Iterable, Optional

import torch

from fbgemm_gpu.split_table_batched_embeddings_ops_inference import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
)
from torchrec.distributed.embedding_types import GroupedEmbeddingConfig

FUSED_PARAM_REGISTER_TBE_BOOL: str = "__register_tbes_in_named_modules"
FUSED_PARAM_QUANT_STATE_DICT_SPLIT_SCALE_BIAS: str = (
    "__register_quant_state_dict_split_scale_bias"
)


class TBEToRegisterMixIn:
    def get_tbes_to_register(
        self,
    ) -> Dict[IntNBitTableBatchedEmbeddingBagsCodegen, GroupedEmbeddingConfig]:
        raise NotImplementedError


def get_tbes_to_register_from_iterable(
    iterable: Iterable[torch.nn.Module],
) -> Dict[IntNBitTableBatchedEmbeddingBagsCodegen, GroupedEmbeddingConfig]:
    tbes: Dict[IntNBitTableBatchedEmbeddingBagsCodegen, GroupedEmbeddingConfig] = {}
    for m in iterable:
        if isinstance(m, TBEToRegisterMixIn):
            tbes.update(m.get_tbes_to_register())
    return tbes


def is_fused_param_register_tbe(fused_params: Optional[Dict[str, Any]]) -> bool:
    return (
        fused_params
        and FUSED_PARAM_REGISTER_TBE_BOOL in fused_params
        and fused_params[FUSED_PARAM_REGISTER_TBE_BOOL]
    )


def is_fused_param_quant_state_dict_split_scale_bias(
    fused_params: Optional[Dict[str, Any]]
) -> bool:
    return (
        fused_params
        and FUSED_PARAM_QUANT_STATE_DICT_SPLIT_SCALE_BIAS in fused_params
        and fused_params[FUSED_PARAM_QUANT_STATE_DICT_SPLIT_SCALE_BIAS]
    )


def tbe_fused_params(
    fused_params: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    if not fused_params:
        return None

    fused_params_for_tbe = dict(fused_params)
    if FUSED_PARAM_REGISTER_TBE_BOOL in fused_params_for_tbe:
        fused_params_for_tbe.pop(FUSED_PARAM_REGISTER_TBE_BOOL)
    if FUSED_PARAM_QUANT_STATE_DICT_SPLIT_SCALE_BIAS in fused_params_for_tbe:
        fused_params_for_tbe.pop(FUSED_PARAM_QUANT_STATE_DICT_SPLIT_SCALE_BIAS)

    return fused_params_for_tbe
