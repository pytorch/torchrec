#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

#!/usr/bin/env python3

from typing import Dict, List, Optional, Tuple

import torch
from torchrec.sparse.jagged_tensor import (
    _all_keys_used_once,
    _desugar_keyed_tensors,
    _remap_to_groups,
    KeyedTensor,
)


try:
    torch.ops.load_library(
        "//deeplearning/fbgemm/fbgemm_gpu:permute_pooled_embedding_ops_gpu"
    )
    torch.ops.load_library(
        "//deeplearning/fbgemm/fbgemm_gpu:permute_pooled_embedding_ops_cpu"
    )
except OSError:
    pass


@torch.fx.wrap
def _concat_values(kts: List[KeyedTensor], dim: int) -> torch.Tensor:
    return torch.cat([kt.values() for kt in kts], dim=dim)


@torch.fx.wrap
def _permuted_values(
    kts: List[KeyedTensor], remap: List[Tuple[int, str]], dim: int
) -> torch.Tensor:
    embedding_dicts = [kt.to_dict() for kt in kts]
    values = [embedding_dicts[idx][key] for (idx, key) in remap]
    return torch.cat(values, dim=dim)


@torch.fx.wrap
def _build_dict(
    keys: List[str], values: torch.Tensor, splits: List[int], dim: int
) -> Dict[str, torch.Tensor]:
    return {
        key: tensor for key, tensor in zip(keys, torch.split(values, splits, dim=dim))
    }


class KTRegroupAsDict(torch.nn.Module):
    """
    KTRegroupAsDict is a nn.Module that mirrors beahvior of static method KeyedTensor.regroup_as_dict()

    The advantage of using this module it caches the regrouping logic after first batch.

    Args:
        groups (List[List[str]]): features per output group
        keys (List[str]): key of each output group

    Example::

        keys = ['object', 'user']
        groups = [['f1', 'f2'], ['f3']]
        regroup_module = KTRegroupAsDict(groups, keys)


        tensor_list = [torch.randn(2, 4), torch.randn(2, 8), torch.randn(2, 2)]
        kts = [KeyedTensor.from_tensor_list(['f1', 'f2', 'f3' ], tensor_list)]
        out = regroup_module(kts)

    """

    def __init__(self, groups: List[List[str]], keys: List[str]) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torchrec.modules.{self.__class__.__name__}")
        assert len(groups) == len(keys), "Groups and keys should have same length"
        self._groups = groups
        self._keys = keys
        self._is_inited = False

        # cached values populated on first forward call
        self.device: Optional[torch.device] = None
        self._concat_dim: int = 1
        self._use_fbgemm_regroup: bool = False
        self._splits: List[int] = []
        self._idx_key_pairs: List[Tuple[int, str]] = []
        self._permute_tensor: Optional[torch.Tensor] = None
        self._inv_permute_tensor: Optional[torch.Tensor] = None
        self._offsets_tensor: Optional[torch.Tensor] = None
        self._inv_offsets_tensor: Optional[torch.Tensor] = None

    def _init_fbgemm_regroup(self, kts: List[KeyedTensor]) -> None:
        self._use_fbgemm_regroup = True
        keys, lengths, values = _desugar_keyed_tensors(kts)
        permute, inv_permute, offsets, inv_offsets, splits = _remap_to_groups(
            keys, lengths, self._groups
        )
        # no need to pin_memory() or to(..., non_blocking=True) since occurs only once
        self._permute_tensor = permute.to(self.device)
        self._inv_permute_tensor = inv_permute.to(self.device)
        self._offsets_tensor = offsets.to(self.device)
        self._inv_offsets_tensor = inv_offsets.to(self.device)
        self._splits = splits

    def _init_regroup(self, kts: List[KeyedTensor]) -> None:
        lengths = [kt.length_per_key() for kt in kts]
        indices = [kt._key_indices() for kt in kts]

        key_to_idx: dict[str, int] = {}
        for i, kt in enumerate(kts):
            for key in kt.keys():
                if key in key_to_idx:
                    raise RuntimeError(
                        f"Duplicate key {key} found in KeyedTensors, undefined behavior"
                    )
                key_to_idx[key] = i

        splits: List[int] = []
        idx_key_pairs: List[Tuple[int, str]] = []
        for group in self._groups:
            group_length = 0
            for name in group:
                idx_key_pairs.append((key_to_idx[name], name))
                group_length += lengths[key_to_idx[name]][
                    indices[key_to_idx[name]][name]
                ]
            splits.append(group_length)

        self._splits = splits
        self._idx_key_pairs = idx_key_pairs

    def forward(self, keyed_tensors: List[KeyedTensor]) -> Dict[str, torch.Tensor]:
        if not self._is_inited:
            assert len(keyed_tensors) > 0, "Empty list provided"
            assert all(
                kt.device() == keyed_tensors[0].device() for kt in keyed_tensors
            ), "All inputs should be on the same device."
            self.device = keyed_tensors[0].device()
            assert all(
                kt.key_dim() == keyed_tensors[0].key_dim() for kt in keyed_tensors
            ), "All inputs should have the same key_dim"
            self._dim = keyed_tensors[0].key_dim()

            if _all_keys_used_once(keyed_tensors, self._groups) and self._dim == 1:
                self._init_fbgemm_regroup(keyed_tensors)
            else:
                self._init_regroup(keyed_tensors)
            self._is_inited = True

        if self._use_fbgemm_regroup:
            values = _concat_values(keyed_tensors, self._dim)
            permuted_values = torch.ops.fbgemm.permute_pooled_embs_auto_grad(
                values,
                self._offsets_tensor,
                self._permute_tensor,
                self._inv_offsets_tensor,
                self._inv_permute_tensor,
            )
        else:
            permuted_values = _permuted_values(
                keyed_tensors, self._idx_key_pairs, self._dim
            )

        return _build_dict(self._keys, permuted_values, self._splits, self._dim)
