#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch.profiler import record_function
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor


@torch.fx.wrap
def _fx_to_list(tensor: torch.Tensor) -> List[int]:
    return tensor.long().tolist()


def extract_module_or_tensor_callable(
    module_or_callable: Union[
        Callable[[], torch.nn.Module],
        torch.nn.Module,
        Callable[[torch.Tensor], torch.Tensor],
    ]
) -> Union[torch.nn.Module, Callable[[torch.Tensor], torch.Tensor]]:
    try:
        # pyre-ignore[20]: PositionalOnly call expects argument in position 0
        module = module_or_callable()
        if isinstance(module, torch.nn.Module):
            return module
        else:
            raise ValueError(
                "Expected callable that takes no input to return "
                "a torch.nn.Module, but got: {}".format(type(module))
            )
    except TypeError as e:
        if "required positional argument" in str(e):
            # pyre-ignore[7]: Expected `Union[typing.Callable[[torch.Tensor], torch.Tensor], torch.nn.Module]`
            return module_or_callable
        raise


def get_module_output_dimension(
    module: Union[Callable[[torch.Tensor], torch.Tensor], torch.nn.Module],
    in_features: int,
) -> int:
    input = torch.zeros(1, in_features)
    output = module(input)
    return output.size(-1)


def check_module_output_dimension(
    module: Union[Iterable[torch.nn.Module], torch.nn.Module],
    in_features: int,
    out_features: int,
) -> bool:
    """
    Verify that the out_features of a given module or a list of modules matches the
    specified number. If a list of modules or a ModuleList is given, recursively check
    all the submodules.
    """
    if isinstance(module, list) or isinstance(module, torch.nn.ModuleList):
        return all(
            check_module_output_dimension(submodule, in_features, out_features)
            for submodule in module
        )
    else:
        # pyre-fixme[6]: Expected `Union[typing.Callable[[torch.Tensor],
        #  torch.Tensor], torch.nn.Module]` for 1st param but got
        #  `Union[Iterable[torch.nn.Module], torch.nn.Module]`.
        return get_module_output_dimension(module, in_features) == out_features


def init_mlp_weights_xavier_uniform(m: torch.nn.Module) -> None:
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)


def construct_modulelist_from_single_module(
    module: torch.nn.Module, sizes: Tuple[int, ...]
) -> torch.nn.Module:
    """
    Given a single module, construct a (nested) ModuleList of size of sizes by making
    copies of the provided module and reinitializing the Linear layers.
    """
    if len(sizes) == 1:
        return torch.nn.ModuleList(
            [
                copy.deepcopy(module).apply(init_mlp_weights_xavier_uniform)
                for _ in range(sizes[0])
            ]
        )
    else:
        # recursively create nested ModuleList
        return torch.nn.ModuleList(
            [
                construct_modulelist_from_single_module(module, sizes[1:])
                for _ in range(sizes[0])
            ]
        )


def convert_list_of_modules_to_modulelist(
    modules: Iterable[torch.nn.Module], sizes: Tuple[int, ...]
) -> torch.nn.Module:
    assert (
        # pyre-fixme[6]: Expected `Sized` for 1st param but got
        #  `Iterable[torch.nn.Module]`.
        len(modules)
        == sizes[0]
    ), f"the counts of modules ({len(modules)}) do not match with the required counts {sizes}"
    if len(sizes) == 1:
        return torch.nn.ModuleList(modules)
    else:
        # recursively create nested list
        return torch.nn.ModuleList(
            convert_list_of_modules_to_modulelist(m, sizes[1:]) for m in modules
        )


def construct_jagged_tensors(
    embeddings: torch.Tensor,
    features: KeyedJaggedTensor,
    embedding_names: List[str],
    need_indices: bool = False,
    features_to_permute_indices: Optional[Dict[str, List[int]]] = None,
    original_features: Optional[KeyedJaggedTensor] = None,
    reverse_indices: Optional[torch.Tensor] = None,
) -> Dict[str, JaggedTensor]:
    with record_function("## construct_jagged_tensors ##"):
        if original_features is not None:
            features = original_features
        if reverse_indices is not None:
            embeddings = torch.index_select(
                embeddings, 0, reverse_indices.to(torch.int32)
            )

        ret: Dict[str, JaggedTensor] = {}
        stride = features.stride()
        length_per_key = features.length_per_key()
        values = features.values()

        lengths = features.lengths().view(-1, stride)
        lengths_tuple = torch.unbind(lengths.view(-1, stride), dim=0)
        embeddings_list = torch.split(embeddings, length_per_key, dim=0)
        values_list = torch.split(values, length_per_key) if need_indices else None

        key_indices = defaultdict(list)
        for i, key in enumerate(embedding_names):
            key_indices[key].append(i)
        for key, indices in key_indices.items():
            # combines outputs in correct order for CW sharding
            indices = (
                _permute_indices(indices, features_to_permute_indices[key])
                if features_to_permute_indices and key in features_to_permute_indices
                else indices
            )
            ret[key] = JaggedTensor(
                lengths=lengths_tuple[indices[0]],
                values=(
                    embeddings_list[indices[0]]
                    if len(indices) == 1
                    else torch.cat([embeddings_list[i] for i in indices], dim=1)
                ),
                # pyre-ignore
                weights=values_list[indices[0]] if need_indices else None,
            )
        return ret


def construct_jagged_tensors_inference(
    embeddings: torch.Tensor,
    lengths: torch.Tensor,
    values: torch.Tensor,
    embedding_names: List[str],
    need_indices: bool = False,
    features_to_permute_indices: Optional[Dict[str, List[int]]] = None,
    reverse_indices: Optional[torch.Tensor] = None,
) -> Dict[str, JaggedTensor]:
    with record_function("## construct_jagged_tensors_inference ##"):
        if reverse_indices is not None:
            embeddings = torch.index_select(
                embeddings, 0, reverse_indices.to(torch.int32)
            )

        ret: Dict[str, JaggedTensor] = {}
        length_per_key: List[int] = _fx_to_list(
            torch.sum(lengths.view(len(embedding_names), -1), dim=1)
        )

        lengths = lengths.view(len(embedding_names), -1)
        lengths_tuple = torch.unbind(lengths, dim=0)
        embeddings_list = torch.split(embeddings, length_per_key, dim=0)
        values_list = torch.split(values, length_per_key) if need_indices else None

        key_indices = defaultdict(list)
        for i, key in enumerate(embedding_names):
            key_indices[key].append(i)
        for key, indices in key_indices.items():
            # combines outputs in correct order for CW sharding
            indices = (
                _permute_indices(indices, features_to_permute_indices[key])
                if features_to_permute_indices and key in features_to_permute_indices
                else indices
            )
            ret[key] = JaggedTensor(
                lengths=lengths_tuple[indices[0]],
                values=(
                    embeddings_list[indices[0]]
                    if len(indices) == 1
                    else torch.cat([embeddings_list[i] for i in indices], dim=1)
                ),
                # pyre-ignore
                weights=values_list[indices[0]] if need_indices else None,
            )
        return ret


def _permute_indices(indices: List[int], permute: List[int]) -> List[int]:
    permuted_indices = [0] * len(indices)
    for i, permuted_index in enumerate(permute):
        permuted_indices[i] = indices[permuted_index]
    return permuted_indices
