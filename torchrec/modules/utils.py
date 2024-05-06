#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import threading
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch.profiler import record_function
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor


lib = torch.library.Library("custom", "FRAGMENT")


try:
    if torch.jit.is_scripting():
        raise Exception()

    from torch.compiler import (
        is_compiling as is_compiler_compiling,
        is_dynamo_compiling as is_torchdynamo_compiling,
    )

    def is_non_strict_exporting() -> bool:
        return not is_torchdynamo_compiling() and is_compiler_compiling()

except Exception:

    def is_non_strict_exporting() -> bool:
        return False


class OpRegistryState:
    """
    State of operator registry.

    We can only register the op schema once. So if we're registering multiple
    times we need a lock and check if they're the same schema
    """

    op_registry_lock = threading.Lock()

    # operator schema: {class}.{id} => op_name
    op_registry_schema: Dict[str, str] = {}
    # operator counter: {class} => count
    op_registry_counter: Dict[str, int] = defaultdict(int)


operator_registry_state = OpRegistryState()


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


#  register a customized operator that takes a list of tensors as input and returns
#  a list of tensors as output. The operator is registered with the name of
#  {module_class_name}_{instance_count}
def register_custom_op(
    module: torch.nn.Module, dims: List[int]
) -> Callable[[List[Optional[torch.Tensor]], int], List[torch.Tensor]]:
    """
    Register a customized operator.

    Args:
        module: customized module instance
        dims: output dimensions
    """

    global operator_registry_state

    m_name: str = type(module).__name__
    op_id: str = f"{m_name}_{id(module)}"
    with operator_registry_state.op_registry_lock:
        if op_id in operator_registry_state.op_registry_schema:
            op_name: str = operator_registry_state.op_registry_schema[op_id]
        else:
            operator_registry_state.op_registry_counter[m_name] += 1
            op_name: str = (
                f"{m_name}_{operator_registry_state.op_registry_counter[m_name]}"
            )
            operator_registry_state.op_registry_schema[op_id] = op_name

            def custom_op(
                values: List[Optional[torch.Tensor]],
                batch_size: int,
            ) -> List[torch.Tensor]:
                device = None
                for v in values:
                    if v is not None:
                        device = v.device
                        break
                else:
                    raise AssertionError(
                        f"Custom op {op_name} expects at least one input tensor"
                    )

                return [
                    torch.empty(
                        batch_size,
                        dim,
                        device=device,
                    )
                    for dim in dims
                ]

            schema_string = f"{op_name}(Tensor?[] values, int batch_size) -> Tensor[]"
            operator_registry_state.op_registry_schema[op_name] = schema_string
            # Register schema
            lib.define(schema_string)

            # Register implementation
            lib.impl(op_name, custom_op, "CPU")
            lib.impl(op_name, custom_op, "CUDA")

            # Register meta formula
            lib.impl(op_name, custom_op, "Meta")

        return getattr(torch.ops.custom, op_name)
