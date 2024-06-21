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
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch.profiler import record_function
from torchrec.sparse.jagged_tensor import (
    _permute_tensor_by_segments,
    JaggedTensor,
    KeyedJaggedTensor,
)
from torchrec.streamable import Multistreamable


lib = torch.library.Library("custom", "FRAGMENT")


class OpRegistryState:
    """
    State of operator registry.

    We can only register the op schema once. So if we're registering multiple
    times we need a lock and check if they're the same schema
    """

    op_registry_lock = threading.Lock()

    # operator schema: {class}.{id} => op_name
    op_registry_schema: Dict[str, str] = {}


operator_registry_state = OpRegistryState()


@dataclass
class SequenceVBEContext(Multistreamable):
    recat: torch.Tensor
    unpadded_lengths: torch.Tensor
    reindexed_lengths: torch.Tensor
    reindexed_length_per_key: List[int]
    reindexed_values: Optional[torch.Tensor] = None

    def record_stream(self, stream: torch.Stream) -> None:
        self.recat.record_stream(stream)
        self.unpadded_lengths.record_stream(stream)
        self.reindexed_lengths.record_stream(stream)
        if self.reindexed_values is not None:
            self.reindexed_values.record_stream(stream)


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


def _vbe_reindex(
    embeddings: torch.Tensor,
    seq_vbe_ctx: SequenceVBEContext,
) -> Tuple[torch.Tensor, torch.Tensor, List[int], Optional[torch.Tensor]]:
    """
    Reindexes embeddings for variable batch size per feature scenarios.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, List[int], torch.Tensor]: the reindexed
            embeddings, lengths, length_per_key, and values
    """
    dim = embeddings.shape[1]
    output_size = sum(seq_vbe_ctx.reindexed_length_per_key) * dim
    reindexed_embeddings, _ = _permute_tensor_by_segments(
        tensor=embeddings.flatten(),
        segment_sizes=seq_vbe_ctx.unpadded_lengths * dim,
        recat=seq_vbe_ctx.recat,
        weights=None,
        output_size=output_size,
    )
    reindexed_embeddings = reindexed_embeddings.view(-1, dim)
    assert len(seq_vbe_ctx.reindexed_lengths.shape) == 2
    return (
        reindexed_embeddings,
        seq_vbe_ctx.reindexed_lengths,
        seq_vbe_ctx.reindexed_length_per_key,
        seq_vbe_ctx.reindexed_values,
    )


def construct_jagged_tensors(
    embeddings: torch.Tensor,
    features: KeyedJaggedTensor,
    embedding_names: List[str],
    need_indices: bool = False,
    features_to_permute_indices: Optional[Dict[str, List[int]]] = None,
    original_features: Optional[KeyedJaggedTensor] = None,
    reverse_indices: Optional[torch.Tensor] = None,
    seq_vbe_ctx: Optional[SequenceVBEContext] = None,
) -> Dict[str, JaggedTensor]:
    with record_function("## construct_jagged_tensors ##"):
        if original_features is not None:
            features = original_features
        if reverse_indices is not None:
            embeddings = torch.index_select(
                embeddings, 0, reverse_indices.to(torch.int32)
            )
        ret: Dict[str, JaggedTensor] = {}

        if seq_vbe_ctx is not None:
            embeddings, lengths, length_per_key, values = _vbe_reindex(
                embeddings=embeddings, seq_vbe_ctx=seq_vbe_ctx
            )
        else:
            lengths = features.lengths().view(-1, features.stride())
            length_per_key = features.length_per_key()
            values = features.values()

        lengths_tuple = torch.unbind(lengths, dim=0)
        embeddings_list = torch.split(embeddings, length_per_key, dim=0)
        values_list = (
            torch.split(values, length_per_key)
            if need_indices and values is not None
            else None
        )

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
                weights=(
                    values_list[indices[0]]
                    if need_indices and values_list is not None
                    else None
                ),
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
    module_name: str,
    dims: List[int],
) -> Callable[[List[Optional[torch.Tensor]], int], List[torch.Tensor]]:
    """
    Register a customized operator.

    Args:
        module: customized module instance
        dims: output dimensions
    """

    global operator_registry_state

    dims_str = "_".join([str(d) for d in dims])
    with operator_registry_state.op_registry_lock:
        op_name: str = f"{module_name}_{dims_str}"

        if op_name in operator_registry_state.op_registry_schema:
            return getattr(torch.ops.custom, op_name)

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

        schema_string = f"{op_name}(Tensor?[] values, SymInt batch_size) -> Tensor[]"
        operator_registry_state.op_registry_schema[op_name] = schema_string
        # Register schema
        lib.define(schema_string)

        # Register implementation
        lib.impl(op_name, custom_op, "CPU")
        lib.impl(op_name, custom_op, "CUDA")

        # Register meta formula
        lib.impl(op_name, custom_op, "Meta")

        return getattr(torch.ops.custom, op_name)


def register_custom_ops_for_nodes(
    nodes: List[torch.fx.Node],
) -> None:
    """
    Given a list of nodes, register custom ops if they exist in the nodes.
    Required for deserialization if in different runtime environments

    Args:
        nodes: list of nodes
    """

    for node in nodes:
        if "custom." in str(node.target):
            # torch.ops.custom.EmbeddingBagCollection_100.default
            # number represents dimension
            op_name = str(node.target).split(".")[-2]
            register_custom_op(
                op_name.split("_")[0],
                [int(dim) for dim in op_name.split("_")[1:]],
            )


@torch.fx.wrap
def jagged_index_select_with_empty(
    values: torch.Tensor,
    ids: torch.Tensor,
    offsets: torch.Tensor,
    output_offsets: torch.Tensor,
) -> torch.Tensor:
    if ids.size()[0] == 0:
        return torch.empty(0, device=values.device, dtype=values.dtype)
    output_values = torch.ops.fbgemm.jagged_index_select_2d_forward_v2(
        values.flatten().unsqueeze(-1),
        ids,
        offsets.long(),
        output_offsets.long(),
    )
    return output_values


def deterministic_dedup(ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    To remove race condition in conflict update, remove duplicated IDs. Only the last existence of duplicated ID will be kept.
    Return sorted unique ids and the position of the last existence
    """
    sorted_id_values, sorted_id_indices = ids.sort()
    sorted_unique_ids, sorted_unique_inverses = sorted_id_values.unique_consecutive(
        return_counts=False,
        return_inverse=True,
    )
    last_existence_index = torch.scatter_reduce(
        input=torch.zeros_like(sorted_unique_ids, dtype=torch.int64),
        dim=0,
        index=sorted_unique_inverses,
        src=sorted_id_indices,
        reduce="amax",
    )

    return sorted_unique_ids.view(-1), last_existence_index.flatten()
