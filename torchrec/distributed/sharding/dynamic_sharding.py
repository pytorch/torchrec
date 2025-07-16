#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed._shard.sharded_tensor import Shard
from torchrec.distributed.types import (
    EmbeddingModuleShardingPlan,
    ParameterSharding,
    ShardedModule,
    ShardedTensor,
    ShardingEnv,
    ShardingType,
)

OrderedShardNamesWithSizes = List[Tuple[str, List[int]]]
"""
A type alias to represent an ordered shard name and the corresponding shard_size
in dim 0 & 1 that were sent to the current rank.
This is a flattened and pruned nested list, which orders the shards names and
sizes in the following priority:
1. Rank order
2. Table order
3. Shard order

<table_x, shard_y> in below examples represent the 2d tensor correlated to a
certain table `x`, allocated to rank `z`. The `y` here denotes the order of shards
in the module attributes such as state_dict, sharding_plan, etc..

`z` != `y` numerically, but the order of shards is based on the order of ranks allocated

Example 1 NOTE: the ordering by rank:
Rank 0 sends table_0, shard_0 to Rank 1.
Rank 2 sends table_1, shard_0 to Rank 1.
Rank 2 sends table_1, shard_1 to Rank 0
Rank 3 sends table_0, shard_1 to Rank 0

NOTE: table_1 comes first due to its source rank being 'first'
On Rank 0:output_tensor = [
    <table_1, shard_0>, # from rank 2
    <table_0, shard_1>  # from rank 3
]

On Rank 1: output_tensor = [
    <table_0, shard_0>, # from rank 0
    <table_1, shard_0>  # from rank 2
]

Example 2: NOTE: ordered by table when ranks are the same
Rank 0 sends table_1 to Rank 1
Rank 0 sends table_2 to Rank 1

output_tensor = [
    <table_0, shard_y>,
    <table_1, shard_y>
]

Example 3: NOTE: ordered by shard if table and rank are the same
Rank 0 sends table_1, shard_0 to Rank 1
Rank 0 sends table_1, shard_1 to Rank 1

Rank 1: output_tensor = [
    <table_0, shard_0>,
    <table_1, shard_1>
]
"""


def _generate_shard_allocation_metadata(
    shard_name: str,
    source_params: ParameterSharding,
    destination_params: ParameterSharding,
) -> Dict[int, List[Tuple[int, List[int]]]]:
    """
    Generates a mapping of shards to ranks for redistribution of data.

    This function creates a mapping from source ranks to destination ranks
    based on the sharding specifications provided in the source and destination
    parameters. It calculates the shard dimensions and allocates them to the
    appropriate ranks in a greedy manner.

    Args:
        shard_name (str): The name of the shard being processed.
        source_params (ParameterSharding): The sharding parameters for the source.
        destination_params (ParameterSharding): The sharding parameters for the destination.

    Returns:
        Dict[int, List[Tuple[int, List[int]]]]: A dictionary mapping source ranks to a list of tuples,
        where each tuple contains a destination rank and the corresponding shard offsets.
    """
    shard_to_rank_mapping: Dict[int, List[Tuple[int, List[int]]]] = {}
    src_rank_index = 0
    dst_rank_index = 0
    curr_source_offset = 0
    curr_dst_offset = 0

    assert source_params.ranks is not None
    assert destination_params.ranks is not None

    assert source_params.sharding_spec is not None
    assert destination_params.sharding_spec is not None

    # Initialize dictionary keys for all source ranks
    # Pyre-ignore
    for rank in source_params.ranks:
        shard_to_rank_mapping[rank] = []

    # Pyre-ignore
    while src_rank_index < len(source_params.ranks) and dst_rank_index < len(
        destination_params.ranks  # Pyre-ignore
    ):
        # Pyre-ignore
        src_shard_size = source_params.sharding_spec.shards[src_rank_index].shard_sizes
        dst_shard_size = destination_params.sharding_spec.shards[
            dst_rank_index
        ].shard_sizes

        shard_dim = min(
            src_shard_size[1] - curr_source_offset, dst_shard_size[1] - curr_dst_offset
        )

        next_source_offset = curr_source_offset + shard_dim
        next_dst_offset = curr_dst_offset + shard_dim

        # Greedy way of allocating shards to ranks
        # Pyre-ignore
        shard_to_rank_mapping[source_params.ranks[src_rank_index]].append(
            (
                destination_params.ranks[dst_rank_index],
                [curr_source_offset, next_source_offset],
            )
        )
        curr_source_offset = next_source_offset
        curr_dst_offset = next_dst_offset

        if next_source_offset >= src_shard_size[1]:
            src_rank_index += 1
            curr_source_offset = 0

        if next_dst_offset >= dst_shard_size[1]:
            dst_rank_index += 1
            curr_dst_offset = 0
    return shard_to_rank_mapping


def _process_shard_redistribution_metadata(
    rank: int,
    shard_name: str,
    max_dim_0: int,
    max_dim_1: int,
    shard_to_rank_mapping: Dict[int, List[Tuple[int, List[int]]]],
    sharded_tensor: ShardedTensor,
    input_splits_per_rank: List[List[int]],
    output_splits_per_rank: List[List[int]],
    shard_names_to_lengths_by_src_rank: List[List[Tuple[str, List[int]]]],
    local_table_to_input_tensor_by_dst_rank: List[List[torch.Tensor]],
    local_table_to_opt_by_dst_rank: List[List[torch.Tensor]],
    optimizer_state: Optional[Dict[str, Dict[str, Dict[str, ShardedTensor]]]] = None,
    extend_shard_name: Callable[[str], str] = lambda x: x,
) -> Tuple[int, int]:
    """
    calculates shard redistribution metadata across ranks and processes optimizer state if present.

    This function handles the redistribution of tensor shards from source ranks to destination ranks
    based on the provided shard-to-rank mapping. It also processes optimizer state if available,
    ensuring that the data is correctly padded and split for communication between ranks.

    Args:
        rank (int): The current rank of the process.
        shard_name (str): The name of the shard being processed.
        max_dim_0 (int): The maximum dimension size of dim 0 for padding.
        max_dim_1 (int): The maximum dimension size of dim 1 for padding.
        shard_to_rank_mapping (Dict[int, List[Tuple[int, List[int]]]]): Mapping of source ranks to destination ranks and split offsets.
        sharded_tensor (ShardedTensor): The sharded tensor to be redistributed.
        input_splits_per_rank (List[List[int]]): Input split sizes for each rank.
        output_splits_per_rank (List[List[int]]): Output split sizes for each rank.
        shard_names_to_lengths_by_src_rank (List[List[Tuple[str, List[int]]]]): List of shard names and sizes by source rank.
        local_table_to_input_tensor_by_dst_rank (List[List[torch.Tensor]]): Local input tensors by destination rank.
        local_table_to_opt_by_dst_rank (List[List[torch.Tensor]]): Local optimizer tensors by destination rank.
        optimizer_state (Optional[Dict[str, Dict[str, Dict[str, ShardedTensor]]]]): Optimizer state if available.
        extend_shard_name (Callable[[str], str]): Function to extend shard names.

    Returns:
        Tuple[int, int]: Counts of output tensors and optimizer tensors processed.
    """

    output_tensor_count = 0
    output_optimizer_count = 0
    has_optimizer = optimizer_state is not None

    # Process each shard mapping from source to destination
    for src_rank, dsts in shard_to_rank_mapping.items():

        for dst_rank, split_offsets in dsts:

            # Get shard metadata
            shard_metadata = sharded_tensor.metadata().shards_metadata[0]
            shard_size = shard_metadata.shard_sizes

            assert split_offsets[0] >= 0
            assert split_offsets[1] <= shard_size[1]
            # Update the shard size with new size
            shard_size = [shard_size[0], split_offsets[1] - split_offsets[0]]
            # Update split sizes for communication
            input_splits_per_rank[src_rank][dst_rank] += max_dim_0
            output_splits_per_rank[dst_rank][src_rank] += max_dim_0
            if has_optimizer:
                input_splits_per_rank[src_rank][dst_rank] += max_dim_0
                output_splits_per_rank[dst_rank][src_rank] += max_dim_0

            # Process data being sent from current rank
            if src_rank == rank:
                # Handle optimizer state if present
                if has_optimizer and optimizer_state is not None:

                    local_optimizer_shards = optimizer_state["state"][
                        extend_shard_name(shard_name)
                    ][tmp_momentum_extender(shard_name)].local_shards()
                    assert (
                        len(local_optimizer_shards) == 1
                    ), "Expected exactly one local optimizer shard"

                    local_optimizer_tensor = local_optimizer_shards[0].tensor
                    if len(local_optimizer_tensor.size()) == 1:  # 1D Optimizer Tensor
                        # Convert to 2D Tensor, transpose, for AllToAll
                        local_optimizer_tensor = local_optimizer_tensor.view(
                            local_optimizer_tensor.size(0), 1
                        )
                    padded_optimizer_tensor = pad_tensor_to_max_dims(
                        local_optimizer_tensor, max_dim_0, max_dim_1
                    )
                    local_table_to_opt_by_dst_rank[dst_rank].append(
                        padded_optimizer_tensor
                    )

                # Handle main tensor data
                local_shards = sharded_tensor.local_shards()
                assert len(local_shards) == 1, "Expected exactly one local shard"

                # cut the tensor based on split points
                dst_t = local_shards[0].tensor[:, split_offsets[0] : split_offsets[1]]

                padded_tensor = pad_tensor_to_max_dims(dst_t, max_dim_0, max_dim_1)
                local_table_to_input_tensor_by_dst_rank[dst_rank].append(padded_tensor)

            # Process data being received at current rank
            if dst_rank == rank:
                shard_names_to_lengths_by_src_rank[src_rank].append(
                    (shard_name, shard_size)
                )
                output_tensor_count += max_dim_0
                if has_optimizer:
                    output_optimizer_count += max_dim_0

    return output_tensor_count, output_optimizer_count


def _create_local_shard_tensors(
    ordered_shard_names_and_lengths: OrderedShardNamesWithSizes,
    output_tensor: torch.Tensor,
    max_dim_0: int,
) -> Dict[str, torch.Tensor]:
    """
    Creates local shard tensors from the output tensor based on the ordered shard names and lengths.

    This function slices the output tensor into smaller tensors (shards) according to the specified
    dimensions in `ordered_shard_names_and_lengths`. It pads each shard to the maximum dimensions
    and concatenates them if multiple shards exist for the same shard name.

    Args:
        ordered_shard_names_and_lengths (OrderedShardNamesWithSizes): A list of tuples containing shard names
            and their corresponding sizes.
        output_tensor (torch.Tensor): The tensor containing all shards received by the current rank.
        max_dim_0 (int): The maximum dimension size of dim 0 for slicing the output tensor.

    Returns:
        Dict[str, torch.Tensor]: A dictionary mapping shard names to their corresponding local output tensors.
    """
    slice_index = 0
    shard_name_to_local_output_tensor: Dict[str, torch.Tensor] = {}
    for shard_name, shard_size in ordered_shard_names_and_lengths:
        end_slice_index = slice_index + max_dim_0
        cur_t = output_tensor[slice_index:end_slice_index]
        cur_t = pad_tensor_to_max_dims(cur_t, shard_size[0], shard_size[1])
        if shard_name not in shard_name_to_local_output_tensor.keys():
            shard_name_to_local_output_tensor[shard_name] = cur_t
        else:
            # CW sharding may have multiple shards per rank in many to one case, so we need to concatenate them
            shard_name_to_local_output_tensor[shard_name] = torch.cat(
                (shard_name_to_local_output_tensor[shard_name], cur_t), dim=1
            )
        slice_index = end_slice_index
    return shard_name_to_local_output_tensor


def shards_all_to_all(
    module: ShardedModule[Any, Any, Any, Any],  # pyre-ignore
    state_dict: Dict[str, ShardedTensor],
    device: torch.device,
    changed_sharding_params: Dict[str, ParameterSharding],
    env: ShardingEnv,
    max_dim_0: int,
    max_dim_1: int,
    extend_shard_name: Callable[[str], str] = lambda x: x,
    optimizer_state: Optional[Dict[str, Dict[str, Dict[str, ShardedTensor]]]] = None,
) -> Tuple[OrderedShardNamesWithSizes, torch.Tensor]:
    """
    Performs an all-to-all communication to redistribute shards across ranks based on new sharding parameters.
    Assumes ranks are ordered in ParameterSharding.ranks. Implements padding for concatenating, sending and
    receiving tensors of different sizes in dim 0 or 1.

    Args:
        module (ShardedModule[Any, Any, Any, Any]): The module containing sharded tensors to be redistributed.
            TODO: Update to support more modules, currently only supports ShardedEmbeddingBagCollection.

        state_dict (Dict[str, ShardedTensor]): The state dictionary containing the current sharded tensors.

        device (torch.device): The device on which the output tensors will be placed.

        changed_sharding_params (Dict[str, ParameterSharding]): A dictionary mapping shard names to their new sharding parameters.

        env (ShardingEnv): The sharding environment containing world size and other distributed information.

        extend_shard_name (Callable[[str], str], optional): A function to extend shard names to the full name in state_dict.

        max_dim_0 (int): The maximum dimension size of dim 0 across all tables in the module.

        max_dim_1 (int): The maximum dimension size of dim 1 across all tables in the module.

    Returns:
        Tuple[List[Tuple[str, List[int]]], torch.Tensor]: Two outputs containing:
            - A list of shard name and the corresponding shard_size in dim 0 & 1 that were sent to the current rank.
                This is a flattened and pruned nested list, which orders the shards names and sizes by source rank, then shard order.
            - The tensor containing all shards received by the current rank after the all-to-all operation.
    """
    if env.output_dtensor:
        raise RuntimeError("We do not yet support DTensor for resharding yet")
        return

    # Module sharding plan is used to get the source ranks for each shard
    assert hasattr(module, "module_sharding_plan")

    has_optimizer = optimizer_state is not None

    world_size = env.world_size
    rank = dist.get_rank()
    input_splits_per_rank = [[0] * world_size for _ in range(world_size)]
    output_splits_per_rank = [[0] * world_size for _ in range(world_size)]

    output_tensor_tensor_count = 0
    output_optimizer_tensor_count = 0
    shard_names_to_lengths_by_src_rank = [[] for _ in range(world_size)]
    local_table_to_input_tensor_by_dst_rank = [[] for _ in range(world_size)]
    local_table_to_opt_by_dst_rank = [[] for _ in range(world_size)]

    for shard_name, param in changed_sharding_params.items():
        sharded_t = state_dict[extend_shard_name(shard_name)]
        assert param.ranks is not None

        # pyre-ignore
        src_params = module.module_sharding_plan[shard_name]

        assert (
            param.sharding_type == ShardingType.COLUMN_WISE.value
            or param.sharding_type == ShardingType.TABLE_WISE.value
        )

        shard_mapping = _generate_shard_allocation_metadata(
            shard_name=shard_name,
            source_params=src_params,
            destination_params=param,
        )

        tensor_count, optimizer_count = _process_shard_redistribution_metadata(
            rank=rank,
            shard_name=shard_name,
            max_dim_0=max_dim_0,
            max_dim_1=max_dim_1,
            shard_to_rank_mapping=shard_mapping,
            sharded_tensor=sharded_t,
            input_splits_per_rank=input_splits_per_rank,
            output_splits_per_rank=output_splits_per_rank,
            shard_names_to_lengths_by_src_rank=shard_names_to_lengths_by_src_rank,
            local_table_to_input_tensor_by_dst_rank=local_table_to_input_tensor_by_dst_rank,
            local_table_to_opt_by_dst_rank=local_table_to_opt_by_dst_rank,
            optimizer_state=optimizer_state,
            extend_shard_name=extend_shard_name,
        )

        output_tensor_tensor_count += tensor_count
        if has_optimizer:
            output_optimizer_tensor_count += optimizer_count

    local_input_splits = input_splits_per_rank[rank]
    local_output_splits = output_splits_per_rank[rank]

    local_input_tensor = torch.empty([0, max_dim_1], device=device)
    for sub_l in local_table_to_input_tensor_by_dst_rank:
        for shard_info in sub_l:
            local_input_tensor = torch.cat(
                (
                    local_input_tensor,
                    shard_info,
                ),
                dim=0,
            )

    for sub_l in local_table_to_opt_by_dst_rank:
        for shard_info in sub_l:
            local_input_tensor = torch.cat(
                (
                    local_input_tensor,
                    shard_info,
                ),
                dim=0,
            )

    receive_count = output_tensor_tensor_count + output_optimizer_tensor_count
    max_embedding_size = max_dim_1
    local_output_tensor = torch.empty(
        [
            receive_count,
            max_embedding_size,
        ],
        device=device,
    )

    assert sum(local_output_splits) == len(local_output_tensor)
    assert sum(local_input_splits) == len(local_input_tensor)

    dist.all_to_all_single(
        output=local_output_tensor,
        input=local_input_tensor,
        output_split_sizes=local_output_splits,
        input_split_sizes=local_input_splits,
        group=env.process_group,  # TODO: 2D uses env.sharding_pg
    )

    flattened_output_names_lengths = [
        shard_info
        for sub_l in shard_names_to_lengths_by_src_rank
        for shard_info in sub_l
    ]

    return flattened_output_names_lengths, local_output_tensor


def update_state_dict_post_resharding(
    state_dict: Dict[str, ShardedTensor],
    ordered_shard_names_and_lengths: OrderedShardNamesWithSizes,
    output_tensor: torch.Tensor,
    new_sharding_params: Dict[str, ParameterSharding],
    curr_rank: int,
    max_dim_0: int,
    extend_shard_name: Callable[[str], str] = lambda x: x,
) -> Dict[str, ShardedTensor]:
    """
    Updates and returns the given state_dict with new placements and
    local_shards based on the output tensor of the AllToAll collective.
    Removes padding from the output tensor in dim 0 and 1 if necessary.

    Args:
        state_dict (Dict[str, Any]): The state dict to be updated with new shard placements and local shards.

        ordered_shard_names_and_lengths (List[Tuple[str, List[int]]]): A list of shard name and the corresponding shard_size.
            This is a flattened and pruned nested list, which orders the shards names and sizes by rank, then shard order.

        output_tensor (torch.Tensor): The tensor containing the output data from the AllToAll operation.

        new_sharding_params (Dict[str, ParameterSharding]): A dictionary mapping shard names to their new sharding parameters.
            This should only contain shard names that were updated during the AllToAll operation.

        curr_rank (int): The current rank of the process in the distributed environment.

        max_dim_0 (int): The maximum dimension size of dim 0 across all tables in the module. Only dim 0
            is needed here to slice the output tensor correctly, as removing the padding will only reference
            the original shard sizes stored in ordered_shard_names_and_lengths.

        extend_shard_name (Callable[[str], str], optional): A function to extend shard names to the full name in state_dict.

    Returns:
        Dict[str, ShardedTensor]: The updated state dictionary with new shard placements and local shards.
    """

    shard_name_to_local_output_tensor: Dict[str, torch.Tensor] = (
        _create_local_shard_tensors(
            ordered_shard_names_and_lengths, output_tensor, max_dim_0
        )
    )

    for shard_name, param in new_sharding_params.items():
        extended_name = extend_shard_name(shard_name)
        # pyre-ignore
        for i in range(len(param.ranks)):
            # pyre-ignore
            r = param.ranks[i]
            sharded_t = state_dict[extended_name]
            # Update placements

            if len(sharded_t.metadata().shards_metadata) > i:
                # pyre-ignore
                sharded_t.metadata().shards_metadata[i] = param.sharding_spec.shards[i]
            else:
                sharded_t.metadata().shards_metadata.append(
                    param.sharding_spec.shards[i]
                )
            # Update local shards
            if r == curr_rank:
                assert len(output_tensor) > 0
                # slice output tensor for correct size.
                sharded_t._local_shards = [
                    Shard(
                        tensor=shard_name_to_local_output_tensor[shard_name],
                        metadata=param.sharding_spec.shards[i],
                    )
                ]
                break
            else:
                sharded_t._local_shards = []

    return state_dict


def update_optimizer_state_post_resharding(
    old_opt_state: Dict[str, Dict[str, Dict[str, ShardedTensor]]],
    new_opt_state: Dict[str, Dict[str, Dict[str, ShardedTensor]]],
    ordered_shard_names_and_lengths: OrderedShardNamesWithSizes,
    output_tensor: torch.Tensor,
    max_dim_0: int,
) -> Dict[str, Dict[str, Dict[str, ShardedTensor]]]:
    new_opt_state_state = new_opt_state["state"]
    old_opt_state_state = old_opt_state["state"]

    # Remove padding and store tensors by shard name

    shard_name_to_local_output_tensor: Dict[str, torch.Tensor] = (
        _create_local_shard_tensors(
            ordered_shard_names_and_lengths, output_tensor, max_dim_0
        )
    )

    for extended_shard_name, item in new_opt_state_state.items():
        if extended_shard_name in old_opt_state_state:
            new_opt_state_state[extended_shard_name] = old_opt_state_state[
                extended_shard_name
            ]
        else:
            shard_name = extract_shard_name(extended_shard_name)
            momentum_name = tmp_momentum_extender(shard_name)
            sharded_t = item[momentum_name]
            assert len(sharded_t._local_shards) == 1
            # local_tensor is updated in-pace for CW sharding
            local_tensor = shard_name_to_local_output_tensor[shard_name]
            if len(sharded_t._local_shards[0].tensor.size()) == 1:
                # Need to transpose 1D optimizer tensor, due to previous conversion
                local_tensor = local_tensor.T[0]
            sharded_t._local_shards = [
                Shard(
                    tensor=local_tensor,
                    metadata=shard.metadata,
                )
                for shard in sharded_t._local_shards
            ]

    return new_opt_state


def update_module_sharding_plan(
    module: ShardedModule[Any, Any, Any, Any],  # pyre-ignore
    changed_sharding_params: Dict[str, ParameterSharding],
) -> None:
    if not hasattr(module, "module_sharding_plan"):
        return

    # pyre-ignore
    current_plan: Dict[str, ParameterSharding] = module.module_sharding_plan
    for table_name, param_sharding in changed_sharding_params.items():
        current_plan[table_name] = param_sharding
    return


def get_largest_dims_from_state_dict(
    state_dict: Dict[str, ShardedTensor],
) -> Tuple[int, int]:
    """
    Returns the largest dimension size of dim 0 and 1 across all tables in a module.

    Args:
        state_dict (Dict[str, ShardedTensor]): The state dict containing the sharded tensors.

    Returns:
        List[int]: A list of the largest dimension size of each table in the state_dict.
    """
    max_dim_0 = 0
    max_dim_1 = 0
    for sharded_t in state_dict.values():
        for shard in sharded_t.metadata().shards_metadata:
            max_dim_0 = max(max_dim_0, shard.shard_sizes[0])
            max_dim_1 = max(max_dim_1, shard.shard_sizes[1])

    return max_dim_0, max_dim_1


def get_largest_dims_from_sharding_plan_updates(
    sharding_plan_updates: Dict[str, ParameterSharding],
) -> Tuple[int, int]:
    """
    Returns the largest dimension size of dim 0 and 1 across all tables in a module.

    Args:
        state_dict (Dict[str, ShardedTensor]): The state dict containing the sharded tensors.

    Returns:
        List[int]: A list of the largest dimension size of each table in the state_dict.
    """
    max_dim_0 = 0
    max_dim_1 = 0
    for _, param in sharding_plan_updates.items():
        assert hasattr(param.sharding_spec, "shards")
        for shard in param.sharding_spec.shards:  # pyre-ignore
            max_dim_0 = max(max_dim_0, shard.shard_sizes[0])
            max_dim_1 = max(max_dim_1, shard.shard_sizes[1])

    return max_dim_0, max_dim_1


def pad_tensor_to_max_dims(
    t: torch.Tensor,
    expected_dim_0: int,
    expected_dim_1: int,
) -> torch.Tensor:
    """
    Pads a tensor on the right and bottom with zeros.

    Args:
        tensor (torch.Tensor): The tensor to be padded.
        pad_right (int): The number of zeros to pad on the right.
        pad_bottom (int): The number of zeros to pad on the bottom.

    Returns:
        torch.Tensor: The padded tensor.
    """
    pad_right = expected_dim_1 - t.size(1)
    pad_bottom = expected_dim_0 - t.size(0)
    pad = (0, pad_right, 0, pad_bottom)
    return F.pad(
        input=t,
        pad=pad,
        mode="constant",
        value=0,
    )


# Utils
def output_sharding_plan_delta(
    old_plan: EmbeddingModuleShardingPlan, new_plan: EmbeddingModuleShardingPlan
) -> EmbeddingModuleShardingPlan:
    """
    Compute and return a new sharding plan that is the delta
    between new and old embedding module plans. Assumes that the old and new plan
    have the same number of parameters/tables.

    This is useful for Dynamic Sharding since Resharding API takes in only the
    ParameterSharding or shards that needs to be moved.
    """
    assert len(old_plan) == len(new_plan)
    return EmbeddingModuleShardingPlan(
        {
            k: copy.deepcopy(v)
            for k, v in new_plan.items()
            if v.ranks != old_plan[k].ranks
        }
    )


"""
Utils for Optimizer State accessing
"""


def tmp_momentum_extender(name: str) -> str:
    return name + ".momentum1"


def extract_shard_name(name: str) -> str:
    return name.split(".")[-2]
