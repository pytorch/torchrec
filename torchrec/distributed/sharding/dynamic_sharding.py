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

ShardToRankMapping = Dict[int, List[Tuple[int, int, int, List[int]]]]
"""
ShardToRankMapping is a type alias for a dictionary that maps source ranks to a list of shard metadata tuples.
Each key in the dictionary is an integer representing a source rank.
The value associated with each key is a list of tuples, where each tuple contains metadata about a shard.
The structure of each tuple is as follows:
1. First Element (int): Represents the position of the shard metadata in the original module sharding plan.
   This indicates the order or index of the shard within the sharding plan.
2. Second Element (int): Represents the position of the shard tensor in the original state dictionary.
   This helps in identifying the specific shard tensor within the state dictionary.
3. Third Element (int): Represents the destination rank.
   This indicates the rank to which the shard is being redistributed.
4. Fourth Element (List[int]): A list representing the shard size.
   This provides the dimensions or size of the shard being handled.

   E.g [0,(1,0,2,[10,4])] means that the source rank 0 has a shard with size (10,4) and it's  in the first position of the
    source modulesharding plan. and the data can be accessed through rank=1, first local tensor and this is being sent to rank 2,
    the new shard size is (10,4)
"""


def _generate_shard_allocation_metadata(
    shard_name: str,
    source_params: ParameterSharding,
    destination_params: ParameterSharding,
    rank: int,
    world_size: int,
) -> ShardToRankMapping:
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
    shard_to_rank_mapping: ShardToRankMapping = {}
    src_rank_index = 0
    dst_rank_index = 0
    curr_source_offset = 0
    curr_dst_offset = 0
    local_shard_indices = [0 for _ in range(world_size)]

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
                src_rank_index,
                local_shard_indices[source_params.ranks[src_rank_index]],
                destination_params.ranks[dst_rank_index],
                [curr_source_offset, next_source_offset],
            )
        )
        curr_source_offset = next_source_offset
        curr_dst_offset = next_dst_offset

        if next_source_offset >= src_shard_size[1]:
            local_shard_indices[source_params.ranks[src_rank_index]] += 1
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
    shard_to_rank_mapping: ShardToRankMapping,
    sharded_tensor: ShardedTensor,
    input_splits_per_rank: List[List[int]],
    output_splits_per_rank: List[List[int]],
    shard_names_to_lengths_by_src_rank: List[List[Tuple[str, List[int]]]],
    local_table_to_input_tensor_by_dst_rank: List[List[torch.Tensor]],
    local_table_to_opt_by_dst_rank: List[List[torch.Tensor]],
    optimizer_state: Optional[Dict[str, Dict[str, Dict[str, ShardedTensor]]]] = None,
    extend_shard_name: Callable[[str], str] = lambda x: x,
    has_optimizer: bool = False,
) -> Tuple[int, int]:
    """
    Processes the metadata for shard redistribution across ranks.

    This function handles the redistribution of shards from source ranks to destination ranks
    based on the provided shard-to-rank mapping. It updates the input and output splits for
    each rank and manages the optimizer state if present.

    Args:
        rank (int): The current rank of the process.
        shard_name (str): The name of the shard being processed.
        max_dim_0 (int): The maximum dimension size of dim 0 for padding.
        max_dim_1 (int): The maximum dimension size of dim 1 for padding.
        shard_to_rank_mapping ShardToRankMapping: Mapping of source ranks to destination ranks and shard offsets.
        sharded_tensor (ShardedTensor): The sharded tensor being processed.
        input_splits_per_rank (List[List[int]]): Input split sizes for each rank.
        output_splits_per_rank (List[List[int]]): Output split sizes for each rank.
        shard_names_to_lengths_by_src_rank (List[List[Tuple[str, List[int]]]]): List of shard names and sizes by source rank.
        local_table_to_input_tensor_by_dst_rank (List[List[torch.Tensor]]): Local input tensors for each destination rank.
        local_table_to_opt_by_dst_rank (List[List[torch.Tensor]]): Local optimizer tensors for each destination rank.
        optimizer_state (Optional[Dict[str, Dict[str, Dict[str, ShardedTensor]]]]): The optimizer state, if present.
        extend_shard_name (Callable[[str], str]): Function to extend shard names.
        has_optimizer (bool): Flag indicating whether any rank has an optimizer state for this shard. Destination ranks
            initally may not have an optimizer state, but if any source rank has an optimizer state, then all destinations
            should recreate that optimizer state.

    Returns:
        Tuple[int, int]: Counts of output tensors and optimizer tensors processed.
    """

    output_tensor_count = 0
    output_optimizer_count = 0
    has_local_optimizer = (
        optimizer_state is not None
    )  # local optimizer represents wether sharding table has optimizer state locally.

    # Process each shard mapping from source to destination
    for src_rank, dsts in shard_to_rank_mapping.items():

        for shard_id, local_shard_id, dst_rank, split_offsets in dsts:

            # Get shard metadata
            shard_metadata = sharded_tensor.metadata().shards_metadata[shard_id]
            shard_size = shard_metadata.shard_sizes

            assert split_offsets[0] >= 0
            assert split_offsets[1] <= shard_size[1]
            # Update the shard size with new size
            shard_size = [shard_size[0], split_offsets[1] - split_offsets[0], shard_id]
            # Update split sizes for communication
            input_splits_per_rank[src_rank][dst_rank] += shard_size[0]
            output_splits_per_rank[dst_rank][src_rank] += shard_size[0]

            # Process data being sent from current rank
            if src_rank == rank:
                # Handle optimizer state if present
                extended_shard_name: str = extend_shard_name(shard_name)
                if has_local_optimizer:
                    momentun_name = tmp_momentum_extender(shard_name)

                    # Pyre-ignore
                    local_optimizer_shards = optimizer_state["state"][
                        extended_shard_name
                    ][momentun_name].local_shards()

                    local_optimizer_tensor = local_optimizer_shards[
                        local_shard_id
                    ].tensor

                    if len(local_optimizer_tensor.size()) == 1:  # 1D Optimizer Tensor
                        # Convert to 2D Tensor, transpose, for AllToAll
                        local_optimizer_tensor = local_optimizer_tensor.view(
                            local_optimizer_tensor.size(0), 1
                        )
                    else:
                        local_optimizer_tensor = local_optimizer_tensor[
                            :, split_offsets[0] : split_offsets[1]
                        ]

                    padded_optimizer_tensor = pad_tensor_to_max_dims(
                        local_optimizer_tensor, shard_size[0], max_dim_1
                    )
                    local_table_to_opt_by_dst_rank[dst_rank].append(
                        padded_optimizer_tensor
                    )

                    input_splits_per_rank[src_rank][dst_rank] += shard_size[0]

                local_shards = sharded_tensor.local_shards()

                local_tensor = local_shards[local_shard_id].tensor

                dst_t = local_tensor[:, split_offsets[0] : split_offsets[1]]

                padded_tensor = pad_tensor_to_max_dims(dst_t, shard_size[0], max_dim_1)
                local_table_to_input_tensor_by_dst_rank[dst_rank].append(padded_tensor)

            # Process data being received at current rank
            if dst_rank == rank:
                shard_names_to_lengths_by_src_rank[src_rank].append(
                    (shard_name, shard_size)
                )

                output_tensor_count += shard_size[0]
                if has_optimizer:

                    output_optimizer_count += shard_size[0]
                    output_splits_per_rank[dst_rank][src_rank] += shard_size[0]

    return output_tensor_count, output_optimizer_count


def _create_local_shard_tensors(
    ordered_shard_names_and_lengths: OrderedShardNamesWithSizes,
    output_tensor: torch.Tensor,
    has_optimizer: bool = False,
    optimizer_mode: bool = False,
    new_state: Optional[Dict[str, Dict[str, ShardedTensor]]] = None,
    extend_shard_name: Optional[Callable[[str], str]] = None,
) -> Dict[str, List[torch.Tensor]]:
    """
    Creates local shard tensors from the output tensor based on the ordered shard names and lengths.

    This function slices the output tensor into smaller tensors (shards) according to the specified
    dimensions in `ordered_shard_names_and_lengths`. It pads each shard to the maximum dimensions
    and concatenates them if multiple shards exist for the same shard name.

    Args:
        ordered_shard_names_and_lengths (OrderedShardNamesWithSizes): A list of tuples containing shard names
            and their corresponding sizes.
        output_tensor (torch.Tensor): The tensor containing all shards received by the current rank.

    Returns:
        Dict[str, torch.Tensor]: A dictionary mapping shard names to their corresponding local output tensors.
        has_optimizer (bool): Flag indicating whether optimizer is enabled and optimizer weights are present.
            It is helpful, to determine the split indexes of the output_tensor.

        e.g output_tensor_format when has_optimizer enabled = [ST1,OPW1,ST2,OPW2,ST3,OPW3,ST4,OPW4]
        e.g output_tensor_format when has_optimizer disabled = [ST1,ST2,ST3,ST4]
    """

    shard_name_to_local_output_tensor: Dict[str, List[torch.Tensor]] = {}

    slice_index = 0

    splitted_shards_with_names: Dict[str, List[Tuple[int, torch.Tensor]]] = {}
    for i, (shard_name, shard_size) in enumerate(ordered_shard_names_and_lengths):
        if i == 0:
            slice_index = 0 if not optimizer_mode else shard_size[0]
        shard_id = shard_size[2]
        end_slice_index = slice_index + shard_size[0]
        cur_t = output_tensor[slice_index:end_slice_index]
        cur_t = cur_t[: shard_size[0], : shard_size[1]]

        extended_shard_name = (
            extend_shard_name(shard_name) if extend_shard_name else shard_name
        )
        new_state = new_state if new_state else {}

        momentum_name = tmp_momentum_extender(shard_name)

        if (
            optimizer_mode
            and new_state is not None
            and extended_shard_name in new_state.keys()
        ):
            sharded_t = new_state[extended_shard_name][momentum_name]
            assert len(sharded_t._local_shards) == 1

            if len(sharded_t._local_shards[0].tensor.size()) == 1:
                cur_t.mul_(shard_size[1])  # Supporting RowWise Adagrad operation

        if shard_name not in splitted_shards_with_names:
            splitted_shards_with_names[shard_name] = [(shard_id, cur_t)]
        else:
            splitted_shards_with_names[shard_name].append((shard_id, cur_t))

        slice_index = (
            end_slice_index
            if not has_optimizer
            else (
                end_slice_index + shard_size[0]
                if not optimizer_mode
                else (
                    end_slice_index + ordered_shard_names_and_lengths[i + 1][1][0]
                    if i < len(ordered_shard_names_and_lengths) - 1
                    else end_slice_index
                )
            )
        )

    # Assuming splitted_shards_with_names is already populated
    for shard_name, shards in splitted_shards_with_names.items():
        # Sort shards by shard_id if needed, since, CW sharding can have multiple shards for the same table
        shards.sort(key=lambda x: x[0])

        for _, curr_t in shards:
            # Initialize shard_name_to_local_output_tensor[shard_name] if it doesn't exist
            if shard_name not in shard_name_to_local_output_tensor:
                # Initialize with a list containing the first tensor
                shard_name_to_local_output_tensor[shard_name] = [curr_t]
            else:
                # Since we always assume one tensor in the list, concatenate with it
                # TODO: Extend this for multiple shards per table for same rank for new state.
                # TODO: Although original plan supports min_partition, we assume changing plan has only one shard per table IN CW sharding
                concatenated_tensor = torch.cat(
                    (shard_name_to_local_output_tensor[shard_name][0], curr_t), dim=1
                )
                # Replace the existing tensor with the concatenated one
                shard_name_to_local_output_tensor[shard_name][0] = concatenated_tensor

    return shard_name_to_local_output_tensor


def move_sharded_tensors_to_cpu(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively traverse a state dictionary and move all local shard tensors to CPU.
    This helps reduce GPU memory usage by keeping tensors in CPU memory until needed.

    Args:
        state_dict: The state dictionary to traverse (can be model or optimizer state dict)

    Returns:
        The modified state dictionary with tensors moved to CPU
    """

    # Pyre-ignore
    def _process_item(item: Any) -> Any:
        if isinstance(item, ShardedTensor):
            # For ShardedTensor, move all local shards to CPU
            for shard in item.local_shards():
                if shard.tensor.device.type == "cuda":
                    shard.tensor = shard.tensor.cpu()
            return item
        elif isinstance(item, dict):
            # Recursively process dictionaries
            return {k: _process_item(v) for k, v in item.items()}
        elif isinstance(item, list):
            # Recursively process lists
            return [_process_item(v) for v in item]
        elif isinstance(item, tuple):
            # Recursively process tuples
            return tuple(_process_item(v) for v in item)
        else:
            # Return other types unchanged
            return item

    processed_dict = _process_item(state_dict)
    torch.cuda.empty_cache()
    return processed_dict


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
    has_optimizer: bool = False,
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

        optimizer_state (Optional[Dict[str, Dict[str, Dict[str, ShardedTensor]]]]): The optimizer state, if present.
            A dictionary mapping shard names to their optimizer states, which are themselves
            dictionaries mapping parameter names to their optimizer states.

        has_optimizer (bool): Flag indicating if optimizer state is present.

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

    has_local_optimizer = has_optimizer and optimizer_state is not None

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
            rank=rank,
            world_size=world_size,
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
            has_optimizer=has_optimizer,
        )

        output_tensor_tensor_count += tensor_count
        if has_optimizer:
            output_optimizer_tensor_count += optimizer_count

    local_input_splits = input_splits_per_rank[rank]
    local_output_splits = output_splits_per_rank[rank]

    total_input_size = sum(local_input_splits)
    local_input_tensor = torch.zeros(
        [total_input_size, max_dim_1], device=torch.device("cpu")
    )

    current_pos = 0

    for i, sub_l in enumerate(local_table_to_input_tensor_by_dst_rank):
        batch_size = local_input_splits[i]
        if batch_size == 0:
            continue

        # Create a view into the pre-allocated tensor for this destination rank
        batch_view = local_input_tensor[current_pos : current_pos + batch_size]

        current_row = 0
        for j, shard_info_cpu in enumerate(sub_l):
            if shard_info_cpu is not None:
                rows = shard_info_cpu.size(0)

                # Copying data to input tensor uvm->uvm operation
                batch_view[current_row : current_row + rows] = shard_info_cpu
                current_row += rows

            # Free CPU memory by removing reference
            local_table_to_input_tensor_by_dst_rank[i][j] = None

            if has_local_optimizer:
                opt_shard_info_cpu = local_table_to_opt_by_dst_rank[i][j]
                if opt_shard_info_cpu is not None:
                    opt_rows = opt_shard_info_cpu.size(0)

                    batch_view[current_row : current_row + opt_rows] = (
                        opt_shard_info_cpu
                    )
                    current_row += opt_rows

                    # Free CPU memory by removing reference
                    local_table_to_opt_by_dst_rank[i][j] = None

        # Move position pointer forward
        current_pos += batch_size

    receive_count = output_tensor_tensor_count + output_optimizer_tensor_count
    max_embedding_size = max_dim_1
    local_output_tensor_cpu = torch.empty(
        [
            receive_count,
            max_embedding_size,
        ],
        device=torch.device("cpu"),
    )

    assert sum(local_output_splits) == len(local_output_tensor_cpu)
    assert sum(local_input_splits) == len(local_input_tensor)

    # TODO: move this to hireachical process creation if possible for scaling beyond 32
    if not hasattr(env, "cpu_process_group"):
        # Create a CPU process group with Gloo backend
        # Pyre-ignore
        env.cpu_process_group = dist.new_group(
            ranks=list(range(env.world_size)),
            backend="gloo",  # Use Gloo backend for CPU operations
        )

    dist.all_to_all_single(
        output=local_output_tensor_cpu,
        input=local_input_tensor,
        output_split_sizes=local_output_splits,
        input_split_sizes=local_input_splits,
        group=env.cpu_process_group,  # TODO: 2D uses env.sharding_pg
    )
    del local_input_tensor

    flattened_output_names_lengths = [
        shard_info
        for sub_l in shard_names_to_lengths_by_src_rank
        for shard_info in sub_l
    ]

    return flattened_output_names_lengths, local_output_tensor_cpu


def update_state_post_resharding(
    old_state: Dict[str, ShardedTensor],
    new_state: Dict[str, ShardedTensor],
    ordered_shard_names_and_lengths: OrderedShardNamesWithSizes,
    output_tensor: torch.Tensor,
    extend_shard_name: Callable[[str], str] = lambda x: x,
    has_optimizer: bool = False,
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

    shard_name_to_local_output_tensor: Dict[str, List[torch.Tensor]] = (
        _create_local_shard_tensors(
            ordered_shard_names_and_lengths,
            output_tensor,
            has_optimizer=has_optimizer,
            optimizer_mode=False,
        )
    )

    for extended_shard_name, item in new_state.items():
        shard_name = extract_shard_name(extended_shard_name)
        if (
            old_state is not None
            and extended_shard_name in old_state
            and shard_name not in shard_name_to_local_output_tensor.keys()
        ):

            sharded_t = new_state[extended_shard_name]
            sharded_t_old = old_state[extended_shard_name]

            local_shards = sharded_t._local_shards
            for i, shard in enumerate(local_shards):
                shard.tensor.copy_(
                    sharded_t_old._local_shards[i].tensor, non_blocking=True
                )
                shard.metadata = sharded_t_old._local_shards[i].metadata
        else:

            sharded_t = item
            assert len(sharded_t._local_shards) == 1
            # local_tensor is updated in-pace for CW sharding
            local_tensor = shard_name_to_local_output_tensor[shard_name][0]

            for i, shard in enumerate(sharded_t._local_shards):
                shard.tensor.copy_(local_tensor, non_blocking=True)
                shard.metadata = sharded_t._local_shards[i].metadata

    return new_state


def update_optimizer_state_post_resharding(
    old_opt_state: Dict[str, Dict[str, Dict[str, ShardedTensor]]],
    new_opt_state: Dict[str, Dict[str, Dict[str, ShardedTensor]]],
    ordered_shard_names_and_lengths: OrderedShardNamesWithSizes,
    output_tensor: torch.Tensor,
    max_dim_0: int,
    extend_shard_name: Callable[[str], str] = lambda x: x,
) -> Dict[str, Dict[str, Dict[str, ShardedTensor]]]:
    new_opt_state_state = new_opt_state["state"] if new_opt_state else None
    old_opt_state_state = old_opt_state["state"] if old_opt_state else None

    # Remove padding and store tensors by shard name
    shard_name_to_local_output_tensor: Dict[str, List[torch.Tensor]] = (
        _create_local_shard_tensors(
            ordered_shard_names_and_lengths,
            output_tensor,
            has_optimizer=True,
            optimizer_mode=True,
            new_state=new_opt_state_state,
            extend_shard_name=extend_shard_name,
        )
    )

    if new_opt_state_state is None or len(new_opt_state_state) == 0:
        return new_opt_state

    for extended_shard_name, item in new_opt_state_state.items():
        shard_name = extract_shard_name(extended_shard_name)
        momentum_name = tmp_momentum_extender(shard_name)

        if (
            old_opt_state_state is not None
            and extended_shard_name in old_opt_state_state
            and shard_name not in shard_name_to_local_output_tensor.keys()
        ):

            sharded_t = new_opt_state_state[extended_shard_name][momentum_name]
            sharded_t_old = old_opt_state_state[extended_shard_name][momentum_name]
            local_shards = sharded_t._local_shards
            for i, shard in enumerate(local_shards):
                shard.tensor.copy_(
                    sharded_t_old._local_shards[i].tensor, non_blocking=True
                )
                shard.metadata = sharded_t_old._local_shards[i].metadata
        else:

            sharded_t = item[momentum_name]
            assert len(sharded_t._local_shards) == 1
            # local_tensor is updated in-pace for CW sharding
            local_tensor = shard_name_to_local_output_tensor[shard_name][0]
            if len(sharded_t._local_shards[0].tensor.size()) == 1:
                # Need to transpose 1D optimizer tensor, due to previous conversion

                local_tensor_dim = local_tensor.size()[1]
                squared_sum_t = torch.sum(local_tensor, dim=1, keepdim=True)
                mean_squared_sum_t = torch.div(squared_sum_t, local_tensor_dim)
                local_tensor = mean_squared_sum_t.T[0]

            for i, shard in enumerate(sharded_t._local_shards):
                shard.tensor.copy_(local_tensor, non_blocking=True)
                shard.metadata = sharded_t._local_shards[i].metadata

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
    if expected_dim_0 == t.size(0) and expected_dim_1 == t.size(1):
        return t
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
    old_plan: EmbeddingModuleShardingPlan,
    new_plan: EmbeddingModuleShardingPlan,
    return_data_volume: bool = False,
) -> Tuple[float, EmbeddingModuleShardingPlan]:
    """
    Compute and return a new sharding plan that is the delta
    between new and old embedding module plans. Assumes that the old and new plan
    have the same number of parameters/tables.

    This is useful for Dynamic Sharding since Resharding API takes in only the
    ParameterSharding or shards that needs to be moved.
    """
    assert len(old_plan) == len(new_plan)
    diff = EmbeddingModuleShardingPlan(
        {
            k: copy.deepcopy(v)
            for k, v in new_plan.items()
            if v.ranks != old_plan[k].ranks
        }
    )
    data_volume: float = 0
    if return_data_volume:
        for _, v in diff.items():
            # Pyre-ignore
            for shard in v.sharding_spec.shards:
                data_volume += (
                    shard.shard_sizes[0] * shard.shard_sizes[1] * 4 / (1024 * 1024)
                )  # Asumming float datatype

    return (data_volume, diff)


"""
Utils for Optimizer State accessing
"""


def tmp_momentum_extender(name: str) -> str:
    return name + ".momentum1"


def extract_shard_name(name: str) -> str:
    return name.split(".")[-2]
