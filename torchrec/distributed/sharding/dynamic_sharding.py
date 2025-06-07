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
        dst_ranks = param.ranks
        # pyre-ignore
        src_ranks = module.module_sharding_plan[shard_name].ranks

        # TODO: Implement changing rank sizes for beyond TW sharding
        assert len(dst_ranks) == len(src_ranks)

        # index needed to distinguish between multiple shards
        # within the same shardedTensor for each table
        for i in range(len(src_ranks)):

            # 1 to 1 mapping from src to dst
            dst_rank = dst_ranks[i]
            src_rank = src_ranks[i]

            shard_size = sharded_t.metadata().shards_metadata[i].shard_sizes
            input_splits_per_rank[src_rank][dst_rank] += max_dim_0
            output_splits_per_rank[dst_rank][src_rank] += max_dim_0
            if has_optimizer:
                input_splits_per_rank[src_rank][dst_rank] += max_dim_0
                output_splits_per_rank[dst_rank][src_rank] += max_dim_0

            # If sending from current rank
            if src_rank == rank:
                if has_optimizer:
                    # pyre-ignore
                    local_optimizer = optimizer_state["state"][
                        extend_shard_name(shard_name)
                    ][tmp_momentum_extender(shard_name)].local_shards()
                    assert len(local_optimizer) == 1
                    padded_local_optimizer = pad_tensor_to_max_dims(
                        local_optimizer[0].tensor, max_dim_0, max_dim_1
                    )
                    local_table_to_opt_by_dst_rank[dst_rank].append(
                        padded_local_optimizer
                    )
                local_shards = sharded_t.local_shards()
                assert len(local_shards) == 1
                cur_t = pad_tensor_to_max_dims(
                    local_shards[0].tensor, max_dim_0, max_dim_1
                )
                local_table_to_input_tensor_by_dst_rank[dst_rank].append(cur_t)

            # If recieving from current rank
            if dst_rank == rank:
                shard_names_to_lengths_by_src_rank[src_rank].append(
                    (shard_name, shard_size)
                )
                output_tensor_tensor_count += max_dim_0
                if has_optimizer:
                    output_optimizer_tensor_count += max_dim_0

    local_input_splits = input_splits_per_rank[rank]
    local_output_splits = output_splits_per_rank[rank]

    local_input_tensor = torch.empty([0], device=device)
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

    max_embedding_size = max_dim_1
    local_output_tensor = torch.empty(
        [
            output_tensor_tensor_count + output_optimizer_tensor_count,
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
    slice_index = 0

    shard_name_to_local_output_tensor: Dict[str, torch.Tensor] = {}

    for shard_name, shard_size in ordered_shard_names_and_lengths:
        end_slice_index = slice_index + max_dim_0
        cur_t = output_tensor[slice_index:end_slice_index]
        cur_t = pad_tensor_to_max_dims(
            cur_t, shard_size[0], shard_size[1], remove_padding=True
        )
        shard_name_to_local_output_tensor[shard_name] = cur_t
        slice_index = end_slice_index

    for shard_name, param in new_sharding_params.items():
        extended_name = extend_shard_name(shard_name)
        # pyre-ignore
        for i in range(len(param.ranks)):
            # pyre-ignore
            r = param.ranks[i]
            sharded_t = state_dict[extended_name]
            # Update placements
            sharded_t.metadata().shards_metadata[i].placement = (
                torch.distributed._remote_device(f"rank:{r}/cuda:{r}")
            )
            if r == curr_rank:
                assert len(output_tensor) > 0
                # slice output tensor for correct size.
                sharded_t._local_shards = [
                    Shard(
                        tensor=shard_name_to_local_output_tensor[shard_name],
                        metadata=state_dict[extended_name]
                        .metadata()
                        .shards_metadata[i],
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
    slice_index = 0
    shard_name_to_local_output_tensor: Dict[str, torch.Tensor] = {}
    for shard_name, shard_size in ordered_shard_names_and_lengths:
        end_slice_index = slice_index + max_dim_0
        cur_t = output_tensor[slice_index:end_slice_index]
        cur_t = pad_tensor_to_max_dims(
            cur_t, shard_size[0], shard_size[1], remove_padding=True
        )
        shard_name_to_local_output_tensor[shard_name] = cur_t
        slice_index = end_slice_index

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
            # TODO: support multiple shards in CW sharding
            sharded_t._local_shards = [
                Shard(
                    tensor=shard_name_to_local_output_tensor[shard_name],
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
    remove_padding: bool = False,
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
    return F.pad(
        input=t,
        pad=(
            0,
            pad_right,
            0,
            pad_bottom,
        ),  # right and bottom
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
