#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Callable, Dict, List, Tuple

import torch
import torch.distributed as dist
from torch.distributed._shard.sharded_tensor import Shard
from torchrec.distributed.types import (
    ParameterSharding,
    ShardedModule,
    ShardedTensor,
    ShardingEnv,
)


def shards_all_to_all(
    module: ShardedModule[Any, Any, Any, Any],  # pyre-ignore
    state_dict: Dict[str, ShardedTensor],
    device: torch.device,
    changed_sharding_params: Dict[str, ParameterSharding],
    env: ShardingEnv,
    extend_shard_name: Callable[[str], str] = lambda x: x,
) -> Tuple[List[Tuple[str, int]], torch.Tensor]:
    """
    Performs an all-to-all communication to redistribute shards across ranks based on new sharding parameters.
    Assumes ranks are ordered in ParameterSharding.ranks.

    Args:
        module (ShardedModule[Any, Any, Any, Any]): The module containing sharded tensors to be redistributed.
            TODO: Update to support more modules, currently only supports ShardedEmbeddingBagCollection.

        state_dict (Dict[str, ShardedTensor]): The state dictionary containing the current sharded tensors.

        device (torch.device): The device on which the output tensors will be placed.

        changed_sharding_params (Dict[str, ParameterSharding]): A dictionary mapping shard names to their new sharding parameters.

        env (ShardingEnv): The sharding environment containing world size and other distributed information.

        extend_shard_name (Callable[[str], str], optional): A function to extend shard names to the full name in state_dict.

    Returns:
        Tuple[List[Tuple[str, int]], torch.Tensor]: A tuple containing:
            - A list of shard name and the corresponding shard_size in dim 1 that were sent to the current rank.
                This is a flattened and pruned nested list, which orders the shards names and sizes by rank, then shard order.
            - The tensor containing all shards received by the current rank after the all-to-all operation.
    """
    if env.output_dtensor:
        raise RuntimeError("We do not yet support DTensor for resharding yet")
        return

    # Module sharding plan is used to get the source ranks for each shard
    assert hasattr(module, "module_sharding_plan")

    world_size = env.world_size
    rank = dist.get_rank()
    input_splits_per_rank = [[0] * world_size for _ in range(world_size)]
    output_splits_per_rank = [[0] * world_size for _ in range(world_size)]

    # 0 by default, as current rank may be recieving 0 shards
    num_embeddings_received = 0
    output_tensor_tensor_count = 0
    shard_names_to_lengths_by_src_rank = [[] for _ in range(world_size)]
    local_table_to_input_tensor_by_dst_rank = [[] for _ in range(world_size)]
    for shard_name, param in changed_sharding_params.items():
        sharded_t = state_dict[extend_shard_name(shard_name)]
        assert param.ranks is not None
        dst_ranks = param.ranks
        state_dict[extend_shard_name(shard_name)]
        # pyre-ignore
        src_ranks = module.module_sharding_plan[shard_name].ranks

        # TODO: Implement changing rank sizes for beyond TW sharding
        assert len(dst_ranks) == len(src_ranks)

        # index needed to distinguish between multiple shards
        # within the same shardedTensor for each table
        for i in range(len(src_ranks)):
            dst_rank = dst_ranks[i]
            src_rank = src_ranks[i]

            shard_size = sharded_t.metadata().shards_metadata[i].shard_sizes
            shard_size_dim_1 = shard_size[1]
            input_splits_per_rank[src_rank][dst_rank] += shard_size_dim_1
            output_splits_per_rank[dst_rank][src_rank] += shard_size_dim_1
            if src_rank == rank:
                local_shards = sharded_t.local_shards()
                assert len(local_shards) == 1
                local_table_to_input_tensor_by_dst_rank[dst_rank].append(
                    sharded_t.local_shards()[0].tensor
                )
            if dst_rank == rank:
                shard_names_to_lengths_by_src_rank[src_rank].append(
                    (shard_name, shard_size_dim_1)
                )
                # NOTE: Only need to update num_embeddings_received to be the
                # num_embeddings of shards if this rank is actually recieving
                # any tensors
                if num_embeddings_received == 0:
                    num_embeddings_received = shard_size[0]
                else:
                    # TODO: for 2D and row-wise, shard_sizes in dim 0 may be variable
                    # For now, assume that shard_sizes in dim 0 are all the same
                    assert num_embeddings_received == shard_size[0]
                output_tensor_tensor_count += shard_size[1]

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
                dim=1,
            )

    # Transposing the Tensors - because we are concatenating them along dimension 1
    # This is because dim 0 size may be different for different shards
    # whereas dim 1 size is the same for all shards as dim 1 size = num_embeddings per table
    local_output_tensor = torch.empty(
        [output_tensor_tensor_count, num_embeddings_received], device=device
    )
    local_input_tensor = local_input_tensor.T.contiguous()

    assert sum(local_output_splits) == len(local_output_tensor)
    assert sum(local_input_splits) == len(local_input_tensor)
    dist.all_to_all_single(
        output=local_output_tensor,
        input=local_input_tensor,
        output_split_sizes=local_output_splits,
        input_split_sizes=local_input_splits,
        group=dist.group.WORLD,
    )

    flattened_output_names_lengths = [
        shard_info
        for sub_l in shard_names_to_lengths_by_src_rank
        for shard_info in sub_l
    ]

    return flattened_output_names_lengths, local_output_tensor


def update_state_dict_post_resharding(
    state_dict: Dict[str, ShardedTensor],
    ordered_shard_names_and_lengths: List[Tuple[str, int]],
    output_tensor: torch.Tensor,
    new_sharding_params: Dict[str, ParameterSharding],
    curr_rank: int,
    extend_shard_name: Callable[[str], str] = lambda x: x,
) -> Dict[str, ShardedTensor]:
    """
    Updates and returns the given state_dict with new placements and
    local_shards based on the output tensor of the AllToAll collective.

    Args:
        state_dict (Dict[str, Any]): The state dict to be updated with new shard placements and local shards.

        shard_names_by_src_rank (List[Tuple[str, int]]): A list of shard name and the corresponding shard_size in dim 1
        that were sent to the current rank. This is a flattened and pruned nested list, which orders the shards names and
        sizes by rank, then shard order.

        output_tensor (torch.Tensor): The tensor containing the output data from the AllToAll operation.

        new_sharding_params (Dict[str, ParameterSharding]): A dictionary mapping shard names to their new sharding parameters.
            This should only contain shard names that were updated during the AllToAll operation.

        curr_rank (int): The current rank of the process in the distributed environment.

        extend_shard_name (Callable[[str], str], optional): A function to extend shard names to the full name in state_dict.

    Returns:
        Dict[str, ShardedTensor]: The updated state dictionary with new shard placements and local shards.
    """
    slice_index = 0

    shard_name_to_local_output_tensor: Dict[str, torch.Tensor] = {}

    for shard_name, shard_size in ordered_shard_names_and_lengths:
        end_slice_index = slice_index + shard_size
        shard_name_to_local_output_tensor[shard_name] = output_tensor[
            slice_index:end_slice_index
        ].T
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
