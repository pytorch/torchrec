#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
from torch.distributed._shard.sharded_tensor import Shard
from torchrec.distributed.types import ParameterSharding, ShardedTensor, ShardingEnv


def extend_shard_name(shard_name: str) -> str:
    return f"embedding_bags.{shard_name}.weight"


def shards_all_to_all(
    module: torch.nn.Module,
    device: torch.device,
    changed_sharding_params: Dict[str, ParameterSharding],
    env: ShardingEnv,
) -> Tuple[List[List[str]], torch.Tensor]:
    """
    Performs an all-to-all communication to redistribute shards across ranks based on new sharding parameters.
    Assumes ranks are ordered in ParameterSharding.ranks.

    Args:
        module (torch.nn.Module): The module containing sharded tensors to be redistributed.
        device (torch.device): The device on which the output tensors will be placed.
        changed_sharding_params (Dict[str, ParameterSharding]): A dictionary mapping shard names to their new sharding parameters.
        env (ShardingEnv): The sharding environment containing world size and other distributed information.

    Returns:
        Tuple[List[List[str]], torch.Tensor]: A tuple containing:
            - A list of lists where each sublist contains shard names that were sent from a specific rank to the current rank.
            - The tensor containing all shards received by the current rank after the all-to-all operation.
    """
    if env.output_dtensor:
        raise RuntimeError("We do not yet support DTensor for resharding yet")
        return
    world_size = env.world_size
    rank = dist.get_rank()
    input_splits_per_rank = [[0] * world_size for _ in range(world_size)]
    output_splits_per_rank = [[0] * world_size for _ in range(world_size)]
    local_input_tensor = torch.empty([0], device=device)
    local_output_tensor = torch.empty([0], device=device)

    local_output_by_src_rank = [[] for _ in range(world_size)]
    for shard_name, param in changed_sharding_params.items():
        # pyre-ignore
        sharded_t = module._model_parallel_name_to_sharded_tensor[shard_name]
        assert param.ranks is not None
        dst_ranks = param.ranks
        # pyre-ignore
        src_ranks = module.module_sharding_plan[shard_name].ranks

        # TODO: Implement changing rank sizes for beyond TW sharding
        assert len(dst_ranks) == len(src_ranks)
        for i in range(len(src_ranks)):
            dst_rank = dst_ranks[i]
            src_rank = src_ranks[i]
            if dst_rank == rank:  # Limit to only dst_rank is here
                local_output_by_src_rank[src_rank].append(shard_name)

            shard_size = sharded_t.metadata().shards_metadata[i].shard_sizes
            shard_size_dim_0 = shard_size[0]
            input_splits_per_rank[src_rank][dst_rank] += shard_size_dim_0
            output_splits_per_rank[dst_rank][src_rank] += shard_size_dim_0
            if src_rank == rank:
                local_shards = sharded_t.local_shards()
                assert len(local_shards) == 1
                local_input_tensor = torch.cat(
                    (
                        local_input_tensor,
                        sharded_t.local_shards()[0].tensor,
                    )
                )
            if dst_rank == rank:
                local_output_tensor = torch.cat(
                    (local_output_tensor, torch.empty(shard_size, device=device))
                )

    local_input_splits = input_splits_per_rank[rank]
    local_output_splits = output_splits_per_rank[rank]

    assert sum(local_output_splits) == len(local_output_tensor)
    assert sum(local_input_splits) == len(local_input_tensor)
    dist.all_to_all_single(
        output=local_output_tensor,
        input=local_input_tensor,
        output_split_sizes=local_output_splits,
        input_split_sizes=local_input_splits,
        group=dist.group.WORLD,
    )

    return local_output_by_src_rank, local_output_tensor


def update_state_dict_post_resharding(
    update_state_dict: Dict[str, ShardedTensor],
    local_output_by_src_rank: List[List[str]],
    local_output_tensor: torch.Tensor,
    changed_sharding_params: Dict[str, ParameterSharding],
    curr_rank: int,
) -> Dict[str, ShardedTensor]:
    """
    Updates and returns the given state_dict with new placements and
    local_shards based on the output tensor of the AllToAll collective.

    Args:
        update_state_dict (Dict[str, Any]): The state dict to be updated with new shard placements and local shards.

        local_output_by_src_rank (List[List[str]]): A list of len(world_size) containing sublists,
            local_output_by_src_rank[i] contains the sublist of shard_names that were sent from rank i to the current
            rank in order relative to the output tensor.

        local_output_tensor (torch.Tensor): The tensor containing the output data from the AllToAll operation.

        changed_sharding_params (Dict[str, ParameterSharding]): A dictionary mapping shard names to their new sharding parameters.
            This should only contain shard names that were updated during the AllToAll operation.

        curr_rank (int): The current rank of the process in the distributed environment.

    Returns:
        Dict[str, ShardedTensor]: The updated state dictionary with new shard placements and local shards.
    """
    slice_index = 0
    local_output_tensor_order = []
    for i in range(len(local_output_by_src_rank)):
        local_output_tensor_order.extend(local_output_by_src_rank[i])

    shard_name_to_local_output_tensor: Dict[str, torch.Tensor] = {}
    for i in range(len(local_output_tensor_order)):
        shard_name = local_output_tensor_order[i]
        shard_size = update_state_dict[extend_shard_name(shard_name)].size(0)
        end_slice_index = slice_index + shard_size
        shard_name_to_local_output_tensor[shard_name] = local_output_tensor[
            slice_index:end_slice_index
        ]
        slice_index = end_slice_index

    for shard_name, param in changed_sharding_params.items():
        extended_name = extend_shard_name(shard_name)
        # pyre-ignore
        for i in range(len(param.ranks)):
            # pyre-ignore
            r = param.ranks[i]
            # Update placements
            update_state_dict[extended_name].metadata().shards_metadata[i].placement = (
                torch.distributed._remote_device(f"rank:{r}/cuda:{r}")
            )
            if r == curr_rank:
                assert len(local_output_tensor) > 0
                # slice output tensor for correct size.
                update_state_dict[extended_name]._local_shards = [
                    Shard(
                        tensor=shard_name_to_local_output_tensor[shard_name],
                        metadata=update_state_dict[extended_name]
                        .metadata()
                        .shards_metadata[i],
                    )
                ]
                break
            else:
                update_state_dict[extended_name]._local_shards = []

    return update_state_dict
