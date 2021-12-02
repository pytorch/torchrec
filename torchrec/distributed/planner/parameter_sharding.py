#!/usr/bin/env python3
import abc
import itertools
import math
from typing import List, Tuple, cast

import torch
from torch.distributed._sharding_spec import EnumerableShardingSpec, ShardMetadata
from torchrec.distributed.planner.types import ParameterInfo
from torchrec.distributed.types import ShardingType, ParameterSharding


def _twrw_shard_table_rows(
    table_node: int,
    hash_size: int,
    embedding_dim: int,
    world_size: int,
    local_size: int,
) -> Tuple[List[int], List[int], List[int]]:

    block_size = math.ceil(hash_size / local_size)
    last_rank_offset = hash_size // block_size
    last_block_size = hash_size - block_size * (last_rank_offset)

    first_local_rank = (table_node) * local_size
    last_local_rank = first_local_rank + last_rank_offset
    local_rows: List[int] = []
    local_cols: List[int] = []
    local_row_offsets: List[int] = []
    cumul_row_offset = 0
    for rank in range(world_size):
        local_col = embedding_dim
        if rank < first_local_rank:
            local_row = 0
            local_col = 0
        elif rank < last_local_rank:
            local_row = block_size
        elif rank == last_local_rank:
            local_row = last_block_size
        else:
            local_row = 0
        local_rows.append(local_row)
        local_cols.append(local_col)
        local_row_offsets.append(cumul_row_offset)
        cumul_row_offset += local_row

    return (local_rows, local_cols, local_row_offsets)


def _rw_shard_table_rows(hash_size: int, world_size: int) -> Tuple[List[int], int, int]:
    block_size = (hash_size + world_size - 1) // world_size
    last_rank = hash_size // block_size
    last_block_size = hash_size - block_size * last_rank
    local_rows: List[int] = []
    for rank in range(world_size):
        if rank < last_rank:
            local_row = block_size
        elif rank == last_rank:
            local_row = last_block_size
        else:
            local_row = 0
        local_rows.append(local_row)
    return (local_rows, block_size, last_rank)


def _device_placement(
    compute_device_type: str,
    rank: int,
    local_size: int,
) -> str:
    param_device = torch.device("cpu")
    if compute_device_type == "cuda":
        param_device = torch.device("cuda", rank % local_size)
    return f"rank:{rank}/{param_device}"


class ParameterShardingFactory(abc.ABC):
    @staticmethod
    def shard_parameters(
        param_info: ParameterInfo,
        compute_device_type: str,
        world_size: int,
        local_size: int,
    ) -> ParameterSharding:
        sharding_option = param_info.sharding_options[0]
        sharding_type = sharding_option.sharding_type
        if sharding_type == ShardingType.TABLE_WISE.value:
            parameter_sharding = TwParameterSharding.shard_parameters(
                param_info, compute_device_type, world_size, local_size
            )
        elif sharding_type == ShardingType.ROW_WISE.value:
            parameter_sharding = RwParameterSharding.shard_parameters(
                param_info, compute_device_type, world_size, local_size
            )
        elif sharding_type == ShardingType.TABLE_ROW_WISE.value:
            parameter_sharding = TwRwParameterSharding.shard_parameters(
                param_info, compute_device_type, world_size, local_size
            )
        elif sharding_type == ShardingType.COLUMN_WISE.value:
            parameter_sharding = CwParameterSharding.shard_parameters(
                param_info, compute_device_type, world_size, local_size
            )
        elif sharding_type == ShardingType.DATA_PARALLEL.value:
            parameter_sharding = DpParameterSharding.shard_parameters(
                param_info, compute_device_type, world_size, local_size
            )
        else:
            raise ValueError(
                f"unsupported {sharding_option.sharding_type} sharding type"
            )
        return parameter_sharding


class TwParameterSharding:
    @classmethod
    def shard_parameters(
        cls,
        param_info: ParameterInfo,
        compute_device_type: str,
        world_size: int,
        local_size: int,
    ) -> ParameterSharding:
        sharding_option = param_info.sharding_options[0]
        tensor = param_info.param
        # pyre-fixme [16]
        rank = sharding_option.ranks[0]
        shards = [
            ShardMetadata(
                shard_sizes=[
                    tensor.shape[0],
                    tensor.shape[1],
                ],
                shard_offsets=[0, 0],
                placement=_device_placement(compute_device_type, rank, local_size),
            )
        ]
        return ParameterSharding(
            sharding_spec=EnumerableShardingSpec(shards),
            sharding_type=sharding_option.sharding_type,
            compute_kernel=sharding_option.compute_kernel,
            ranks=sharding_option.ranks,
        )


class RwParameterSharding:
    @classmethod
    def shard_parameters(
        cls,
        param_info: ParameterInfo,
        compute_device_type: str,
        world_size: int,
        local_size: int,
    ) -> ParameterSharding:
        sharding_option = param_info.sharding_options[0]
        tensor = param_info.param
        local_rows, block_size, last_rank = _rw_shard_table_rows(
            tensor.shape[0], world_size
        )
        shards = [
            ShardMetadata(
                shard_sizes=[
                    local_rows[rank],
                    tensor.shape[1],
                ],
                shard_offsets=[block_size * min(rank, last_rank), 0],
                placement=_device_placement(compute_device_type, rank, local_size),
            )
            for rank in range(world_size)
        ]
        return ParameterSharding(
            sharding_type=sharding_option.sharding_type,
            compute_kernel=sharding_option.compute_kernel,
            ranks=sharding_option.ranks,
            sharding_spec=EnumerableShardingSpec(shards),
        )


class TwRwParameterSharding:
    @classmethod
    def shard_parameters(
        cls,
        param_info: ParameterInfo,
        compute_device_type: str,
        world_size: int,
        local_size: int,
    ) -> ParameterSharding:
        sharding_option = param_info.sharding_options[0]
        tensor = param_info.param
        # pyre-fixme [16]
        rank = sharding_option.ranks[0]
        table_node = rank // local_size
        local_rows, local_cols, local_row_offsets = _twrw_shard_table_rows(
            table_node=table_node,
            hash_size=tensor.shape[0],
            embedding_dim=tensor.shape[1],
            world_size=world_size,
            local_size=local_size,
        )
        shards = [
            ShardMetadata(
                shard_sizes=[
                    local_rows[rank],
                    local_cols[rank],
                ],
                shard_offsets=[local_row_offsets[rank], 0],
                placement=_device_placement(compute_device_type, rank, local_size),
            )
            for rank in range(table_node * local_size, (table_node + 1) * local_size)
        ]

        return ParameterSharding(
            sharding_type=sharding_option.sharding_type,
            compute_kernel=sharding_option.compute_kernel,
            ranks=sharding_option.ranks,
            sharding_spec=EnumerableShardingSpec(shards),
        )


class CwParameterSharding:
    @classmethod
    def shard_parameters(
        cls,
        param_info: ParameterInfo,
        compute_device_type: str,
        world_size: int,
        local_size: int,
    ) -> ParameterSharding:
        sharding_option = param_info.sharding_options[0]
        tensor = param_info.param
        # pyre-fixme [6]
        ranks = sorted(sharding_option.ranks)
        block_size = cast(int, sharding_option.col_wise_shard_dim)
        num_col_wise_shards, residual = divmod(tensor.shape[1], block_size)
        sizes = [block_size] * num_col_wise_shards
        if residual > 0:
            sizes += [residual]
        merged_sizes = []
        merged_ranks = []
        for i, rank in enumerate(ranks):
            if rank not in merged_ranks:
                merged_ranks.append(rank)
                merged_sizes.append(sizes[i])
            else:
                merged_sizes[-1] += sizes[i]
        offsets = [0] + list(itertools.accumulate(merged_sizes))[:-1]
        shards = [
            ShardMetadata(
                shard_sizes=[
                    tensor.shape[0],
                    merged_sizes[i],
                ],
                shard_offsets=[0, offsets[i]],
                placement=_device_placement(compute_device_type, rank, local_size),
            )
            for i, rank in enumerate(merged_ranks)
        ]
        return ParameterSharding(
            sharding_type=sharding_option.sharding_type,
            compute_kernel=sharding_option.compute_kernel,
            ranks=merged_ranks,
            sharding_spec=EnumerableShardingSpec(shards),
        )


class DpParameterSharding:
    @classmethod
    def shard_parameters(
        cls,
        param_info: ParameterInfo,
        compute_device_type: str,
        world_size: int,
        local_size: int,
    ) -> ParameterSharding:
        sharding_option = param_info.sharding_options[0]
        return ParameterSharding(
            sharding_type=sharding_option.sharding_type,
            compute_kernel=sharding_option.compute_kernel,
            ranks=sharding_option.ranks,
        )
