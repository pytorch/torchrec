#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
from functools import reduce
from typing import Any, cast, Dict, Iterable, List, Optional, Type, Union

import torch
from torchrec.distributed.planner.types import Perf, ShardingOption, Storage
from torchrec.distributed.types import ShardingType

# pyre-ignore[2]
def sharder_name(t: Type[Any]) -> str:
    return t.__module__ + "." + t.__name__


def bytes_to_gb(num_bytes: int) -> float:
    return float(num_bytes / (1024 * 1024 * 1024))


def bytes_to_mb(num_bytes: Union[float, int]) -> float:
    return float(num_bytes / (1024 * 1024))


def gb_to_bytes(gb: float) -> int:
    return int(gb * 1024 * 1024 * 1024)


def prod(iterable: Iterable[int]) -> int:
    return reduce(operator.mul, iterable, 1)


def placement(
    compute_device: str,
    rank: int,
    local_size: int,
) -> str:
    """
    Returns placement, formatted as string
    """

    param_device = compute_device
    if compute_device in {"cuda", "mtia"}:
        param_device = torch.device(compute_device, rank % local_size)
    return f"rank:{rank}/{param_device}"


def storage_repr_in_gb(storage: Optional[Storage]) -> str:
    if storage is None:
        return ""
    return (
        f"Storage(hbm = {round(bytes_to_gb(storage.hbm), 3)} GB, "
        f"ddr = {round(bytes_to_gb(storage.ddr), 3)} GB)"
    )


def reset_shard_rank(proposal: List[ShardingOption]) -> None:
    for sharding_option in proposal:
        for shard in sharding_option.shards:
            shard.rank = None


def _find_imbalance_tables(
    sharding_options: List[ShardingOption], target_imbalance: str = "perf"
) -> List[ShardingOption]:
    """
    Find the tables that are causing the imbalance, and return their names.
    """
    rank_to_target_stats: Dict[int, float] = {}

    # populate rank_to_target_stats
    for sharding_option in sharding_options:
        for shard in sharding_option.shards:
            rank = cast(int, shard.rank)
            if rank not in rank_to_target_stats:
                rank_to_target_stats[rank] = 0

            if target_imbalance == "perf":
                rank_to_target_stats[rank] += cast(Perf, shard.perf).total
            elif target_imbalance == "hbm":
                rank_to_target_stats[rank] += cast(Storage, shard.storage).hbm
            else:
                raise ValueError(f"Unknown target imbalance {target_imbalance}")

    if len(rank_to_target_stats.values()) <= 1:
        # world_size is 1
        return []

    max_value = max(rank_to_target_stats.values())
    max_value_ranks = {
        rank for rank, value in rank_to_target_stats.items() if value == max_value
    }

    # find tables
    tables_in_max_value_ranks: List[ShardingOption] = []
    for sharding_option in sharding_options:
        sharding_option_ranks = [shard.rank for shard in sharding_option.shards]
        if set(
            sharding_option_ranks
        ) >= max_value_ranks and sharding_option.sharding_type not in [
            ShardingType.DATA_PARALLEL.value,
            ShardingType.ROW_WISE.value,
        ]:
            tables_in_max_value_ranks.append(sharding_option)

    if target_imbalance == "perf":
        # sort tables by total perf from largest to smallest
        tables_in_max_value_ranks.sort(
            key=lambda sharding_option: sharding_option.shards[0].perf.total,
            reverse=True,
        )
    elif target_imbalance == "hbm":
        # sort tables by hbm from largest to smallest
        tables_in_max_value_ranks.sort(
            key=lambda sharding_option: sharding_option.shards[0].storage.hbm,
            reverse=True,
        )
    else:
        raise ValueError(f"Unknown target imbalance {target_imbalance}")

    return tables_in_max_value_ranks
