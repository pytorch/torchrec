#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections import defaultdict
from typing import Union, Tuple, Optional, Any, List, Dict, cast

from torchrec.distributed.planner.types import (
    ShardingOption,
    Stats,
    Topology,
    ParameterConstraints,
    Storage,
)
from torchrec.distributed.planner.utils import bytes_to_gb
from torchrec.distributed.types import ShardingType, ParameterSharding, ShardingPlan


logger: logging.Logger = logging.getLogger(__name__)


STATS_DIVIDER = "####################################################################################################"
STATS_BAR = f"#{'------------------------------------------------------------------------------------------------': ^98}#"


class EmbeddingStats(Stats):
    """
    Stats for a sharding planner execution.
    """

    def log(
        self,
        sharding_plan: ShardingPlan,
        topology: Topology,
        num_proposals: int,
        num_plans: int,
        best_plan: List[ShardingOption],
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
    ) -> None:
        """
        Log stats for a given sharding plan to stdout.

        Provide a tabular view of stats for the given sharding plan with per device
        storage usage (HBM and DDR), perf, input (pooling factors), output (embedding
        dimension), and number and type of shards.

        Args:
            sharding_plan (ShardingPlan): sharding plan chosen by the ShardingPlanner.
            topology (Topology): device topology.
            num_proposals (int): number of proposals evaluated
            num_plans (int): number of proposals successfully partitioned
            best_plan (List[ShardingOption]): plan with expected performance
            constraints (Optional[Dict[str, ParameterConstraints]]): dict of parameter
                names to provided ParameterConstraints.
        """

        shard_by_fqn = {
            module_name + "." + param_name: value
            for module_name, param_dict in sharding_plan.plan.items()
            for param_name, value in param_dict.items()
        }
        stats: Dict[int, Dict[str, Any]] = {
            rank: {"type": {}, "pooling_factor": 0.0, "embedding_dims": 0}
            for rank in range(topology.world_size)
        }

        used_sharding_types = set()
        compute_kernels_to_count = defaultdict(int)

        for sharding_option in best_plan:
            fqn = sharding_option.fqn

            if shard_by_fqn.get(fqn) is None:
                continue
            shard: ParameterSharding = shard_by_fqn[fqn]

            ranks, pooling_factor, emb_dims = self._get_shard_stats(
                shard=shard,
                sharding_option=sharding_option,
                world_size=topology.world_size,
                local_size=topology.local_world_size,
                constraints=constraints,
            )
            sharding_type_abbr = _get_sharding_type_abbr(shard.sharding_type)
            used_sharding_types.add(sharding_type_abbr)
            compute_kernels_to_count[sharding_option.compute_kernel] += 1

            for i, rank in enumerate(ranks):
                count = stats[rank]["type"].get(sharding_type_abbr, 0)
                stats[rank]["type"][sharding_type_abbr] = count + 1
                stats[rank]["pooling_factor"] += pooling_factor[i]
                stats[rank]["embedding_dims"] += emb_dims[i]

        used_hbm = [0] * topology.world_size
        used_ddr = [0] * topology.world_size
        perf = [0.0] * topology.world_size
        for sharding_option in best_plan:
            for shard in sharding_option.shards:
                storage = cast(Storage, shard.storage)
                rank = cast(int, shard.rank)
                used_hbm[rank] += storage.hbm
                used_ddr[rank] += storage.ddr
                perf[rank] += cast(float, shard.perf)

        table: List[List[Union[str, int]]] = [
            ["Rank", "HBM (GB)", "DDR (GB)", "Perf", "Input", "Output", "Shards"],
            [
                "------",
                "----------",
                "----------",
                "------",
                "-------",
                "--------",
                "--------",
            ],
        ]

        for rank, device in enumerate(topology.devices):
            used_hbm_gb = bytes_to_gb(used_hbm[rank])
            used_hbm_ratio = (
                used_hbm[rank] / device.storage.hbm
                if topology.compute_device == "cuda"
                else 0
            )
            used_ddr_gb = bytes_to_gb(used_ddr[rank])
            used_ddr_ratio = used_ddr[rank] / device.storage.ddr
            for sharding_type in used_sharding_types:
                if sharding_type not in stats[rank]["type"]:
                    stats[rank]["type"][sharding_type] = 0

            rank_hbm = f"{used_hbm_gb:.1f} ({used_hbm_ratio:.0%})"
            rank_ddr = f"{used_ddr_gb:.1f} ({used_ddr_ratio:.0%})"
            rank_perf = f"{perf[rank] / 1000:,.0f}"
            rank_pooling = f"{int(stats[rank]['pooling_factor']):,}"
            rank_dims = f"{stats[rank]['embedding_dims']:,}"
            rank_shards = " ".join(
                f"{sharding_type}: {num_tables}"
                for sharding_type, num_tables in sorted(stats[rank]["type"].items())
            )
            table.append(
                [
                    rank,
                    rank_hbm,
                    rank_ddr,
                    rank_perf,
                    rank_pooling,
                    rank_dims,
                    rank_shards,
                ]
            )

        logger.info(STATS_DIVIDER)
        header_text = "--- Planner Statistics ---"
        logger.info(f"#{header_text: ^98}#")

        iter_text = (
            f"--- Evalulated {num_proposals} proposal(s), "
            f"found {num_plans} possible plan(s) ---"
        )
        logger.info(f"#{iter_text: ^98}#")
        logger.info(STATS_BAR)

        formatted_table = _format_table(table)
        for row in formatted_table:
            logger.info(f"# {row: <97}#")

        logger.info(f"#{'' : ^98}#")
        legend = "Input: pooling factor, Output: embedding dimension, Shards: number of tables"
        logger.info(f"# {legend: <97}#")
        logger.info(f"#{'' : ^98}#")

        compute_kernels_count = [
            f"{compute_kernel}: {count}"
            for compute_kernel, count in sorted(compute_kernels_to_count.items())
        ]
        logger.info(f"# {'Compute Kernels:' : <97}#")
        for compute_kernel_count in compute_kernels_count:
            logger.info(f"#   {compute_kernel_count : <95}#")

        logger.info(STATS_DIVIDER)

    def _get_shard_stats(
        self,
        shard: ParameterSharding,
        sharding_option: ShardingOption,
        world_size: int,
        local_size: int,
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
    ) -> Tuple[List[int], List[float], List[int]]:
        """
        Gets ranks, pooling factors, and embedding dimensions per shard.

        Returns:
            ranks: list of ranks.
            pooling_factor: list of pooling factors across ranks.
            emb_dims: list of embedding dimensions across ranks.
        """

        ranks = list(range(world_size))
        pooling_factor = [
            sum(constraints[sharding_option.name].pooling_factors)
            if constraints and constraints.get(sharding_option.name)
            else 0.0
        ]
        emb_dims = [sharding_option.tensor.shape[1]]

        if shard.sharding_type == ShardingType.DATA_PARALLEL.value:
            emb_dims = emb_dims * len(ranks)
            pooling_factor = pooling_factor * len(ranks)

        elif shard.sharding_type == ShardingType.TABLE_WISE.value:
            assert shard.ranks
            ranks = shard.ranks

        elif shard.sharding_type == ShardingType.COLUMN_WISE.value:
            assert shard.ranks
            ranks = shard.ranks
            emb_dims = [
                int(shard.shard_sizes[1])
                # pyre-ignore [16]
                for shard in shard.sharding_spec.shards
            ]
            pooling_factor = pooling_factor * len(ranks)

        elif shard.sharding_type == ShardingType.ROW_WISE.value:
            pooling_factor = [pooling_factor[0] / world_size] * len(ranks)
            emb_dims = emb_dims * len(ranks)

        elif shard.sharding_type == ShardingType.TABLE_ROW_WISE.value:
            assert shard.ranks
            host_id = shard.ranks[0] // local_size
            ranks = list(range(host_id * local_size, (host_id + 1) * local_size))
            pooling_factor = [pooling_factor[0] / local_size] * len(ranks)
            emb_dims = emb_dims * len(ranks)

        elif shard.sharding_type == ShardingType.TABLE_COLUMN_WISE.value:
            assert shard.ranks
            ranks = shard.ranks
            pooling_factor = pooling_factor * len(ranks)
            emb_dims = [
                int(shard.shard_sizes[1]) for shard in shard.sharding_spec.shards
            ]

        else:
            raise ValueError(
                f"Unrecognized or unsupported sharding type provided: {shard.sharding_type}"
            )

        return ranks, pooling_factor, emb_dims


def _get_sharding_type_abbr(sharding_type: str) -> str:
    if sharding_type == ShardingType.DATA_PARALLEL.value:
        return "DP"
    elif sharding_type == ShardingType.TABLE_WISE.value:
        return "TW"
    elif sharding_type == ShardingType.COLUMN_WISE.value:
        return "CW"
    elif sharding_type == ShardingType.ROW_WISE.value:
        return "RW"
    elif sharding_type == ShardingType.TABLE_ROW_WISE.value:
        return "TWRW"
    elif sharding_type == ShardingType.TABLE_COLUMN_WISE.value:
        return "TWCW"
    else:
        raise ValueError(
            f"Unrecognized or unsupported sharding type provided: {sharding_type}"
        )


def _format_table(table: List[List[Union[str, int]]]) -> List[str]:
    longest_cols = [
        (max([len(str(row[i])) for row in table]) + 3) for i in range(len(table[0]))
    ]
    row_format = "".join(
        ["{:>" + str(longest_col) + "}" for longest_col in longest_cols]
    )
    return [row_format.format(*row) for row in table]
