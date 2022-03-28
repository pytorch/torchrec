#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections import defaultdict
from typing import Union, Tuple, Optional, Any, List, Dict, cast

from torchrec.distributed.planner.constants import BIGINT_DTYPE
from torchrec.distributed.planner.types import (
    ShardingOption,
    Stats,
    Topology,
    ParameterConstraints,
    Storage,
)
from torchrec.distributed.planner.utils import bytes_to_gb, bytes_to_mb
from torchrec.distributed.types import ShardingType, ParameterSharding, ShardingPlan


logger: logging.Logger = logging.getLogger(__name__)


MIN_WIDTH = 90


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
        debug: bool = False,
    ) -> None:
        """
        Logs stats for a given sharding plan to stdout.

        Provides a tabular view of stats for the given sharding plan with per device
        storage usage (HBM and DDR), perf, input, output, and number/type of shards.

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
            rank: {"type": {}, "input_sizes": 0.0, "output_sizes": 0.0}
            for rank in range(topology.world_size)
        }

        used_sharding_types = set()
        compute_kernels_to_count = defaultdict(int)

        for sharding_option in best_plan:
            fqn = sharding_option.fqn

            if shard_by_fqn.get(fqn) is None:
                continue
            shard: ParameterSharding = shard_by_fqn[fqn]

            ranks, input_sizes, output_sizes = self._get_shard_stats(
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
                stats[rank]["input_sizes"] += input_sizes[i]
                stats[rank]["output_sizes"] += output_sizes[i]

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
            [
                "Rank",
                "HBM (GB)",
                "DDR (GB)",
                "Perf (ms)",
                "Input (MB)",
                "Output (MB)",
                "Shards",
            ],
            [
                "------",
                "----------",
                "----------",
                "-----------",
                "------------",
                "-------------",
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

            rank_hbm = f"{round(used_hbm_gb, 1)} ({used_hbm_ratio:.0%})"
            rank_ddr = f"{round(used_ddr_gb, 1)} ({used_ddr_ratio:.0%})"
            rank_perf = f"{round(perf[rank], 2)}"
            rank_input = f"{round(stats[rank]['input_sizes'], 2)}"
            rank_output = f"{round(stats[rank]['output_sizes'], 2)}"
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
                    rank_input,
                    rank_output,
                    rank_shards,
                ]
            )
        formatted_table = _format_table(table)
        width = max(MIN_WIDTH, len(formatted_table[0]) + 8)

        if debug:
            param_table: List[List[Union[str, int]]] = [
                ["FQN", "Sharding", "Compute Kernel", "Perf (ms)", "Ranks"],
                [
                    "-----",
                    "----------",
                    "----------------",
                    "-----------",
                    "-------",
                ],
            ]
            for so in best_plan:
                # pyre-ignore[6]
                ranks = sorted([shard.rank for shard in so.shards])
                if len(ranks) > 1 and ranks == list(range(min(ranks), max(ranks) + 1)):
                    ranks = [f"{min(ranks)}-{max(ranks)}"]
                shard_perfs = str(
                    round(sum([cast(float, shard.perf) for shard in so.shards]), 2)
                )
                param_table.append(
                    [
                        so.fqn,
                        _get_sharding_type_abbr(so.sharding_type),
                        so.compute_kernel,
                        shard_perfs,
                        ",".join([str(rank) for rank in ranks]),
                    ]
                )
            formatted_param_table = _format_table(param_table)
            width = max(width, len(formatted_param_table[0]) + 6)

        logger.info("#" * width)
        header_text = "--- Planner Statistics ---"
        logger.info(f"#{header_text: ^{width-2}}#")

        iter_text = (
            f"--- Evalulated {num_proposals} proposal(s), "
            f"found {num_plans} possible plan(s) ---"
        )
        logger.info(f"#{iter_text: ^{width-2}}#")

        divider = "-" * (width - 4)
        logger.info(f"#{divider: ^{width-2}}#")

        for row in formatted_table:
            logger.info(f"# {row: <{width-3}}#")

        logger.info(f"#{'' : ^{width-2}}#")
        legend = "Input: MB/iteration, Output: MB/iteration, Shards: number of tables"
        logger.info(f"# {legend: <{width-3}}#")
        hbm_info = "HBM: est. peak memory usage for shards - parameter, comms, optimizer, and gradients"
        logger.info(f"# {hbm_info: <{width-3}}#")
        logger.info(f"#{'' : ^{width-2}}#")

        compute_kernels_count = [
            f"{compute_kernel}: {count}"
            for compute_kernel, count in sorted(compute_kernels_to_count.items())
        ]
        logger.info(f"# {'Compute Kernels:' : <{width-3}}#")
        for compute_kernel_count in compute_kernels_count:
            logger.info(f"#   {compute_kernel_count : <{width-5}}#")

        if debug:
            logger.info(f"#{'' : ^{width-2}}#")
            logger.info(f"# {'Parameter Info:' : <{width-3}}#")

            for row in formatted_param_table:
                logger.info(f"# {row: <{width-3}}#")

        logger.info("#" * width)

    def _get_shard_stats(
        self,
        shard: ParameterSharding,
        sharding_option: ShardingOption,
        world_size: int,
        local_size: int,
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
    ) -> Tuple[List[int], List[float], List[float]]:
        """
        Gets ranks, input sizes, and output sizes per shard.
        Input size is a function of pooling factor.
        Output size is a function of embedding dimension * number of features.

        Returns:
            ranks: list of ranks.
            input_sizes: input size per iter in MB across ranks for given shard.
            output_sizes: output size per iter in MB across ranks for given shard.
        """
        assert shard.ranks
        ranks = shard.ranks

        batch_size = world_size * sharding_option.batch_size
        input_data_type_size = BIGINT_DTYPE
        pooling_factor = (
            float(sum(constraints[sharding_option.name].pooling_factors))
            if constraints and constraints.get(sharding_option.name)
            else 1.0
        )
        num_features = len(sharding_option.input_lengths)
        output_data_type_size = sharding_option.tensor.element_size()
        num_outputs = 1  # for pooled embeddings

        if shard.sharding_type == ShardingType.DATA_PARALLEL.value:
            batch_size = sharding_option.batch_size
        elif shard.sharding_type == ShardingType.ROW_WISE.value:
            pooling_factor /= world_size
        elif shard.sharding_type == ShardingType.TABLE_ROW_WISE.value:
            pooling_factor /= local_size

        input_sizes = [
            bytes_to_mb(batch_size * pooling_factor * input_data_type_size)
        ] * len(ranks)
        output_sizes = (
            [
                bytes_to_mb(
                    batch_size
                    * num_outputs
                    * sharding_option.tensor.shape[1]  # embedding dim
                    * num_features
                    * output_data_type_size
                )
            ]
            * len(ranks)
            if shard.sharding_type == ShardingType.DATA_PARALLEL.value
            else [
                bytes_to_mb(
                    batch_size
                    * num_outputs
                    * int(shard.shard_sizes[1])  # embedding dim
                    * num_features
                    * output_data_type_size
                )
                # pyre-ignore [16]
                for shard in shard.sharding_spec.shards
            ]
        )
        return ranks, input_sizes, output_sizes


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
