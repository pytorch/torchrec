#!/usr/bin/env python3

import logging
from typing import Tuple, Optional, Any, List, Dict

from tabulate import tabulate
from torchrec.distributed.planner.new.types import (
    PlacerStats,
    ShardingOption,
    Stats,
    Topology,
    InputStats,
)
from torchrec.distributed.planner.utils import bytes_to_gb, bytes_to_tb
from torchrec.distributed.types import ShardingType, ParameterSharding, ShardingPlan


logger: logging.Logger = logging.getLogger(__name__)


STATS_DIVIDER = "####################################################################################################"
STATS_BAR = f"#{'------------------------------------------------------------------------------------------------': ^98}#"


class EmbeddingShardingStats(Stats):
    def run(
        self,
        sharding_plan: ShardingPlan,
        topology: Topology,
        placer_stats: PlacerStats,
        input_stats: Optional[Dict[str, InputStats]] = None,
    ) -> None:
        shard_by_fqn = {
            module_name + "." + param_name: value
            for module_name, param_dict in sharding_plan.plan.items()
            for param_name, value in param_dict.items()
        }
        stats: Dict[int, Dict[str, Any]] = {
            rank: {"type": {}, "pooling_factor": 0.0, "embedding_dims": 0}
            for rank in range(topology.world_size)
        }
        sharding_solution = (
            placer_stats.sharding_solution if placer_stats.sharding_solution else []
        )
        if not placer_stats.topology_solution:
            return
        topology_solution = placer_stats.topology_solution

        for sharding_option in sharding_solution:
            fqn = sharding_option.fqn
            if shard_by_fqn.get(fqn) is None:
                continue
            shard: ParameterSharding = shard_by_fqn[fqn]

            ranks, pooling_factor, emb_dims = self._get_shard_stats(
                shard=shard,
                sharding_option=sharding_option,
                world_size=topology.world_size,
                local_size=topology.local_world_size,
                input_stats=input_stats,
            )
            sharding_type_abbr = _get_sharding_type_abbr(shard.sharding_type)

            for i, rank in enumerate(ranks):
                count = stats[rank]["type"].get(sharding_type_abbr, 0)
                stats[rank]["type"][sharding_type_abbr] = count + 1
                stats[rank]["pooling_factor"] += pooling_factor[i]
                stats[rank]["embedding_dims"] += emb_dims[i]

        logger.info(STATS_DIVIDER)
        header_text = "--- Planner Statistics ---"
        logger.info(f"#{header_text: ^98}#")

        num_iterations = placer_stats.num_iterations
        num_errors = placer_stats.num_errors
        iter_text = (
            f"--- Ran {num_iterations} iteration(s), "
            f"found {num_iterations - num_errors} possible plan(s) ---"
        )
        logger.info(f"#{iter_text: ^98}#")

        logger.info(STATS_BAR)

        headers = ["Rank", "HBM (GB)", "DDR (TB)", "Cost", "Input", "Output", "Shards"]
        table = []
        for rank, (initial_device, solution_device) in enumerate(
            zip(topology.devices, topology_solution.devices)
        ):
            used_hbm = bytes_to_gb(
                initial_device.storage.hbm - solution_device.storage.hbm
            )
            used_hbm_ratio = (
                1 - solution_device.storage.hbm / initial_device.storage.hbm
            )
            used_ddr = bytes_to_tb(
                initial_device.storage.ddr - solution_device.storage.ddr
            )
            used_ddr_ratio = (
                1 - solution_device.storage.ddr / initial_device.storage.ddr
            )

            hbm = f"{used_hbm:.1f} ({used_hbm_ratio:.0%})"
            ddr = f"{used_ddr:.1f} ({used_ddr_ratio:.0%})"
            cost = f"{solution_device.cost / 1000:,.0f}"
            pooling = f"{int(stats[rank]['pooling_factor']):,}"
            dims = f"{stats[rank]['embedding_dims']:,}"
            shards = " ".join(
                f"{sharding_type}: {num_tables}"
                for sharding_type, num_tables in stats[rank]["type"].items()
            )
            table.append([rank, hbm, ddr, cost, pooling, dims, shards])

        table = tabulate(table, headers=headers).split("\n")
        for row in table:
            logger.info(f"# {row: <97}#")

        logger.info(STATS_DIVIDER)

    def _get_shard_stats(
        self,
        shard: ParameterSharding,
        sharding_option: ShardingOption,
        world_size: int,
        local_size: int,
        input_stats: Optional[Dict[str, InputStats]] = None,
    ) -> Tuple[List[int], List[float], List[int]]:
        """
        Gets ranks, pooling factors, and embedding dimensions per shard

        Returns:
            ranks: list of ranks
            pooling_factor: list of mean pooling factor
            emb_dims: list of embedding dimensions
        """
        ranks = list(range(world_size))
        pooling_factor = [
            sum(input_stats[sharding_option.fqn].pooling_factors)
            if input_stats and input_stats.get(sharding_option.fqn)
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
                int(shard.shard_lengths[1])
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
    else:
        raise ValueError(f"Unrecognized sharding type provided: {sharding_type}")
