#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections import defaultdict
from typing import Any, cast, Dict, List, Optional, Tuple, Union

from torchrec.distributed.planner.constants import BIGINT_DTYPE
from torchrec.distributed.planner.shard_estimators import _calculate_shard_io_sizes
from torchrec.distributed.planner.storage_reservations import (
    FixedPercentageStorageReservation,
    HeuristicalStorageReservation,
    InferenceStorageReservation,
)
from torchrec.distributed.planner.types import (
    ParameterConstraints,
    ShardingOption,
    Stats,
    Storage,
    StorageReservation,
    Topology,
)
from torchrec.distributed.planner.utils import bytes_to_gb, bytes_to_mb
from torchrec.distributed.types import ParameterSharding, ShardingPlan, ShardingType


logger: logging.Logger = logging.getLogger(__name__)


MIN_WIDTH = 90


class EmbeddingStats(Stats):
    """
    Stats for a sharding planner execution.
    """

    def __init__(self) -> None:
        self._width: int = MIN_WIDTH
        self._stats_table: List[str] = []

    def log(
        self,
        sharding_plan: ShardingPlan,
        topology: Topology,
        batch_size: int,
        storage_reservation: StorageReservation,
        num_proposals: int,
        num_plans: int,
        run_time: float,
        best_plan: List[ShardingOption],
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
        debug: bool = True,
    ) -> None:
        """
        Logs stats for a given sharding plan.

        Provides a tabular view of stats for the given sharding plan with per device
        storage usage (HBM and DDR), perf, input, output, and number/type of shards.

        Args:
            sharding_plan (ShardingPlan): sharding plan chosen by the planner.
            topology (Topology): device topology.
            batch_size (int): batch size.
            storage_reservation (StorageReservation): reserves storage for unsharded
                parts of the model
            num_proposals (int): number of proposals evaluated.
            num_plans (int): number of proposals successfully partitioned.
            run_time (float): time taken to find plan (in seconds).
            best_plan (List[ShardingOption]): plan with expected performance.
            constraints (Optional[Dict[str, ParameterConstraints]]): dict of parameter
                names to provided ParameterConstraints.
            debug (bool): whether to enable debug mode.
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

        reserved_hbm_percent = (
            storage_reservation._percentage
            if isinstance(
                storage_reservation,
                (
                    FixedPercentageStorageReservation,
                    HeuristicalStorageReservation,
                    InferenceStorageReservation,
                ),
            )
            else 0.0
        )
        dense_storage = (
            storage_reservation._dense_storage
            if isinstance(
                storage_reservation,
                (HeuristicalStorageReservation, InferenceStorageReservation),
            )
            and storage_reservation._dense_storage is not None
            else Storage(0, 0)
        )
        assert dense_storage
        kjt_storage = (
            storage_reservation._kjt_storage
            if isinstance(
                storage_reservation,
                (HeuristicalStorageReservation, InferenceStorageReservation),
            )
            and storage_reservation._kjt_storage
            else Storage(0, 0)
        )
        assert kjt_storage

        for sharding_option in best_plan:
            fqn = sharding_option.fqn

            if shard_by_fqn.get(fqn) is None:
                continue
            shard: ParameterSharding = shard_by_fqn[fqn]

            ranks, input_sizes, output_sizes = self._get_shard_stats(
                shard=shard,
                sharding_option=sharding_option,
                world_size=topology.world_size,
                local_world_size=topology.local_world_size,
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
                shard_storage = cast(Storage, shard.storage)
                rank = cast(int, shard.rank)
                used_hbm[rank] += shard_storage.hbm
                used_ddr[rank] += shard_storage.ddr
                perf[rank] += cast(float, shard.perf)

        used_hbm = [hbm + dense_storage.hbm + kjt_storage.hbm for hbm in used_hbm]
        used_ddr = [ddr + dense_storage.ddr + kjt_storage.ddr for ddr in used_ddr]

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
                used_hbm[rank] / ((1 - reserved_hbm_percent) * device.storage.hbm)
                if topology.compute_device == "cuda"
                else 0
            )
            used_ddr_gb = bytes_to_gb(used_ddr[rank])
            used_ddr_ratio = (
                used_ddr[rank] / device.storage.ddr if device.storage.ddr > 0 else 0
            )
            for sharding_type in used_sharding_types:
                if sharding_type not in stats[rank]["type"]:
                    stats[rank]["type"][sharding_type] = 0

            rank_hbm = f"{round(used_hbm_gb, 1)} ({used_hbm_ratio:.0%})"
            rank_ddr = f"{round(used_ddr_gb, 1)} ({used_ddr_ratio:.0%})"
            rank_perf = f"{round(perf[rank], 3)}"
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
        self._width = max(self._width, len(formatted_table[0]) + 8)

        if debug:
            param_table: List[List[Union[str, int]]] = [
                [
                    "FQN",
                    "Sharding",
                    "Compute Kernel",
                    "Perf (ms)",
                    "Pooling Factor",
                    "Output",
                    "Features",
                    "Emb Dim",
                    "Hash Size",
                    "Ranks",
                ],
                [
                    "-----",
                    "----------",
                    "----------------",
                    "-----------",
                    "----------------",
                    "--------",
                    "----------",
                    "--------",
                    "-----------",
                    "-------",
                ],
            ]
            for so in best_plan:
                ranks = sorted([cast(int, shard.rank) for shard in so.shards])
                ranks = _collapse_consecutive_ranks(ranks)
                shard_perfs = str(
                    round(sum([cast(float, shard.perf) for shard in so.shards]), 3)
                )
                pooling_factor = str(round(sum(so.input_lengths), 3))
                output = "pooled" if so.is_pooled else "sequence"
                num_features = len(so.input_lengths)
                embedding_dim = so.tensor.shape[1]
                hash_size = so.tensor.shape[0]
                param_table.append(
                    [
                        so.fqn,
                        _get_sharding_type_abbr(so.sharding_type),
                        so.compute_kernel,
                        shard_perfs,
                        pooling_factor,
                        output,
                        num_features,
                        embedding_dim,
                        hash_size,
                        ",".join(ranks),
                    ]
                )
            formatted_param_table = _format_table(param_table)
            self._width = max(self._width, len(formatted_param_table[0]) + 6)

        self._stats_table.clear()
        self._stats_table.append("#" * self._width)
        header_text = "--- Planner Statistics ---"
        self._stats_table.append(f"#{header_text: ^{self._width-2}}#")

        iter_text = (
            f"--- Evaluated {num_proposals} proposal(s), "
            f"found {num_plans} possible plan(s), "
            f"ran for {run_time:.2f}s ---"
        )
        self._stats_table.append(f"#{iter_text: ^{self._width-2}}#")

        divider = "-" * (self._width - 4)
        self._stats_table.append(f"#{divider: ^{self._width-2}}#")

        for row in formatted_table:
            self._stats_table.append(f"# {row: <{self._width-3}}#")

        legend = "Input: MB/iteration, Output: MB/iteration, Shards: number of tables"
        hbm_info = "HBM: estimated peak memory usage for shards, dense tensors, and features (KJT)"
        self._stats_table.append(f"#{'' : ^{self._width-2}}#")
        self._stats_table.append(f"# {legend: <{self._width-3}}#")
        self._stats_table.append(f"# {hbm_info: <{self._width-3}}#")

        if debug:
            self._stats_table.append(f"#{'' : ^{self._width-2}}#")
            self._stats_table.append(f"# {'Parameter Info:' : <{self._width-3}}#")
            for row in formatted_param_table:
                self._stats_table.append(f"# {row: <{self._width-3}}#")

        batch_size_text = f"Batch Size: {batch_size}"
        self._stats_table.append(f"#{'' : ^{self._width-2}}#")
        self._stats_table.append(f"# {batch_size_text : <{self._width-3}}#")

        self._log_compute_kernel_stats(compute_kernels_to_count)

        if debug:
            self._log_max_perf_and_max_hbm(perf, used_hbm)
            self._log_storage_reservation_stats(
                storage_reservation,
                topology,
                reserved_hbm_percent,
                dense_storage,
                kjt_storage,
            )

        self._stats_table.append("#" * self._width)

        for row in self._stats_table:
            logger.info(row)

    def _get_shard_stats(
        self,
        shard: ParameterSharding,
        sharding_option: ShardingOption,
        world_size: int,
        local_world_size: int,
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

        num_poolings = (
            cast(List[float], constraints[sharding_option.name].num_poolings)
            if constraints
            and constraints.get(sharding_option.name)
            and constraints[sharding_option.name].num_poolings
            else [1.0] * sharding_option.num_inputs
        )
        batch_sizes = (
            cast(List[int], constraints[sharding_option.name].batch_sizes)
            if constraints
            and constraints.get(sharding_option.name)
            and constraints[sharding_option.name].batch_sizes
            else [sharding_option.batch_size] * sharding_option.num_inputs
        )
        input_data_type_size = BIGINT_DTYPE
        output_data_type_size = sharding_option.tensor.element_size()

        input_sizes, output_sizes = _calculate_shard_io_sizes(
            sharding_type=sharding_option.sharding_type,
            batch_sizes=batch_sizes,
            world_size=world_size,
            local_world_size=local_world_size,
            input_lengths=sharding_option.input_lengths,
            emb_dim=sharding_option.tensor.shape[1],
            shard_sizes=[shard.size for shard in sharding_option.shards],
            input_data_type_size=input_data_type_size,
            output_data_type_size=output_data_type_size,
            num_poolings=num_poolings,
            is_pooled=sharding_option.is_pooled,
        )

        input_sizes = [bytes_to_mb(input_size) for input_size in input_sizes]
        output_sizes = [bytes_to_mb(output_size) for output_size in output_sizes]

        return ranks, input_sizes, output_sizes

    def _log_max_perf_and_max_hbm(self, perf: List[float], used_hbm: List[int]) -> None:
        max_perf = max(perf)
        max_perf_indices = [i for i in range(len(perf)) if perf[i] == max_perf]
        rank_text = "ranks" if len(max_perf_indices) > 1 else "rank"
        max_perf_indices = _collapse_consecutive_ranks(max_perf_indices)
        max_perf_ranks = f"{rank_text} {','.join(max_perf_indices)}"
        longest_critical_path = (
            f"Longest Critical Path: {round(max_perf, 3)} ms on {max_perf_ranks}"
        )
        self._stats_table.append(f"#{'' : ^{self._width-2}}#")
        self._stats_table.append(f"# {longest_critical_path : <{self._width-3}}#")

        max_hbm = max(used_hbm)
        max_hbm_indices = [i for i in range(len(used_hbm)) if used_hbm[i] == max_hbm]
        rank_text = "ranks" if len(max_hbm_indices) > 1 else "rank"
        max_hbm_indices = _collapse_consecutive_ranks(max_hbm_indices)
        max_hbm_ranks = f"{rank_text} {','.join(max_hbm_indices)}"
        peak_memory_pressure = f"Peak Memory Pressure: {round(bytes_to_gb(max_hbm), 3)} GB on {max_hbm_ranks}"
        self._stats_table.append(f"#{'' : ^{self._width-2}}#")
        self._stats_table.append(f"# {peak_memory_pressure : <{self._width-3}}#")

    def _log_storage_reservation_stats(
        self,
        storage_reservation: StorageReservation,
        topology: Topology,
        reserved_hbm_percent: float,
        dense_storage: Storage,
        kjt_storage: Storage,
    ) -> None:
        device_storage = topology.devices[0].storage
        usable_hbm = round(
            bytes_to_gb(int((1 - reserved_hbm_percent) * device_storage.hbm)), 3
        )
        usable_ddr = round(bytes_to_gb(int(device_storage.ddr)), 3)
        usable_memory = f"HBM: {usable_hbm} GB, DDR: {usable_ddr} GB"
        usable_hbm_percentage = (
            f"Percent of Total HBM: {(1 - reserved_hbm_percent):.0%}"
        )
        self._stats_table.append(f"#{'' : ^{self._width-2}}#")
        self._stats_table.append(f"# {'Usable Memory:' : <{self._width-3}}#")
        self._stats_table.append(f"#    {usable_memory : <{self._width-6}}#")
        self._stats_table.append(f"#    {usable_hbm_percentage : <{self._width-6}}#")

        if isinstance(storage_reservation, HeuristicalStorageReservation):
            dense_hbm = round(bytes_to_gb(dense_storage.hbm), 3)
            dense_ddr = round(bytes_to_gb(dense_storage.ddr), 3)
            dense_storage_text = f"HBM: {dense_hbm} GB, DDR: {dense_ddr} GB"
            self._stats_table.append(f"#{'' : ^{self._width-2}}#")
            self._stats_table.append(
                f"# {'Dense Storage (per rank): ' : <{self._width-3}}#"
            )
            self._stats_table.append(f"#    {dense_storage_text : <{self._width-6}}#")

            kjt_hbm = round(bytes_to_gb(kjt_storage.hbm), 3)
            kjt_ddr = round(bytes_to_gb(kjt_storage.ddr), 3)
            kjt_storage_text = f"HBM: {kjt_hbm} GB, DDR: {kjt_ddr} GB"
            self._stats_table.append(f"#{'' : ^{self._width-2}}#")
            self._stats_table.append(
                f"# {'KJT Storage (per rank): ' : <{self._width-3}}#"
            )
            self._stats_table.append(f"#    {kjt_storage_text : <{self._width-6}}#")

    def _log_compute_kernel_stats(
        self, compute_kernels_to_count: Dict[str, int]
    ) -> None:
        compute_kernels_count = [
            f"{compute_kernel}: {count}"
            for compute_kernel, count in sorted(compute_kernels_to_count.items())
        ]
        self._stats_table.append(f"#{'' : ^{self._width-2}}#")
        self._stats_table.append(f"# {'Compute Kernels:' : <{self._width-3}}#")
        for compute_kernel_count in compute_kernels_count:
            self._stats_table.append(f"#    {compute_kernel_count : <{self._width-6}}#")


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


def _collapse_consecutive_ranks(ranks: List[int]) -> List[str]:
    if len(ranks) > 1 and ranks == list(range(min(ranks), max(ranks) + 1)):
        return [f"{min(ranks)}-{max(ranks)}"]
    else:
        return [str(rank) for rank in ranks]
