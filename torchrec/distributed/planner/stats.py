#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import logging
import math
import statistics
from collections import defaultdict
from typing import Any, Callable, cast, Dict, Iterable, List, Optional, Tuple, Union

from torch import nn
from torchrec.distributed.planner.constants import BIGINT_DTYPE, NUM_POOLINGS
from torchrec.distributed.planner.shard_estimators import _calculate_shard_io_sizes
from torchrec.distributed.planner.storage_reservations import (
    FixedPercentageStorageReservation,
    HeuristicalStorageReservation,
    InferenceStorageReservation,
)
from torchrec.distributed.planner.types import (
    ParameterConstraints,
    Perf,
    ShardingOption,
    Stats,
    Storage,
    StorageReservation,
    Topology,
)
from torchrec.distributed.planner.utils import (
    _find_imbalance_tables,
    bytes_to_gb,
    bytes_to_mb,
    sharder_name as get_sharder_name,
)
from torchrec.distributed.types import (
    ModuleSharder,
    ParameterSharding,
    ShardingPlan,
    ShardingType,
)
from torchrec.distributed.utils import none_throws

logger: logging.Logger = logging.getLogger(__name__)


MIN_WIDTH = 90


def _normalize_float(p: List[float]) -> List[float]:
    p_total = sum(p)
    return [p_i / p_total for p_i in p]


def _normalize_int(p: List[int]) -> List[float]:
    p_total = sum(p)
    return [p_i * 1.0 / p_total for p_i in p]


def _total_variation(p: List[float]) -> float:
    k = len(p)
    if not k:
        return -1.0
    return max(abs(pi - 1.0 / k) for pi in p)


def _total_distance(p: List[float]) -> float:
    k = len(p)
    if not k:
        return -1.0
    return sum(abs(pi - 1.0 / k) for pi in p)


def _chi_divergence(p: List[float], alpha: float = 1.0) -> float:
    assert alpha >= 1
    k = len(p)
    if not k:
        return -1.0
    return sum(abs(pi - 1.0 / k) ** alpha * k ** (alpha - 1.0) for pi in p)


def _kl_divergence(p: List[float]) -> float:
    k = len(p)
    if not k:
        return -1.0
    return sum(pi * math.log(k * pi) for pi in p if pi > 0)


def _calc_max_chi_divergence(N: int, alpha: float) -> float:
    assert N > 0
    # Upper bound for chi divergence in our case given sample size of distribution (N) and alpha
    return (N - 1) ** alpha * (1.0 / N) + (N - 1) * (1.0 / N)


def _calc_max_kl_divergence(N: int) -> float:
    assert N > 0
    # Upper bound for KL divergence in our case given sample size of distribution (N)
    return math.log(N)


CHI_DIVERGENCE_ALPHA = 1.0

IMBALANCE_STAT_MEASURE: Dict[str, Tuple[Callable[..., float], Dict[str, Any]]] = {
    "Total Variation": (_total_variation, {}),
    "Total Distance": (_total_distance, {}),
    "Chi Divergence": (_chi_divergence, {"alpha": CHI_DIVERGENCE_ALPHA}),
    "KL Divergence": (_kl_divergence, {}),
}


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
        sharders: Optional[List[ModuleSharder[nn.Module]]] = None,
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
            # pyre-ignore - this is a EmbeddingShardingPlan below
            for param_name, value in param_dict.items()
        }
        stats: Dict[int, Dict[str, Any]] = {
            rank: {"type": {}, "input_sizes": 0.0, "output_sizes": 0.0}
            for rank in topology.device_ranks
        }

        used_sharding_types = set()
        compute_kernels_to_count = defaultdict(int)
        compute_kernels_to_storage = defaultdict(lambda: Storage(0, 0))

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

            compute_kernels_to_count[sharding_option.compute_kernel] += 1
            compute_kernels_to_storage[
                sharding_option.compute_kernel
            ] += sharding_option.total_storage

            # for shard in sharding_option.shards:
            # compute_kernels_to_storage[sharding_option.compute_kernel] += shard.hbm

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

            for i, rank in enumerate(ranks):
                count = stats[rank]["type"].get(sharding_type_abbr, 0)
                stats[rank]["type"][sharding_type_abbr] = count + 1
                stats[rank]["input_sizes"] += input_sizes[i]
                stats[rank]["output_sizes"] += output_sizes[i]

        used_hbm_and_ddr: Dict[int, Storage] = {
            rank: Storage(hbm=0, ddr=0) for rank in topology.device_ranks
        }
        perf: Dict[int, Perf] = {
            rank: Perf(fwd_compute=0, fwd_comms=0, bwd_compute=0, bwd_comms=0)
            for rank in topology.device_ranks
        }
        for sharding_option in best_plan:
            for shard in sharding_option.shards:
                shard_storage = cast(Storage, shard.storage)
                rank = shard.rank
                if rank == -1:
                    continue
                assert (
                    rank is not None and rank in used_hbm_and_ddr
                ), f"Unexpected rank {rank} in plan {best_plan}"
                mem = cast(Storage, used_hbm_and_ddr.get(rank))
                mem.hbm += shard_storage.hbm
                mem.ddr += shard_storage.ddr
                perf[rank] += cast(Perf, shard.perf)

        # add dense and kjt storage
        for mem in used_hbm_and_ddr.values():
            mem.hbm += dense_storage.hbm + kjt_storage.hbm
            mem.ddr += dense_storage.ddr + kjt_storage.ddr

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

        for device in topology.devices:
            rank = device.rank
            assert (
                rank in used_hbm_and_ddr
            ), f"Unexpected rank: {rank} is not in {list(used_hbm_and_ddr.keys())}"
            used_hbm_bytes = none_throws(used_hbm_and_ddr.get(rank)).hbm
            used_ddr_bytes = none_throws(used_hbm_and_ddr.get(rank)).ddr
            used_hbm_gb = bytes_to_gb(used_hbm_bytes)
            used_hbm_ratio = (
                used_hbm_bytes / ((1 - reserved_hbm_percent) * device.storage.hbm)
                if topology.compute_device == "cuda"
                and ((1 - reserved_hbm_percent) * device.storage.hbm) != 0
                else 0
            )
            used_ddr_gb = bytes_to_gb(used_ddr_bytes)
            used_ddr_ratio = (
                used_ddr_bytes / device.storage.ddr if device.storage.ddr > 0 else 0
            )
            for sharding_type in used_sharding_types:
                if sharding_type not in stats[rank]["type"]:
                    stats[rank]["type"][sharding_type] = 0

            rank_hbm = f"{round(used_hbm_gb, 3)} ({used_hbm_ratio:.0%})"
            rank_ddr = f"{round(used_ddr_gb, 3)} ({used_ddr_ratio:.0%})"
            rank_perf = _format_perf_breakdown(perf[rank])
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
                    "Storage (HBM, DDR)",
                    "Cache Load Factor",
                    "Pooling Factor",
                    "Num Poolings",
                    "Output",
                    "Weighing",
                    "Sharder",
                    "Features",
                    "Emb Dim (CW Dim)",
                    "Hash Size",
                    "Ranks",
                ],
                [
                    "-----",
                    "----------",
                    "----------------",
                    "-----------",
                    "--------------------",
                    "-------------------",
                    "----------------",
                    "--------------",
                    "--------",
                    "----------",
                    "---------",
                    "----------",
                    "------------------",
                    "-----------",
                    "-------",
                ],
            ]
            feat_batch_sizes = [
                (
                    constraints[so.name].batch_sizes
                    if constraints and constraints.get(so.name)
                    else None
                )
                for so in best_plan
            ]

            sharder_map: Dict[str, ModuleSharder[nn.Module]] = {
                get_sharder_name(sharder.module_type): sharder
                # pyre-ignore - this is a ModuleSharder below
                for sharder in sharders
                if sharders
            }

            if include_batch_sizes := any(feat_batch_sizes):
                param_table[0].append("Batch Sizes")
                param_table[1].append("-------------")
            for i, so in enumerate(best_plan):
                ranks = sorted([cast(int, shard.rank) for shard in so.shards])
                ranks = _collapse_consecutive_ranks(ranks)

                so_perf = Perf(fwd_compute=0, fwd_comms=0, bwd_compute=0, bwd_comms=0)
                for shard in so.shards:
                    so_perf += cast(Perf, shard.perf)

                shard_perfs = _format_perf_breakdown(so_perf)

                so_storage = Storage(hbm=0, ddr=0)
                for shard in so.shards:
                    so_storage += cast(Storage, shard.storage)

                shard_storages = _format_storage_breakdown(so_storage)

                pooling_factor = str(round(sum(so.input_lengths), 3))
                num_poolings = (
                    cast(List[float], constraints[so.name].num_poolings)
                    if constraints
                    and constraints.get(so.name)
                    and constraints[so.name].num_poolings
                    else [NUM_POOLINGS] * len(so.input_lengths)
                )
                num_poolings = str(round(sum(num_poolings), 3))
                output = "pooled" if so.is_pooled else "sequence"
                weighing = "weighted" if so.is_weighted else "unweighted"
                sharder = sharder_map.get(get_sharder_name(type(so.module[1])), None)
                sharder_name = type(sharder).__name__
                num_features = len(so.input_lengths)
                embedding_dim = (
                    f"{so.tensor.shape[1]} ({so.shards[0].size[1]})"
                    if so.sharding_type == ShardingType.COLUMN_WISE.value
                    or so.sharding_type == ShardingType.TABLE_COLUMN_WISE.value
                    else f"{so.tensor.shape[1]}"
                )
                sharder_cache_load_factor = (
                    sharder.fused_params.get("cache_load_factor")  # pyre-ignore[16]
                    if hasattr(sharder, "fused_params") and sharder.fused_params
                    else None
                )
                cache_load_factor = str(
                    so.cache_load_factor
                    if so.cache_load_factor is not None
                    else sharder_cache_load_factor
                )
                hash_size = so.tensor.shape[0]
                param_table.append(
                    [
                        so.fqn,
                        _get_sharding_type_abbr(so.sharding_type),
                        so.compute_kernel,
                        shard_perfs,
                        shard_storages,
                        cache_load_factor,
                        pooling_factor,
                        num_poolings,
                        output,
                        weighing,
                        sharder_name,
                        num_features,
                        embedding_dim,
                        hash_size,
                        ",".join(ranks) if sharding_plan.plan else "None",
                    ]
                )
                if include_batch_sizes:
                    bs = feat_batch_sizes[i]
                    param_table[-1].append(_reduce_int_list(bs) if bs else "n/a")
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

        if sharding_plan.plan:
            for row in formatted_table:
                self._stats_table.append(f"# {row: <{self._width-3}}#")

            perf_breakdown = "Perf: Total perf (Forward compute, Forward comms, Backward compute, Backward comms, Prefetch compute)"
            legend = (
                "Input: MB/iteration, Output: MB/iteration, Shards: number of tables"
            )
            hbm_info = "HBM: estimated peak memory usage for shards, dense tensors, and features (KJT)"
            self._stats_table.append(f"#{'' : ^{self._width-2}}#")
            self._stats_table.append(f"# {perf_breakdown: <{self._width-3}}#")
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

        if not sharding_plan.plan:
            rank_size_text = f"World Size: {topology.world_size}"
            self._stats_table.append(f"#{'' : ^{self._width-2}}#")
            self._stats_table.append(f"# {rank_size_text : <{self._width-3}}#")

        self._log_compute_kernel_stats(
            compute_kernels_to_count, description="Compute Kernels Count"
        )
        self._log_compute_kernel_stats(
            {
                k: f"HBM: {round(bytes_to_gb(s.hbm),3)} GB, DDR: {round(bytes_to_gb(s.ddr),3)} GB"
                for k, s in compute_kernels_to_storage.items()
            },
            description="Compute Kernels Storage",
        )

        if debug:
            if sharding_plan.plan:
                # Plan imbalance stats for perf and storage
                self._log_plan_imbalance_stats(
                    perf,
                    used_hbm_and_ddr,
                )

                # Max perf and HBM to help root cause imbalance
                self._log_max_perf_and_max_hbm(perf, used_hbm_and_ddr)
            self._log_storage_reservation_stats(
                storage_reservation,
                topology,
                reserved_hbm_percent,
                dense_storage,
                kjt_storage,
            )
            if sharding_plan.plan:
                self._log_imbalance_tables(best_plan)

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

    def _log_dist_imbalance_stats(
        self,
        normalized_dist: List[float],
    ) -> None:
        for name, (measure, kwargs) in IMBALANCE_STAT_MEASURE.items():
            result_txt = f"{name}: {measure(normalized_dist, **kwargs):.3f}"
            self._stats_table.append(f"# {result_txt : <{self._width-3}}#")

    def _log_plan_imbalance_stats(
        self, perf: Dict[int, Perf], used_hbm_and_ddr: Dict[int, Storage]
    ) -> None:
        imbalance_logged = False
        total_perfs = [perf_i.total for perf_i in perf.values()]

        # Can extend with fwd/bwd perfs if needed
        perf_dists = [
            ("Total", total_perfs),
        ]

        for name, perf_dist in perf_dists:
            if sum(perf_dist) > 0:
                imbalance_logged = True
                self._stats_table.append(f"#{'' : ^{self._width-2}}#")
                self._stats_table.append(
                    f"# {name + ' Perf Imbalance Statistics' : <{self._width-3}}#"
                )
                normalized_perf_dist = _normalize_float(perf_dist)
                self._log_dist_imbalance_stats(normalized_perf_dist)

        used_hbm = [storage.hbm for storage in used_hbm_and_ddr.values()]
        used_ddr = [storage.ddr for storage in used_hbm_and_ddr.values()]

        if sum(used_hbm) > 0:
            imbalance_logged = True
            normalized_used_hbm = _normalize_int(used_hbm)
            self._stats_table.append(f"#{'' : ^{self._width-2}}#")
            self._stats_table.append(
                f"# {'HBM Imbalance Statistics' : <{self._width-3}}#"
            )
            self._log_dist_imbalance_stats(normalized_used_hbm)

        if sum(used_ddr) > 0:
            imbalance_logged = True
            normalized_used_ddr = _normalize_int(used_ddr)
            self._stats_table.append(f"#{'' : ^{self._width-2}}#")
            self._stats_table.append(
                f"# {'DDR Imbalance Statistics' : <{self._width-3}}#"
            )
            self._log_dist_imbalance_stats(normalized_used_ddr)

        if imbalance_logged:
            self._stats_table.append(f"#{'' : ^{self._width-2}}#")
            self._stats_table.append(
                f"# {'Total Variation: higher means more imbalanced (ranges 0 to 1)' : <{self._width-3}}#"
            )
            self._stats_table.append(
                f"# {'Total Distance: higher means more imbalanced (ranges 0 to 1)' : <{self._width-3}}#"
            )
            N = len(perf)  # world size
            if N > 0:
                max_chi_divergence = _calc_max_chi_divergence(
                    N=N, alpha=CHI_DIVERGENCE_ALPHA
                )
                self._stats_table.append(
                    f"# {f'Chi Divergence: higher means more imbalanced (ranges 0 to {max_chi_divergence:.3f})' : <{self._width-3}}#"
                )
                max_kl_divergence = _calc_max_kl_divergence(N)
                self._stats_table.append(
                    f"# {f'KL Divergence: higher means more imbalanced (ranges 0 to {max_kl_divergence:.3f})' : <{self._width-3}}#"
                )

    def _log_max_perf_and_max_hbm(
        self, perfs: Dict[int, Perf], used_mem: Dict[int, Storage]
    ) -> None:

        max_total_perf_text = f"Longest Critical Path (Maximum of Total Perf): {_generate_max_text(perfs, lambda p: p.total)}"
        max_fwd_compute_perf_text = f"Maximum of Forward Compute: {_generate_max_text(perfs, lambda p: p.fwd_compute)}"
        max_fwd_comms_perf_text = f"Maximum of Forward Comms: {_generate_max_text(perfs, lambda p: p.fwd_comms)}"
        max_bwd_compute_perf_text = f"Maximum of Backward Compute: {_generate_max_text(perfs, lambda p: p.bwd_compute)}"
        max_bwd_comms_perf_text = f"Maximum of Backward Comms: {_generate_max_text(perfs, lambda p: p.bwd_comms)}"
        max_prefetch_compute_perf_text = f"Maximum of Prefetch Compute: {_generate_max_text(perfs, lambda p: p.prefetch_compute)}"

        sum_of_maxima = (
            max(p.fwd_compute for p in perfs.values())
            + max(p.fwd_comms for p in perfs.values())
            + max(p.bwd_compute for p in perfs.values())
            + max(p.bwd_comms for p in perfs.values())
            + max(p.prefetch_compute for p in perfs.values())
        )
        sum_of_maxima_text = f"Sum of Maxima: {round(sum_of_maxima, 3)} ms"

        self._stats_table.append(f"#{'' : ^{self._width-2}}#")
        self._stats_table.append(f"# {max_total_perf_text : <{self._width-3}}#")
        self._stats_table.append(f"# {max_fwd_compute_perf_text : <{self._width-3}}#")
        self._stats_table.append(f"# {max_fwd_comms_perf_text : <{self._width-3}}#")
        self._stats_table.append(f"# {max_bwd_compute_perf_text : <{self._width-3}}#")
        self._stats_table.append(f"# {max_bwd_comms_perf_text : <{self._width-3}}#")
        self._stats_table.append(
            f"# {max_prefetch_compute_perf_text : <{self._width-3}}#"
        )
        self._stats_table.append(f"# {sum_of_maxima_text : <{self._width-3}}#")

        used_hbm: Dict[int, int] = {
            rank: storage.hbm for rank, storage in used_mem.items()
        }
        self._stats_table.append(f"#{'' : ^{self._width-2}}#")
        self._stats_table.append(
            f"# {'Estimated Sharding Distribution' : <{self._width-2}}#"
        )
        self._stats_table.append(
            f"# {'Max HBM: '+_generate_rank_hbm_stats(used_hbm, max) : <{self._width-3}}#"
        )
        self._stats_table.append(
            f"# {'Min HBM: '+_generate_rank_hbm_stats(used_hbm, min) : <{self._width-3}}#"
        )
        self._stats_table.append(
            f"# {'Mean HBM: '+_generate_rank_hbm_stats(used_hbm, statistics.mean) : <{self._width-3}}#"
        )
        self._stats_table.append(
            f"# {'Low Median HBM: '+_generate_rank_hbm_stats(used_hbm, statistics.median_low) : <{self._width-3}}#"
        )
        self._stats_table.append(
            f"# {'High Median HBM: '+_generate_rank_hbm_stats(used_hbm, statistics.median_high) : <{self._width-3}}#"
        )

        self._stats_table.append(f"#{'' : ^{self._width-2}}#")
        per_rank_hbm: Dict[int, int] = copy.copy(used_hbm)
        NUM_PEAK_RANK = 5
        peak_memory_pressure = []

        top_hbm_usage_estimation = (
            "Top HBM Memory Usage Estimation: "
            f"{round(bytes_to_gb(max(used_hbm.values())), 3)} GB"
        )
        self._stats_table.append(f"#{'' : ^{self._width-2}}#")
        self._stats_table.append(f"# {top_hbm_usage_estimation : <{self._width-3}}#")

        for top in range(NUM_PEAK_RANK):
            if not per_rank_hbm:
                break
            max_hbm = max(per_rank_hbm.values())
            max_hbm_indices: List[int] = []
            remaining: Dict[int, int] = {}
            for rank, hbm in per_rank_hbm.items():
                if math.isclose(bytes_to_mb(hbm), bytes_to_mb(max_hbm), abs_tol=1.0):
                    max_hbm_indices.append(rank)
                else:
                    remaining[rank] = hbm
            rank_text = "ranks" if len(max_hbm_indices) > 1 else "rank"
            max_hbm_ranks = (
                f"{rank_text} {','.join(_collapse_consecutive_ranks(max_hbm_indices))}"
            )
            peak_memory_pressure.append(
                (
                    f"Top Tier #{top+1} Estimated Peak HBM Pressure: "
                    f"{round(bytes_to_gb(max_hbm), 3)} GB on {max_hbm_ranks}"
                )
            )
            per_rank_hbm = remaining

        for peak_rank in reversed(peak_memory_pressure):
            self._stats_table.append(f"# {peak_rank : <{self._width-3}}#")

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
        reserved_hbm = round(
            bytes_to_gb(int(reserved_hbm_percent * device_storage.hbm)), 3
        )
        reserved_memory = f"HBM: {reserved_hbm} GB"
        reserved_hbm_percentage = f"Percent of Total HBM: {reserved_hbm_percent:.0%}"
        usable_ddr = round(bytes_to_gb(int(device_storage.ddr)), 3)
        usable_memory = f"HBM: {usable_hbm} GB, DDR: {usable_ddr} GB"
        usable_hbm_percentage = (
            f"Percent of Total HBM: {(1 - reserved_hbm_percent):.0%}"
        )
        self._stats_table.append(f"#{'' : ^{self._width-2}}#")
        self._stats_table.append(f"# {'Reserved Memory:' : <{self._width-3}}#")
        self._stats_table.append(f"#    {reserved_memory : <{self._width-6}}#")
        self._stats_table.append(f"#    {reserved_hbm_percentage : <{self._width-6}}#")
        self._stats_table.append(f"# {'Planning Memory:' : <{self._width-3}}#")
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

    def _log_imbalance_tables(self, best_plan: List[ShardingOption]) -> None:
        self._stats_table.append(f"#{'' : ^{self._width-2}}#")
        perf_imbalance_tables = _find_imbalance_tables(best_plan)
        hbm_imbalance_tables = _find_imbalance_tables(best_plan, target_imbalance="hbm")
        self._stats_table.append(
            f"# {'Top 5 Tables Causing Max Perf:' : <{self._width-3}}#"
        )
        for sharding_option in perf_imbalance_tables[0:5]:
            self._stats_table.append(f"#    {sharding_option.name : <{self._width-6}}#")
        self._stats_table.append(
            f"# {'Top 5 Tables Causing Max HBM:' : <{self._width-3}}#"
        )
        for sharding_option in hbm_imbalance_tables[0:5]:
            storage = sharding_option.shards[0].storage
            assert storage is not None  # linter friendly optional check

            rank_text = "ranks" if len(sharding_option.shards) > 1 else "rank"
            top_table = (
                f"{sharding_option.name}: {round(bytes_to_gb(storage.hbm),3)} GB on {rank_text} "
                f"{[shard.rank for shard in sharding_option.shards]}"
            )
            self._stats_table.append(f"#    {top_table : <{self._width-6}}#")

    def _log_compute_kernel_stats(
        self, compute_kernels_stats: Dict[str, Any], description: str
    ) -> None:
        compute_kernels_count = [
            f"{compute_kernel}: {count}"
            for compute_kernel, count in sorted(compute_kernels_stats.items())
        ]
        self._stats_table.append(f"#{'' : ^{self._width-2}}#")
        self._stats_table.append(f"# {description+':' : <{self._width-3}}#")
        for compute_kernel_count in compute_kernels_count:
            self._stats_table.append(f"#    {compute_kernel_count : <{self._width-6}}#")


def _generate_rank_hbm_stats(
    per_rank_hbm: Dict[int, int], func: Callable[[Iterable[float]], float]
) -> str:
    stats = round(func(per_rank_hbm))
    stats_indicies = [
        rank
        for rank, hbm in per_rank_hbm.items()
        if math.isclose(bytes_to_mb(hbm), bytes_to_mb(stats), abs_tol=1.0)
    ]
    rank_text = "ranks" if len(stats_indicies) > 1 else "rank"
    return f"{round(bytes_to_gb(stats), 3)} GB on {rank_text} {stats_indicies}"


def _generate_max_text(perfs: Dict[int, Perf], getter: Callable[[Perf], float]) -> str:
    max_perf = max(getter(p) for p in perfs.values())

    max_perf_indices = [
        rank for rank, perf in perfs.items() if getter(perf) == max_perf
    ]
    rank_text = "ranks" if len(max_perf_indices) > 1 else "rank"
    max_perf_indices = _collapse_consecutive_ranks(max_perf_indices)
    max_perf_ranks = f"{rank_text} {','.join(max_perf_indices)}"

    return f"{round(max_perf, 3)} ms on {max_perf_ranks}"


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


def _format_perf_breakdown(perf: Perf) -> str:
    breakdown = [
        perf.fwd_compute,
        perf.fwd_comms,
        perf.bwd_compute,
        perf.bwd_comms,
        perf.prefetch_compute,
    ]
    breakdown_string = ",".join(
        [str(round(num)) if num >= 1 else round_to_one_sigfig(num) for num in breakdown]
    )

    return f"{str(round(perf.total, 3))} ({breakdown_string})"


def _format_storage_breakdown(storage: Storage) -> str:
    storage_hbm = round(bytes_to_gb(storage.hbm), 3)
    storage_ddr = round(bytes_to_gb(storage.ddr), 3)
    return f"({storage_hbm} GB, {storage_ddr} GB)"


def round_to_one_sigfig(x: float) -> str:
    return f'{float(f"{x:.1g}"):g}'


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


def _reduce_int_list(input_list: List[int]) -> str:
    if len(input_list) == 0:
        return ""
    reduced = []
    count = 1
    prev_num = input_list[0]

    for num in input_list[1:]:
        if num == prev_num:
            count += 1
        else:
            if count > 1:
                reduced.append(f"{prev_num} * {count}")
            else:
                reduced.append(str(prev_num))
            prev_num = num
            count = 1

    # Handle the last number
    if count > 1:
        reduced.append(f"{prev_num}*{count}")
    else:
        reduced.append(str(prev_num))

    return ", ".join(reduced)


class NoopEmbeddingStats(Stats):
    """
    Noop Stats for a sharding planner execution.
    """

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
        sharders: Optional[List[ModuleSharder[nn.Module]]] = None,
        debug: bool = True,
    ) -> None:
        pass
