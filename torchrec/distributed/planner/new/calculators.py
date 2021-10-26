#!/usr/bin/env python3

from typing import Tuple, List

from torchrec.distributed.planner.new.constants import (
    BIGINT_DTYPE,
    KERNEL_LOOKUP_BW,
    INTRA_NODE_BANDWIDTH,
    CROSS_NODE_BANDWIDTH,
)
from torchrec.distributed.planner.new.types import CostCalc, Topology, ShardingOption
from torchrec.distributed.types import ShardingType


def cost_func_emb_wall_time(
    shard_lengths: List[List[int]],
    compute_kernel: str,
    compute_device: str,
    sharding_type: str,
    batch_size: int,
    world_size: int,
    local_world_size: int,
    input_lengths: List[float],
    input_data_type_size: float,
    output_data_type_size: float,
    bw_intra_host: int,
    bw_inter_host: int,
    has_input_dist: bool = True,
    has_output_dist: bool = True,
) -> List[float]:
    """
    Attempts to model costs as a function of relative wall times.
    Only models forward costs (ignores backward costs)
    The computation cost estimation is based on EmbeddingBagCollectionSharder
    (pooledEmbedding)

    shard_lengths: the list of (local_rows, local_cols) pf each shard
    input_lengths: the list of the average number of lookups of each input query feature
    bw_intra_host: the bandwidth within the single host like multiple threads
    bw_inter_host: the bandwidth between two hosts like multiple machines
    input_dist:
        tw_sharding: https://fburl.com/code/uxueh8wh
        rw_sharding: https://fburl.com/code/zemh4rzw
        cw_sharding: same as tw, consider as multiple tables (cw_emb_dim * num_sharding = tw_emb_dim)
        twrw_sharding: https://fburl.com/code/vrweq0ri
    output_dist:
        tw_sharding: https://fburl.com/code/ete7schi
        rw_sharding: https://fburl.com/code/gl9186u1
        cw_sharding: same as tw, consider as multiple tables (cw_emb_dim * num_sharding = tw_emb_dim)
        twrw_sharding: https://fburl.com/code/z9nyjflj

    Note: the computation of the output cost will count len(input_length) due to pooling

    """
    shard_costs = []
    B = 1.0 * world_size * batch_size  # global batch size
    device_bw = KERNEL_LOOKUP_BW[(compute_device, compute_kernel)]

    for _, emb_dim in shard_lengths:

        if sharding_type is ShardingType.TABLE_WISE.value:
            input_cost, compute_cost, output_cost = _get_tw_sharding_cost(
                B,
                world_size,
                input_lengths,
                emb_dim,
                input_data_type_size,
                output_data_type_size,
                device_bw,
                bw_inter_host,
            )
        elif sharding_type is ShardingType.COLUMN_WISE.value:
            input_cost, compute_cost, output_cost = _get_cw_sharding_cost(
                B,
                world_size,
                input_lengths,
                emb_dim,
                input_data_type_size,
                output_data_type_size,
                device_bw,
                bw_inter_host,
            )
        elif sharding_type is ShardingType.ROW_WISE.value:
            input_cost, compute_cost, output_cost = _get_rw_sharding_cost(
                B,
                world_size,
                input_lengths,
                emb_dim,
                input_data_type_size,
                output_data_type_size,
                device_bw,
                bw_inter_host,
            )
        elif sharding_type is ShardingType.TABLE_ROW_WISE.value:
            input_cost, compute_cost, output_cost = _get_twrw_sharding_cost(
                B,
                world_size,
                local_world_size,
                input_lengths,
                emb_dim,
                input_data_type_size,
                output_data_type_size,
                device_bw,
                bw_inter_host,
                bw_intra_host,
            )
        elif sharding_type is ShardingType.DATA_PARALLEL.value:
            input_cost, compute_cost, output_cost = _get_dp_sharding_cost(
                batch_size,
                input_lengths,
                emb_dim,
                output_data_type_size,
                device_bw,
            )
        else:
            raise RuntimeError(f"Unexpected sharding type: {sharding_type}")

        shard_cost = 0
        shard_cost += input_cost if has_input_dist else 0
        shard_cost += compute_cost
        shard_cost += output_cost if has_output_dist else 0
        shard_costs.append(shard_cost)

    return shard_costs


def _get_tw_sharding_cost(
    global_batch_size: float,
    world_size: int,
    input_lengths: List[float],
    emb_dim: int,
    input_data_type_size: float,
    output_data_type_size: float,
    device_bw: float,
    bw_inter_host: int,
) -> Tuple[float, float, float]:
    input_cost = (
        global_batch_size * sum(input_lengths) * input_data_type_size / bw_inter_host
    )
    compute_cost = (
        global_batch_size
        * sum(input_lengths)
        * emb_dim
        * output_data_type_size
        / device_bw
    )
    output_cost = (
        global_batch_size
        * emb_dim
        * len(input_lengths)
        * output_data_type_size
        / bw_inter_host
    )
    return (input_cost, compute_cost, output_cost)


def _get_cw_sharding_cost(
    global_batch_size: float,
    world_size: int,
    input_lengths: List[float],
    emb_dim: int,
    input_data_type_size: float,
    output_data_type_size: float,
    device_bw: float,
    bw_inter_host: int,
) -> Tuple[float, float, float]:
    input_cost = (
        global_batch_size * sum(input_lengths) * input_data_type_size / bw_inter_host
    )
    compute_cost = (
        global_batch_size
        * sum(input_lengths)
        * emb_dim
        * output_data_type_size
        / device_bw
    )
    output_cost = (
        global_batch_size
        * emb_dim
        * len(input_lengths)
        * output_data_type_size
        / bw_inter_host
    )
    return (input_cost, compute_cost, output_cost)


def _get_rw_sharding_cost(
    global_batch_size: float,
    world_size: int,
    input_lengths: List[float],
    emb_dim: int,
    input_data_type_size: float,
    output_data_type_size: float,
    device_bw: float,
    bw_inter_host: int,
) -> Tuple[float, float, float]:
    input_cost = (
        global_batch_size
        * sum(input_lengths)
        / world_size
        * input_data_type_size
        / bw_inter_host
    )
    compute_cost = (
        global_batch_size
        * sum(input_lengths)
        / world_size
        * emb_dim
        * output_data_type_size
        / device_bw
    )
    output_cost = (
        global_batch_size
        * emb_dim
        * len(input_lengths)
        * output_data_type_size
        / bw_inter_host
    )
    return (input_cost, compute_cost, output_cost)


def _get_twrw_sharding_cost(
    global_batch_size: float,
    world_size: int,
    local_world_size: int,
    input_lengths: List[float],
    emb_dim: int,
    input_data_type_size: float,
    output_data_type_size: float,
    device_bw: float,
    bw_inter_host: int,
    bw_intra_host: int,
) -> Tuple[float, float, float]:
    input_cost = (
        global_batch_size
        * sum(input_lengths)
        / local_world_size
        * input_data_type_size
        / bw_inter_host
    )
    compute_cost = (
        global_batch_size
        * sum(input_lengths)
        / local_world_size
        * emb_dim
        * output_data_type_size
        / device_bw
    )
    output_cost = (
        global_batch_size
        * emb_dim
        * len(input_lengths)
        * output_data_type_size
        / bw_intra_host
        + global_batch_size
        * emb_dim
        * len(input_lengths)
        * output_data_type_size
        * (local_world_size / world_size)
        / bw_inter_host
    )
    return (input_cost, compute_cost, output_cost)


def _get_dp_sharding_cost(
    batch_size: float,
    input_lengths: List[float],
    emb_dim: int,
    output_data_type_size: float,
    device_bw: float,
) -> Tuple[float, float, float]:
    input_cost = 0
    compute_cost = (
        batch_size * sum(input_lengths) * emb_dim * output_data_type_size / device_bw
    )
    output_cost = 0
    return (input_cost, compute_cost, output_cost)


class EmbeddingWTCostCalculator(CostCalc):
    """
    Embedding Wall Time Cost Calculator
    """

    def __init__(self, topology: Topology) -> None:
        self._topology = topology

    def run(self, sharding_option: ShardingOption) -> None:
        shard_costs = cost_func_emb_wall_time(
            # pyre-ignore [6]: Incompatible parameter type
            shard_lengths=sharding_option.shard_lengths,
            compute_kernel=sharding_option.compute_kernel,
            compute_device=self._topology.compute_device,
            sharding_type=sharding_option.sharding_type,
            batch_size=sharding_option.batch_size,
            world_size=self._topology.world_size,
            local_world_size=self._topology.local_world_size,
            input_lengths=sharding_option.input_lengths,
            input_data_type_size=BIGINT_DTYPE,
            output_data_type_size=sharding_option.tensor.element_size(),
            bw_intra_host=getattr(
                self._topology, "intra_host_bw", INTRA_NODE_BANDWIDTH
            ),
            bw_inter_host=getattr(
                self._topology, "inter_host_bw", CROSS_NODE_BANDWIDTH
            ),
            has_input_dist=True if sharding_option.upstream_modules else False,
            has_output_dist=False if sharding_option.downstream_modules else True,
        )

        sharding_option.shard_costs = shard_costs
        # set costs to sum of shard costs
        sharding_option.cost = sum(shard_costs)
