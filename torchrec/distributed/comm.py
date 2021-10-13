#!/usr/bin/env python3

import logging
import os
from typing import List, Tuple, Optional

import torch.distributed as dist

logger: logging.Logger = logging.getLogger(__name__)

# Global, only should be accessed via intra_and_cross_node_pg()
_INTRA_PG: Optional[dist.ProcessGroup] = None
_CROSS_PG: Optional[dist.ProcessGroup] = None


def _env2int(env_list: List[str], default: int = -1) -> int:
    for e in env_list:
        val = int(os.environ.get(e, -1))
        if val >= 0:
            return val
    return default


def get_local_size() -> int:
    """
    Get the local world size (see https://pytorch.org/docs/stable/elastic/run.html)
    This is usually the size of workers on each node, or nproc_per_node
    """
    local_size = _env2int(
        [
            "LOCAL_WORLD_SIZE",
            "MPI_LOCALNRANKS",
            "OMPI_COMM_WORLD_LOCAL_SIZE",
            "MV2_COMM_WORLD_LOCAL_SIZE",
        ],
        8,
    )

    world_size = dist.get_world_size()
    if local_size == -1 or world_size % local_size != 0:
        logging.warning(
            "Could not determine LOCAL_WORLD_SIZE from environment, falling back to WORLD_SIZE."
        )
        local_size = world_size
    return local_size


def get_local_rank() -> int:
    """
    Get the local rank of the local processes (see https://pytorch.org/docs/stable/elastic/run.html)
    This is usually the rank of the worker on its node
    """
    my_local_rank = _env2int(
        [
            "LOCAL_RANK",
            "MPI_LOCALRANKID",
            "OMPI_COMM_WORLD_LOCAL_RANK",
            "MV2_COMM_WORLD_LOCAL_RANK",
        ],
        -1,
    )
    local_size = get_local_size()

    if my_local_rank == -1 or my_local_rank >= local_size:
        logging.warning(
            "Could not determine LOCAL_RANK from environment, falling back to GLOBAL_RANK % LOCAL_SIZE."
        )
        my_local_rank = dist.get_rank() % local_size
    return my_local_rank


def get_group_rank() -> int:
    """
    Get the group rank of the worker group. Also available with GROUP_RANK environment varible
    A number between 0 and get_num_groups() (See https://pytorch.org/docs/stable/elastic/run.html)
    """
    return dist.get_rank() // get_local_size()


def get_num_groups() -> int:
    """
    Get the number of worker groups.
    Usually equivalent to max_nnodes (See https://pytorch.org/docs/stable/elastic/run.html)
    """
    return dist.get_world_size() // get_local_size()


def intra_and_cross_node_pg(
    backend: str = "nccl",
) -> Tuple[dist.ProcessGroup, dist.ProcessGroup]:
    """
    This function creates sub process groups (intra and cross node)
    """
    global _INTRA_PG  # intra node process group
    global _CROSS_PG  # cross node process group

    my_rank = dist.get_rank()
    my_local_rank = get_local_rank()
    local_size = get_local_size()
    my_group_rank = get_group_rank()
    group_count = get_num_groups()

    logger.info(
        f"[{my_rank}] my_local_rank = {my_local_rank}, local_size = {local_size},"
        f"my_group_rank = {my_group_rank}, group_count = {group_count}"
    )
    if _INTRA_PG is None:
        for group_rank in range(group_count):
            peers = [group_rank * local_size + r for r in range(local_size)]
            curr_intra_group_pg = dist.new_group(backend=backend, ranks=peers)
            if my_group_rank == group_rank:
                logger.warning(
                    "[Connection] intra_group: [%d] -> [%s]" % (my_rank, peers)
                )
                _INTRA_PG = curr_intra_group_pg

    dist.barrier()

    if _CROSS_PG is None:
        for l_rank in range(local_size):
            peers = [l_rank + g * local_size for g in range(group_count)]
            curr_cross_group_pg = dist.new_group(backend=backend, ranks=peers)
            if l_rank == my_local_rank:
                logger.warning(
                    "[Connection] cross_group: [%d] -> [%s]" % (my_rank, peers)
                )
                _CROSS_PG = curr_cross_group_pg

    dist.barrier()

    # pyre-ignore [7]: Incompatible return type
    return _INTRA_PG, _CROSS_PG
