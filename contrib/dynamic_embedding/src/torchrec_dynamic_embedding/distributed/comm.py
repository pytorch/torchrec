from typing import List, Optional

import torch
import torch.distributed as dist

__all__ = []


def gather_global_ids(global_ids: List[torch.Tensor], group):
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    concat_global_ids = torch.cat(global_ids)

    concat_numel = torch.tensor(concat_global_ids.numel(), dtype=torch.int64)
    concat_numel_list = [torch.tensor(0, dtype=torch.int64) for _ in range(world_size)]
    dist.all_gather(concat_numel_list, concat_numel, group=group, async_op=False)

    max_numel = max(concat_numel_list)
    concat_global_ids.resize_(max_numel)

    if rank == 0:
        concat_global_ids_list = [
            torch.empty_like(concat_global_ids) for _ in range(world_size)
        ]
        dist.gather(concat_global_ids, concat_global_ids_list, 0, group, async_op=False)
        return [
            concat_global_ids_list[i][: concat_numel_list[i]] for i in range(world_size)
        ], concat_numel_list
    else:
        dist.gather(concat_global_ids, None, 0, group, async_op=False)
        return None, concat_numel_list


def scatter_cache_ids(
    cache_ids_list: Optional[List[torch.Tensor]], concat_numel_list: List[int], group
):
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    max_numel = max(concat_numel_list)

    concat_cache_ids = torch.empty(max_numel, dtype=torch.int64)
    if rank == 0:
        concat_cache_ids_list = [concat_cache_ids] + [
            cache_ids.resize_(max_numel)
            for cache_ids in cache_ids_list[-world_size + 1 :]
        ]
        assert len(concat_cache_ids_list) == world_size
        dist.scatter(concat_cache_ids, concat_cache_ids_list, group=group)
    else:
        dist.scatter(concat_cache_ids, None, group=group)
        offset = 0
        for cache_ids in cache_ids_list:
            cache_ids[:] = concat_cache_ids[offset : offset + cache_ids.numel()]
            offset += cache_ids.numel()


def broadcast_transform_result(
    success: bool, ids_to_fetch: Optional[torch.Tensor], group
):
    if dist.get_rank() == 0:
        success_and_numel = torch.tensor(
            [1 if success else 0, ids_to_fetch.numel()], dtype=torch.int64
        )
        dist.broadcast(success_and_numel, src=0, group=group)
    else:
        success_and_numel = torch.tensor([0, 0], dtype=torch.int64)
        dist.broadcast(success_and_numel, src=0, group=group)
        success, numel = success_and_numel.tolist()
        success = success != 0
        ids_to_fetch = torch.empty((numel // 2, 2), dtype=torch.int64)

    if ids_to_fetch.numel() > 0:
        dist.broadcast(ids_to_fetch, src=0, group=group)
    return success, ids_to_fetch


def broadcast_ids_to_evict(ids, group):
    if dist.get_rank() == 0:
        numel = torch.tensor(ids.numel(), dtype=torch.int64)
        dist.broadcast(numel, src=0, group=group)
    else:
        numel = torch.tensor(0, dtype=torch.int64)
        dist.broadcast(numel, src=0, group=group)
        numel = numel.item()
        ids = torch.empty((numel // 2, 2), dtype=torch.int64)

    if numel > 0:
        dist.broadcast(ids, src=0, group=group)
    return ids
