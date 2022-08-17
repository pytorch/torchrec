from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.distributed as dist

from torchrec import KeyedJaggedTensor

__all__ = ["gatherv_kjts", "default_group"]

_group = [None]


@dataclass
class GatherResult:
    values_list: List[List[torch.Tensor]]
    offset_per_key_list: List[List[torch.Tensor]]


def default_group(group):
    if group is not None:
        return group

    if _group[0] is None:
        _group[0] = dist.new_group(backend="gloo")

    return _group[0]


def gather_tensor_list(
    tensors: List[torch.Tensor], numel_lists: List[List[int]], dst=0, group=None
) -> Optional[List[List[torch.Tensor]]]:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if dst == rank:
        if len(numel_lists) != world_size:
            raise ValueError("dst rank should know size of tensors on all ranks")
    else:
        if len(numel_lists) != 1:
            raise ValueError("non dst rank should pass its own tensor sizes")

    group = default_group(group)
    dtype = tensors[0].dtype
    device = tensors[0].device

    concated_tensor = torch.cat(tensors)

    if rank == dst:
        # gather can only accept same-size tensors.
        max_numel = max(sum(numel_list) for numel_list in numel_lists)
        concat_results = [
            torch.empty(max_numel, dtype=dtype, device=device)
            for _ in range(world_size)
        ]
        dist.gather(
            concated_tensor,
            gather_list=concat_results,
            dst=dst,
            group=group,
            async_op=False,
        )

        results = []
        for i in range(world_size):
            splited_tensors = []
            offset = 0
            for numel in numel_lists[i]:
                splited_tensors.append(concat_results[i][offset : offset + numel])
                offset += numel
            results.append(splited_tensors)

        return results
    else:
        dist.gather(
            concated_tensor, gather_list=None, dst=dst, group=group, async_op=False
        )

        return None


def gather_kjts(kjts: List[KeyedJaggedTensor], dst=0, group=None) -> GatherResult:
    world_size = dist.get_world_size()
    if world_size == 1:
        return GatherResult(
            values_list=[[kjt.values()] for kjt in kjts],
            offset_per_key_list=[[kjt.offset_per_key()] for kjt in kjts],
        )

    rank = dist.get_rank()
    group = default_group(group)

    offset_per_key_list = [torch.tensor(kjt.offset_per_key()) for kjt in kjts]
    values_list = [kjt.values() for kjt in kjts]

    offset_numel_list = [tensor.numel() for tensor in offset_per_key_list]
    values_numel_list = [tensor.numel() for tensor in values_list]

    if rank == dst:
        global_offset_numel_list = [offset_numel_list] * world_size
        offset_results = gather_tensor_list(
            offset_per_key_list,
            numel_lists=global_offset_numel_list,
            dst=dst,
            group=group,
        )

        global_values_numel_list = [
            [offset[-1].item() for offset in offsets] for offsets in offset_results
        ]

        values_result = gather_tensor_list(
            values_list, numel_lists=global_values_numel_list, dst=dst, group=group
        )

        return GatherResult(
            values_list=values_result, offset_per_key_list=offset_results
        )
    else:
        gather_tensor_list(
            offset_per_key_list, numel_lists=[offset_numel_list], dst=dst, group=group
        )

        gather_tensor_list(
            values_list, numel_lists=[values_numel_list], dst=dst, group=group
        )

        return None
