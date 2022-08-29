import os
from typing import Callable, Dict, List, Tuple, Union

import torch
import torch.nn as nn
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torchrec.distributed.model_parallel import DistributedModelParallel as DMP
from torchrec.distributed.types import ParameterSharding

from .tensor_list import TensorList

try:
    torch.ops.load_library(os.path.join(os.path.dirname(__file__), "tde_cpp.so"))
except Exception as ex:
    print(f"File tde_cpp.so not found {ex}")


__all__ = ["PS", "PSCollection"]


class PS:
    def __init__(
        self,
        table_name: str,
        tensors: Union[List[torch.Tensor], List[ShardedTensor]],
        ps_config: str,
    ):
        shards = torch.classes.tde.LocalShardList()
        num_optimizer_stats = len(tensors)
        if isinstance(tensors[0], ShardedTensor):
            # Here we assume the shard metadata of optimizer state and weight are the same.
            for i, shard in enumerate(tensors[0].local_shards()):
                local_tensors = [tensor.local_shards()[i].tensor for tensor in tensors]
                shards.append(
                    shard.metadata.shard_offsets[0],
                    shard.metadata.shard_offsets[1],
                    shard.metadata.shard_sizes[0],
                    shard.metadata.shard_sizes[1],
                    TensorList(local_tensors).tensor_list,
                )
                # This assumes all shard have the same column size.
                col_size = shard.tensor.shape[1]
        elif isinstance(tensors[0], torch.Tensor):
            tensors
            shards.append(
                0,
                0,
                tensors[0].shape[0],
                tensors[0].shape[1],
                TensorList(tensors).tensor_list,
            )
            col_size = tensors[0].shape[1]
        self._ps = torch.classes.tde.PS(
            table_name, shards, col_size, num_optimizer_stats, ps_config
        )

    def evict(self, ids_to_evict: torch.Tensor):
        self._ps.evict(ids_to_evict)

    def fetch(
        self,
        ids_to_fetch: torch.Tensor,
        reinit: bool = False,
        weight_init_max: float = 0,
        weight_init_min: float = 0,
    ):
        self._ps.fetch(ids_to_fetch, reinit, weight_init_max, weight_init_min)


class PSCollection:
    """
    PS for one table.
    """

    def __init__(
        self,
        path: str,
        plan: Dict[str, Tuple[ParameterSharding, Union[torch.Tensor, ShardedTensor]]],
        ps_config: Union[str, Callable[[str], str]],
    ):
        self._path = path
        self._ps_collection = {}
        for table_name, (param_plan, tensor) in plan.items():
            if isinstance(ps_config, str):
                table_config = ps_config
            else:
                table_config = ps_config[table_name]
            self._ps_collection[table_name] = PS(
                f"{path}.{table_name}", tensor, table_config
            )

    def keys(self):
        return self._ps_collection.keys()

    def __getitem__(self, table_name):
        return self._ps_collection[table_name]
