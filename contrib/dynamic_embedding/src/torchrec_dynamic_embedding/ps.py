import os
from typing import Callable, Dict, List, Tuple, Union

import torch
import torch.nn as nn
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torchrec.distributed.model_parallel import DistributedModelParallel as DMP
from torchrec.distributed.types import ParameterSharding, ShardingPlan

from .tensor_list import TensorList

try:
    torch.ops.load_library(os.path.join(os.path.dirname(__file__), "tde_cpp.so"))
except Exception as ex:
    print(f"File tde_cpp.so not found {ex}")


__all__ = ["PS", "PSCollection", "get_ps"]


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


def get_sharded_modules_recursive(
    module: nn.Module,
    path: str,
    plan: ShardingPlan,
) -> Dict[str, nn.Module]:
    params_plan = plan.get_plan_for_module(path)
    if params_plan:
        return {path: (module, params_plan)}

    res = {}
    for name, child in module.named_children():
        new_path = f"{path}.{name}" if path else name
        res.update(get_sharded_modules_recursive(child, new_path, plan))
    return res


def get_ps(
    module: DMP,
    num_optimizer_stats: int,
    ps_config: Union[str, Callable[[str, str], str]],
):
    # Note that `num_optimizer_stats` here does not take the weight into account,
    # while in C++ side, the weight is also considered a optimizer stat.
    if num_optimizer_stats > 2 or num_optimizer_stats < 0:
        raise ValueError(
            f"num_optimizer_stats must be in [0, 2], got {num_optimizer_stats}"
        )

    plan = module.plan
    sharded_modules = get_sharded_modules_recursive(module.module, "", plan)

    ps_list = {}

    for path, (sharded_module, params_plan) in sharded_modules.items():
        state_dict = sharded_module.state_dict()
        optimizer_state_dict = sharded_module.fused_optimizer.state_dict()["state"]
        tensor_infos = {}
        for key, tensor in state_dict.items():
            # Here we use the fact that state_dict will be shape of
            # `embeddings.xxx.weight` or `embeddingbags.xxx.weight`
            if len(key.split(".")) <= 1 or key.split(".")[1] not in params_plan:
                continue
            table_name = key.split(".")[1]
            param_plan = params_plan.pop(table_name)
            tensors = [tensor]
            # This is really hardcoded right now...
            optimizer_state = optimizer_state_dict[key]
            for i in range(num_optimizer_stats):
                tensors.append(optimizer_state[f"{table_name}.momentum{i+1}"])
            tensor_infos[table_name] = (param_plan, tensors)

        assert (
            len(params_plan) == 0
        ), f"There are sharded param not found, leaving: {params_plan}."

        if isinstance(ps_config, str):
            collection_config = ps_config
        else:
            collection_config = lambda table_name: ps_config(path, table_name)

        ps_list[path] = PSCollection(path, tensor_infos, collection_config)
    return ps_list
