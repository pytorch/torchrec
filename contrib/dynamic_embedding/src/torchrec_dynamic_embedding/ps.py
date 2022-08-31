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
        schema: str,
    ):
        """
        PS table of an embedding table.

        Args:
            table_name: name of the table.
            tensors: tensors of the table, the first one is the parameter tensor, others are
                tenors of optimizers, e.g. for Adam, it will be [weight, m, v].
            schema: schema of the PS.
        """
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
            table_name, shards, col_size, num_optimizer_stats, schema
        )

    def evict(self, ids_to_evict: torch.Tensor):
        """
        Evict the `ids_to_evict` to PS.
        """
        self._ps.evict(ids_to_evict)

    def fetch(
        self,
        ids_to_fetch: torch.Tensor,
        time: int,
        reinit: bool = False,
        weight_init_max: float = 0,
        weight_init_min: float = 0,
    ):
        """
        Fetch `ids_to_fetch` from tensor. If `reinit` is set to `True`, will
        reinitialize the embedding if the global id is not in PS.
        """
        return self._ps.fetch(
            ids_to_fetch, time, reinit, weight_init_max, weight_init_min
        )


class PSCollection:
    """
    PS tables correspond to an EmbeddingCollection or EmbeddingBagCollection.
    """

    def __init__(
        self,
        path: str,
        plan: Dict[str, Tuple[ParameterSharding, Union[torch.Tensor, ShardedTensor]]],
        schema: Union[str, Callable[[str], str]],
    ):
        """
        Args:
            path: module path.
            plan: dict keyed by table name of ParameterSharding and tensor of the table.
            schema: schema of the PS.
        """
        self._path = path
        self._ps_collection = {}
        for table_name, (param_plan, tensor) in plan.items():
            if isinstance(schema, str):
                table_config = schema
            else:
                table_config = schema[table_name]
            self._ps_collection[table_name] = PS(
                f"{path}.{table_name}", tensor, table_config
            )

    def table_names(self):
        return self._ps_collection.keys()

    def __getitem__(self, table_name):
        return self._ps_collection[table_name]

    @staticmethod
    def fromModule(path, sharded_module, params_plan, ps_config):
        """
        Create PSCollection for `sharded_module`, whose module path is `path`

        Args:
            path: module path of the sharded module.
            sharded_module: the sharded module.
            params_plan: the sharding plan of `sharded_module`.
            ps_config: configuration for PS. Required fields are "schema", which designates the schema of
                the PS server, e.g. redis://192.168.3.1:3948 and "num_optimizer_stats", which tell PS server
                how many optimizer states for the parameter, for intance, the value is 2 for Adam optimizer.

        Return:
            PSCollection of the sharded module.
        """
        if "schema" not in ps_config:
            raise ValueError("schema not found in ps_config")
        schema = ps_config["schema"]

        if "num_optimizer_stats" not in ps_config:
            raise ValueError("num_optimizer_stats not found in ps_config")
        num_optimizer_stats = ps_config["num_optimizer_stats"]
        # Note that `num_optimizer_stats` here does not take the weight into account,
        # while in C++ side, the weight is also considered a optimizer stat.
        if num_optimizer_stats > 2 or num_optimizer_stats < 0:
            raise ValueError(
                f"num_optimizer_stats must be in [0, 2], got {num_optimizer_stats}"
            )

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

        if isinstance(schema, str):
            collection_schema = schema
        else:
            collection_schema = lambda table_name: schema(path, table_name)

        return PSCollection(path, tensor_infos, collection_schema)
