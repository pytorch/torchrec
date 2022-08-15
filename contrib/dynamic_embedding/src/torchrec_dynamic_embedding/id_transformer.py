import json
import os
from typing import List

import torch

try:
    torch.ops.load_library(os.path.join(os.path.dirname(__file__), "tde_cpp.so"))
except Exception as ex:
    print(f"File tde_cpp.so not found {ex}")


__all__ = ["IDTransformer", "TensorList"]


class TensorList:
    def __init__(self, tensors: List[torch.Tensor]):
        self.tensor_list = torch.classes.tde.TensorList()
        for tensor in tensors:
            self.tensor_list.append(tensor)

    def __len__(self):
        return len(self.tensor_list)

    def __getitem__(self, i):
        return self.tensor_list[i]


class IDTransformer:
    def __init__(self, num_embedding, eviction_config=None, transform_config=None):
        if not eviction_config:
            eviction_config = {"type": "mixed_lru_lfu"}
        if not transform_config:
            transform_config = {"type": "naive"}
        config = json.dumps(
            {
                "lxu_strategy": eviction_config,
                "id_transformer": transform_config,
            }
        )
        self._transformer = torch.classes.tde.IDTransformer(num_embedding, config)
        self._time = 0

    def transform(self, global_ids: TensorList, cache_ids: TensorList):
        self._time += 1
        return self._transformer.transform(
            global_ids.tensor_list, cache_ids.tensor_list, self._time
        )

    def evict(self, num_to_evict):
        return self._transformer.evict(num_to_evict)
