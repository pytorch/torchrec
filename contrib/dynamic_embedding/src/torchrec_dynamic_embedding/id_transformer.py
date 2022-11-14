import json
import os

import torch

from .tensor_list import TensorList

try:
    torch.ops.load_library(os.path.join(os.path.dirname(__file__), "tde_cpp.so"))
except Exception as ex:
    print(f"File tde_cpp.so not found {ex}")


__all__ = []


class IDTransformer:
    def __init__(self, num_embedding, eviction_config=None, transform_config=None):
        self._num_embedding = num_embedding
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

    def transform(self, global_ids: TensorList, cache_ids: TensorList, time: int):
        """
        Transform `global_ids` and store the results in `cache_ids`.
        """
        result = self._transformer.transform(
            global_ids.tensor_list, cache_ids.tensor_list, time
        )
        return result.success, result.ids_to_fetch

    def evict(self, num_to_evict):
        """
        Evict `num_to_evict` ids from the transformer.
        """
        return self._transformer.evict(num_to_evict)

    def save(self):
        """
        Get ids to save.
        """
        return self._transformer.save()
