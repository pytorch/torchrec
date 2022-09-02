import queue
import threading
from typing import Dict, List, Union

import torch
from torch.utils.data._utils import MP_STATUS_CHECK_INTERVAL
from torchrec import EmbeddingBagConfig, EmbeddingConfig
from torchrec.distributed.model_parallel import DistributedModelParallel

from .id_transformer_group import IDTransformerGroup


# Similar to torch.utils.data._utils.pin_memory._pin_memory_loop
def transform_loop(dataset, transform_fn, out_queue, done_event):
    # This setting is thread local, and prevents the copy in pin_memory from
    # consuming all CPU cores.
    torch.set_num_threads(1)

    for data in dataset:
        if done_event.is_set():
            break
        transformed_data = transform_fn(data)

        while not done_event.is_set():
            try:
                out_queue.put(transformed_data, timeout=MP_STATUS_CHECK_INTERVAL)
                break
            except queue.Full:
                continue
        # save memory
        del transformed_data


class DataLoaderIter:
    def __init__(self, dataset, transform_fn, num_prefetch=0):
        self._data_queue = queue.Queue(maxsize=num_prefetch)
        self._done_event = threading.Event()
        self._transform_thread = threading.Thread(
            target=transform_loop,
            args=(dataset, transform_fn, self._data_queue, self._done_event),
        )
        self._transform_thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        return self._get_data()

    def __del__(self):
        self._done_event.set()

    def _get_data(self):
        if not self._transform_thread.is_alive():
            raise RuntimeError("Transform thread exited unexpectedly")
        data, handles = self._data_queue.get()
        for handle in handles:
            handle.wait()
        return data


class DataLoader:
    def __init__(
        self,
        module: DistributedModelParallel,
        configs_dict: Dict[str, Union[List[EmbeddingBagConfig], List[EmbeddingConfig]]],
        dataset,
        *,
        data_info: Dict[int, str],
        ps_config,
        eviction_config=None,
        transform_config=None,
        parallel=True,
        num_prefetch=0,
    ):
        """
        DataLoader to transform data from global id to cache id.

        Args:
            module: DMP module that need dynamic embedding.
            configs_dict: a dictionary that maps the module path of the sharded module to its embedding
                configs or embeddingbag configs. The plan of `module` should contain the module path
                in `configs_dict`.
            dataset: dataset to transform.
            data_info: dict keyed by int index of module path. For example, if the dataset produces
                `label, kjt1, kjt2` each iteration and `kjt1` and `kjt2` are inputs to modules of path
                `emb1` and `emb2` respectively, then `data_info` should be `{ 1: "emb1", 2: "emb2" }`.
            ps_config: configuration for PS. Required fields are "schema", which designates the schema of
                the PS server, e.g. redis://192.168.3.1:3948 and "num_optimizer_stats", which tell PS server
                how many optimizer states for the parameter, for intance, the value is 2 for Adam optimizer.
            eviction_config: configuration for eviction policy. Default is `{"type": "mixed_lru_lfu"}`
            transform_config: configuration for the transformer. Default is `{"type": "naive"}`
            parallel: Whether the IDTransformerCollections will run paralell. When set to True,
                IDTransformerGroup will start a thread for each IDTransformerCollection.
            num_prefetch: number of samples to prefetch.

        Example:
            class Model(nn.Module):
                def __init__(self, config1, config2):
                    super().__init__()
                    self.emb1 = EmbeddingCollection(tables=config1, device=torch.device("meta"))
                    self.emb2 = EmbeddingCollection(tables=config2, device=torch.device("meta"))
                    ...

                def forward(self, kjt1, kjt2):
                    ...

            m = Model(config1, config2)
            m = DistributedModelParallel(m)
            dataloader = DataLoader(
                m,
                { "emb1": config1, "emb2": config2 },
                dataset,
                data_info={ 1: "emb1", 2: "emb2" }
                ps_configs={
                    "num_optimizer_stats": 2,
                    "schema": "memory://"
                },
                num_prefetch=1)

            for label, kjt1, kjt2 in dataloader:
                output = m(kjt1, kjt2)
                ...
        """
        self._id_transformer_group = IDTransformerGroup(
            module,
            configs_dict,
            ps_config=ps_config,
            eviction_config=eviction_config,
            transform_config=transform_config,
            parallel=parallel,
        )

        for _, path in data_info.items():
            if path not in self._id_transformer_group:
                raise ValueError(
                    f"invalid path `{path}` data_info. No id transformer for this path."
                )

        self._data_info = data_info

        self._data_queue = queue.Queue(maxsize=num_prefetch)
        self._done_event = threading.Event()

        self._dataset = dataset
        self._num_prefetch = num_prefetch

    def _transform_fn(self, data):
        """
        transform data with `data_info`
        """
        global_kjts = {path: data[idx] for idx, path in self._data_info.items()}
        cache_kjts, fetch_handles = self._id_transformer_group.transform(global_kjts)
        data = list(data)
        for idx, path in self._data_info.items():
            data[idx] = cache_kjts[path]
        return tuple(data), fetch_handles

    def __iter__(self):
        return DataLoaderIter(
            self._dataset, self._transform_fn, num_prefetch=self._num_prefetch
        )

    def __len__(self):
        return len(self._dataset)
