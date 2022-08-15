import queue
import threading

import torch
from torch.utils.data._utils import MP_STATUS_CHECK_INTERVAL


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
        return self._data_queue.get()


class DataLoader:
    def __init__(self, dataset, transform_fn, num_prefetch=0):
        self._dataset = dataset
        self._transform_fn = transform_fn
        self._num_prefetch = num_prefetch

    def __iter__(self):
        return DataLoaderIter(
            self._dataset, self._transform_fn, num_prefetch=self._num_prefetch
        )

    def __len__(self):
        return len(self._dataset)
