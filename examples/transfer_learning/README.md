# Transfer Learning example

This examples showcases training a distributed model using TorchRec. The embeddings are initialized with pretrained values (assumed to be loaded from storage, such as parquet). The value is large enough that we use the `share_memory_` API to load the tensors from shared memory.

See [`torch.multiprocessing`](https://pytorch.org/docs/stable/multiprocessing.html) and [best practices](https://pytorch.org/docs/1.6.0/notes/multiprocessing.html?highlight=multiprocessing) for more information on shared memory.

## Running

`python train_from_pretrained_embedding.py`
