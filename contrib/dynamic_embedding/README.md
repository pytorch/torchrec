# TorchRec Dynamic Embedding

This folder contains the extension to support dynamic embedding for torchrec. Specifically, this extension enable torchrec to attach an external PS, so that when local GPU embedding is not big enough, we could pull/evict embeddings from/to the PS.

## Installation

After install torchrec, please clone the torchrec repo and manually install the dynamic embedding:

```bash
git clone git@github.com:pytorch/torchrec.git
cd contrib/dynamic_embedding
python setup.py install
```

And the dynamic embedding will be installed as a separate package named torchrec_dynamic_embedding.

We incorporate `gtest` for the C++ code and use unittest for the python APIs. The tests make sure that the implementation does not have any precision loss. Please turn on the `TDE_WITH_TESTING` in `setup.py` to run tests. Note that for the python test, one needs to set the environment variable `TDE_MEMORY_IO_PATH` to the path of the compiled `memory_io.so`.

## Usage

The dynamic embedding extension has only one api, `tde.wrap`, when wrapping the dataloader and model with it, we will automatically pipeline the data processing and model training. And example of `tde.wrap` is:

```python
import torchrec_dynamic_embedding as tde

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
dataloader = tde.wrap(
    "redis://127.0.0.1:6379/?prefix=model",
    dataloader,
    m,
    # configs of the embedding collections in the model
    { "emb1": config1, "emb2": config2 })

for label, kjt1, kjt2 in dataloader:
    output = m(kjt1, kjt2)
    ...
```

The internal of `tde.wrap` is in `src/torchrec_dynamic_embedding/dataloader.py`, where we will attach hooks to the embedding tensor as well as creating the dataloader thread for pipelining.

## Custom PS Extension

The dynamic embedding extension supports connecting with your PS cluster. To write your own PS extension, you need to create an dynamic library (`*.so`) with these 4 functions and 1 variable:

```c++
const char* IO_type = "your-ps";

void* IO_Initialize(const char* cfg);

void IO_Finalize(void* instance);

void IO_Pull(void* instance, IOPullParameter cfg);

void IO_Push(void* instance, IOPushParameter cfg);
```

And then use the following python API to register it:

```python
torch.ops.tde.register_io(so_path)
```

After that, you could use your own PS extension by passing the corresponding URL into `tde.wrap`, where the protocol name would be the `IO_type` and the string after `"://"` will be passed to `IO_Finalize` (`"type://cfg"`).
