# TorchRec retrieval example

This example demonstrates training a distributed TwoTower (i.e. User-Item) Retrieval model that is sharded using TorchRec. The projected item embeddings are added to an [IVFPQ](https://github.com/facebookresearch/faiss/wiki/Lower-memory-footprint) [FAISS]((https://github.com/facebookresearch/faiss)) index for candidate generation. The retrieval model and KNN lookup are bundled in a Pytorch model for efficient end-to-end retrieval.

This example contains two scripts:

`two_tower_train.py` trains a simple Two Tower (UV) model, which is a simplified version of [A Dual Augmented Two-tower Model for Online Large-scale Recommendation](https://dlp-kdd.github.io/assets/pdf/DLP-KDD_2021_paper_4.pdf). Torchrec is used to shard the model, and is pipelined so that dataloading, data-parallel to model-parallel comms, and forward/backward are overlapped. It is trained on random data in the format of [MovieLens 20M](https://grouplens.org/datasets/movielens/20m/) dataset in SPMD fashion. The distributed model is gathered to CPU. The item (movie) towers embeddings are used to train a FAISS [IVFPQ](https://github.com/facebookresearch/faiss/wiki/Lower-memory-footprint) index, which is serialized. The resulting `KNNIndex` can be queried with batched `torch.Tensor`, and will return the distances and indices for the approximate K nearest neighbors of the query embeddings. The model itself is also serialized.

`two_tower_retrieval.py` loads the serialized model and FAISS index from `two_tower_train.py`. A `TwoTowerRetrieval` model is instantiated, which wraps the `KNNIndex`, the query (user) tower and the candidate item (movie) tower inside an `nn.Module`. The retreival model is quantized using [`torchrec.quant`](https://pytorch.org/torchrec/torchrec.quant.html). The serialized `TwoTower` model weights trained before are converted into `TwoTowerRetrieval` which are loaded into the retrieval model. The seralized trained FAISS index is also loaded. The entire retreival model can be queried with a batch of candidate (user) ids and returns logits which can be used in ranking.

# Running

## Installation

`conda install -c conda-forge pytorch faiss-gpu`
`pip install torchx`

## Torchx
We recommend using [torchx](https://pytorch.org/torchx/main/quickstart.html) to run. Here we use the [DDP builtin](https://pytorch.org/torchx/main/components/distributed.html)

1. (optional) setup a slurm or kubernetes cluster
2.
    a. locally: `torchx run -s local_cwd dist.ddp -j 1x2 --gpu 2 --script two_tower_train.py` -- --save_dir <path_to_save_shards>
    b. remotely: `torchx run -s slurm dist.ddp -j 1x8 --gpu 8 --script two_tower_train.py` -- --save_dir <path_to_save_shards>
3. CUDA_VISIBLE_DEVICES=0,1 python two_tower_retrieval.py --load_dir <path_to_save_shards>


## Notes

There are are few things which a user of this example will likely want to do that aren't included in this example:
1. Set up recall statistics (e.g Top-K retrieval accuracy). These can be used to tune the FAISS parameters. e.g which nprobe one would want to use (and nlist) would depend upon desired recall parameters. nprobe = nlist is equivalent to a brute-force search through everything. Sweeping powers of 2 (e.g., 1, 2, 4, 8, ..,) for nprobe and picking one of those based on recall statistics is typically done.
2. Experiments to understand the relationship between the number of devices used in training and inference, model quantity and cost will likely be used to set the number of devices uses in inference.
