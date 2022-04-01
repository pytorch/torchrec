# TorchRec Bert4Rec Example

`bert4rec_main.py` trains, validates, and tests a [BERT4REC](https://arxiv.org/abs/1904.06690) with TorchRec. The Bert4Rec is a Sequential Recommendation with Bidirectional Encoder Representations from Transformer and the model contains both data parallel components (e.g. transformation blocks) and model parallel components (e.g. item embeddings). The Bert4Rec model can be run with either a random dataloader or the movielens dataset (https://grouplens.org/datasets/movielens/). Notice that the final output layer of the model is different from the original paper which used the embedding layer weight to get the final output by matmul, from our experimentation, we used a linear layer instead which brings us higher performance

It has been tested on the following cloud instance types:
| Cloud | Instance Type | GPUs     | vCPUs | Memory (GB) |
|-------|---------------|----------|-------|-------------|
| AWS   | p3.16xlarge   | 8 x V100 | 64    | 488         |

# Running

## Torchx
We recommend using [torchx](https://pytorch.org/torchx/main/quickstart.html) to run. Here we use the [DDP builtin](https://pytorch.org/torchx/main/components/distributed.html)

1. pip install torchx
2. (optional) setup a slurm or kubernetes cluster
3.
    a. locally for DMP: `torchx run -s local_cwd dist.ddp -j 1x2 --script bert4rec_main.py -- --dataset_name random --mode dmp`
    b. locally for DDP: `torchx run -s local_cwd dist.ddp -j 1x2 --script bert4rec_main.py -- --dataset_name random --mode ddp`
    c. remotely for DMP: `torchx run -s slurm dist.ddp -j 1x8 --script bert4rec_main.py -- --dataset_name random --mode dmp`
    d. remotely for DDP: `torchx run -s slurm dist.ddp -j 1x8 --script bert4rec_main.py -- --dataset_name random --mode ddp`


## Preliminary Training Results

**Setup:**
* Dataset: Random Dataset
* CUDA 11.3, NCCL 2.10.3.
* AWS p3.16xlarge instances, each with 8 Tesla V100s.
* Shared flags for all runs include: `--dataset_name random --random_user_count 3000 --random_item_count 500000 --random_size 8000000 --lr 0.001 --mask_prob 0.2 --train_batch_size 8 --val_batch_size 8 --max_len 16 --emb_dim 128 --num_epochs 1 --mode dmp`


**Results**

###### Recommendation Metrics Reproduce

| Dataset | Metrics | Paper  | Repro  |
|---------|---------|--------|--------|
| ML-1M   | HR@1    | 0.2863 | 0.2636 |
|         | HR@5    | 0.5876 | 0.5895 |
|         | HR@10   | 0.6970 | 0.7078 |
|         | NDCG@5  | 0.4454 | 0.4356 |
|         | NDCG@10 | 0.4818 | 0.4740 |
| ML-20M  | HR@1    | 0.3440 | 0.3273 |
|         | HR@5    | 0.6323 | 0.6399 |
|         | HR@10   | 0.7473 | 0.7601 |
|         | NDCG@5  | 0.4967 | 0.4929 |
|         | NDCG@10 | 0.5340 | 0.5319 |

###### Infra Metrics Improvement

|Number of GPUs|Local Batch Size|Global Batch Size|DDP Train Records/Second|DMP Train Records/Second|Improvements |
| --- | --- | --- | --- | --- | --- | ---
|1|8|8|242.56 rec/s| 324.24 rec/s | 33.6%
|4|8|32|757.12 rec/s| 1058.56 rec/s | 39.8%
|8|8|64|1592.32 rec/s | 2225.28 rec/s | 39.8%

**Reproduce**

Run the following command to reproduce the results for a single node (8 GPUs) on AWS.

```
torchx run -s local_cwd dist.ddp -j 1x8 --script bert4rec_folder/bert4rec_main.py -- --dataset_name random --random_user_count 3000 --random_item_count 500000 --random_size 8000000 --lr 0.001 --mask_prob 0.2 --train_batch_size 8 --val_batch_size 8 --max_len 16 --emb_dim 128 --num_epochs 1 --mode dmp
```
Upon scheduling the job, there should be an output that looks like this:

```
torchx 2022-03-23 05:38:40 INFO     Log files located in: /tmp/torchx_fby9w0w9/torchx/bert4rec_main-txq11m6736w20c/bert4rec_main/0
local_cwd://torchx/bert4rec_main-txq11m6736w20c
torchx 2022-03-23 05:38:40 INFO     Waiting for the app to finish...
bert4rec_main/0 WARNING:__main__:
bert4rec_main/0 *****************************************
bert4rec_main/0 Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
bert4rec_main/0 *****************************************
bert4rec_main/0 [2]:Filtering triplets
bert4rec_main/0 [7]:Filtering triplets
bert4rec_main/0 [4]:Filtering triplets
bert4rec_main/0 [6]:Filtering triplets
bert4rec_main/0 [3]:Filtering triplets
bert4rec_main/0 [5]:Filtering triplets
bert4rec_main/0 [2]:Densifying index
bert4rec_main/0 [7]:Densifying index
bert4rec_main/0 [4]:Densifying index
bert4rec_main/0 [6]:Densifying index
bert4rec_main/0 [3]:Densifying index
bert4rec_main/0 [5]:Densifying index
bert4rec_main/0 [1]:Filtering triplets
bert4rec_main/0 [0]:Filtering triplets
bert4rec_main/0 [1]:Densifying index
bert4rec_main/0 [0]:Densifying index
bert4rec_main/0 [2]:Splitting
bert4rec_main/0 [2]:<pandas.core.groupby.generic.DataFrameGroupBy object at 0x7f52d986dcd0>
bert4rec_main/0 [6]:Splitting
bert4rec_main/0 [6]:<pandas.core.groupby.generic.DataFrameGroupBy object at 0x7f22b398bbe0>
bert4rec_main/0 [7]:Splitting
bert4rec_main/0 [7]:<pandas.core.groupby.generic.DataFrameGroupBy object at 0x7f90fe1b3be0>
bert4rec_main/0 [4]:Splitting
bert4rec_main/0 [4]:<pandas.core.groupby.generic.DataFrameGroupBy object at 0x7f6f53e3ccd0>
bert4rec_main/0 [3]:Splitting
bert4rec_main/0 [3]:<pandas.core.groupby.generic.DataFrameGroupBy object at 0x7f3babf7ecd0>
bert4rec_main/0 [5]:Splitting
bert4rec_main/0 [5]:<pandas.core.groupby.generic.DataFrameGroupBy object at 0x7f675bedbcd0>
bert4rec_main/0 [1]:Splitting
bert4rec_main/0 [1]:<pandas.core.groupby.generic.DataFrameGroupBy object at 0x7f84351f7cd0>
```

**TODO/Work In Progress**
* Run QPS test on A100s machines
* Use Ads Synthetic data to get the QPS number
