# TorchRec Benchmarks for `EmbeddingBag`

We evaluate the performance of two EmbeddingBagCollection modules:

1. `EmbeddingBagCollection` (EBC) ([code](https://pytorch.org/torchrec/torchrec.modules.html#torchrec.modules.embedding_modules.EmbeddingBagCollection)): a simple module backed by [torch.nn.EmbeddingBag](https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html).

2. `FusedEmbeddingBagCollection` (Fused EBC) ([code](https://github.com/pytorch/torchrec/blob/main/torchrec/modules/fused_embedding_bag_collection.py#L299)): a module backed by [FBGEMM](https://github.com/pytorch/FBGEMM) kernels which enables more efficient, high-performance operations on embedding tables. It is equipped with a fused optimizer, and UVM caching/management that makes much larger memory available for GPUs.


## Module architecture and running setup

We chose the embedding tables (sparse arch) in ML Perf [DLRM](https://github.com/facebookresearch/dlrm/tree/main/torchrec_dlrm) as the model to compare the performance difference between EBC and Fused EBC. Below are the settings on the embedding tables:
```
num_embeddings_per_feature = [45833188, 36746, 17245, 7413, 20243, 3, 7114, 1441, 62, 29275261, 1572176, 345138, 10, 2209, 11267, 128, 4, 974, 14, 48937457, 11316796, 40094537, 452104, 12606, 104, 35]
embedding_dim_size = 128
```

Other setup includes:
-   Optimizer: Stochastic Gradient Descent (SGD)
-   Dataset: Random dataset ([code](https://pytorch.org/torchrec/torchrec.datasets.html#module-torchrec.datasets.random))
-   CUDA 11.7, NCCL 2.11.4.
-   AWS EC2 instance with 8 16GB NVIDIA Tesla V100


## How to run

After the installation of Torchrec (see "Binary" in the "Installation" section,  [link](https://github.com/pytorch/torchrec)), run the following command under the benchmark directory (/torchrec/torchrec/benchmarks):

```
python ebc_benchmarks.py [--mode MODE] [--cpu_only]
```

where `MODE` can be specified as `ebc_comparison_dlrm` (default) / `fused_ebc_uvm` / `ebc_comparison_scaling` to see different comparisons.


## Results

### Methodology

To ease the reading, we use "DLRM EMB" to abbreviate "DLRM embedding tables" from below. Since 1 GPU can't accommondate the full sized tables in DLRM, we need to reduce the `embedding_dim` of the 5 largest tables to some degree (see the "Note" column in the following tables for the reduction degree). For the metrics, we use the average training time over 100 epochs to represent the performance of each module. `speedup` (defined as `training time using EBC` divided by `training time using Fused EBC`) is also computed to demonstrate the degree of improvement from EBC to Fused EBC.

### 1. Comparison between EBC and FusedEBC on DLRM EMB (`ebc_comparison_dlrm`)

We see that Fused EBC has much faster training efficiency compared to EBC. The speedup from EBC to Fused EBC is 13X, 18X and 23X when the DLRM EMB is reduced by 128 times, 64 times and 32 times, respectively.

| Module | Time to train one epoch | Note |
| ------ | ---------------------- | ---- |
| EBC | 0.267 (+/- 0.002) second | DLRM EMB with sizes of the 5 largest tables reduced by 128 times |
| EBC | 0.332 (+/- 0.002) second | DLRM EMB with sizes of the 5 largest tables reduced by 64 times |
| EBC | 0.462 (+/- 0.002) second | DLRM EMB with sizes of the 5 largest tables reduced by 32 times |
| Fused EBC | 0.019 (+/- 0.001) second | DLRM EMB with sizes of the 5 largest tables reduced by 128 times |
| Fused EBC | 0.019 (+/- 0.001) second | DLRM EMB with sizes of the 5 largest tables reduced by 64 times |
| Fused EBC | 0.019 (+/- 0.009) second | DLRM EMB with sizes of the 5 largest tables reduced by 32 times |

### 2. Full sized DLRM EMB w/ UVM/UVM-caching w/ FusedEBC (`fused_ebc_uvm`)

Here, we demonstrate the advantage of UVM/UVM-caching with Fused EBC. With UVM caching enabled, we can put larger sized tables in DLRM EMB in Fused EBC without significant sacrifice on the efficiency. With UVM enabled, we can allocate full sized DLRM EMB in UVM, with expected slower training performance because of the extra sync points between host and GPU (see [this example](https://github.com/pytorch/torchrec/blob/main/examples/sharding/uvm.ipynb) for more UVM explanation/usage).

| Module | Time to train one epoch | Note |
| ------ | ---------------------- | ---- |
|Fused EBC with UVM caching | 0.06 (+/- 0.37) second | DLRM EMB with size of the 5 largest tables reduced by 2 |
|Fused EBC with UVM | 0.62 (+/- 5.34) second | full sized DLRM EMB |

The above performance comparison is also put in a bar chart for better visualization.
![EBC_benchmarks_dlrm_emb](https://github.com/pytorch/torchrec/tree/main/benchmarks/EBC_benchmarks_dlrm_emb.png)


### 3. Comparison between EBC and fused_EBC on different sized embedding tables (`ebc_comparison_scaling`)

Here, we study how the scaling on the embedding table affects the performance difference between EBC and Fused EBC. In doing so, we vary three parameters, `num_tables`, `embedding_dim` and `num_embeddings`, and present `speedup` from EBC to Fused EBC in the following tables. In each table, we observe that `embedding_dim` and `num_embeddings` do not have significant effect on speedup. However, as `num_tables` increases, the improvement from EBC to Fused EBC becomes higher (speedup increases), suggesting the benefit of Fused EBC when it is to deal with many embedding tables.


-  `num_tables` = 10

|||——————|——————| `embedding_dim` |——————|——————|—————>|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
||| **4** | **8** | **16** | **32** | **64** | **128** |
|&#124;|    **4** | 2.87 | 2.79 | 2.79 | 2.79 | 2.76 | 2.8 |
|&#124;|    **8** | 2.71 | 3.11 | 2.97 | 3.02 | 2.99 | 2.95 |
|&#124;|   **16** | 2.98 | 2.97 | 2.98 | 2.97 | 3.0 | 3.05 |
|&#124;|   **32** | 3.01 | 2.95 | 2.99 | 2.98 | 2.98 | 3.01 |
|&#124;|   **64** | 3.0 | 3.02 | 3.0 | 2.97 | 2.96 | 2.97 |
|**`num_embeddings`**|  **128** | 3.03 | 2.96 | 3.02 | 3.0 | 3.02 | 3.05 |
|&#124;|  **256** | 3.01 | 2.95 | 3.0 | 3.03 | 3.05 | 3.02 |
|&#124;| **1024** | 3.0 | 3.05 | 3.05 | 3.08 | 5.89 | 3.07 |
|&#124;| **2048** | 2.99 | 3.03 | 3.0 | 3.05 | 3.0 | 3.06 |
|&#124;| **4096** | 3.0 | 3.03 | 3.05 | 3.02 | 3.07 | 3.05 |
|V|      **8192** | 3.0 | 3.08 | 3.04 | 3.02 | 3.09 | 3.1 |


-  `num_tables` = 100

|||——————|——————| `embedding_dim` |——————|——————|—————>|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
||| **4** | **8** | **16** | **32** | **64** | **128** |
|&#124;|    **4** | 10.33 | 10.36 | 10.26 | 10.24 | 10.28 | 10.24 |
|&#124;|    **8** | 10.34 | 10.47 | 10.29 | 10.25 | 10.23 | 10.19 |
|&#124;|   **16** | 10.18 | 10.36 | 10.2 | 10.28 | 10.25 | 10.26 |
|&#124;|   **32** | 10.41 | 10.2 | 10.19 | 10.2 | 10.04 | 9.89 |
|&#124;|   **64** | 9.93 | 9.9 | 9.73 | 9.89 | 10.17 | 10.16 |
|**`num_embeddings`**|  **128** | 10.32 | 10.11 | 10.12 | 10.08 | 10.01 | 10.05 |
|&#124;|  **256** | 10.57 | 8.39 | 10.36 | 10.21 | 10.14 | 10.43 |
|&#124;| **1024** | 10.39 | 9.67 | 8.46 | 10.23 | 10.29 | 10.11 |
|&#124;| **2048** | 10.0 | 9.74 | 10.0 | 9.67 | 10.08 | 11.87 |
|&#124;| **4096** | 9.94 | 9.82 | 10.17 | 9.66 | 9.87 | 9.95 |
|V|      **8192** | 9.81 | 10.23 | 10.12 | 10.18 | 10.36 | 9.57 |


-  `num_tables` = 1000

|||——————|——————| `embedding_dim` |——————|——————|—————>|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
||| **4** | **8** | **16** | **32** | **64** | **128** |
|&#124;|    **4** | 13.81 | 13.56 | 13.33 | 13.33 | 13.24 | 12.86 |
|&#124;|    **8** | 13.44 | 13.4 | 13.39 | 13.41 | 13.39 | 13.09 |
|&#124;|   **16** | 12.55 | 12.88 | 13.22 | 13.19 | 13.27 | 12.95 |
|&#124;|   **32** | 13.17 | 12.84 | 12.8 | 12.78 | 13.13 | 13.07 |
|&#124;|   **64** | 13.06 | 12.84 | 12.84 | 12.9 | 12.83 | 12.89 |
|**`num_embeddings`**|  **128**| 13.14 | 13.04 | 13.16 | 13.21 | 13.08 | 12.91 |
|&#124;|  **256** | 13.71 | 13.59 | 13.76 | 13.24 | 13.36 | 13.59 |
|&#124;| **1024** | 13.24 | 13.29 | 13.56 | 13.64 | 13.68 | 13.79 |
|&#124;| **2048** | 13.2 | 13.19 | 13.35 | 12.44 | 13.32 | 13.17 |
|&#124;| **4096** | 12.96 | 13.24 | 12.51 | 12.99 | 12.47 | 12.34 |
|V|      **8192** | 12.84 | 13.32 | 13.27 | 13.06 | 12.35 | 12.58 |
