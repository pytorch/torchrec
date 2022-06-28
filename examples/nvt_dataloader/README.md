# Running torchrec using NVTabular DataLoader

First run nvtabular preprocessing to first convert the criteo TSV files to parquet, and perform offline preprocessing. For example
```
cd torchrec/torchrce/datasets/scripts/nvt/
bash nvt_preproc.sh /data/criteo_tb /data 8096

```

To run locally
```
torchx run -s local_cwd dist.ddp -j 1x8 --script train_torchrec.py -- --num_embeddings_per_feature 45833188,36746,17245,7413,20243,3,7114,1441,62,29275261,1572176,345138,10,2209,11267,128,4,974,14,48937457,11316796,40094537,452104,12606,104,35 --over_arch_layer_sizes 1024,1024,512,256,1 --binary_path /data/home/renfeichen/criteo_correct_output/criteo_binary/split/ --change_lr --learning_rate 15.0 --validation_freq_within_epoch 5000 --throughput_check_freq_within_epoch 200 --batch_size 2048
```

# Test on A100s

## Preliminary Training Results

**Setup:**
* Dataset: Criteo 1TB Click Logs dataset
* CUDA 11.1, NCCL 2.10.3.
* AWS p4d24xlarge instances, each with 8 40GB NVIDIA A100s.

**Results**

Common settings across all runs:

```
--num_embeddings_per_feature 45833188,36746,17245,7413,20243,3,7114,1441,62,29275261,1572176,345138,10,2209,11267,128,4,974,14,48937457,11316796,40094537,452104,12606,104,35 --over_arch_layer_sizes 1024,1024,512,256,1 --binary_path /data/home/renfeichen/criteo_correct_output/criteo_binary/split/ --change_lr --learning_rate 15.0 --validation_freq_within_epoch 5000 --throughput_check_freq_within_epoch 200 --batch_size 2048
```

|Number of GPUs|Collective Size of Embedding Tables (GiB)|Local Batch Size|Global Batch Size|AUROC over Val Set After 1 Epoch|AUROC Over Test Set After 1 Epoch|Train Records/Second|Time to Train 1 Epoch | Unique Flags |
--- | --- | --- | --- | --- | --- | --- | --- | ---
|8|91.10|2048|16384|0.8039939403533936|0.7984522581100464|~1,995,163 rec/s| 37m12s | `--batch_size 2048 --lr_change_point 0.80 --lr_after_change_point 0.20` |
