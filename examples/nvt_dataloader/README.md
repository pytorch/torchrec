# Running torchrec using NVTabular DataLoader

First run nvtabular preprocessing to first convert the criteo TSV files to parquet, and perform offline preprocessing.

Please follow the installation instructions in the [README](https://github.com/pytorch/torchrec/tree/main/torchrec/datasets/scripts/nvt) of torchrec/torchrec/datasets/scripts/nvt.

Afterward, to run the model across 8 GPUs, use the below command

```
torchx run -s local_cwd dist.ddp -j 1x8 --script train_torchrec.py -- --num_embeddings_per_feature 40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36 --over_arch_layer_sizes 1024,1024,512,256,1 --dense_arch_layer_sizes 512,256,128 --embedding_dim 128 --binary_path <path_to_nvt_output>/criteo_binary/split/ --learning_rate 1.0 --validation_freq_within_epoch 1000000 --throughput_check_freq_within_epoch 1000000 --batch_size 256
```

To run with adagrad as an optimizer, use the below flag

```
---adagrad
```

# Test on A100s

## Preliminary Training Results

**Setup:**
* Dataset: Criteo 1TB Click Logs dataset
* CUDA 11.1, NCCL 2.10.3.
* AWS p4d24xlarge instances, each with 8 40GB NVIDIA A100s.

**Results**

Reproducing MLPerfV1 settings
1. Embedding per features + model architecture
2. Learning Rate fixed at 1.0 with SGD
3. Dataset setup:
    - No frequency thresholding
4. Report > .8025 on validation set (0.8027645945549011 from above script)
5. Global batch size 2048
