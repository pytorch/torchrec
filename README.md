# TorchRec (🚨 Warning: Unstable Prototype 🚨)

TorchRec is a PyTorch domain library built to provide common sparsity & parallelism primitives needed for large-scale recommender systems (RecSys). It allows authors to train models with large embedding tables sharded across many GPUs.

## TorchRec contains:
- Parallelism primitives that enable easy authoring of large, performant multi-device/multi-node models using hybrid data-parallelism/model-parallelism.
- The TorchRec sharder can shard embedding tables with different sharding strategies including data-parallel, table-wise, row-wise, table-wise-row-wise, and column-wise sharding.
- The TorchRec planner can automatically generate optimized sharding plans for models.
- Pipelined training overlaps dataloading device transfer (copy to GPU), inter-device communications (input_dist), and computation (forward, backward) for increased performance.
- Optimized kernels for RecSys powered by FBGEMM.
- Quantization support for reduced precision training and inference.
- Common modules for RecSys.
- Production-proven model architectures for RecSys.
- RecSys datasets (criteo click logs and movielens)
- Examples of end-to-end training such the dlrm event prediction model trained on criteo click logs dataset.

## Installation instructions
[Install FBGEMM] (https://github.com/pytorch/FBGEMM/tree/main/fbgemm_gpu#build-notes)
`pip install -e .`
