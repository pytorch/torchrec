# TorchRec FAQ

Frequently asked questions about TorchRec

## Table of Contents

- [General Concepts](#general-concepts)
- [Sharding and Distributed Training](#sharding-and-distributed-training)
- [Memory Management and Performance](#memory-management-and-performance)
- [Integrating with Existing Systems](#integrating-with-existing-systems)
- [Technical Challenges](#technical-challenges)
- [Model Design and Evaluation](#model-design-and-evaluation)

## General Concepts

### What are TorchRec and FSDP, and when should they be used?

**TorchRec** is a PyTorch domain library with primitives for large-scale distributed embeddings, usually used in recommendation systems. Use it when dealing with models containing massive embedding tables that exceed single-GPU memory.

**FSDP (Fully Sharded Data Parallel)** is a PyTorch distributed training technique that shards dense model parameters, gradients, and optimizer states across GPUs, reducing memory footprint for large models. Use it for training large language models or other general deep learning architectures that require scaling across multiple GPUs.

### Can TorchRec do everything FSDP can do for sparse embeddings, and vice versa?

- **TorchRec** offers specialized sharding strategies and optimized kernels designed for sparse embeddings, making it more efficient for this specific task.
- **FSDP** can work with models containing sparse embeddings, but it might not be as optimized or feature-rich as TorchRec for this specific task. For recommendation systems, TorchRec's methods are often more memory efficient due to their focus on sparse data characteristics.
- For optimal results in recommendation systems with large sparse embeddings, combine TorchRec for embeddings and FSDP for the dense parts of the model.

### Does TorchRec support DTensor?

Yes, TorchRec models can benefit from DTensor support in PyTorch distributed components, like FSDP2. This improves distributed training performance, efficiency, and interoperability between TorchRec and other DTensor-based components.

## Sharding and Distributed Training

### How do you choose the best sharding strategy for embedding tables?

TorchRec offers multiple sharding strategies:
- Table-Wise (TW)
- Row-Wise (RW)
- Column-Wise (CW)
- Table-Wise-Row-Wise (TWRW)
- Grid-Shard (GS)
- Data Parallel (DP)

Consider factors like embedding table size, memory constraints, communication patterns, and load balancing when selecting a strategy.

The TorchRec Planner can automatically find an optimal sharding plan based on your hardware and settings.

### How does the TorchRec planner work, and can it be customized?

The Planner aims to balance memory and computation across devices. You can influence the planner using ParameterConstraints, providing information like pooling factors. TorchRec also features automated sharding based on cost modeling and deep reinforcement learning called AutoShard.


## Memory Management and Performance

### How do you manage the memory footprint of large embedding tables?

- Choose an optimal sharding strategy
- If GPU memory is not sufficient, TorchRec provides options to offload embeddings to CPU (UVM) and SSD memory


## Integrating with Existing Systems

### Can TorchRec modules be easily converted to TorchScript for deployment and inference in C++ environments?

Yes, TorchRec modules can be traced and scripted for TorchSCript inference in C++ environments. However, it's recommended to script only the non-embedding layers for better performance and to handle potential limitations with sharded embedding modules in TorchScript.

## Technical Challenges

### Why are you getting row-wise alltoall errors when combining different pooling types?

This can occur due to incompatible sharding and pooling types, resulting in communication mismatches during data aggregation. Ensure your sharding and pooling choices align with the communication patterns required.

### How do you handle floating point exceptions when using quantized embeddings with float32 data types?

- Implement gradient clipping
- Monitor gradients and weights for numerical issues
- Consider using different scaling strategies like amp
- Accumulate gradients over mini-batches

### What are best practices for handling scenarios with empty batches for EmbeddingCollection?

Handle empty batches by filtering them out, skipping lookups, using default values, or padding and masking them accordingly.

### What are common causes of issues during the forward() graph and optimizer step()?

- Incorrect input data format, type, or device
- Invalid embedding lookups (out-of-range indices, mismatched names)
- Issues in the computational graph preventing gradient flow
- Incorrect optimizer setup, learning rate, or fusion settings

### What is the role of fused optimizers in TorchRec?
TorchRec uses fused optimizers, often with DistributedModelParallel, where the optimizer update is integrated into the backward pass. This prevents the materialization of embedding gradients, leading to significant memory savings. You can also opt for a dense optimizer for more control.

## Model Design and Evaluation

### What are best practices for designing recommendation models with TorchRec?

- Carefully select and preprocess features
- Choose suitable model architectures for your recommendation task
- Leverage TorchRec components like EmbeddingBagCollection and optimized kernels
- Design the model with distributed training in mind, considering sharding and communication patterns

### What are the most effective methods for evaluating recommendation systems built with TorchRec?

**Offline Evaluation**:
- Use metrics like AUC, Recall@K, Precision@K, and NDCG@K
- Employ train-test splits, cross-validation, and negative sampling

**Online Evaluation**:
- Conduct A/B tests in production
- Measure metrics like click-through rate, conversion rate, and user engagement
- Gather user feedback
