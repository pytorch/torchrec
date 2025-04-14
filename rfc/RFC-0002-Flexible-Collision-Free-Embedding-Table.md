# RFC: Flexible Collision Free Embedding Table

| Status | Published |
| :---- | :---- |
| Author(s) | Emma Lin, Joe Wang, Kaustubh Vartak, Dennis van der Staay, Huanyu He|
| Updated | 04-13-2025 |

## Motivation

PyTorch and FBGemm utilize fixed-size continuous embedding memory to handle sparse features, employing a uniform hash function to map raw IDs to a limited index space. This approach has resulted in a high rate of hash collisions and inefficient storage.

The Zero Collision Hash (ZCH) technique allows models to train individual IDs uniquely, leading to notable enhancements in model freshness and user engagement. When properly configured, we've seen improvements in freshness late stage ranking and early stage ranking models. However, the remapping-based ZCH solutions have presented several scalability challenges.

## Objective

This RFC proposes a new embedding table format that natively supports collision-free features, enhancing the scalability and usability of embedding tables.

The approach involves utilizing an extremely large hash size (e.g., 2^63) to map raw IDs to a vast ID space, significantly reducing the likelihood of collisions during hashing.

Instead of mapping IDs to a fixed table size (e.g., 1 billion rows), this format reserves memory spaces for each ID, effectively eliminating collisions and achieving native collision-free results.

Notably, this design eliminates the need for a remapper to find available slots for colliding IDs, streamlining the process and improving overall efficiency, embedding scalability and usability.

## Design Proposal

### Bucket Aware Sharding and Resharding Algorithm

To address the dynamic nature of sparse IDs and their distribution, we propose introducing a bucket concept. We will provide a large default bucket number configuration and an extremely large table size. The mapping from ID to bucket ID can be done in two ways:

* Interleave-based: bucket\_id \= hash\_id % total\_bucket\_number
  This approach is similar to the sharding solution used in MPZCH, allowing for seamless migration without requiring ID resharding.
* Chunk-based:
  * bucket\_size \= table\_size / total\_bucket\_number,
  * bucket\_id \= id / bucket\_size


Both options will be configurable.

After sharding IDs into buckets, we will distribute the buckets sequentially across trainers. For example, with 1000 buckets and 100 trainers, each trainer would own 10 consecutive buckets.

T1: b0-b9

T2: b10-b19

...

When resharding is necessary, we will move buckets around instead of individual rows. For instance, reducing the number of trainers from 100 to 50 would result in each trainer owning 20 consecutive buckets.

T1: b0-b19

T2: b20-b39

...

The row count within each bucket can vary from 0 to the maximum bucket size, depending on the ID distribution. However, using a larger bucket number should lead to a more even distribution of IDs.

#### Benefit

The bucket number remains unchanged when scaling the model or adjusting the number of trainers, making it easier to move buckets around without introducing additional overhead to reshard every ID's new location.

Resharding every ID can be an expensive operation, especially for large tables (e.g., over 1 billion rows).

### Bucketized Torchrec Sharding and Input Distribution

Based on the proposed sharding definition, the TorchRec sharder needs to be aware of the bucket configuration from the embedding table.

Input distribution needs to take into account the bucket configuration, and then distribute input to the corresponding trainer.

Here is the code [reference](https://github.com/pytorch/torchrec/blob/f36d26db4432bd7335f6df9e7e75d8643b7ffb04/torchrec/distributed/sharding/rw_sequence_sharding.py#L129C16-L129C36).

### FBGemm Operators Optimization for Collision Free EmbeddingTable

FBGEMM\_GPU (FBGEMM GPU Kernels Library) is highly optimized for fixed sized tables, with continuous memory space, including in HBM, UVM or CPU memory.
However, when we apply collision free idea, there are several assumptions of FBGEMM are broken:

* Table size is not fixed. It could grow over training iterations or shrink after eviction.
* Embedding lookup input is not embedding offset anymore, so we need to maintain an explicit mapping from input to the embedding value.
* Table size could exceed memory limitation, but actual trained id size is finite, so we cannot preserve memory based on table configuration.

We’re looking for an optimized K/V FBGemm version to support flexible memory management.

#### Training Operator (from [Joe Wang](mailto:wangj@meta.com))

* Optimized CPU memory management with K/V format
  * Reduce memory fragmentation
  * efficient memory utilization
  * Fast lookup performance
  * Flexible eviction policy
* Collision free LXU cache to avoid extra memory copy from CPU to UVM and UVM memory read during embedding lookup.
  * The current LXU cache used by FBGemm could cause id collisions. When collision happens, prefetch won’t be able to load embedding value to HBM, which will fallback to UVM memory read during embedding lookup. This can impact training QPS in two ways:
    * Introduce one extra CPU memory copy, since data needs to be copied from CPU to UVM, since the CPU embedding data in k/v format might not be accessible from the GPU card.
    * Introduced H2D data copy in embedding lookup.

[Here](https://github.com/pytorch/FBGEMM/blob/main/fbgemm_gpu/fbgemm_gpu/tbe/ssd/training.py) is the code reference of a k/v format SSD offloading operator, and provides a backend interface to hook up other k/v store implementations.
We propose to implement a new k/v store backend, to decouple SSD and rocksDB dependency, but the SSD backend operator can be used for extra large embeddings which do not fit into host memory.

#### Inference Operator

On top of training operators functionality, the inference operator needs to support dequantization from nbit int value after embedding is queried out from the embedding store. We’d like to have a fast inference operator with additional requirements:

* Optimized CPU memory management with k/v format
* Collision free LXU cache
* Keep the fast nbit Int data format support, with pooling, dequantization features.
* Support decoupled large tensor loading and reset, to allow model state in-place update.
  [Here](https://github.com/pytorch/FBGEMM/blob/main/fbgemm_gpu/fbgemm_gpu/split_table_batched_embeddings_ops_inference.py) is the current inference operator which only supports offset based access for now.

### Enhancing TorchRec's Prefetch Pipeline for Synchronized Training

TorchRec offers multiple training [pipelines](https://github.com/pytorch/torchrec/blob/main/torchrec/distributed/train_pipeline/train_pipelines.py) that help overlap communication and computation, reducing embedding lookup latency and improving training QPS. Specifically, PrefetchTrainPipelineSparseDist supports synchronized training, while TrainPipelineSemiSync supports asynchronous training.

Our goal is to further enhance the prefetch pipeline to enable prefetching multiple training data batches ahead of time while maintaining synchronized training and avoiding step misalignment.

**Design Principles:**

* Zero Collision Cache: Enable zero collision cache in GPU to cache embeddings for multiple batches without collision-induced cache misses.
* Forward Pass: Perform embedding lookups only in HBM during forward passes to improve performance.
* Backward Pass: Update embeddings in HBM synchronously during backward passes to ensure all embedding lookup results are up-to-date.
* Asynchronous UVM Embedding Update: Update UVM embeddings asynchronously after embedding updates in HBM.

**End Goal:**

Achieve on-par performance with GPU HBM-based training while scaling up sparse embedding tables in CPU memory.

### Warm Start and Transfer Learning with Collision-Free Embedding Tables

Warm start, or transfer learning, is a widely used technique in industry to facilitate model iteration while maintaining on-par topline metrics. However, the introduction of collision-free embedding tables poses challenges to this process.

With the ability to increase table size and feature hash size to \~2^60, collision-free embedding tables offer improved efficiency. However, since id hash size is changed and sharding solution is different, when resuming training from a non-zero collision table to a zero-collision table, the redistribution of IDs across trainers becomes computationally expensive.

#### Solution: Backfilling Embedding Values

To address this challenge, we propose the following solution:

* Create Two Embedding Tables: One table is copied from the old model, and the other is the new table.

* Freeze Old Embedding Table: The old embedding table is set to read-only mode in the new model.

* Training Forward Loop: During the forward pass, if an embedding value is not found in the new table, the underlying embedding lookup operator searches the old embedding table for a pre-trained value.

  * This requires an additional all-to-all call using TorchRec to retrieve the old embedding value.

  * We need to leverage the prefetch process to hide the extra latency.

* Stop Backfilling Process: Once the new table is sufficiently populated, the backfilling process can be stopped.

This approach enables efficient warm start and transfer learning with collision-free embedding tables, reducing the computational overhead associated with ID redistribution.
