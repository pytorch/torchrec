.. meta::
   :description: TorchRec High Level Architecture
   :keywords: recommendation systems, sharding, distributed training, torchrec, architecture

.. _overview_label:

==================
TorchRec High Level Architecture
==================

In this section, you will learn about the high-level architecture of TorchRec, designed to optimize large-scale recommendation systems using PyTorch. You will learn how TorchRec employs model parallelism to distribute complex models across multiple GPUs, enhancing memory management and GPU utilization, as well as get introduced to TorchRec's base components and sharding strategies.

Modern deep learning models include an ever increasing number of parameters as well as the size of the dataset growing substantially. These models have gotten to a point where distributed deep learning is required to successfully train these models in sufficient time. In this paradigm, two main approaches have been developed: data parallelism and model parallelism. TorchRec focuses on the latter for sharding of embedding tables.

In effect, TorchRec provides parallelism primitives allowing hybrid data parallelism/model parallelism, embedding table sharding, planner to generate sharding plans, pipelined training, and more.

TorchRec's Parallelism Strategy: Model Parallelism
------------------

Model parallelism focusses on dividing the model into pieces and placing them on to different GPUs. We can divide the model into pieces and place a few consecutive layers on each GPU and calculate their activations and gradients.The diagram below displays the difference between the data parallelism and model parallelism approaches:

.. figure:: img/model_parallel.png
   :alt: Visualizing the difference of sharding a model in model parallel or data parallel approach
   :align: center

   Figure 1. Comparison between model parallelism and data parallelism approach

   As you can see in the diagram above, in model parallelism, we divide the model into different segments and distribute these across multiple GPUs. This allows each segment to process data independently, which is particularly useful for large models that do not fit on a single GPU. In contrast, data parallelism involves distributing the entire model across several GPUs, where each GPU processes a subset of the data and contributes to the overall computation. This method is effective for models that can fit within a single GPU but need to handle large datasets efficiently. Model parallelism is especially beneficial for recommendation systems with large embedding tables, as it allows for the distribution of these tables across more GPUs, optimizing memory usage and computational efficiency. Moreover, in a DLRM type architecture, we can compute the embeddings in parallel, unlike a multilayer neural network where each layer depends on the output of the previous layer.

Embedding Tables
------------------

For TorchRec to figure out what to recommend, we need to be able to represent entities and their relationships, this is what embeddings are used for. Embeddings are vectors of real numbers in a high dimensional space used to represent meaning in complex data like words, images, or users. An embedding table is an aggregation of multiple embeddings into one matrix. Most commonly, embedding tables are represented as a 2D matrix with dimensions (B, N).
- B is the number of embeddings stored by the table
- N is number of dimensions per embedding

Each of B can also be referred to as an ID (representing information such as movie title, user, ad, and so on), when accessing an ID we are returned the corresponding embedding vector which has size of embedding dimension N.

There is also the choice of pooling embeddings, often, we’re looking up multiple rows for a given feature which gives rise to the question of what we do with looking up multiple embedding vectors. Pooling is a common technique where we combine the embedding vectors, usually through sum or mean of the rows, to produce one embedding vector. This is the main difference between the PyTorch nn.Embedding and nn.EmbeddingBag.

PyTorch represents embeddings through nn.Embedding and nn.EmbeddingBag. Building on these modules, TorchRec introduces EmbeddingCollection and EmbeddingBagCollection, which are collections of the corresponding PyTorch modules. This extension enables TorchRec to batch tables and perform lookups on multiple embeddings in a single kernel call, improving efficiency.

Here is a end to end flow of how embeddings are used in the training process for recommendation models:

.. figure:: img/torchrec_forward.png
   :alt: Demonstrating the full training loop from embedding lookup to optimizer update in backward
   :align: center

   Figure 2. TorchRec End-to-end Embedding Flow

   In the diagram above, we show the general TorchRec end to end embedding lookup process. In the forward we do the embedding lookup and pooling, in the backward we compute the gradients of the output embedding lookups and apply them to the appropriate embedding table gradients and pass it into the optimizer. Note here, these gradients are grayed out since we don’t fully materialize these into memory and instead fuse them with the optimizer update. This results in a significant memory reduction which we detail later in the optimizer section.

We recommend going through the TorchRec Concepts page as well to get a understanding of the fundamentals of how everything ties together end-to-end.

See also
------------------
+ `PyTorch docs on DistributedDataParallel <https://pytorch.org/tutorials/beginner/ddp_series_theory.html>`_
