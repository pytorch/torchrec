.. meta::
   :description: TorchRec Concepts
   :keywords: recommendation systems, sharding, distributed training, torchrec, embedding bags, embeddings, keyedjaggedtensor, row wise, table wise, column wise, table row wise, planner, sharder

###################
 TorchRec Concepts
###################

In this section, you will learn about the key concepts of TorchRec,
designed to optimize large-scale recommendation systems using PyTorch.
You will learn how each concept works in detail and is used with the
rest of TorchRec.

TorchRec has specific input/output data types of its modules to
efficiently represent sparse features,

-  **JaggedTensor:** a wrapper around the lengths/offsets and values
   tensors for a singular sparse feature
-  **KeyedJaggedTensor:** efficiently represent multiple sparse
   features, can think of it as multiple ``JaggedTensor``\s
-  **KeyedTensor:** a wrapper around ``torch.Tensor`` that allows access
   to tensor values through keys

With the goal of high performance and efficiency, canonical
``torch.Tensor`` is highly inefficient for representing sparse data.
TorchRec introduces these new data types because they provide efficient
storage and representation of sparse input data. These data types are
particularly effective in recommender systems as they heavily deal with
sparse data. As you will see later on, the ``KeyedJaggedTensor`` makes
communication of input data in a distributed environment very efficient
leading to one of the key performance advantages that TorchRec provides.

In the end to end training loop, TorchRec comprises of the following
main components,

-  **Planner:** Take in configuration of embedding tables, environment
   setup, and generate optimized sharding plan for model

-  **Sharder:** Shard model according to sharding plan with different
   sharding strategies including data-parallel, table-wise, row-wise,
   table-wise-row-wise, column-wise, table-wise-column-wise sharding.

-  **DistributedModelParallel:** Combines sharder, optimizer, and
   provides entry point into training the model in a distributed manner.

**************
 JaggedTensor
**************

A ``JaggedTensor`` represents a sparse feature through lengths, values,
and offsets. It’s denoted as jagged as it helps efficiently represent
data with variable-length sequences. A canonical ``torch.Tensor`` each
sequence has the same length but in real world data each sequence can
have varying lengths. A ``JaggedTensor`` allows representation of this
data without padding making it highly efficient.

Key Components: Lengths: A list of integers representing the number of
elements for each entity. Offsets: A list of integers representing the
starting index of each sequence in the flattened values tensor. These
provide an alternative to lengths. Values: A 1D tensor containing the
actual values for each entity, stored contiguously.

The use of offsets provides the same information as lengths but in a
slightly different form. While lengths tell you how many interactions
each user had, offsets tell you where each user’s interactions begin.

Here is a simple example demonstrating how each of the components would
look like

.. code:: python

   # User interactions:
   # - User 1 interacted with 2 items
   # - User 2 interacted with 3 items
   # - User 3 interacted with 1 item
   lengths = [2, 3, 1]
   offsets = [0, 2, 5]  # Starting index of each user's interactions
   values = torch.Tensor([101, 102, 201, 202, 203, 301])  # Item IDs interactedwith
   jt = JaggedTensor(lengths=lengths, values=values)
   # OR
   jt = JaggedTensor(offsets=offsets, values=values)

*******************
 KeyedJaggedTensor
*******************

A ``KeyedJaggedTensor`` extends the functionality of ``JaggedTensor`` by
introducing keys (which are typically feature names) to label different
groups of features (e.g., user features and item features). This is the
data type used in ``forward`` of ``EmbeddingBagCollection`` and
``EmbeddingCollection`` as they are used to represent multiple features
in a table.

A ``KeyedJaggedTensor`` has an implied batch size which is the number of
features divided by length of ``lengths`` tensor. The example below has
a batch size of 2. Just like a ``JaggedTensor`` the ``offsets`` and
``lengths`` work the same way. You can also access the ``lengths``,
``offsets``, and ``values`` of a feature by accessing the key from the
``KeyedJaggedTensor``.

.. code:: python

   keys = ["user_features", "item_features"]
   # Lengths of interactions:
   # - User features: 2 users, with 2 and 3 interactions respectively
   # - Item features: 2 items, with 1 and 2 interactions respectively
   lengths = [2, 3, 1, 2]
   values = torch.Tensor([11, 12, 21, 22, 23, 101, 102, 201])
   # Create a KeyedJaggedTensor
   kjt = KeyedJaggedTensor(keys=keys, lengths=lengths, values=values)
   # Access the features by key
   print(kjt["user_features"])
   # Outputs user features
   print(kjt["item_features"])

*********
 Planner
*********

The TorchRec planner helps determine the best sharding configuration for
a model. What it does it evaluates multiple possibilities of how
embedding tables can be sharded and then optimizes for performance. The
planner,

-  Assesses the memory constraints of hardware
-  Estimates compute based on memory fetches as embedding lookups,
-  Addresses data specific factors
-  Considers other hardware specifics like bandwidth to generate an
   optimal sharding plan

To help with accurate consideration of these factors, the Planner can
take in data about the embedding tables, constraints, hardware
information, and topology to help in generating an optimal plan.

*****************************
 Sharding of EmbeddingTables
*****************************

TorchRec sharder provides multiple sharding strategies for various use
cases, we outline some of the sharding strategies and how they work as
well as their benefits and limitations. Generally, we recommend using
the TorchRec planner to generate a sharding plan for you as it will find
the optimal sharding strategy for each embedding table in your model.

Each sharding strategy determines how to do the table split, whether the
table should be cut up and how, whether to keep one or a few copies of
some tables, and so on. Each piece of the table from the outcome of
sharding, whether it is one embedding table or part of it, is referred
to as a shard.

.. figure:: _static/img/sharding.png
   :alt: Visualizing the difference of sharding types offered in TorchRec
   :align: center

   *Figure 1: Visualizing the placement of table shards under different sharding schemes offered in TorchRec*

There is also a combination of these strategies such as table-wise
row-wise and table-wise column-wise. Where we place a table on a node
and then column wise or row wise shard it within the node.

Once sharded, the modules are converted to sharded versions of
themselves, known as ``ShardedEmbeddingBag`` and
``ShardedEmbeddingBagCollection`` in TorchRec. These modules handle the
communication of input data, embedding lookups, and gradients.

There is a cost associated with sharding, which largely determines which
sharding strategy is best for a model.

Without sharding, where each GPU keeps a copy of the embedding table,
the main cost is computation in which each GPU looks up the embedding
vectors in its memory in the forward pass and updates the gradients in
the backward.

With sharding, there is an added communication cost: each GPU needs to
ask the other GPUs for embedding vector lookup and communicate the
gradients computed as well. This is usually referred to as all2all
communication. In TorchRec, for input data on a given GPU, we determine
where the embedding shard for each part of the data is located and send
it to the target GPU. That target GPU then returns the embedding vectors
back to the original GPU. In the backward pass, the gradients are sent
back to the target GPU and the shards are updated accordingly with the
optimizer.

As described above, sharding requires us to communicate the input data
and embedding lookups. TorchRec handles this in three main stages, we’ll
refer to this as the sharded embedding module forward that is used in
training and inference of a TorchRec model,

-  Feature All to All/Input distribution (``input_dist``)

   -  Communicate input data (in the form of a ``KeyedJaggedTensor``) to
      the appropriate device containing relevant embedding table shard

-  Embedding Lookup

   -  Lookup embeddings with new input data formed after feature all to
      all exchange

-  Embedding All to All/Output Distribution (``output_dist``)

   -  Communicate embedding lookup data back to the appropriate device
      that asked for it (in accordance with the input data the device
      received)

-  The backward pass does the same operations but in reverse order.

We show this below in the diagram,

.. figure:: _static/img/torchrec_forward.png
   :alt: Visualizing the forward pass including the input_dist, lookup, and output_dist of a sharded TorchRec module
   :align: center

   *Figure 2: Visualizing the forward pass including the input_dist, lookup, and output_dist of a sharded TorchRec module*

**************************
 DistributedModelParallel
**************************

All of the above culminates into the main entrypoint that TorchRec uses
to shard and integrate the plan. At a high level,
``DistributedModelParallel`` does,

-  Initialize environment by setting up process groups and assigning
   device type

-  Uses default shaders if no shaders are provided, default includes
   ``EmbeddingBagCollectionSharder``

-  Takes in provided sharding plan, if none provided it generates one

-  Creates sharded version of modules and replaces the original modules
   with them, such as ``EmbeddingCollection`` to
   ``ShardedEmbeddingCollection``

-  By default, wraps the ``DistributedModelParallel`` with
   ``DistributedDataParallel`` to make the module both model and data
   parallel

***********
 Optimizer
***********

TorchRec modules provide a seamless API to fuse the backwards pass and
optimize step in training, providing a significant optimization in
performance and decreasing the memory used, alongside granularity in
assigning distinct optimizers to distinct model parameters.

.. figure:: _static/img/fused_backward_optimizer.png
   :alt: Visualizing fusing of optimizer in backward to update sparse embedding table
   :align: center

   *Figure 3: Fusing embedding backward with sparse optimizer*

***********
 Inference
***********

Inference environments are different from training, they are very
sensitive to performance and size of the model. There are two key
differences TorchRec inference optimizes for,

-  Quantization: inference models are quantized for lower latency and
   reduced model size, this lets us use as few devices as possible for
   inference to minimize latency.

-  C++ environment: to minimize latency even further, the model is ran
   in a C++ environment

TorchRec provides the following to convert a TorchRec model into being
inference ready. * APIs for quantizing the model, including
optimizations automatically with FBGEMM TBE * Sharding embeddings for
distributed inference * Compiling the model to TorchScript (compatible
in C++)

**********
 See Also
**********

-  `TorchRec Interactive Notebook using the concepts
   <https://github.com/pytorch/torchrec/blob/main/TorchRec_Interactive_Tutorial_Notebook_OSS_version.ipynb>`_
