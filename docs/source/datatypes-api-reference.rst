Data Types
-------------------


TorchRec contains data types for representing embedding, otherwise known as sparse features.
Sparse features are typically indices that are meant to be fed into embedding tables. For a given
batch, the number of embedding lookup indices are variable. Therefore, there is a need for a **jagged**
dimension to represent the variable amount of embedding lookup indices for a batch.

This section covers the classes for the 3 TorchRec data types for representing sparse features:
**JaggedTensor**, **KeyedJaggedTensor**, and **KeyedTensor**.

.. automodule:: torchrec.sparse.jagged_tensor

.. autoclass:: JaggedTensor
    :members:

.. autoclass:: KeyedJaggedTensor
    :members:

.. autoclass:: KeyedTensor
    :members:
