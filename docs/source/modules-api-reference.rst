Modules
----------------------------------

Standard TorchRec modules represent collections of embedding tables:

* ``EmbeddingBagCollection`` is a collection of ``torch.nn.EmbeddingBag``
* ``EmbeddingCollection`` is a collection of ``torch.nn.Embedding``

These modules are constructed through standardized config classes:

* ``EmbeddingBagConfig`` for ``EmbeddingBagCollection``
* ``EmbeddingConfig`` for ``EmbeddingCollection``

.. automodule:: torchrec.modules.embedding_configs

.. autoclass:: EmbeddingBagConfig
    :show-inheritance:

.. autoclass:: EmbeddingConfig
    :show-inheritance:

.. autoclass:: BaseEmbeddingConfig

.. automodule:: torchrec.modules.embedding_modules

.. autoclass:: EmbeddingBagCollection
    :members:

.. autoclass:: EmbeddingCollection
    :members:
