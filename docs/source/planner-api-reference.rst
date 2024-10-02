Planner
----------------------------------

The TorchRec Planner is responsible for determining the most performant, balanced
sharding plan for distributed training and inference.

The main API for generating a sharding plan is ``EmbeddingShardingPlanner.plan``

.. automodule:: torchrec.distributed.types

.. autoclass:: ShardingPlan
    :members:

.. automodule:: torchrec.distributed.planner.planners

.. autoclass:: EmbeddingShardingPlanner
    :members:

.. automodule:: torchrec.distributed.planner.enumerators

.. autoclass:: EmbeddingEnumerator
    :members:

.. automodule:: torchrec.distributed.planner.partitioners

.. autoclass:: GreedyPerfPartitioner
    :members:


.. automodule:: torchrec.distributed.planner.storage_reservations

.. autoclass:: HeuristicalStorageReservation
    :members:

.. automodule:: torchrec.distributed.planner.proposers

.. autoclass:: GreedyProposer
    :members:


.. automodule:: torchrec.distributed.planner.shard_estimators

.. autoclass:: EmbeddingPerfEstimator
    :members:


.. automodule:: torchrec.distributed.planner.shard_estimators

.. autoclass:: EmbeddingStorageEstimator
    :members:
