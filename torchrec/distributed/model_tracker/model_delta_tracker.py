#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from typing import Dict, List, Optional, Union

import torch

from torch import nn
from torchrec.distributed.embedding import ShardedEmbeddingCollection
from torchrec.distributed.embeddingbag import ShardedEmbeddingBagCollection
from torchrec.distributed.model_tracker.types import (
    DeltaRows,
    EmbdUpdateMode,
    TrackingMode,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

UPDATE_MODE_MAP: Dict[TrackingMode, EmbdUpdateMode] = {
    # Only IDs are tracked, no additional state is stored.
    TrackingMode.ID_ONLY: EmbdUpdateMode.NONE,
    # TrackingMode.EMBEDDING utilizes EmbdUpdateMode.FIRST to ensure that
    # the earliest embedding values are stored since the last checkpoint or snapshot.
    # This mode is used for computing topk delta rows, which is currently achieved by running (new_emb - old_emb).norm().topk().
    TrackingMode.EMBEDDING: EmbdUpdateMode.FIRST,
}

# Tracking is current only supported for ShardedEmbeddingCollection and ShardedEmbeddingBagCollection.
SUPPORTED_MODULES = Union[ShardedEmbeddingCollection, ShardedEmbeddingBagCollection]


class ModelDeltaTracker:
    r"""

    ModelDeltaTracker provides a way to track and retrieve unique IDs for supported modules, along with optional support
    for tracking corresponding embeddings or states. This is useful for identifying and retrieving the latest delta or
    unique rows for a given model, which can help compute topk or to stream updated embeddings from predictors to trainers during
    online training. Unique IDs or states can be retrieved by calling the get_unique() method.

    Args:
        model (nn.Module): the model to track.
        consumers (List[str], optional): list of consumers to track. Each consumer will
            have its own batch offset index. Every get_unique_ids invocation will
            only return the new ids for the given consumer since last get_unique_ids
            call.
        delete_on_read (bool, optional): whether to delete the tracked ids after all consumers have read them.
        mode (TrackingMode, optional): tracking mode to use from supported tracking modes. Default: TrackingMode.ID_ONLY.
    """

    DEFAULT_CONSUMER: str = "default"

    def __init__(
        self,
        model: nn.Module,
        consumers: Optional[List[str]] = None,
        delete_on_read: bool = True,
        mode: TrackingMode = TrackingMode.ID_ONLY,
    ) -> None:
        self._model = model
        self._consumers: List[str] = consumers or [self.DEFAULT_CONSUMER]
        self._delete_on_read = delete_on_read
        self._mode = mode
        pass

    def record_lookup(self, kjt: KeyedJaggedTensor, states: torch.Tensor) -> None:
        """
        Record Ids from a given KeyedJaggedTensor and embeddings/ parameter states.

        Args:
            kjt (KeyedJaggedTensor): the KeyedJaggedTensor to record.
            states (torch.Tensor): the states to record.
        """
        pass

    def get_delta(self, consumer: Optional[str] = None) -> Dict[str, DeltaRows]:
        """
        Return a dictionary of hit local IDs for each sparse feature. The IDs are first keyed by submodule FQN.

        Args:
            consumer (str, optional): The consumer to retrieve IDs for. If not specified, "default" is used as the default consumer.
        """
        return {}

    def fqn_to_feature_names(self, module: nn.Module) -> Dict[str, List[str]]:
        """
        Returns a mapping from FQN to feature names for a given module.

        Args:
            module (nn.Module): the module to retrieve feature names for.
        """
        return {}

    def clear(self, consumer: Optional[str] = None) -> None:
        """
        Clear tracked IDs for a given consumer.

        Args:
            consumer (str, optional): The consumer to clear IDs/States for. If not specified, "default" is used as the default consumer.
        """
        pass

    def compact(self, start_idx: int, end_idx: int) -> None:
        """
        Compact tracked IDs for a given range of indices.

        Args:
            start_idx (int): Starting index for compaction.
            end_idx (int): Ending index for compaction.
        """
        pass
