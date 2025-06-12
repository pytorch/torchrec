#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import logging as logger
from collections import Counter, OrderedDict
from typing import Dict, Iterable, List, Optional, Union

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
SUPPORTED_MODULES = (ShardedEmbeddingCollection, ShardedEmbeddingBagCollection)


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
        fqns_to_skip (Iterable[str], optional): list of FQNs to skip tracking. Default: None.

    """

    DEFAULT_CONSUMER: str = "default"

    def __init__(
        self,
        model: nn.Module,
        consumers: Optional[List[str]] = None,
        delete_on_read: bool = True,
        mode: TrackingMode = TrackingMode.ID_ONLY,
        fqns_to_skip: Iterable[str] = (),
    ) -> None:
        self._model = model
        self._consumers: List[str] = consumers or [self.DEFAULT_CONSUMER]
        self._delete_on_read = delete_on_read
        self._mode = mode
        self._fqn_to_feature_map: Dict[str, List[str]] = {}
        self._fqns_to_skip: Iterable[str] = fqns_to_skip
        self.fqn_to_feature_names()
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

    def fqn_to_feature_names(self) -> Dict[str, List[str]]:
        """
        Returns a mapping of FQN to feature names from all Supported Modules [EmbeddingCollection and EmbeddingBagCollection] present in the given model.
        """
        if (self._fqn_to_feature_map is not None) and len(self._fqn_to_feature_map) > 0:
            return self._fqn_to_feature_map

        table_to_feature_names: Dict[str, List[str]] = OrderedDict()
        table_to_fqn: Dict[str, str] = OrderedDict()
        for fqn, named_module in self._model.named_modules():
            split_fqn = fqn.split(".")
            # Skipping partial FQNs present in fqns_to_skip
            # TODO: Validate if we need to support more complex patterns for skipping fqns
            should_skip = False
            for fqn_to_skip in self._fqns_to_skip:
                if fqn_to_skip in split_fqn:
                    logger.info(f"Skipping {fqn} because it is part of fqns_to_skip")
                    should_skip = True
                    break
            if should_skip:
                continue

            # Using FQNs of the embedding and mapping them to features as state_dict() API uses these to key states.
            if isinstance(named_module, SUPPORTED_MODULES):
                for table_name, config in named_module._table_name_to_config.items():
                    logger.info(
                        f"Found {table_name} for {fqn} with features {config.feature_names}"
                    )
                    table_to_feature_names[table_name] = config.feature_names
            for table_name in table_to_feature_names:
                # Using the split FQN to get the exact table name matching. Otherwise, checking "table_name in fqn"
                # will incorrectly match fqn with all the table names that have the same prefix
                if table_name in split_fqn:
                    embedding_fqn = fqn.replace("_dmp_wrapped_module.module.", "")
                    if table_name in table_to_fqn:
                        # Sanity check for validating that we don't have more then one table mapping to same fqn.
                        logger.warning(
                            f"Override {table_to_fqn[table_name]} with {embedding_fqn} for entry {table_name}"
                        )
                    table_to_fqn[table_name] = embedding_fqn
            logger.info(f"Table to fqn: {table_to_fqn}")
        flatten_names = [
            name for names in table_to_feature_names.values() for name in names
        ]
        # TODO: Validate if there is a better way to handle duplicate feature names.
        # Logging a warning if duplicate feature names are found across tables, but continue execution as this could be a valid case.
        if len(set(flatten_names)) != len(flatten_names):
            counts = Counter(flatten_names)
            duplicates = [item for item, count in counts.items() if count > 1]
            logger.warning(f"duplicate feature names found: {duplicates}")

        fqn_to_feature_names: Dict[str, List[str]] = OrderedDict()
        for table_name in table_to_feature_names:
            if table_name not in table_to_fqn:
                # This is likely unexpected, where we can't locate the FQN associated with this table.
                logger.warning(
                    f"Table {table_name} not found in {table_to_fqn}, skipping"
                )
                continue
            fqn_to_feature_names[table_to_fqn[table_name]] = table_to_feature_names[
                table_name
            ]
        self._fqn_to_feature_map = fqn_to_feature_names
        return fqn_to_feature_names

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
