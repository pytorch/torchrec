# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List

import os
import nvtabular as nvt
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = os.environ["LOCAL_RANK"]

from nvtabular.loader.torch import TorchAsyncItr
from torch.utils.data import DataLoader

from torchrec.datasets.criteo import (
    CAT_FEATURE_COUNT,
    DEFAULT_CAT_NAMES,
    DEFAULT_INT_NAMES,
    DEFAULT_LABEL_NAME,
)
from torchrec.datasets.utils import Batch

from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class NvtCriteoDataloader:
    def __init__(
        self,
        paths: List[str],
        batch_size: int,
        world_size: int,
        rank: int,
    ):
        self.paths = paths
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self._num_ids_in_batch: int = CAT_FEATURE_COUNT * batch_size
        self.keys: List[str] = DEFAULT_CAT_NAMES
        self.lengths: torch.Tensor = torch.ones(
            (self._num_ids_in_batch,), dtype=torch.int32
        )
        self.offsets: torch.Tensor = torch.arange(
            0, self._num_ids_in_batch + 1, dtype=torch.int32
        )
        self.stride = batch_size
        self.length_per_key: List[int] = CAT_FEATURE_COUNT * [batch_size]
        self.offset_per_key: List[int] = [
            batch_size * i for i in range(CAT_FEATURE_COUNT + 1)
        ]
        self.index_per_key: Dict[str, int] = {
            key: i for (i, key) in enumerate(self.keys)
        }

    def seed_fn(self):
        """
        Generate consistent dataloader shuffle seeds across workers
        Reseeds each worker's dataloader each epoch to get fresh a shuffle
        that's consistent across workers.

        TODO there is something wrong with the seed_fn example. Return 0 for now
        """
        return 0

    def get_nvt_criteo_dataloader(self):
        dataset = TorchAsyncItr(
            nvt.Dataset(self.paths, part_size="144MB"),
            batch_size=self.batch_size,
            cats=DEFAULT_CAT_NAMES,
            conts=DEFAULT_INT_NAMES,
            labels=[DEFAULT_LABEL_NAME],
            device=self.rank,
            global_size=self.world_size,
            global_rank=self.rank,
            shuffle=True,
            seed_fn=self.seed_fn,
        )

        def collate_fn(attr_dict):
            batch_features, labels = attr_dict
            # We know that all categories are one-hot. However, this may not generalize
            # We should work with nvidia to allow nvtabular to natively transform to
            # a KJT format.
            return Batch(
                dense_features=torch.cat(
                    [batch_features[feature] for feature in DEFAULT_INT_NAMES], dim=1
                ),
                sparse_features=KeyedJaggedTensor(
                    keys=DEFAULT_CAT_NAMES,
                    values=torch.cat(
                        [batch_features[feature] for feature in DEFAULT_CAT_NAMES]
                    ).view(-1),
                    lengths=self.lengths,
                    offsets=self.offsets,
                    stride=self.stride,
                    length_per_key=self.length_per_key,
                    offset_per_key=self.offset_per_key,
                    index_per_key=self.index_per_key,
                ),
                labels=labels,
            )

        # Don't pin memory since the batches are already on cuda!
        # Num worker is set to zero as well, because it is on GPU
        return DataLoader(
            dataset,
            batch_size=None,
            collate_fn=collate_fn,
            pin_memory=False,
            num_workers=0,
        )
