#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterator, List, Optional

import torch
from pyre_extensions import none_throws
from torch.utils.data.dataset import IterableDataset
from torchrec.datasets.utils import Batch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class _RandomRecBatch:
    generator: Optional[torch.Generator]

    def __init__(
        self,
        keys: List[str],
        batch_size: int,
        hash_size: Optional[int],
        hash_sizes: Optional[List[int]],
        ids_per_feature: int,
        num_dense: int,
        manual_seed: Optional[int] = None,
    ) -> None:
        if (hash_size is None and hash_sizes is None) or (
            hash_size is not None and hash_sizes is not None
        ):
            raise ValueError(
                "One - and only one - of hash_size or hash_sizes must be set."
            )

        self.keys = keys
        self.keys_length: int = len(keys)
        self.batch_size = batch_size
        self.hash_size = hash_size
        self.hash_sizes = hash_sizes
        self.ids_per_feature = ids_per_feature
        self.num_dense = num_dense

        if manual_seed is not None:
            self.generator = torch.Generator()
            # pyre-ignore[16]
            self.generator.manual_seed(manual_seed)
        else:
            self.generator = None

        self.iter_num = 0
        self._num_ids_in_batch: int = (
            self.ids_per_feature * self.keys_length * self.batch_size
        )
        self.max_values: Optional[torch.Tensor] = None
        if hash_sizes is not None:
            self.max_values: torch.Tensor = torch.tensor(
                [
                    hash_size
                    for hash_size in hash_sizes
                    for b in range(batch_size)
                    for i in range(ids_per_feature)
                ]
            )
        self._generated_batches: List[Batch] = [self._generate_batch()] * 10
        self.batch_index = 0

    def __iter__(self) -> "_RandomRecBatch":
        return self

    def __next__(self) -> Batch:
        batch = self._generated_batches[self.batch_index % len(self._generated_batches)]
        self.batch_index += 1
        return batch

    def _generate_batch(self) -> Batch:
        if self.hash_sizes is None:
            # pyre-ignore[28]
            values = torch.randint(
                high=self.hash_size,
                size=(self._num_ids_in_batch,),
                generator=self.generator,
            )
        else:
            values = (
                torch.rand(
                    self._num_ids_in_batch,
                    generator=self.generator,
                )
                * none_throws(self.max_values)
            ).type(torch.LongTensor)
        sparse_features = KeyedJaggedTensor.from_offsets_sync(
            keys=self.keys,
            values=values,
            offsets=torch.tensor(
                list(
                    range(
                        0,
                        self._num_ids_in_batch + 1,
                        self.ids_per_feature,
                    )
                ),
                dtype=torch.int32,
            ),
        )

        dense_features = torch.randn(
            self.batch_size,
            self.num_dense,
            generator=self.generator,
        )
        # pyre-ignore[28]
        labels = torch.randint(
            low=0,
            high=2,
            size=(self.batch_size,),
            generator=self.generator,
        )

        batch = Batch(
            dense_features=dense_features,
            sparse_features=sparse_features,
            labels=labels,
        )
        return batch


class RandomRecDataset(IterableDataset[Batch]):
    """
    Random iterable dataset used to generate batches for recommender systems
    (RecSys). Currently produces unweighted sparse features only. TODO: Add
    weighted sparse features.

    Args:
        keys (List[str]): List of feature names for sparse features.
        batch_size (int): batch size.
        hash_size (Optional[int]): Max sparse id value. All sparse IDs will be taken
            modulo this value.
        hash_sizes (Optional[List[int]]): Max sparse id value per feature in keys. Each
            sparse ID will be taken modulo the corresponding value from this argument.
        ids_per_feature (int): Number of IDs per sparse feature.
        num_dense (int): Number of dense features.
        manual_seed (int): Seed for deterministic behavior.

    Example:
        >>> dataset = RandomRecDataset(
        >>>     keys=["feat1", "feat2"],
        >>>     batch_size=16,
        >>>     hash_size=100_000,
        >>>     ids_per_feature=1,
        >>>     num_dense=13,
        >>> ),
        >>> example = next(iter(dataset))
    """

    def __init__(
        self,
        keys: List[str],
        batch_size: int,
        hash_size: Optional[int] = 100,
        hash_sizes: Optional[List[int]] = None,
        ids_per_feature: int = 2,
        num_dense: int = 50,
        manual_seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.batch_generator = _RandomRecBatch(
            keys=keys,
            batch_size=batch_size,
            hash_size=hash_size,
            hash_sizes=hash_sizes,
            ids_per_feature=ids_per_feature,
            num_dense=num_dense,
            manual_seed=manual_seed,
        )

    def __iter__(self) -> Iterator[Batch]:
        return iter(self.batch_generator)
