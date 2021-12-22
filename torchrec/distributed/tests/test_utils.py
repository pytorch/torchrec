#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import math
import os
import random
import unittest
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from hypothesis import Verbosity, strategies as st, settings, given
from torchrec.distributed.embedding_sharding import bucketize_kjt_before_all2all
from torchrec.distributed.embeddingbag import (
    EmbeddingBagCollectionSharder,
)
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.tests.test_model import TestSparseNN
from torchrec.distributed.utils import get_unsharded_module_names
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.sparse.tests.tests_utils import keyed_jagged_tensor_equals
from torchrec.tests.utils import get_free_port


class UtilsTest(unittest.TestCase):
    def test_get_unsharded_module_names(self) -> None:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = str("localhost")
        os.environ["MASTER_PORT"] = str(get_free_port())
        os.environ["GLOO_DEVICE_TRANSPORT"] = "TCP"
        device = torch.device("cpu")
        backend = "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        tables = [
            EmbeddingBagConfig(
                num_embeddings=10,
                embedding_dim=4,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(2)
        ]
        weighted_tables = [
            EmbeddingBagConfig(
                num_embeddings=10,
                embedding_dim=4,
                name="weighted_table_" + str(i),
                feature_names=["weighted_feature_" + str(i)],
            )
            for i in range(2)
        ]
        m = TestSparseNN(
            tables=tables,
            weighted_tables=weighted_tables,
            dense_device=device,
            sparse_device=device,
        )
        dmp = DistributedModelParallel(
            module=m,
            init_data_parallel=False,
            device=device,
            sharders=[
                EmbeddingBagCollectionSharder(),
            ],
        )

        np.testing.assert_array_equal(
            sorted(get_unsharded_module_names(dmp)),
            sorted(["module.over", "module.dense"]),
        )


def _compute_translated_lengths(
    row_indices: List[int],
    indices_offsets: List[int],
    lengths_size: int,
    trainers_size: int,
    block_sizes: List[int],
) -> List[int]:
    translated_lengths = [0] * trainers_size * lengths_size

    batch_size = int(lengths_size / len(block_sizes))
    iteration = feature_offset = batch_iteration = 0
    for start_offset, end_offset in zip(indices_offsets, indices_offsets[1:]):
        # iterate all rows that belong to current feature and batch iteration
        for row_idx in row_indices[start_offset:end_offset]:
            # compute the owner of this row
            trainer_offset = int(row_idx / block_sizes[feature_offset])
            # we do not have enough trainers to handle this row
            if trainer_offset >= trainers_size:
                continue
            trainer_lengths_offset = trainer_offset * lengths_size
            # compute the offset in lengths that is local in each trainer
            local_lengths_offset = feature_offset * batch_size + batch_iteration
            # increment the corresponding length in the trainer
            translated_lengths[trainer_lengths_offset + local_lengths_offset] += 1
        # bookkeeping
        iteration += 1
        feature_offset = int(iteration / batch_size)
        batch_iteration = (batch_iteration + 1) % batch_size
    return translated_lengths


def _compute_translated_indices_with_weights(
    translated_lengths: List[int],
    row_indices: List[int],
    indices_offsets: List[int],
    lengths_size: int,
    weights: Optional[List[int]],
    trainers_size: int,
    block_sizes: List[int],
) -> List[Tuple[int, int]]:
    translated_indices_with_weights = [(0, 0)] * len(row_indices)

    translated_indices_offsets = np.cumsum([0] + translated_lengths)
    batch_size = int(lengths_size / len(block_sizes))
    iteration = feature_offset = batch_iteration = 0
    for start_offset, end_offset in zip(indices_offsets, indices_offsets[1:]):
        # iterate all rows that belong to current feature and batch iteration
        # and assign the translated row index to the corresponding offset in output
        for current_offset in range(start_offset, end_offset):
            row_idx = row_indices[current_offset]
            feature_block_size = block_sizes[feature_offset]
            # compute the owner of this row
            trainer_offset = int(row_idx / feature_block_size)
            if trainer_offset >= trainers_size:
                continue
            trainer_lengths_offset = trainer_offset * lengths_size
            # compute the offset in lengths that is local in each trainer
            local_lengths_offset = feature_offset * batch_size + batch_iteration
            # since we know the number of rows belonging to each trainer,
            # we can figure out the corresponding offset in the translated indices list
            # for the current translated index
            translated_indices_offset = translated_indices_offsets[
                trainer_lengths_offset + local_lengths_offset
            ]
            translated_indices_with_weights[translated_indices_offset] = (
                row_idx % feature_block_size,
                weights[current_offset] if weights else 0,
            )
            # the next row that goes to this trainer for this feature and batch
            # combination goes to the next offset
            translated_indices_offsets[
                trainer_lengths_offset + local_lengths_offset
            ] += 1
        # bookkeeping
        iteration += 1
        feature_offset = int(iteration / batch_size)
        batch_iteration = (batch_iteration + 1) % batch_size
    return translated_indices_with_weights


def block_bucketize_ref(
    keyed_jagged_tensor: KeyedJaggedTensor,
    trainers_size: int,
    block_sizes: torch.Tensor,
) -> KeyedJaggedTensor:
    lengths_list = keyed_jagged_tensor.lengths().view(-1).tolist()
    indices_list = keyed_jagged_tensor.values().view(-1).tolist()
    weights_list = (
        keyed_jagged_tensor.weights().view(-1).tolist()
        if keyed_jagged_tensor.weights() is not None
        else None
    )
    block_sizes_list = block_sizes.view(-1).tolist()
    lengths_size = len(lengths_list)

    """
    each element in indices_offsets signifies both the starting offset, in indices_list,
    that corresponds to all rows in a particular feature and batch iteration,
    and the ending offset of the previous feature/batch iteration

    For example:
    given that features_size = 2 and batch_size = 2, an indices_offsets of
    [0,1,4,6,6] signifies that:

    elements in indices_list[0:1] belongs to feature 0 batch 0
    elements in indices_list[1:4] belongs to feature 0 batch 1
    elements in indices_list[4:6] belongs to feature 1 batch 0
    elements in indices_list[6:6] belongs to feature 1 batch 1
    """
    indices_offsets = np.cumsum([0] + lengths_list)

    translated_lengths = _compute_translated_lengths(
        row_indices=indices_list,
        indices_offsets=indices_offsets,
        lengths_size=lengths_size,
        trainers_size=trainers_size,
        block_sizes=block_sizes_list,
    )
    translated_indices_with_weights = _compute_translated_indices_with_weights(
        translated_lengths=translated_lengths,
        row_indices=indices_list,
        indices_offsets=indices_offsets,
        lengths_size=lengths_size,
        weights=weights_list,
        trainers_size=trainers_size,
        block_sizes=block_sizes_list,
    )

    translated_indices = [
        translated_index for translated_index, _ in translated_indices_with_weights
    ]

    translated_weights = [
        translated_weight for _, translated_weight in translated_indices_with_weights
    ]

    expected_keys = [
        key for index in range(trainers_size) for key in keyed_jagged_tensor.keys()
    ]

    return KeyedJaggedTensor(
        keys=expected_keys,
        lengths=torch.tensor(
            translated_lengths, dtype=keyed_jagged_tensor.lengths().dtype
        )
        .view(-1)
        .cuda(),
        values=torch.tensor(
            translated_indices, dtype=keyed_jagged_tensor.values().dtype
        ).cuda(),
        weights=torch.tensor(translated_weights).float().cuda()
        if weights_list
        else None,
    )


class KJTBucketizeTest(unittest.TestCase):
    @unittest.skipIf(
        torch.cuda.device_count() <= 0,
        "CUDA is not available",
    )
    # pyre-ignore[56]
    @given(
        index_type=st.sampled_from([torch.int, torch.long]),
        offset_type=st.sampled_from([torch.int, torch.long]),
        world_size=st.integers(1, 129),
        num_features=st.integers(1, 15),
        batch_size=st.integers(1, 15),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=5, deadline=None)
    def test_kjt_bucketize_before_all2all(
        self,
        index_type: torch.dtype,
        offset_type: torch.dtype,
        world_size: int,
        num_features: int,
        batch_size: int,
    ) -> None:
        MAX_BATCH_SIZE = 15
        MAX_LENGTH = 10
        # max number of rows needed for a given feature to have unique row index
        MAX_ROW_COUNT = MAX_LENGTH * MAX_BATCH_SIZE

        lengths_list = [
            random.randrange(MAX_LENGTH + 1) for _ in range(num_features * batch_size)
        ]
        keys_list = [f"feature_{i}" for i in range(num_features)]
        # for each feature, generate unrepeated row indices
        indices_lists = [
            random.sample(
                range(MAX_ROW_COUNT),
                # number of indices needed is the length sum of all batches for a feature
                sum(
                    lengths_list[
                        feature_offset * batch_size : (feature_offset + 1) * batch_size
                    ]
                ),
            )
            for feature_offset in range(num_features)
        ]
        indices_list = list(itertools.chain(*indices_lists))

        weights_list = [random.randint(1, 100) for _ in range(len(indices_list))]

        # for each feature, calculate the minimum block size needed to
        # distribute all rows to the available trainers
        block_sizes_list = [
            math.ceil((max(feature_indices_list) + 1) / world_size)
            if feature_indices_list
            else 1
            for feature_indices_list in indices_lists
        ]

        kjt = KeyedJaggedTensor(
            keys=keys_list,
            lengths=torch.tensor(lengths_list, dtype=offset_type)
            .view(num_features * batch_size)
            .cuda(),
            values=torch.tensor(indices_list, dtype=index_type).cuda(),
            weights=torch.tensor(weights_list, dtype=torch.float).cuda(),
        )
        """
        each entry in block_sizes identifies how many hashes for each feature goes
        to every rank; we have three featues in `self.features`
        """
        block_sizes = torch.tensor(block_sizes_list, dtype=index_type).cuda()

        block_bucketized_kjt, _ = bucketize_kjt_before_all2all(
            kjt, world_size, block_sizes, False, False
        )

        expected_block_bucketized_kjt = block_bucketize_ref(
            kjt,
            world_size,
            block_sizes,
        )

        self.assertTrue(
            keyed_jagged_tensor_equals(
                block_bucketized_kjt, expected_block_bucketized_kjt
            )
        )
