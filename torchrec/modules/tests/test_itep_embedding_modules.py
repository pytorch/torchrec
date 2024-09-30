#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import random
import unittest
from typing import Dict, List
from unittest.mock import MagicMock, patch

import torch
from torchrec import KeyedJaggedTensor
from torchrec.distributed.embedding_types import ShardedEmbeddingTable
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection

from torchrec.modules.itep_embedding_modules import ITEPEmbeddingBagCollection
from torchrec.modules.itep_modules import GenericITEPModule

MOCK_NS: str = "torchrec.modules.itep_modules"


class TestITEPEmbeddingBagCollection(unittest.TestCase):
    # Setting up the environment for the tests.
    def setUp(self) -> None:
        # Embedding bag configurations for testing
        embedding_bag_config1 = EmbeddingBagConfig(
            name="table1",
            embedding_dim=4,
            num_embeddings=50,
            feature_names=["feature1"],
        )
        embedding_bag_config2 = EmbeddingBagConfig(
            name="table2",
            embedding_dim=4,
            num_embeddings=40,
            feature_names=["feature2"],
        )
        unpruned_hash_size_1, unpruned_hash_size_2 = (100, 80)
        self._table_name_to_pruned_hash_sizes = {"table1": 50, "table2": 40}
        self._table_name_to_unpruned_hash_sizes = {
            "table1": unpruned_hash_size_1,
            "table2": unpruned_hash_size_2,
        }
        self._feature_name_to_unpruned_hash_sizes = {
            "feature1": unpruned_hash_size_1,
            "feature2": unpruned_hash_size_2,
        }
        self._batch_size = 8

        # Util function for creating sharded embedding tables from embedding bag configurations.
        def embedding_bag_config_to_sharded_table(
            config: EmbeddingBagConfig,
        ) -> ShardedEmbeddingTable:
            return ShardedEmbeddingTable(
                name=config.name,
                embedding_dim=config.embedding_dim,
                num_embeddings=config.num_embeddings,
                feature_names=config.feature_names,
            )

        sharded_et1 = embedding_bag_config_to_sharded_table(embedding_bag_config1)
        sharded_et2 = embedding_bag_config_to_sharded_table(embedding_bag_config2)

        # Create test ebc
        self._embedding_bag_collection = EmbeddingBagCollection(
            tables=[
                embedding_bag_config1,
                embedding_bag_config2,
            ],
            device=torch.device("cuda"),
        )

        # Create a mock object for tbe lookups
        self._mock_list_emb_tables = [
            sharded_et1,
            sharded_et2,
        ]
        self._mock_lookups = [MagicMock()]
        self._mock_lookups[0]._emb_modules = [MagicMock()]
        self._mock_lookups[0]._emb_modules[0]._config = MagicMock()
        self._mock_lookups[0]._emb_modules[
            0
        ]._config.embedding_tables = self._mock_list_emb_tables

    def generate_input_kjt_cuda(
        self, feature_name_to_unpruned_hash_sizes: Dict[str, int], use_vbe: bool = False
    ) -> KeyedJaggedTensor:
        keys = []
        values = []
        lengths = []
        cuda_device = torch.device("cuda")

        # Input KJT uses unpruned hash size (same as sigrid hash), and feature names
        for key, unpruned_hash_size in feature_name_to_unpruned_hash_sizes.items():
            value = []
            length = []
            for _ in range(self._batch_size):
                L = random.randint(0, 8)
                for _ in range(L):
                    index = random.randint(0, unpruned_hash_size - 1)
                    value.append(index)
                length.append(L)
            keys.append(key)
            values += value
            lengths += length

        # generate kjt
        if use_vbe:
            inverse_indices_list = []
            inverse_indices = None
            num_keys = len(keys)
            deduped_batch_size = len(lengths) // num_keys
            # Fix the number of samples after duplicate to 2x the number of
            # deduplicated ones
            full_batch_size = deduped_batch_size * 2
            stride_per_key_per_rank = []

            for _ in range(num_keys):
                stride_per_key_per_rank.append([deduped_batch_size])
                # Generate random inverse indices for each key
                keyed_inverse_indices = torch.randint(
                    low=0,
                    high=deduped_batch_size,
                    size=(full_batch_size,),
                    dtype=torch.int32,
                    device=cuda_device,
                )
                inverse_indices_list.append(keyed_inverse_indices)
            inverse_indices = (
                keys,
                torch.stack(inverse_indices_list),
            )

            input_kjt_cuda = KeyedJaggedTensor.from_lengths_sync(
                keys=keys,
                values=torch.tensor(
                    copy.deepcopy(values),
                    dtype=torch.int32,
                    device=cuda_device,
                ),
                lengths=torch.tensor(
                    copy.deepcopy(lengths),
                    dtype=torch.int32,
                    device=cuda_device,
                ),
                stride_per_key_per_rank=stride_per_key_per_rank,
                inverse_indices=inverse_indices,
            )
        else:
            input_kjt_cuda = KeyedJaggedTensor.from_lengths_sync(
                keys=keys,
                values=torch.tensor(
                    copy.deepcopy(values),
                    dtype=torch.int32,
                    device=cuda_device,
                ),
                lengths=torch.tensor(
                    copy.deepcopy(lengths),
                    dtype=torch.int32,
                    device=cuda_device,
                ),
            )

        return input_kjt_cuda

    def generate_expected_address_lookup_buffer(
        self,
        list_et: List[ShardedEmbeddingTable],
        table_name_to_unpruned_hash_sizes: Dict[str, int],
        table_name_to_pruned_hash_sizes: Dict[str, int],
    ) -> torch.Tensor:

        address_lookup = []
        for et in list_et:
            table_name = et.name
            unpruned_hash_size = table_name_to_unpruned_hash_sizes[table_name]
            pruned_hash_size = table_name_to_pruned_hash_sizes[table_name]
            for idx in range(unpruned_hash_size):
                if idx < pruned_hash_size:
                    address_lookup.append(idx)
                else:
                    address_lookup.append(0)

        return torch.tensor(address_lookup, dtype=torch.int64)

    # pyre-ignores
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Skip when not enough GPUs available",
    )
    def test_init_itep_module(self) -> None:
        itep_module = GenericITEPModule(
            table_name_to_unpruned_hash_sizes=self._table_name_to_unpruned_hash_sizes,
            lookups=self._mock_lookups,
            enable_pruning=True,
            pruning_interval=5,
        )

        # Check the address lookup and row util after initialization
        expected_address_lookup = self.generate_expected_address_lookup_buffer(
            self._mock_list_emb_tables,
            self._table_name_to_unpruned_hash_sizes,
            self._table_name_to_pruned_hash_sizes,
        )
        expetec_row_util = torch.zeros(
            expected_address_lookup.shape, dtype=torch.float32
        )
        torch.testing.assert_close(
            expected_address_lookup,
            itep_module.address_lookup.cpu(),
            atol=0,
            rtol=0,
            equal_nan=True,
        )
        torch.testing.assert_close(
            expetec_row_util,
            itep_module.row_util.cpu(),
            atol=1.0e-5,
            rtol=1.0e-5,
            equal_nan=True,
        )

    # pyre-ignores
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Skip when not enough GPUs available",
    )
    def test_init_itep_module_without_pruned_table(self) -> None:
        itep_module = GenericITEPModule(
            table_name_to_unpruned_hash_sizes={},
            lookups=self._mock_lookups,
            enable_pruning=True,
            pruning_interval=5,
        )

        self.assertEqual(itep_module.address_lookup.cpu().shape, torch.Size([0]))
        self.assertEqual(itep_module.row_util.cpu().shape, torch.Size([0]))

    # pyre-ignores
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Skip when not enough GPUs available",
    )
    def test_train_forward(self) -> None:
        itep_module = GenericITEPModule(
            table_name_to_unpruned_hash_sizes=self._table_name_to_unpruned_hash_sizes,
            lookups=self._mock_lookups,
            enable_pruning=True,
            pruning_interval=500,
        )

        itep_ebc = ITEPEmbeddingBagCollection(
            embedding_bag_collection=self._embedding_bag_collection,
            itep_module=itep_module,
        )

        # Test forward 2000 times
        for _ in range(2000):
            input_kjt = self.generate_input_kjt_cuda(
                self._feature_name_to_unpruned_hash_sizes
            )
            _ = itep_ebc(input_kjt)

    # pyre-ignores
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Skip when not enough GPUs available",
    )
    def test_train_forward_vbe(self) -> None:
        itep_module = GenericITEPModule(
            table_name_to_unpruned_hash_sizes=self._table_name_to_unpruned_hash_sizes,
            lookups=self._mock_lookups,
            enable_pruning=True,
            pruning_interval=500,
        )

        itep_ebc = ITEPEmbeddingBagCollection(
            embedding_bag_collection=self._embedding_bag_collection,
            itep_module=itep_module,
        )

        # Test forward 2000 times
        for _ in range(5):
            input_kjt = self.generate_input_kjt_cuda(
                self._feature_name_to_unpruned_hash_sizes, use_vbe=True
            )
            _ = itep_ebc(input_kjt)

    # pyre-ignores
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Skip when not enough GPUs available",
    )
    # Mock out reset_weight_momentum to count calls
    @patch(f"{MOCK_NS}.GenericITEPModule.reset_weight_momentum")
    def test_check_pruning_schedule(
        self,
        mock_reset_weight_momentum: MagicMock,
    ) -> None:
        random.seed(1)
        itep_module = GenericITEPModule(
            table_name_to_unpruned_hash_sizes=self._table_name_to_unpruned_hash_sizes,
            lookups=self._mock_lookups,
            enable_pruning=True,
            pruning_interval=500,
        )

        itep_ebc = ITEPEmbeddingBagCollection(
            embedding_bag_collection=self._embedding_bag_collection,
            itep_module=itep_module,
        )

        # Test forward 2000 times
        for _ in range(2000):
            input_kjt = self.generate_input_kjt_cuda(
                self._feature_name_to_unpruned_hash_sizes
            )
            _ = itep_ebc(input_kjt)

        # Check that reset_weight_momentum is called
        self.assertEqual(mock_reset_weight_momentum.call_count, 5)

    # pyre-ignores
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Skip when not enough GPUs available",
    )
    # Mock out reset_weight_momentum to count calls
    @patch(f"{MOCK_NS}.GenericITEPModule.reset_weight_momentum")
    def test_eval_forward(
        self,
        mock_reset_weight_momentum: MagicMock,
    ) -> None:
        itep_module = GenericITEPModule(
            table_name_to_unpruned_hash_sizes=self._table_name_to_unpruned_hash_sizes,
            lookups=self._mock_lookups,
            enable_pruning=True,
            pruning_interval=500,
        )

        itep_ebc = ITEPEmbeddingBagCollection(
            embedding_bag_collection=self._embedding_bag_collection,
            itep_module=itep_module,
        )

        # Set eval mode
        itep_ebc.eval()

        # Test forward 2000 times
        for _ in range(2000):
            input_kjt = self.generate_input_kjt_cuda(
                self._feature_name_to_unpruned_hash_sizes
            )
            _ = itep_ebc(input_kjt)

        # Check that reset_weight_momentum is not called
        self.assertEqual(mock_reset_weight_momentum.call_count, 0)
