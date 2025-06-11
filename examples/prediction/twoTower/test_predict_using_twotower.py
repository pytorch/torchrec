#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
import sys
import unittest

import torch

# Add the current directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import using the full module path
from torchrec.github.examples.prediction.twoTower.predict_using_twotower import (
    create_kjt_from_ids,
    RecommendationDataset,
    TwoTowerModel,
    TwoTowerRatingWrapper,
)
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection

# TorchRec imports
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class TestRecommendationDataset(unittest.TestCase):
    """Test cases for the RecommendationDataset class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.num_users = 100
        self.num_items = 50
        self.num_samples = 1000
        self.dataset = RecommendationDataset(
            num_users=self.num_users,
            num_items=self.num_items,
            num_samples=self.num_samples,
        )

    def test_init(self) -> None:
        """Test initialization of RecommendationDataset."""
        self.assertEqual(self.dataset.num_users, self.num_users)
        self.assertEqual(self.dataset.num_items, self.num_items)
        self.assertEqual(self.dataset.num_samples, self.num_samples)

        # Check tensor shapes
        self.assertEqual(self.dataset.user_ids.shape, (self.num_samples,))
        self.assertEqual(self.dataset.item_ids.shape, (self.num_samples,))
        self.assertEqual(self.dataset.ratings.shape, (self.num_samples,))
        self.assertEqual(self.dataset.user_features.shape, (self.num_samples, 4))
        self.assertEqual(self.dataset.item_features.shape, (self.num_samples, 4))

    def test_len(self) -> None:
        """Test __len__ method."""
        self.assertEqual(len(self.dataset), self.num_samples)

    def test_getitem(self) -> None:
        """Test __getitem__ method."""
        item = self.dataset[0]
        self.assertIsInstance(item, dict)
        self.assertIn("user_id", item)
        self.assertIn("item_id", item)
        self.assertIn("user_features", item)
        self.assertIn("item_features", item)
        self.assertIn("rating", item)

        # Check tensor shapes for a single item
        self.assertEqual(item["user_features"].shape, (4,))
        self.assertEqual(item["item_features"].shape, (4,))


class TestTwoTowerModel(unittest.TestCase):
    """Test cases for the TwoTowerModel class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.embedding_dim = 32
        self.user_tower_sizes = [64, 32]
        self.item_tower_sizes = [64, 32]

        # Create embedding bag configs
        self.num_users = 100
        self.num_items = 50

        self.user_eb_config = EmbeddingBagConfig(
            name="user_id",
            embedding_dim=self.embedding_dim,
            num_embeddings=self.num_users,
            feature_names=["user_id"],
        )

        self.item_eb_config = EmbeddingBagConfig(
            name="item_id",
            embedding_dim=self.embedding_dim,
            num_embeddings=self.num_items,
            feature_names=["item_id"],
        )

        # Create EmbeddingBagCollections
        self.device = torch.device("cpu")
        self.user_ebc = EmbeddingBagCollection(
            tables=[self.user_eb_config],
            device=self.device,
        )

        self.item_ebc = EmbeddingBagCollection(
            tables=[self.item_eb_config],
            device=self.device,
        )

        # Create model
        self.model = TwoTowerModel(
            user_ebc=self.user_ebc,
            item_ebc=self.item_ebc,
            user_tower_sizes=self.user_tower_sizes,
            item_tower_sizes=self.item_tower_sizes,
            normalize_embeddings=False,
        )

        # Create model with normalization
        self.model_normalized = TwoTowerModel(
            user_ebc=self.user_ebc,
            item_ebc=self.item_ebc,
            user_tower_sizes=self.user_tower_sizes,
            item_tower_sizes=self.item_tower_sizes,
            normalize_embeddings=True,
        )

        # Test data
        self.batch_size = 8

        # Create batch data
        self.user_ids = torch.randint(0, self.num_users, (self.batch_size,))
        self.item_ids = torch.randint(0, self.num_items, (self.batch_size,))

        # Create KeyedJaggedTensor for sparse features
        self.user_features = create_kjt_from_ids(self.user_ids, "user_id", self.device)
        self.item_features = create_kjt_from_ids(self.item_ids, "item_id", self.device)

    def test_init(self) -> None:
        """Test initialization of TwoTowerModel."""
        self.assertIsInstance(self.model.user_ebc, EmbeddingBagCollection)
        self.assertIsInstance(self.model.item_ebc, EmbeddingBagCollection)

        # Check embedding dimensions
        self.assertEqual(
            self.model.user_ebc.embedding_bag_configs()[0].embedding_dim,
            self.embedding_dim,
        )
        self.assertEqual(
            self.model.item_ebc.embedding_bag_configs()[0].embedding_dim,
            self.embedding_dim,
        )

    def test_forward(self) -> None:
        """Test forward pass of TwoTowerModel."""
        # Run forward pass with both user and item features
        output = self.model(self.user_features, self.item_features)

        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, 1))

        # Check output is not NaN
        self.assertFalse(torch.isnan(output).any())

        # Test forward pass with only user features (retrieval mode)
        user_embeddings = self.model(self.user_features, None)

        # Check user embeddings shape - should match the last layer size of user tower
        self.assertEqual(
            user_embeddings.shape, (self.batch_size, self.user_tower_sizes[-1])
        )

        # Check embeddings are not NaN
        self.assertFalse(torch.isnan(user_embeddings).any())

    def test_get_embeddings(self) -> None:
        """Test get_user_embedding and get_item_embedding methods."""
        # Get user embeddings
        user_embeddings = self.model.get_user_embedding(self.user_features)

        # Check user embeddings shape
        self.assertEqual(
            user_embeddings.shape, (self.batch_size, self.user_tower_sizes[-1])
        )

        # Get item embeddings
        item_embeddings = self.model.get_item_embedding(self.item_features)

        # Check item embeddings shape
        self.assertEqual(
            item_embeddings.shape, (self.batch_size, self.item_tower_sizes[-1])
        )

        # Check embeddings are not NaN
        self.assertFalse(torch.isnan(user_embeddings).any())
        self.assertFalse(torch.isnan(item_embeddings).any())

    def test_normalization(self) -> None:
        """Test embedding normalization."""
        # Get normalized user embeddings
        user_embeddings = self.model_normalized.get_user_embedding(self.user_features)

        # Check that embeddings are normalized (L2 norm should be close to 1)
        norms = torch.norm(user_embeddings, p=2, dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-6))

        # Get normalized item embeddings
        item_embeddings = self.model_normalized.get_item_embedding(self.item_features)

        # Check that embeddings are normalized (L2 norm should be close to 1)
        norms = torch.norm(item_embeddings, p=2, dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-6))


class TestTwoTowerRatingWrapper(unittest.TestCase):
    """Test cases for the TwoTowerRatingWrapper class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.embedding_dim = 32
        self.user_tower_sizes = [64, 32]
        self.item_tower_sizes = [64, 32]
        self.device = torch.device("cpu")

        # Create embedding bag configs
        self.num_users = 100
        self.num_items = 50

        self.user_eb_config = EmbeddingBagConfig(
            name="user_id",
            embedding_dim=self.embedding_dim,
            num_embeddings=self.num_users,
            feature_names=["user_id"],
        )

        self.item_eb_config = EmbeddingBagConfig(
            name="item_id",
            embedding_dim=self.embedding_dim,
            num_embeddings=self.num_items,
            feature_names=["item_id"],
        )

        # Create EmbeddingBagCollections
        self.user_ebc = EmbeddingBagCollection(
            tables=[self.user_eb_config],
            device=self.device,
        )

        self.item_ebc = EmbeddingBagCollection(
            tables=[self.item_eb_config],
            device=self.device,
        )

        # Create base model
        self.base_model = TwoTowerModel(
            user_ebc=self.user_ebc,
            item_ebc=self.item_ebc,
            user_tower_sizes=self.user_tower_sizes,
            item_tower_sizes=self.item_tower_sizes,
        )

        # Create wrapper
        self.model_wrapper = TwoTowerRatingWrapper(self.base_model).to(self.device)

        # Test data
        self.batch_size = 8

        # Create batch data
        self.user_ids = torch.randint(0, self.num_users, (self.batch_size,))
        self.item_ids = torch.randint(0, self.num_items, (self.batch_size,))

        # Create KeyedJaggedTensor for sparse features
        self.user_features = create_kjt_from_ids(self.user_ids, "user_id", self.device)
        self.item_features = create_kjt_from_ids(self.item_ids, "item_id", self.device)

    def test_forward(self) -> None:
        """Test forward pass of TwoTowerRatingWrapper."""
        # Run forward pass
        output = self.model_wrapper(self.user_features, self.item_features)

        # Check output has the correct batch size
        self.assertEqual(output.numel(), self.batch_size)

        # Ensure output is 1D or 2D with second dimension of 1
        if output.dim() == 2:
            self.assertEqual(output.shape[1], 1)
            # Squeeze to make it 1D for further checks
            output = output.squeeze()

        # Check output is not NaN
        self.assertFalse(torch.isnan(output).any())

        # Since we're using random initialization, we can't guarantee exact output range
        # Just check that the output is finite
        self.assertTrue(torch.isfinite(output).all())


class TestCreateKJTFromIds(unittest.TestCase):
    """Test cases for the create_kjt_from_ids function."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.batch_size = 8
        self.device = torch.device("cpu")

        # Create IDs
        self.user_ids = torch.randint(0, 100, (self.batch_size,))
        self.feature_name = "user_id"

    def test_create_kjt_from_ids(self) -> None:
        """Test create_kjt_from_ids function."""
        kjt = create_kjt_from_ids(self.user_ids, self.feature_name, self.device)

        # Check that it's a KeyedJaggedTensor
        self.assertIsInstance(kjt, KeyedJaggedTensor)

        # Check keys
        self.assertEqual(kjt.keys(), [self.feature_name])

        # Check values
        self.assertEqual(kjt.values().shape[0], self.batch_size)

        # Check lengths
        self.assertEqual(kjt.lengths().shape[0], self.batch_size)
        self.assertTrue((kjt.lengths() == 1).all())


if __name__ == "__main__":
    unittest.main()
