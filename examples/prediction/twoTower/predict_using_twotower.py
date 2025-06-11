#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.modules.mlp import MLP

# TorchRec imports
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


# Mock dataset for demonstration.
class RecommendationDataset(Dataset):
    """
    A PyTorch Dataset class for generating random user-item interaction data
    for recommendation systems.

    Attributes:
        num_samples (int): Number of samples in the dataset.
        num_users (int): Number of unique users.
        num_items (int): Number of unique items.
        user_ids (torch.Tensor): Tensor of user IDs for each sample.
        item_ids (torch.Tensor): Tensor of item IDs for each sample.
        ratings (torch.Tensor): Tensor of ratings for each sample.
        user_features (torch.Tensor): Tensor of user features for each sample.
        item_features (torch.Tensor): Tensor of item features for each sample.
    """

    def __init__(
        self, num_users: int = 1000, num_items: int = 500, num_samples: int = 10000
    ) -> None:
        """
        Initializes the RecommendationDataset with random data.

        Args:
            num_users (int): Number of unique users.
            num_items (int): Number of unique items.
            num_samples (int): Number of samples to generate.
        """
        self.num_samples: int = num_samples
        self.num_users: int = num_users
        self.num_items: int = num_items

        # Generate random user-item interactions
        self.user_ids: torch.Tensor = torch.randint(0, num_users, (num_samples,))
        self.item_ids: torch.Tensor = torch.randint(0, num_items, (num_samples,))

        # Generate random ratings (0-5)
        self.ratings: torch.Tensor = torch.randint(0, 6, (num_samples,)).float()

        # Generate random user and item features (normalized to [0, 1])
        self.user_features: torch.Tensor = torch.rand(num_samples, 4)
        self.item_features: torch.Tensor = torch.rand(num_samples, 4)

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves a sample from the dataset at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing user_id, item_id, user_features,
                  item_features, and rating for the sample.
        """
        return {
            "user_id": self.user_ids[idx],
            "item_id": self.item_ids[idx],
            "user_features": self.user_features[idx],
            "item_features": self.item_features[idx],
            "rating": self.ratings[idx],
        }


# TorchRec TwoTower model using KeyedJaggedTensor and EmbeddingBagCollection
class TwoTowerModel(nn.Module):
    """
    A Two-Tower model implementation using TorchRec's EmbeddingBagCollection and KeyedJaggedTensor.
    This model follows the architecture described in Google Cloud's article:
    "Scaling deep retrieval with TensorFlow and the two towers architecture"

    The model consists of two separate towers:
    1. User/Query tower: Processes user features to create user embeddings
    2. Item tower: Processes item features to create item embeddings

    The embeddings from both towers are projected into the same semantic space, allowing
    for efficient retrieval using approximate nearest neighbor search.

    Key advantages of this architecture:
    - Item embeddings can be pre-computed offline and indexed for efficient retrieval
    - Only the user/query tower needs to be computed at serving time
    - Scales to large item catalogs by avoiding the need to score all items

    Args:
        user_ebc: EmbeddingBagCollection for user features
        item_ebc: EmbeddingBagCollection for item features
        user_tower_sizes: Layer sizes for the user tower MLP
        item_tower_sizes: Layer sizes for the item tower MLP
        normalize_embeddings: Whether to L2 normalize the final embeddings (recommended for cosine similarity)

    Example:
        ```
        # Create embedding bag configs
        user_eb_config = EmbeddingBagConfig(
            name="user_id",
            embedding_dim=64,
            num_embeddings=1000,
            feature_names=["user_id"],
        )

        item_eb_config = EmbeddingBagConfig(
            name="item_id",
            embedding_dim=64,
            num_embeddings=500,
            feature_names=["item_id"],
        )

        # Create EmbeddingBagCollections
        user_ebc = EmbeddingBagCollection(
            tables=[user_eb_config],
            device=torch.device("cpu"),
        )

        item_ebc = EmbeddingBagCollection(
            tables=[item_eb_config],
            device=torch.device("cpu"),
        )

        # Create TwoTowerModel
        model = TwoTowerModel(
            user_ebc=user_ebc,
            item_ebc=item_ebc,
            user_tower_sizes=[64, 32],
            item_tower_sizes=[64, 32],
        )

        # Forward pass
        batch_size = 2
        user_features = KeyedJaggedTensor(
            keys=["user_id"],
            values=torch.tensor([0, 1]),
            lengths=torch.ones(2, dtype=torch.int32),
        )

        item_features = KeyedJaggedTensor(
            keys=["item_id"],
            values=torch.tensor([0, 1]),
            lengths=torch.ones(2, dtype=torch.int32),
        )

        # Get predictions
        logits = model(user_features, item_features)
        ```
    """

    def __init__(
        self,
        user_ebc: EmbeddingBagCollection,
        item_ebc: EmbeddingBagCollection,
        user_tower_sizes: List[int],
        item_tower_sizes: List[int],
        normalize_embeddings: bool = False,
    ) -> None:
        super().__init__()

        # Embedding collections
        self.user_ebc = user_ebc
        self.item_ebc = item_ebc

        # Get embedding dimensions
        user_embedding_dim = user_ebc.embedding_bag_configs()[0].embedding_dim
        item_embedding_dim = item_ebc.embedding_bag_configs()[0].embedding_dim

        # User tower
        self.user_tower = MLP(
            in_size=user_embedding_dim,
            layer_sizes=user_tower_sizes,
        )

        # Item tower
        self.item_tower = MLP(
            in_size=item_embedding_dim,
            layer_sizes=item_tower_sizes,
        )

        # Store feature names for lookup
        self.user_feature_names: List[str] = [
            config.feature_names[0] for config in user_ebc.embedding_bag_configs()
        ]
        self.item_feature_names: List[str] = [
            config.feature_names[0] for config in item_ebc.embedding_bag_configs()
        ]

        # Whether to normalize embeddings (useful for cosine similarity)
        self.normalize_embeddings = normalize_embeddings

    def forward(
        self,
        user_features: KeyedJaggedTensor,
        item_features: Optional[KeyedJaggedTensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the Two-Tower model.

        This method supports two modes:
        1. Both user_features and item_features provided: Computes similarity scores
        2. Only user_features provided: Returns user embeddings for retrieval

        Args:
            user_features: User sparse features as KeyedJaggedTensor
            item_features: Item sparse features as KeyedJaggedTensor (optional for embedding-only mode)

        Returns:
            torch.Tensor: Relevance scores if both towers are used, otherwise user embeddings
        """
        # Process user features
        user_embeddings = self.user_ebc(user_features)
        user_embedding = torch.cat(
            [user_embeddings[feature] for feature in self.user_feature_names], dim=1
        )
        user_vector = self.user_tower(user_embedding)

        # Normalize if requested
        if self.normalize_embeddings:
            user_vector = torch.nn.functional.normalize(user_vector, p=2, dim=1)

        # If no item features provided, return user embeddings for retrieval
        if item_features is None:
            return user_vector

        # Process item features
        item_embeddings = self.item_ebc(item_features)
        item_embedding = torch.cat(
            [item_embeddings[feature] for feature in self.item_feature_names], dim=1
        )
        item_vector = self.item_tower(item_embedding)

        # Normalize if requested
        if self.normalize_embeddings:
            item_vector = torch.nn.functional.normalize(item_vector, p=2, dim=1)

        # Compute relevance score using dot product
        # We keep the dimension to match the DLRM example output shape
        return torch.sum(user_vector * item_vector, dim=1, keepdim=True)

    def get_user_embedding(self, user_features: KeyedJaggedTensor) -> torch.Tensor:
        """
        Get user embeddings for retrieval purposes.

        Args:
            user_features: User sparse features as KeyedJaggedTensor

        Returns:
            torch.Tensor: User embeddings
        """
        user_embeddings = self.user_ebc(user_features)
        user_embedding = torch.cat(
            [user_embeddings[feature] for feature in self.user_feature_names], dim=1
        )
        user_vector = self.user_tower(user_embedding)

        # Normalize if requested
        if self.normalize_embeddings:
            user_vector = torch.nn.functional.normalize(user_vector, p=2, dim=1)

        return user_vector

    def get_item_embedding(self, item_features: KeyedJaggedTensor) -> torch.Tensor:
        """
        Get item embeddings for indexing purposes.

        Args:
            item_features: Item sparse features as KeyedJaggedTensor

        Returns:
            torch.Tensor: Item embeddings
        """
        item_embeddings = self.item_ebc(item_features)
        item_embedding = torch.cat(
            [item_embeddings[feature] for feature in self.item_feature_names], dim=1
        )
        item_vector = self.item_tower(item_embedding)

        # Normalize if requested
        if self.normalize_embeddings:
            item_vector = torch.nn.functional.normalize(item_vector, p=2, dim=1)

        return item_vector


# TwoTower wrapper for rating prediction
class TwoTowerRatingWrapper(nn.Module):
    """
    Wrapper for TwoTower model to scale the output to [0, 5] for rating prediction.

    Args:
        two_tower_model: The TwoTower model to wrap

    Example:
        ```
        # Create embedding bag configs
        user_eb_config = EmbeddingBagConfig(
            name="user_id",
            embedding_dim=64,
            num_embeddings=1000,
            feature_names=["user_id"],
        )

        item_eb_config = EmbeddingBagConfig(
            name="item_id",
            embedding_dim=64,
            num_embeddings=500,
            feature_names=["item_id"],
        )

        # Create EmbeddingBagCollections
        user_ebc = EmbeddingBagCollection(
            tables=[user_eb_config],
            device=torch.device("cpu"),
        )

        item_ebc = EmbeddingBagCollection(
            tables=[item_eb_config],
            device=torch.device("cpu"),
        )

        # Create base model
        base_model = TwoTowerModel(
            user_ebc=user_ebc,
            item_ebc=item_ebc,
            user_tower_sizes=[64, 32],
            item_tower_sizes=[64, 32],
        )

        # Create wrapper
        model_wrapper = TwoTowerRatingWrapper(base_model)

        # Forward pass
        batch_size = 2
        user_features = KeyedJaggedTensor(
            keys=["user_id"],
            values=torch.tensor([0, 1]),
            lengths=torch.ones(2, dtype=torch.int32),
        )

        item_features = KeyedJaggedTensor(
            keys=["item_id"],
            values=torch.tensor([0, 1]),
            lengths=torch.ones(2, dtype=torch.int32),
        )

        # Get predictions (scaled to 0-5 range)
        predictions = model_wrapper(user_features, item_features)
        ```
    """

    def __init__(self, two_tower_model: TwoTowerModel) -> None:
        super().__init__()
        self.model: TwoTowerModel = two_tower_model
        self.sigmoid: nn.Sigmoid = nn.Sigmoid()

    def forward(
        self, user_features: KeyedJaggedTensor, item_features: KeyedJaggedTensor
    ) -> torch.Tensor:
        """
        Forward pass of the TwoTower wrapper.

        Args:
            user_features: User sparse features as KeyedJaggedTensor
            item_features: Item sparse features as KeyedJaggedTensor

        Returns:
            torch.Tensor: Rating prediction scaled to [0, 5]
        """
        logits = self.model(user_features, item_features)
        # Scale output to [0, 5] for rating prediction
        return self.sigmoid(logits.squeeze()) * 5.0


def create_kjt_from_ids(
    ids: torch.Tensor, feature_name: str, device: torch.device
) -> KeyedJaggedTensor:
    """
    Create a KeyedJaggedTensor from a tensor of IDs.

    Args:
        ids: Tensor of IDs
        feature_name: Name of the feature
        device: Device to place the KeyedJaggedTensor on

    Returns:
        KeyedJaggedTensor: Sparse features in KeyedJaggedTensor format
    """
    batch_size = ids.size(0)
    lengths = torch.ones(batch_size, dtype=torch.int32, device=device)

    # Create KeyedJaggedTensor
    return KeyedJaggedTensor(
        keys=[feature_name],
        values=ids.to(device),
        lengths=lengths,
    )


def train_two_tower_model() -> Tuple[TwoTowerRatingWrapper, str]:
    """
    Trains the Two-Tower model using a specified dataset and hyperparameters.

    Returns:
        tuple: A tuple containing the trained Two-Tower model and the model filename.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Two-Tower model on device: {device}")

    # Hyperparameters
    num_users = 1000
    num_items = 500
    embedding_dim = 64
    batch_size = 256
    learning_rate = 0.001
    num_epochs = 10

    # Create dataset and dataloader
    dataset = RecommendationDataset(num_users, num_items, 10000)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create embedding bag configs
    user_eb_config = EmbeddingBagConfig(
        name="user_id",
        embedding_dim=embedding_dim,
        num_embeddings=num_users,
        feature_names=["user_id"],
    )

    item_eb_config = EmbeddingBagConfig(
        name="item_id",
        embedding_dim=embedding_dim,
        num_embeddings=num_items,
        feature_names=["item_id"],
    )

    # Create EmbeddingBagCollections
    user_ebc = EmbeddingBagCollection(
        tables=[user_eb_config],
        device=device,
    )

    item_ebc = EmbeddingBagCollection(
        tables=[item_eb_config],
        device=device,
    )

    # Create TwoTowerModel
    model = TwoTowerModel(
        user_ebc=user_ebc,
        item_ebc=item_ebc,
        user_tower_sizes=[128, 64, 32],
        item_tower_sizes=[128, 64, 32],
    ).to(device)

    # Create a wrapper to scale the output to [0, 5] for rating prediction
    model_wrapper = TwoTowerRatingWrapper(model).to(device)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model_wrapper.parameters(), lr=learning_rate, weight_decay=1e-5
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    print(
        f"Two-Tower Model parameters: {sum(p.numel() for p in model_wrapper.parameters()):,}"
    )

    # Training loop
    model_wrapper.train()
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            # Move data to device
            user_ids = batch["user_id"].to(device)
            item_ids = batch["item_id"].to(device)
            ratings = batch["rating"].to(device)

            # Create KeyedJaggedTensor for sparse features
            user_features = create_kjt_from_ids(user_ids, "user_id", device)
            item_features = create_kjt_from_ids(item_ids, "item_id", device)

            # Forward pass
            predictions = model_wrapper(user_features, item_features)
            loss = criterion(predictions, ratings)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model_wrapper.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        # Update learning rate
        scheduler.step()

        avg_loss = total_loss / num_batches
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, LR: {current_lr:.6f}"
        )

    print("Two-Tower Training completed!")

    # Save the model
    model_filename = "two_tower_model.pth"
    torch.save(
        {
            "model_state_dict": model_wrapper.state_dict(),
            "embedding_dim": embedding_dim,
            "num_users": num_users,
            "num_items": num_items,
            "user_tower_sizes": [128, 64, 32],
            "item_tower_sizes": [128, 64, 32],
        },
        model_filename,
    )
    print(f"Two-Tower Model saved as {model_filename}")
    return model_wrapper, model_filename


def evaluate_two_tower_model(
    model: nn.Module,
    dataloader: DataLoader[RecommendationDataset],
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate the Two-Tower model on a dataset.

    Args:
        model (nn.Module): The Two-Tower model to evaluate.
        dataloader (DataLoader): DataLoader providing the evaluation dataset.
        device (torch.device): The device to perform evaluation on (CPU or GPU).

    Returns:
        tuple: A tuple containing the average loss and root mean square error (RMSE).
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for batch in dataloader:
            # Move data to device
            user_ids = batch["user_id"].to(device)
            item_ids = batch["item_id"].to(device)
            ratings = batch["rating"].to(device)

            # Create KeyedJaggedTensor for sparse features
            user_features = create_kjt_from_ids(user_ids, "user_id", device)
            item_features = create_kjt_from_ids(item_ids, "item_id", device)

            # Forward pass
            predictions = model(user_features, item_features)
            loss = criterion(predictions, ratings)

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    rmse = np.sqrt(avg_loss)
    print(f"Two-Tower Evaluation - MSE: {avg_loss:.4f}, RMSE: {rmse:.4f}")
    return avg_loss, rmse


def make_two_tower_predictions(
    model: TwoTowerRatingWrapper,
    user_ids: List[int],
    item_ids: List[int],
    device: torch.device,
) -> np.ndarray:
    """Make predictions using Two-Tower model for multiple user-item pairs"""
    model.eval()
    with torch.no_grad():
        batch_size = len(user_ids)
        assert len(item_ids) == batch_size, "Number of user_ids and item_ids must match"

        # Prepare inputs
        user_ids_tensor = torch.tensor(user_ids).to(device)
        item_ids_tensor = torch.tensor(item_ids).to(device)

        # Create KeyedJaggedTensor for sparse features
        user_features = create_kjt_from_ids(user_ids_tensor, "user_id", device)
        item_features = create_kjt_from_ids(item_ids_tensor, "item_id", device)

        # Make predictions
        predictions = model(user_features, item_features)

        # Convert to numpy array and ensure it's a 1D array
        numpy_predictions = predictions.cpu().numpy()
        # Flatten in case it's not already 1D
        return numpy_predictions.flatten()


def remove_model_file(model_filename: str) -> None:
    """
    Removes the model file if it exists.

    Args:
        model_filename (str): The filename of the model to be removed.
    """
    if os.path.exists(model_filename):
        try:
            os.remove(model_filename)
            print(f"Successfully removed the file: {model_filename}")
        except PermissionError:
            print(f"Permission denied: {model_filename}")
        except Exception as e:
            print(f"An error occurred while trying to remove the file: {e}")
    else:
        print(f"File does not exist: {model_filename}")


def main() -> None:
    """
    Main function to orchestrate the training, evaluation, and prediction
    processes of the Two-Tower model.

    This function performs the following steps:
    1. Trains the Two-Tower model using a specified dataset and hyperparameters.
    2. Evaluates the trained model on a separate evaluation dataset.
    3. Makes sample predictions for specific user-item pairs.
    4. Cleans up the model after training to free up resources.
    """
    print("Starting Two-Tower Model Training...")

    try:
        # Train the Two-Tower model
        print("Starting training...")
        trained_model, model_filename = train_two_tower_model()
        print("Training completed successfully!")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback

        traceback.print_exc()
        return

    # Create evaluation dataset
    eval_dataset = RecommendationDataset(1000, 500, 2000)
    eval_dataloader = DataLoader(eval_dataset, batch_size=256, shuffle=False)

    # Evaluate the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nEvaluating Two-Tower model...")
    evaluate_two_tower_model(trained_model, eval_dataloader, device)

    # Example prediction
    print("\nMaking sample Two-Tower predictions...")
    sample_user_ids = [42, 42, 42, 100, 100]
    sample_item_ids = [10, 25, 50, 100, 200]

    predictions = make_two_tower_predictions(
        trained_model,
        sample_user_ids,
        sample_item_ids,
        device,
    )

    print("Two-Tower Predictions:")
    for user_id, item_id, pred in zip(sample_user_ids, sample_item_ids, predictions):
        print(f"  User {user_id}, Item {item_id}: {pred:.2f}")

    # clean the model after training
    print(f"Cleaning the model {model_filename}")
    # comment this line if you want to keep the model file
    remove_model_file(model_filename)


if __name__ == "__main__":
    main()
