#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

# The imports 'Dict' and 'List' from the typing module are not used in the code.
# Therefore, they can be safely removed to fix the unused import warnings.
# The error indicates that the 'main' attribute is missing from the specified module.
# To fix this, we need to ensure that the 'main' function is defined in the module.
# Since the error is related to 'torchrec.github.examples.prediction.predict_using_torchrec',
# we should check that module to ensure it has a 'main' function defined.
# However, since we don't have access to that module here, we can only provide a placeholder solution.


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


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
        user_categories (torch.Tensor): Tensor of user categories for each sample.
        item_categories (torch.Tensor): Tensor of item categories for each sample.
        dense_features (torch.Tensor): Tensor of dense features for each sample.
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

        # Generate some categorical features
        self.user_categories: torch.Tensor = torch.randint(
            0, 10, (num_samples,)
        )  # 10 user categories
        self.item_categories: torch.Tensor = torch.randint(
            0, 20, (num_samples,)
        )  # 20 item categories

        # Generate dense features (normalized to [0, 1])
        self.dense_features: torch.Tensor = torch.rand(
            num_samples, 4
        )  # 4 dense features

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return self.num_samples

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing user_id, item_id, user_category,
                  item_category, dense_features, and rating for the sample.
        """
        return {
            "user_id": self.user_ids[idx],
            "item_id": self.item_ids[idx],
            "user_category": self.user_categories[idx],
            "item_category": self.item_categories[idx],
            "dense_features": self.dense_features[idx],
            "rating": self.ratings[idx],
        }


# DLRM Model Implementation
class DLRM(nn.Module):
    def __init__(
        self,
        embedding_dim=64,
        dense_arch_dims=[4, 32, 16],  # Dense MLP architecture
        top_arch_dims=[128, 64, 1],  # Top MLP architecture
        categorical_feature_sizes=[1000, 500, 10, 20],  # Sizes of categorical features
        interaction_op="cat",  # 'cat' or 'dot'
    ):
        """
        Initializes the DLRM model with specified architecture and parameters.

        Args:
            embedding_dim (int): Dimension of the embedding vectors.
            dense_arch_dims (list): Architecture of the dense MLP.
            top_arch_dims (list): Architecture of the top MLP.
            categorical_feature_sizes (list): Sizes of the categorical features.
            interaction_op (str): Type of interaction operation ('cat' or 'dot').
        """
        super(DLRM, self).__init__()

        self.embedding_dim = embedding_dim
        self.interaction_op = interaction_op
        self.num_categorical_features = len(categorical_feature_sizes)

        # Bottom MLP for dense features
        self.bottom_mlp = self._build_mlp(dense_arch_dims)

        # Embedding tables for categorical features
        self.embedding_tables = nn.ModuleList(
            [nn.Embedding(size, embedding_dim) for size in categorical_feature_sizes]
        )

        # Initialize embeddings
        for embedding in self.embedding_tables:
            nn.init.uniform_(
                embedding.weight, -1.0 / embedding_dim, 1.0 / embedding_dim
            )

        # Projection layer for dense features to match embedding dimension for interactions
        if interaction_op == "dot" and dense_arch_dims[-1] != embedding_dim:
            self.dense_projection = nn.Linear(dense_arch_dims[-1], embedding_dim)
        else:
            self.dense_projection = None

        # Top MLP
        if interaction_op == "cat":
            # Concatenation: dense output + all embeddings
            top_input_dim = (
                dense_arch_dims[-1] + embedding_dim * self.num_categorical_features
            )
        else:  # dot product
            # Dot product interactions between dense and categorical features
            num_interactions = self.num_categorical_features + 1  # +1 for dense
            top_input_dim = (
                int(num_interactions * (num_interactions - 1) / 2) + dense_arch_dims[-1]
            )

        top_arch_dims[0] = top_input_dim
        self.top_mlp = self._build_mlp(top_arch_dims)

        # Sigmoid activation for final output
        self.sigmoid = nn.Sigmoid()

    def _build_mlp(self, arch_dims):
        """
        Builds a Multi-Layer Perceptron (MLP) with the specified architecture.

        Args:
            arch_dims (list): List of dimensions for each layer in the MLP.

        Returns:
            nn.Sequential: A sequential container of MLP layers.
        """
        layers = []
        for i in range(len(arch_dims) - 1):
            layers.append(nn.Linear(arch_dims[i], arch_dims[i + 1]))
            if i < len(arch_dims) - 2:  # No activation on final layer
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.2))
        return nn.Sequential(*layers)

    def forward(self, dense_features, categorical_features):
        """
        Forward pass of the DLRM model.

        Args:
            dense_features (torch.Tensor): Dense input features [batch_size, num_dense_features].
            categorical_features (list): List of categorical features [batch_size] each.

        Returns:
            torch.Tensor: The output predictions of the model.
        """
        # Bottom MLP processing dense features
        dense_output = self.bottom_mlp(dense_features)

        # Embedding lookup for categorical features
        embeddings = []
        for i, cat_feature in enumerate(categorical_features):
            emb = self.embedding_tables[i](cat_feature)
            embeddings.append(emb)

        # Feature interaction
        if self.interaction_op == "cat":
            # Simple concatenation
            all_features = [dense_output] + embeddings
            interaction_output = torch.cat(all_features, dim=1)
        else:
            # Dot product interactions
            # Project dense output to embedding dimension if needed
            if self.dense_projection is not None:
                dense_projected = self.dense_projection(dense_output)
            else:
                dense_projected = dense_output

            all_features = [dense_projected] + embeddings
            interactions = []

            # Add original dense output to interactions
            interactions.append(dense_output)

            # Compute pairwise dot products
            for i in range(len(all_features)):
                for j in range(i + 1, len(all_features)):
                    dot_product = torch.sum(
                        all_features[i] * all_features[j], dim=1, keepdim=True
                    )
                    interactions.append(dot_product)

            interaction_output = torch.cat(interactions, dim=1)

        # Top MLP
        output = self.top_mlp(interaction_output)

        # Scale output to [0, 5] for rating prediction
        output = self.sigmoid(output) * 5.0

        return output.squeeze()


def train_dlrm_model():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training DLRM on device: {device}")

    # Hyperparameters
    num_users = 1000
    num_items = 500
    num_user_categories = 10
    num_item_categories = 20
    embedding_dim = 64
    batch_size = 256
    learning_rate = 0.001
    num_epochs = 10

    # Create dataset and dataloader
    dataset = RecommendationDataset(num_users, num_items, 10000)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create DLRM model
    model = DLRM(
        embedding_dim=embedding_dim,
        dense_arch_dims=[4, 32, 16],  # 4 dense input features
        top_arch_dims=[128, 64, 1],  # Will be adjusted based on interaction
        categorical_feature_sizes=[
            num_users,
            num_items,
            num_user_categories,
            num_item_categories,
        ],
        interaction_op="dot",  # Use dot product for better performance
    ).to(device)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    print(f"DLRM Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            # Move data to device
            user_ids = batch["user_id"].to(device)
            item_ids = batch["item_id"].to(device)
            user_categories = batch["user_category"].to(device)
            item_categories = batch["item_category"].to(device)
            dense_features = batch["dense_features"].to(device)
            ratings = batch["rating"].to(device)

            # Prepare categorical features list
            categorical_features = [
                user_ids,
                item_ids,
                user_categories,
                item_categories,
            ]

            # Forward pass
            predictions = model(dense_features, categorical_features)
            loss = criterion(predictions, ratings)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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

    print("DLRM Training completed!")

    # Save the model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "embedding_dim": embedding_dim,
                "dense_arch_dims": [4, 32, 16],
                "top_arch_dims": [128, 64, 1],
                "categorical_feature_sizes": [
                    num_users,
                    num_items,
                    num_user_categories,
                    num_item_categories,
                ],
                "interaction_op": "dot",
            },
        },
        "dlrm_model.pth",
    )
    print("DLRM Model saved as 'dlrm_model.pth'")

    return model


def evaluate_dlrm_model(model, dataloader, device):
    """Evaluate the DLRM model on a dataset"""
    model.eval()
    total_loss = 0
    num_batches = 0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for batch in dataloader:
            user_ids = batch["user_id"].to(device)
            item_ids = batch["item_id"].to(device)
            user_categories = batch["user_category"].to(device)
            item_categories = batch["item_category"].to(device)
            dense_features = batch["dense_features"].to(device)
            ratings = batch["rating"].to(device)

            categorical_features = [
                user_ids,
                item_ids,
                user_categories,
                item_categories,
            ]
            predictions = model(dense_features, categorical_features)
            loss = criterion(predictions, ratings)

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    rmse = np.sqrt(avg_loss)
    print(f"DLRM Evaluation - MSE: {avg_loss:.4f}, RMSE: {rmse:.4f}")
    return avg_loss, rmse


def make_dlrm_predictions(
    model, user_id, item_ids, user_category, item_categories, device
):
    """Make predictions using DLRM for a user on multiple items"""
    model.eval()
    with torch.no_grad():
        batch_size = len(item_ids)

        # Prepare inputs
        user_ids_tensor = torch.tensor([user_id] * batch_size).to(device)
        item_ids_tensor = torch.tensor(item_ids).to(device)
        user_cats = torch.tensor([user_category] * batch_size).to(device)
        item_cats = torch.tensor(item_categories).to(device)

        # Generate random dense features for demonstration
        dense_features = torch.rand(batch_size, 4).to(device)

        categorical_features = [user_ids_tensor, item_ids_tensor, user_cats, item_cats]
        predictions = model(dense_features, categorical_features)

        return predictions.cpu().numpy()


def main() -> None:
    print("Starting DLRM (Deep Learning Recommendation Model) Training...")

    # Train the DLRM model
    trained_model = train_dlrm_model()

    # Create evaluation dataset
    eval_dataset = RecommendationDataset(1000, 500, 2000)
    eval_dataloader = DataLoader(eval_dataset, batch_size=256, shuffle=False)

    # Evaluate the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nEvaluating DLRM model...")
    evaluate_dlrm_model(trained_model, eval_dataloader, device)

    # Example prediction
    print("\nMaking sample DLRM predictions...")
    sample_user_id = 42
    sample_items = [10, 25, 50, 100, 200]
    sample_user_cat = 3
    sample_item_cats = [5, 12, 8, 15, 2]

    predictions = make_dlrm_predictions(
        trained_model,
        sample_user_id,
        sample_items,
        sample_user_cat,
        sample_item_cats,
        device,
    )

    print(f"DLRM Predictions for user {sample_user_id}:")
    for item_id, pred in zip(sample_items, predictions):
        print(f"  Item {item_id}: {pred:.2f}")


# If the module is supposed to be imported, ensure it is correctly imported.
# import torchrec.github.examples.prediction.predict_using_torchrec as predict_module

# Ensure the main function is called if this script is executed directly.
if __name__ == "__main__":
    main()
