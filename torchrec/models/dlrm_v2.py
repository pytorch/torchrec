#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torchrec.datasets.utils import Batch
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.modules.mlp import MLP
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor


def choose(n: int, k: int) -> int:
    """
    Simple implementation of math.comb for Python 3.7 compatibility.
    """
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in range(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0


class SparseArch(nn.Module):
    """
    Processes the sparse features of DLRM. Does embedding lookups for all EmbeddingBag
    and embedding features of each collection.

    Args:
        embedding_bag_collection (EmbeddingBagCollection): represents a collection of
            pooled embeddings.

    Example::

        eb1_config = EmbeddingBagConfig(
           name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1"]
        )
        eb2_config = EmbeddingBagConfig(
           name="t2", embedding_dim=4, num_embeddings=10, feature_names=["f2"]
        )

        ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])
        sparse_arch = SparseArch(embedding_bag_collection)

        #     0       1        2  <-- batch
        # 0   [0,1] None    [2]
        # 1   [3]    [4]    [5,6,7]
        # ^
        # feature
        features = KeyedJaggedTensor.from_offsets_sync(
           keys=["f1", "f2"],
           values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
           offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8]),
        )

        sparse_embeddings = sparse_arch(features)
    """

    def __init__(self, embedding_bag_collection: EmbeddingBagCollection) -> None:
        super().__init__()
        self.embedding_bag_collection: EmbeddingBagCollection = embedding_bag_collection
        assert (
            self.embedding_bag_collection.embedding_bag_configs
        ), "Embedding bag collection cannot be empty!"
        self.D: int = self.embedding_bag_collection.embedding_bag_configs[
            0
        ].embedding_dim
        self._sparse_feature_names: List[str] = [
            name
            for conf in embedding_bag_collection.embedding_bag_configs
            for name in conf.feature_names
        ]

        self.F: int = len(self._sparse_feature_names)

    def forward(
        self,
        features: KeyedJaggedTensor,
    ) -> torch.Tensor:
        """
        Args:
            features (KeyedJaggedTensor): an input tensor of sparse features.

        Returns:
            torch.Tensor: tensor of shape B X F X D.
        """

        sparse_features: KeyedTensor = self.embedding_bag_collection(features)

        B: int = features.stride()

        sparse: Dict[str, torch.Tensor] = sparse_features.to_dict()
        sparse_values: List[torch.Tensor] = []
        for name in self.sparse_feature_names:
            sparse_values.append(sparse[name])

        return torch.cat(sparse_values, dim=1).reshape(B, self.F, self.D)

    @property
    def sparse_feature_names(self) -> List[str]:
        return self._sparse_feature_names


class DenseArch(nn.Module):
    """
    Processes the dense features of DLRM model.

    Args:
        in_features (int): dimensionality of the dense input features.
        layer_sizes (List[int]): list of layer sizes.
        device (Optional[torch.device]): default compute device.

    Example::

        B = 20
        D = 3
        dense_arch = DenseArch(10, layer_sizes=[15, D])
        dense_embedded = dense_arch(torch.rand((B, 10)))
    """

    def __init__(
        self,
        in_features: int,
        layer_sizes: List[int],
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.model: nn.Module = MLP(
            in_features, layer_sizes, bias=True, activation="relu", device=device
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): an input tensor of dense features.

        Returns:
            torch.Tensor: an output tensor of size B X D.
        """
        return self.model(features)


class InteractionArch(nn.Module):
    """
    Processes the output of both `SparseArch` (sparse_features) and `DenseArch`
    (dense_features).

    When no optional internal MLPs used:
    Returns the pairwise dot product of each sparse feature pair,
    the dot product of each sparse features with the output of the dense layer,
    and the dense layer itself (all concatenated).

    When optional internal MLPs are used:
    Returns (internal MLP 1 output)*(internal MLP 2 output) and the dense layer itself (all concatenated).
    Internal MLPs 1 and 2 are of size BxF1xD and BxDxF2 respectively.

    .. note::
        The dimensionality of the `dense_features` (D) is expected to match the
        dimensionality of the `sparse_features` so that the dot products between them
        can be computed.


    Args:
        num_sparse_features (int): F.
        interaction_subbranch_1_layer_sizes (List[int]): the layer sizes for first branch of
            interaction layer. The output dimension must be a multiple of D.
        interaction_subbranch_2_layer_sizes (List[int]):the layer sizes for second branch of
            interaction layer. The output dimension must be a multiple of D.
        dense_device (Optional[torch.device]): default compute device.

    Example::

        D = 3
        B = 10
        keys = ["f1", "f2"]
        F = len(keys)
        inter_arch = InteractionArch(num_sparse_features=len(keys))

        dense_features = torch.rand((B, D))
        sparse_features = torch.rand((B, F, D))

        #  B X (D + F + F choose 2), when no MLPs used (regular interaction).
        #  B X (D + (F + 1) * F'), when MLPs used
        concat_dense = inter_arch(dense_features, sparse_features)
    """

    def __init__(
        self,
        num_sparse_features: int,
        interaction_subbranch_1: Optional[nn.Module] = None,
        interaction_subbranch_2: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.F: int = num_sparse_features
        self.triu_indices: torch.Tensor = torch.triu_indices(
            self.F + 1, self.F + 1, offset=1
        )
        self.interaction_subbranch_1 = interaction_subbranch_1
        self.interaction_subbranch_2 = interaction_subbranch_2

    def forward(
        self, dense_features: torch.Tensor, sparse_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            dense_features (torch.Tensor): an input tensor of size B X D.
            sparse_features (torch.Tensor): an input tensor of size B X F X D.

        Returns:
            torch.Tensor: an output tensor of size B X (D + F + F choose 2) or--
            when using MLPs--an output tensor of size B X (D + (F * F')).
        """
        if self.F <= 0:
            return dense_features
        (B, D) = dense_features.shape

        combined_values = torch.cat(
            (dense_features.unsqueeze(1), sparse_features), dim=1
        )

        if self.interaction_subbranch_1:
            bmm_input_1 = self.interaction_subbranch_1(
                torch.reshape(combined_values, [B, -1])
            )
        else:
            bmm_input_1 = combined_values


        if self.interaction_subbranch_2:
            bmm_input_2 = self.interaction_subbranch_2(
                torch.reshape(combined_values, [B, -1])
            )
        else:
            bmm_input_2 = torch.transpose(combined_values, 1, 2)


        interactions = torch.bmm(
            bmm_input_1.reshape([B, -1, D]),
            bmm_input_2.reshape([B, D, -1]),
        )

        if self.interaction_subbranch_1 or self.interaction_subbranch_2:
            interactions_flat = torch.reshape(interactions, [B, -1])
        else:
            interactions_flat = interactions[
                :, self.triu_indices[0], self.triu_indices[1]
            ]

        return torch.cat((dense_features, interactions_flat), dim=1)


class OverArch(nn.Module):
    """
    Final Arch of DLRM - simple MLP over OverArch.

    Args:
        in_features (int): size of the input.
        layer_sizes (List[int]): sizes of the layers of the `OverArch`.
        device (Optional[torch.device]): default compute device.

    Example::

        B = 20
        D = 3
        over_arch = OverArch(10, [5, 1])
        logits = over_arch(torch.rand((B, 10)))
    """

    def __init__(
        self,
        in_features: int,
        layer_sizes: List[int],
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if len(layer_sizes) <= 1:
            raise ValueError("OverArch must have multiple layers.")
        self.model: nn.Module = nn.Sequential(
            MLP(
                in_features,
                layer_sizes[:-1],
                bias=True,
                activation="relu",
                device=device,
            ),
            nn.Linear(layer_sizes[-2], layer_sizes[-1], bias=True, device=device),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor):

        Returns:
            torch.Tensor: size B X layer_sizes[-1]
        """
        return self.model(features)


class DLRM(nn.Module):
    """
    Recsys model from "Deep Learning Recommendation Model for Personalization and
    Recommendation Systems" (https://arxiv.org/abs/1906.00091). Processes sparse
    features by learning pooled embeddings for each feature. Learns the relationship
    between dense features and sparse features by projecting dense features into the
    same embedding space. Also, learns the pairwise relationships between sparse
    features.

    The module assumes all sparse features have the same embedding dimension
    (i.e. each EmbeddingBagConfig uses the same embedding_dim).

    The following notation is used throughout the documentation for the models:

    * F: number of sparse features
    * D: embedding_dimension of sparse features
    * B: batch size
    * num_features: number of dense features

    Args:
        embedding_bag_collection (EmbeddingBagCollection): collection of embedding bags
            used to define `SparseArch`.
        dense_in_features (int): the dimensionality of the dense input features.
        dense_arch_layer_sizes (List[int]): the layer sizes for the `DenseArch`.
        over_arch_layer_sizes (List[int]): the layer sizes for the `OverArch`.
            The output dimension of the `InteractionArch` should not be manually
            specified here.
        interaction_subbranch_1_layer_sizes (List[int]): the layer sizes for first branch of
            interaction layer. The output dimension must be a multiple of D.
        interaction_subbranch_2_layer_sizes (List[int]):the layer sizes for second branch of
            interaction layer. The output dimension must be a multiple of D.
        dense_device (Optional[torch.device]): default compute device.

    Example::

        B = 2
        D = 8

        eb1_config = EmbeddingBagConfig(
           name="t1", embedding_dim=D, num_embeddings=100, feature_names=["f1", "f3"]
        )
        eb2_config = EmbeddingBagConfig(
           name="t2",
           embedding_dim=D,
           num_embeddings=100,
           feature_names=["f2"],
        )

        ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])
        model = DLRM(
           embedding_bag_collection=ebc,
           dense_in_features=100,
           dense_arch_layer_sizes=[20],
           over_arch_layer_sizes=[5, 1],
        )

        features = torch.rand((B, 100))

        #     0       1
        # 0   [1,2] [4,5]
        # 1   [4,3] [2,9]
        # ^
        # feature
        sparse_features = KeyedJaggedTensor.from_offsets_sync(
           keys=["f1", "f3"],
           values=torch.tensor([1, 2, 4, 5, 4, 3, 2, 9]),
           offsets=torch.tensor([0, 2, 4, 6, 8]),
        )

        logits = model(
           dense_features=features,
           sparse_features=sparse_features,
        )
    """

    def __init__(
        self,
        embedding_bag_collection: EmbeddingBagCollection,
        dense_in_features: int,
        dense_arch_layer_sizes: List[int],
        over_arch_layer_sizes: List[int],
        interaction_subbranch_1_layer_sizes: Optional[List[int]] = None,
        interaction_subbranch_2_layer_sizes: Optional[List[int]] = None,
        dense_device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        assert (
            len(embedding_bag_collection.embedding_bag_configs) > 0
        ), "At least one embedding bag is required"
        for i in range(1, len(embedding_bag_collection.embedding_bag_configs)):
            conf_prev = embedding_bag_collection.embedding_bag_configs[i - 1]
            conf = embedding_bag_collection.embedding_bag_configs[i]
            assert (
                conf_prev.embedding_dim == conf.embedding_dim
            ), "All EmbeddingBagConfigs must have the same dimension"
        embedding_dim: int = embedding_bag_collection.embedding_bag_configs[
            0
        ].embedding_dim
        if dense_arch_layer_sizes[-1] != embedding_dim:
            raise ValueError(
                f"embedding_bag_collection dimension ({embedding_dim}) and final dense "
                "arch layer size ({dense_arch_layer_sizes[-1]}) must match."
            )

        self.sparse_arch: SparseArch = SparseArch(embedding_bag_collection)
        num_sparse_features: int = len(self.sparse_arch.sparse_feature_names)

        self.dense_arch = DenseArch(
            in_features=dense_in_features,
            layer_sizes=dense_arch_layer_sizes,
            device=dense_device,
        )

        projected_dim_1: int = num_sparse_features + 1
        projected_dim_2: int = num_sparse_features + 1
        interaction_subbranch_1 = None
        interaction_subbranch_2 = None
        if interaction_subbranch_1_layer_sizes:
            if interaction_subbranch_1_layer_sizes[-1] % embedding_dim != 0:
                raise ValueError(
                    "Final interaction branch1 layer size "
                    "({}) is not a multiple of embedding size ({})".format(
                        interaction_subbranch_1_layer_sizes[-1], embedding_dim
                    )
                )
            projected_dim_1 = interaction_subbranch_1_layer_sizes[-1] // embedding_dim
            interaction_subbranch_1 = MLP(
                in_size=num_sparse_features * embedding_dim
                + dense_arch_layer_sizes[-1],
                layer_sizes=interaction_subbranch_1_layer_sizes,
                device=dense_device,
            )
        if interaction_subbranch_2_layer_sizes:
            if interaction_subbranch_2_layer_sizes[-1] % embedding_dim != 0:
                raise ValueError(
                    "Final interaction branch2 layer size "
                    "({}) is not a multiple of embedding size ({})".format(
                        interaction_subbranch_2_layer_sizes[-1], embedding_dim
                    )
                )
            projected_dim_2 = interaction_subbranch_2_layer_sizes[-1] // embedding_dim
            interaction_subbranch_2 = MLP(
                in_size=num_sparse_features * embedding_dim
                + dense_arch_layer_sizes[-1],
                layer_sizes=interaction_subbranch_2_layer_sizes,
                device=dense_device,
            )

        self.inter_arch = InteractionArch(
            num_sparse_features=num_sparse_features,
            interaction_subbranch_1=interaction_subbranch_1,
            interaction_subbranch_2=interaction_subbranch_2,
        )

        over_in_features: int = 0
        if interaction_subbranch_1 or interaction_subbranch_2:
            over_in_features = embedding_dim + projected_dim_1 * projected_dim_2
        else:
            over_in_features = (
                embedding_dim + choose(num_sparse_features, 2) + num_sparse_features
            )

        self.over_arch = OverArch(
            in_features=over_in_features,
            layer_sizes=over_arch_layer_sizes,
            device=dense_device,
        )

    def forward(
        self,
        dense_features: torch.Tensor,
        sparse_features: KeyedJaggedTensor,
    ) -> torch.Tensor:
        """
        Args:
            dense_features (torch.Tensor): the dense features.
            sparse_features (KeyedJaggedTensor): the sparse features.

        Returns:
            torch.Tensor: logits.
        """
        embedded_dense = self.dense_arch(dense_features)
        embedded_sparse = self.sparse_arch(sparse_features)
        concatenated_dense = self.inter_arch(
            dense_features=embedded_dense, sparse_features=embedded_sparse
        )
        logits = self.over_arch(concatenated_dense)
        return logits


class DLRMTrain(nn.Module):
    """
    nn.Module to wrap DLRM model to use with train_pipeline.

    DLRM Recsys model from "Deep Learning Recommendation Model for Personalization and
    Recommendation Systems" (https://arxiv.org/abs/1906.00091). Processes sparse
    features by learning pooled embeddings for each feature. Learns the relationship
    between dense features and sparse features by projecting dense features into the
    same embedding space. Also, learns the pairwise relationships between sparse
    features.

    The module assumes all sparse features have the same embedding dimension
    (i.e, each EmbeddingBagConfig uses the same embedding_dim)

    Args:
        embedding_bag_collection (EmbeddingBagCollection): collection of embedding bags
            used to define SparseArch.
        dense_in_features (int): the dimensionality of the dense input features.
        dense_arch_layer_sizes (list[int]): the layer sizes for the DenseArch.
        over_arch_layer_sizes (list[int]): the layer sizes for the OverArch. NOTE: The
            output dimension of the InteractionArch should not be manually specified
            here.
        dense_device: (Optional[torch.device]).

    Example::

        ebc = EmbeddingBagCollection(config=ebc_config)
        model = DLRMTrain(
           embedding_bag_collection=ebc,
           dense_in_features=100,
           dense_arch_layer_sizes=[20],
           over_arch_layer_sizes=[5, 1],
        )
    """

    def __init__(
        self,
        embedding_bag_collection: EmbeddingBagCollection,
        dense_in_features: int,
        dense_arch_layer_sizes: List[int],
        over_arch_layer_sizes: List[int],
        interaction_subbranch_1_layer_sizes: Optional[List[int]] = None,
        interaction_subbranch_2_layer_sizes: Optional[List[int]] = None,
        dense_device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.model = DLRM(
            embedding_bag_collection=embedding_bag_collection,
            dense_in_features=dense_in_features,
            dense_arch_layer_sizes=dense_arch_layer_sizes,
            over_arch_layer_sizes=over_arch_layer_sizes,
            interaction_subbranch_1_layer_sizes=interaction_subbranch_1_layer_sizes,
            interaction_subbranch_2_layer_sizes=interaction_subbranch_2_layer_sizes,
            dense_device=dense_device,
        )
        self.loss_fn: nn.Module = nn.BCEWithLogitsLoss()

    def forward(
        self, batch: Batch
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Args:
            batch: batch used with criteo and random data from torchrec.datasets
        Returns:
            Tuple[loss, Tuple[loss, logits, labels]]
        """
        logits = self.model(batch.dense_features, batch.sparse_features)
        logits = logits.squeeze()
        loss = self.loss_fn(logits, batch.labels.float())

        return loss, (loss.detach(), logits.detach(), batch.labels.detach())
