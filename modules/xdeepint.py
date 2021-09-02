#!/usr/bin/env python3
# pyre-strict

from typing import Callable, Optional, Union

import torch
from torch import nn
from torchrec.modules.lazy_extension import LazyModuleExtensionMixin


class PINLayer(LazyModuleExtensionMixin, nn.Module):
    """
    Polynomial Interaction Network (PIN) layer for xDeepInt (https://dlp-kdd.github.io/assets/pdf/a2-yan.pdf).

    It computes X_l = X_{l-1} * f(W_{l-1} x d(X0)) + X_{l-1}, where * is element-wise multiplication,
    x is matrix multiplication, + is a residual connection, f is optional activation, and d is optional dropout.
    The kernel W_{l-1} is a square matrix.

    Constructor Args:
        activation (Union[torch.nn.Module, Callable[[torch.Tensor], torch.Tensor]]):
            activation function after computing W_{l-1} * X0
            Default: None
        dropout (float): dropout rate
            Default: None

    Call Args:
        input_x0 (torch.Tensor): X0, tensor of shape (B, F, K),
            where B is batch size, F is the number of features, K is embedding dim.
        curr_embedding (torch.Tensor): X_{l-1}, of the same shape as X0

    Returns:
        output (torch.Tensor): X_l, tensor of shape (B, F, K), the same as X0

    Example:
        >>> batch_size, num_features, embedding_dim = 4, 5, 8
        >>> x0 = torch.randn(batch_size, num_features, embedding_dim)
        >>> x1 = torch.randn(batch_size, num_features, embedding_dim)
        >>> pin_layer = PINLayer()
        >>> output = pin_layer(x0, x1)
        >>> assert list(output.shape) == [batch_size, num_features, embedding_dim]
    """

    def __init__(
        self,
        activation: Optional[
            Union[
                torch.nn.Module,
                Callable[[torch.Tensor], torch.Tensor],
            ]
        ] = None,
        dropout: Optional[float] = None,
    ) -> None:
        super().__init__()

        self._kernel = None  # pyre-ignore [4]

        self._activation = activation

        self._dropout_layer: Optional[nn.Module] = None
        if dropout is not None and dropout > 0.0:
            self._dropout_layer = nn.Dropout(dropout)

    # pyre-fixme[14]: `initialize_parameters` overrides method defined in
    #  `LazyModuleMixin` inconsistently.
    def initialize_parameters(
        self, input_x0: torch.Tensor, curr_embedding: torch.Tensor
    ) -> None:
        _, num_features, _ = input_x0.shape
        self._kernel = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(num_features, num_features))
        )

    def forward(
        self, input_x0: torch.Tensor, curr_embedding: torch.Tensor
    ) -> torch.Tensor:
        if (dropout_layer := self._dropout_layer) is not None:
            input_x0 = dropout_layer(input_x0)  # (B, F, K)
        input_x0 = torch.matmul(self._kernel, input_x0)  # (B, F, K)
        if (activation := self._activation) is not None:
            input_x0 = activation(input_x0)
        return curr_embedding * (input_x0 + 1.0)  # (B, F, K)


class XdeepInt(LazyModuleExtensionMixin, nn.Module):
    """
    XdeepInt implements xDeepInt (https://dlp-kdd.github.io/assets/pdf/a2-yan.pdf).

    XdeepInt learns higher-order vector-wise feature interactions via recursive PIN layers
    and bit-wise feature interactions through subspace-crossing mechanism. With vector-wise
    feature interaction, the bit of each feature vector only interacts with bits at the same
    position of other feature vectors, e.g. inner product. With bit-wise feature interaction,
    the bit of each feature vector interacts with all bits of other feature vectors, e.g.
    outer product. The number of PIN layers controls the degree of interaction. The number of
    subspaces (H) controls the degree of mixture of bit-wise and vector-wise interactions,
    with H=1 being purely vector-wise interaction and H=embedding_dim being purely bit-wise
    interaction.

    Constructor Args:
        num_pin_layers (int): number of PINLayers (N)
        num_subspaces (int): number of subspaces (H), must divide the embedding_dim
            Default: 1
        activation (Union[torch.nn.Module, Callable[[torch.Tensor], torch.Tensor]]):
            activation function to be used in PINLayer
            Default: None
        dropout (float): dropout rate to be used in PINLayer
            Default: None

    Call Args:
        features (torch.Tensor): X0, tensor of shape (B, F, K),
            where B is batch size, F is the number of features, K is embedding_dim

    Returns:
        output (torch.Tensor): tensor of shape (B, F, K * (N + 1)),
            where B is batch size, F is the number of features, K is embedding dim, N is the number of PINLayers

    Example:
        >>> batch_size, num_features, embedding_dim = 4, 5, 8
        >>> num_pinlayers, num_subspaces = 3, 2
        >>> x0 = torch.randn(batch_size, num_features, embedding_dim)
        >>> xdeepint = XdeepInt(num_pinlayers, num_subspaces)
        >>> output = xdeepint(x0)
        >>> assert list(output.shape) == [batch_size, num_features, embedding_dim * (num_pinlayers + 1)]
    """

    def __init__(
        self,
        num_pin_layers: int,
        num_subspaces: int = 1,
        activation: Optional[
            Union[torch.nn.Module, Callable[[torch.Tensor], torch.Tensor]]
        ] = None,
        dropout: Optional[float] = None,
    ) -> None:
        super().__init__()

        self._num_pin_layers = num_pin_layers
        self._num_subspaces = num_subspaces
        self._num_features: int = 0
        self._embedding_dim: int = 0

        self._pin_layers = nn.ModuleList(  # type: ignore
            [
                PINLayer(
                    activation=activation,
                    dropout=dropout,
                )
                for _ in range(self._num_pin_layers)
            ]
        )

    # pyre-fixme[14]: `initialize_parameters` overrides method defined in
    #  `LazyModuleMixin` inconsistently.
    def initialize_parameters(self, features: torch.Tensor) -> None:
        # features tensor: (batch, num_features, embedding_dim)
        _, self._num_features, self._embedding_dim = features.shape
        assert (
            self._embedding_dim % self._num_subspaces == 0
        ), f"the embedding dim {self._embedding_dim} should be divisible by the number of subspaces {self._num_subspaces}"

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # subspacing
        subspace_dim = int(self._embedding_dim / self._num_subspaces)
        features = features.reshape(
            -1,
            self._num_features * self._num_subspaces,
            subspace_dim,
        )  # (B, F * H, K / H)

        # iteratively compute interactions
        curr_embedding = input_x0 = features
        output = [input_x0]
        for pin_layer in self._pin_layers:
            curr_embedding = pin_layer(input_x0, curr_embedding)  # (B, F * H, K / H)
            output.append(curr_embedding)

        # reshape back to orginal axis arrangement
        # the final embedding dim can become larger due to concatenating output from each PIN layer
        output = torch.cat(
            [
                one_order.reshape(
                    -1, self._num_features, self._embedding_dim
                )  # (B, F, K)
                for one_order in output
            ],
            dim=-1,
        )  # (B, F, K * (N + 1))

        return output
