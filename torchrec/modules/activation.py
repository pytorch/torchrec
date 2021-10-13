#!/usr/bin/env python3

from typing import Optional, Union, List

import torch
from torch import nn


class SwishLayerNorm(nn.Module):
    """
    Applies the Swish function with layer normalization:
        'Y = X * Sigmoid(LayerNorm(X)).'

    Call Args:
        input: an input tensor

    Returns:
        output: an output tensor

    Constructor Args:
        input_dims: dimensions to normalize over. E.g., If an input tensor has shape
        [batch_size, d1, d2, d3], set input_dim=[d2, d3] will do the layer normalization
        on last two dimensions.
        device: (Optional[torch.device]).

    Example:
        >>> sln = SwishLayerNorm(100)

    """

    def __init__(
        self,
        input_dims: Union[int, List[int], torch.Size],
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.norm: torch.nn.modules.Sequential = nn.Sequential(
            nn.LayerNorm(input_dims, device=device),
            nn.Sigmoid(),
        )

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        return input * self.norm(input)
