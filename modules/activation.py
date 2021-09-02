#!/usr/bin/env python3

from typing import Callable, Union

import torch


class Swish(torch.nn.Module):
    """
    Applies the generalized Swish function `y = x * sigmoid(beta_fn(x))`.

    More details can be found in the paper:
    Ramachandran, Prajit; Zoph, Barret; Le, Quoc V. (2017-10-16).
    "Swish: A Self-Gated Activation Function"
    https://arxiv.org/pdf/1710.05941v1.pdf

    Constructor Args:
        beta (Union[float, torch.nn.Module, Callable[[torch.Tensor], torch.Tensor]]): the beta to apply on `x`.
            If `beta` is a float, the actual operation is `y = x * sigmoid(beta * x)`.
            If `beta` is a torch.nn.Module, the actual operation is `y = x * sigmoid(beta(x))`.
            If `beta` is a function, the actual operation is `y = x * sigmoid(beta(x))`.

    Call Args:
        input (torch.Tensor): tensor of any shape.

    Returns:
        output (torch.Tensor): tensor of the same shape as `input`.

    Example:
        >>> m = Swish(1.0)
        >>> output = m(torch.randn(3, 4))
        >>>
        >>> m = Swish(torch.nn.LayerNorm(4))
        >>> output = m(torch.randn(3, 4))
    """

    _beta_fn: Callable[[torch.Tensor], torch.Tensor]

    def __init__(
        self,
        beta: Union[
            float,
            torch.nn.Module,
            Callable[[torch.Tensor], torch.Tensor],
        ],
    ) -> None:
        super().__init__()
        if isinstance(beta, float):
            self._beta_fn = lambda input: torch.mul(input, beta)
        else:
            self._beta_fn = beta

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * torch.sigmoid(self._beta_fn(input))
