#!/usr/bin/env python3

from typing import Callable, List, Union

import torch
from torch import nn
from torchrec.modules.linear import MCLinear
from torchrec.modules.utils import extract_module_or_tensor_callable


class Perceptron(torch.nn.Module):
    """
    Applies a linear transformation and activation.

    Constructor Args:
        out_size (int): number of elements in each output sample.
        bias (bool): if set to ``False``, the layer will not learn an additive bias.
            Default: ``True``.
        activation (Union[torch.nn.Module, Callable[[torch.Tensor], torch.Tensor]]):
            the activation function to apply to the output of linear transformation.
            Default: torch.relu.

    Call Args:
        input (torch.Tensor): tensor of shape (B, I) where I is number of elements
            in each input sample.

    Returns:
        output (torch.Tensor): tensor of shape (B, O) where O is number of elements
            per channel in each output sample (i.e. `out_size`).

    Example:
        >>> batch_size = 3
        >>> input = torch.randn(batch_size, 40)
        >>>
        >>> out_size = 16
        >>> perceptron = Perceptron(out_size, bias=True)
        >>>
        >>> output = perceptron(input)
        >>> assert list(output) == [batch_size, out_size]
    """

    def __init__(
        self,
        out_size: int,
        bias: bool = True,
        activation: Union[
            torch.nn.Module,
            Callable[[torch.Tensor], torch.Tensor],
        ] = torch.relu,
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torchrec.modules.{self.__class__.__name__}")
        self._out_size = out_size
        self._linear: nn.Linear = nn.LazyLinear(self._out_size, bias=bias)
        self._activation_fn: Callable[[torch.Tensor], torch.Tensor] = activation

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._activation_fn(self._linear(input))


class MCPerceptron(Perceptron):
    """
    Applies a multi-channel linear transformation and activation.

    Compared to Perceptron, this MCPerceptron module adds an extra "channels" dimension.
    One use case is multi-task learning where the perceptron module for each task accepts
    the same input but learns a different weight for its corresponding task. (here "task" = "channel")

    Constructor Args:
        out_size (int): number of elements per channel in each output sample.
        num_channels (int): number of channels in each output sample.
        bias (bool): if set to False, the layer will not learn an additive bias.
            Default: True.
        activation (Union[torch.nn.Module, Callable[[torch.Tensor], torch.Tensor]]):
            the activation function to apply to the output of linear transformation.
            It needs to support multi-channel (i.e. support B x C x O as input).
            Default: torch.relu.

    Call Args:
        input (torch.Tensor): tensor of shape (B, I) or (C, B, I),
            where C is number of channels, B is batch size and I is number of elements per channel in each input sample.
            When input is of shape (B, I), we assume all channels share the same tensor input.

    Returns:
        output (torch.Tensor): tensor of shape (C, B, O), where C is number of channels, B is batch size
            and O is number of elements per channel in each output sample (i.e. `out_size`).

    Example:
        >>> batch_size = 3
        >>> in_size = 40
        >>> num_channels = 4
        >>> input = torch.randn(batch_size, in_size)
        >>> out_size = 16
        >>>
        >>> multi_channel_perceptron = MCPerceptron(out_size, num_channels, bias=True)
        >>>
        >>> output = multi_channel_perceptron(input)
        >>> assert list(output.shape) == [batch_size, num_channels, out_size]
    """

    def __init__(
        self,
        out_size: int,
        num_channels: int,
        bias: bool = True,
        activation: Union[
            torch.nn.Module, Callable[[torch.Tensor], torch.Tensor]
        ] = torch.relu,
    ) -> None:
        super().__init__(out_size, bias=bias, activation=activation)
        self._linear: MCLinear = MCLinear(self._out_size, num_channels, bias=bias)


class MLP(torch.nn.Module):
    """
    Applies a stack of Perceptron modules sequentially (i.e. Multi-Layer Perceptron).

    Constructor Args:
        layer_sizes (List[int]): `out_size` of each Perceptron module.
        bias (bool): if set to False, the layer will not learn an additive bias.
            Default: True.
        activation (Union[Callable[[], torch.nn.Module], torch.nn.Module, Callable[[torch.Tensor], torch.Tensor]]):
            the activation function to apply to the output of linear transformation of each Perceptron module.
            If `activation` is a `Callable[[], torch.nn.Module]`, `activation()` will be called once per Perceptron module
            to generate the activation module for that Perceptron module, and the parameters won't be shared
            between those activation modules. One use case is when all the activation modules share the same
            constructor arguments, but don't share the actual module parameters.
            Default: torch.relu.

    Call Args:
        input (torch.Tensor): tensor of shape (B, I) where I is number of elements
            in each input sample.

    Returns:
        output (torch.Tensor): tensor of shape (B, O) where O is `out_size` of
            the last Perceptron module.

    Example:
        >>> batch_size = 3
        >>> in_size = 40
        >>> input = torch.randn(batch_size, in_size)
        >>>
        >>> layer_sizes = [16, 8, 4]
        >>> mlp_module = MLP(layer_sizes, bias=True)
        >>> output = mlp_module(input)
        >>> assert list(output.shape) == [batch_size, layer_sizes[-1]]
        >>>
        >>> mlp_module = MLP(layer_sizes, activation=lambda: Swish(LayerNorm(1, affine_axis=1)))
        >>> output = mlp_module(input)
        >>> assert list(output.shape) == [batch_size, layer_sizes[-1]]
    """

    def __init__(
        self,
        layer_sizes: List[int],
        bias: bool = True,
        activation: Union[
            Callable[[], torch.nn.Module],
            torch.nn.Module,
            Callable[[torch.Tensor], torch.Tensor],
        ] = torch.relu,
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torchrec.modules.{self.__class__.__name__}")
        # TODO(T89836159): Use Apex fused MLP when CUDA is available:
        # https://github.com/NVIDIA/apex/blob/master/apex/mlp/mlp.py
        self._perceptrons: torch.nn.Module = torch.nn.Sequential(
            *[
                Perceptron(
                    layer_size,
                    bias=bias,
                    activation=extract_module_or_tensor_callable(activation),
                )
                for layer_size in layer_sizes
            ]
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._perceptrons(input)


class MCMLP(MLP):
    """
    Applies a stack of MCPerceptron modules sequentially.

    Compared to MLP, this MCMLP module adds an extra "channels" dimension.
    One use case is multi-task learning where the MLP module for each task accepts
    the same input but learns a different weight for its corresponding task. (here "task" = "channel")

    Constructor Args:
        layer_sizes (List[int]): `out_size` of each MCPerceptron module.
        num_channels (int): number of channels in each output sample.
        bias (bool): if set to False, the layer will not learn an additive bias.
            Default: True.
        activation (Union[Callable[[], torch.nn.Module], torch.nn.Module, Callable[[torch.Tensor], torch.Tensor]]):
            the activation function to apply to the output of linear transformation of each MCPerceptron module.
            It needs to support multi-channel (i.e. support B x C x O as input).
            If `activation` is a `Callable[[], torch.nn.Module]`, `activation()` will be called once per MCPerceptron module
            to generate the activation module for that MCPerceptron module, and the parameters won't be shared
            between those activation modules. One use case is when all the activation modules share the same
            constructor arguments, but don't share the actual module parameters.
            Default: torch.relu.

    Call Args:
        input (torch.Tensor): tensor of shape (B, I) or (C, B, I),
            where C is number of channels, B is batch size and I is number of elements per channel in each input sample.
            When input is of shape (B, I), we assume all channels share the same tensor input.

    Returns:
        output (torch.Tensor): tensor of shape (C, B, O), where C is number of channels, B is batch size
            and O is number of elements per channel in each output sample (i.e. `out_size` of
            the last MCPerceptron module).

    Example:
        >>> batch_size = 3
        >>> num_channels = 4
        >>> in_size = 40
        >>> layer_sizes = [16, 8, 4]
        >>>
        >>> input = torch.randn(batch_size, in_sizes)
        >>> multi_channel_mlp = MCMLP(layer_sizes, num_channels, bias=True)
        >>> output = multi_channel_mlp(input)
        >>> assert list(output.shape) == [num_channels, batch_size, layer_sizes[-1]]
        >>>
        >>> multi_channel_mlp = MCMLP(layer_sizes, num_channels, activation=lambda: Swish(LayerNorm(2, affine_axis=1)))
        >>> output = multi_channel_mlp(input)
        >>> assert list(output.shape) == [num_channels, batch_size, layer_sizes[-1]]
    """

    def __init__(
        self,
        layer_sizes: List[int],
        num_channels: int,
        bias: bool = True,
        activation: Union[
            Callable[[], torch.nn.Module],
            torch.nn.Module,
            Callable[[torch.Tensor], torch.Tensor],
        ] = torch.relu,
    ) -> None:
        super().__init__(layer_sizes, bias=bias, activation=activation)
        self._perceptrons: torch.nn.Module = torch.nn.Sequential(
            *[
                MCPerceptron(
                    layer_size,
                    num_channels,
                    bias=bias,
                    activation=extract_module_or_tensor_callable(activation),
                )
                for layer_size in layer_sizes
            ]
        )
