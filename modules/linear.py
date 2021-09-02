#!/usr/bin/env python3

from typing import Any, Optional, Type

import torch
from torch.nn.parameter import UninitializedParameter
from torchrec.modules.lazy_extension import LazyModuleExtensionMixin


class MCLinear(LazyModuleExtensionMixin, torch.nn.modules.linear.LazyLinear):
    """
    Applies a multi-channel linear transformation to the incoming data.

    Compared to torch.nn.Linear, this MCLinear module adds an extra "channels" dimension.
    One use case is multi-task learning where the linear module for each task accepts
    the same input but learns a different weight for its corresponding task. (here "task" = "channel")

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    Constructor Args:
        out_features (int): number of elements per channel in each output sample.
        num_channels (int): number of channels in each output sample.
        bias (bool): if set to False, the layer will not learn an additive bias.
            Default: True.

    Call Args:
        input (torch.Tensor): tensor of shape (B, I) or (C, B, I),
            where C is number of channels, B is batch size and I is number of elements per channel in each input sample.
            When input is of shape (B, I), we assume all channels share the same tensor input.

    Returns:
        output (torch.Tensor): tensor of shape (C, B, O), where C is number of channels, B is batch size
            and O is number of elements per channel in each output sample (i.e. `out_features`).

    Example:
        >>> in_features = 5
        >>> out_features = 10
        >>> num_channels = 3
        >>> batch_size = 3
        >>>
        >>> input_tensor = torch.randn(num_channels, batch_size, in_features)
        >>>
        >>> mc_linear_module = MCLinear(out_features, num_channels, bias=True)
        >>> output_tensor = mc_linear_module(input_tensor)
        >>>
        >>> assert list(output_tensor.shape) == [num_channels, batch_size, out_features]
    """

    # pyre-ignore[4]: Attribute `cls_to_become` of class `MCLinear` must have a type
    # that does not contain `Any`.
    # pyre-ignore[15]: `cls_to_become` overrides attribute defined in
    # `torch.nn.modules.linear.LazyLinear` inconsistently. Type `Optional[Type[typing.Any]]`
    # is not a subtype of the overridden attribute `Type[torch.nn.modules.linear.Linear]`.
    cls_to_become: Optional[Type[Any]] = None

    def __init__(
        self,
        out_features: int,
        num_channels: int,
        bias: bool = True,
    ) -> None:
        # bias is hardcoded to False to avoid creating tensor
        # that will soon be overwritten.
        super().__init__(0, bias=False)
        self.weight = UninitializedParameter()
        self.out_features = out_features
        self.num_channels = num_channels
        self._has_bias = bias
        if self._has_bias:
            self.bias: UninitializedParameter = UninitializedParameter()
        self._input_dim: int = -1

    def initialize_parameters(self, input: torch.Tensor) -> None:
        if self.has_uninitialized_params():
            with torch.no_grad():
                assert input.dim() in [2, 3], (
                    "Expected input to have 2 dimensions (batch_size x in_features) "
                    "or 3 dimensions (num_channels x batch_size x in_features), "
                    "but it has only {} dimensions".format(input.dim())
                )
                self._input_dim = input.dim()
                self.in_features = input.shape[-1]
                if self._input_dim == 3:
                    assert (
                        input.shape[0] == self.num_channels
                    ), "Expected input to have {} channels, but got {}".format(
                        self.num_channels, input.shape[0]
                    )
                self.weight.materialize(
                    (self.num_channels, self.in_features, self.out_features)
                )  # C x I x O
                if self._has_bias:
                    self.bias.materialize(
                        (self.num_channels, self.out_features)
                    )  # C x O
                self.reset_parameters()

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, num_channels={}, bias={}".format(
            self.in_features,
            self.out_features,
            self.num_channels,
            self._has_bias,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size = input.shape[1]
        # input = self._input_transform(input)
        if self._input_dim == 2:
            output = torch.matmul(input, self.weight)  # C x B x O
            if self._has_bias:
                # fmt: off
                bias = (
                    self.bias.unsqueeze(1)  # C x O -> C x 1 x O
                )
                # fmt: on
                output = output + bias  # C x B x O
        else:
            if self._has_bias:
                # fmt: off
                bias = (
                    self.bias.unsqueeze(1)  # C x O -> C x 1 x O
                    .expand(self.num_channels, batch_size, self.out_features)  # C x 1 x O -> C x B x O
                )
                # fmt: on
                output = torch.baddbmm(bias, input, self.weight)  # C x B x O
            else:
                output = torch.bmm(input, self.weight)  # C x B x O
        return output
