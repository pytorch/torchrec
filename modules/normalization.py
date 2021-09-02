#!/usr/bin/env python3

from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn.parameter import UninitializedParameter
from torchrec.modules.lazy_extension import LazyModuleExtensionMixin


class LayerNorm(LazyModuleExtensionMixin, torch.nn.modules.LayerNorm):
    """
    Applies Layer Normalization over a mini-batch of inputs as described in
    the paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`__

    For more details, please see docs of `torch.nn.LayerNorm`:
    https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html.

    This LayerNorm module supports "multi-channel" (i.e. "multi-task") use case, an improvement over
    `torch.nn.LayerNorm`. Concretely, if we have input of shape (num_tasks, batch_size, num_features)
    and we want to apply layer normalization for each task independently, with `torch.nn.LayerNorm`
    we have to do a for-loop over the `num_tasks` axis to apply a different `torch.nn.LayerNorm` module per task.
    Whereas with this LayerNorm module, we can set `LayerNorm(norm_axis=2, num_channels=num_tasks)` to have
    one "multi-task" layer normalization module that covers the same need. This way of "fusing" multiple
    layer normalization modules into one also improves performance.

    Constructor Args:
        norm_axis (int): the starting axis for normalization. e.g. if the
            input tensor's shape is (a_0, a_1, ..., a_{k-1}, a_k, ..., a_{n-1}),
            and norm_axis=k, the input tensor will be normalized over dimensions
            a_k through a_{n-1}.
        num_channels (Optional[int]): if not None, it specifies the number of channels this module supports.
            e.g. if the input tensor's shape is (C, a_0, a_1, ..., a_{k-1}, a_k, ..., a_{n-1}),
            norm_axis=k and num_channels=C, the input tensor will be applied a elementwise affine transform
            with weight and bias of shape (C, 1, ... 1, a_k, ..., a_{n-1}).
            If None, multi-channel support is not activated and input tensor is assumed to be single-channel
            with shape (a_0, a_1, ..., a_{k-1}, a_k, ..., a_{n-1}).
            Default: None.
        elementwise_affine (bool): a boolean value that when set to `True`, this module
            has learnable per-element affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: `True`.
        eps (float): a value added to the denominator for numerical stability. Default: 1e-5.

    Call Args:
        input (torch.Tensor): tensor of shape (C, B, I_1, ..., I_n) when `num_channels=C`,
            or (B, I_1, ..., I_n) when `num_channels=None`.

    Returns:
        output (torch.Tensor): tensor of the same shape as `input`.

    Example:
        >>> num_tasks = 5
        >>> batch_size = 20
        >>> input = torch.randn(num_tasks, batch_size, 10, 10)
        >>> # Normalize over axis 2-3, apply multi-task elementwise affine transform.
        >>> m = nn.LayerNorm(2, num_channels=num_tasks)
        >>> output = m(input)
    """

    num_channels: Optional[int]

    def __init__(
        self,
        norm_axis: int,
        num_channels: Optional[int] = None,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
    ) -> None:
        # `elementwise_affine` is hardcoded to False to avoid creating tensor
        # that will soon be overwritten.
        super().__init__([], eps=eps, elementwise_affine=False)
        self.norm_axis = norm_axis
        self.num_channels = num_channels
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight: UninitializedParameter = UninitializedParameter()
            self.bias: UninitializedParameter = UninitializedParameter()

    # pyre-ignore[14]: `torchrec.modules.normalization.LayerNorm.initialize_parameters`
    # overrides method defined in `LazyModuleExtensionMixin` inconsistently.
    def initialize_parameters(self, input: torch.Tensor) -> None:
        assert len(input.shape) > self.norm_axis, (
            "Expected input tensor to have at least {} dimensions, "
            "but got {}".format(self.norm_axis + 1, len(input.shape))
        )
        self.normalized_shape = input.shape[
            self.norm_axis :
        ]  # norm_axis_1 x ... x norm_axis_n

        if self.has_uninitialized_params():
            if self.elementwise_affine:
                num_channels = self.num_channels
                if num_channels is not None:
                    # affine_shape is C x 1 x ... x 1 x norm_axis_1 x ... x norm_axis_n
                    affine_shape = (
                        [num_channels]
                        + [1] * (self.norm_axis - 1)
                        + list(self.normalized_shape)
                    )
                else:
                    # affine_shape is norm_axis_1 x ... x norm_axis_n
                    affine_shape = list(self.normalized_shape)

                self.weight.materialize(affine_shape)
                self.bias.materialize(affine_shape)

        self.reset_parameters()

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        if self.elementwise_affine:
            if self.num_channels is None:
                return F.layer_norm(
                    input,
                    self.normalized_shape,
                    weight=self.weight,
                    bias=self.bias,
                    eps=self.eps,
                )
            else:
                normalized = F.layer_norm(
                    input,
                    self.normalized_shape,
                    eps=self.eps,
                )
                return torch.addcmul(self.bias, normalized, self.weight)
        else:
            return F.layer_norm(
                input,
                self.normalized_shape,
                eps=self.eps,
            )

    def extra_repr(self) -> str:
        return "normalized_shape={normalized_shape}, num_channels={num_channels}, eps={eps}".format(
            **self.__dict__
        )
