#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Sphinx Documentation Text (for user-facing classes only)

"""
.. fb:display_title::
    CrossNet API
=====

These modules do X:Y:Z
    * Class Cross Net does X
    * Class Low Cross Net does Y
    * Class .....

"""

from typing import Optional, Callable, Union

import torch


class CrossNet(torch.nn.Module):
    r"""
    Cross Network: https://arxiv.org/abs/1708.05123

    Cross net is a stack of "crossing" operations on a tensor of shape :math:`(*, N)`
    to the same shape, effectively creating :math:`N` learnable polynomical functions
    over the input tensor.

    In this module, the crossing operations are defined based on a full rank matrix (NxN),
    such that crossing effect can cover all bits on each layer. On each layer l, the tensor
    is transformed into:

    .. math ::    x_{l+1} = x_0 * (W_l \dot x_l + b_l) + x_l

    where :math:`W_l` is a square matrix :math:`(NxN)`, :math:`*` means element-wise multiplication, :math:`\dot` means
    matrix multiplication.

    Constructor Args:
        * in_features (int): the dimension of the input.
        * num_layers (int): the number of layers in the module.

    Call Args:
        * input (torch.Tensor): tensor with shape [batch_size, in_features]

    Returns:
        * output (torch.Tensor): tensor with shape [batch_size, in_features]

    Example:
        >>> batch_size = 3
        >>> num_layers = 2
        >>> in_features = 10
        >>> input = torch.randn(batch_size, in_features)
        >>> dcn = CrossNet(num_layers=num_layers)
        >>> output = dcn(input)
    """

    def __init__(
        self,
        in_features: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self._num_layers = num_layers
        self.kernels: torch.nn.Module = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.nn.init.xavier_normal_(torch.empty(in_features, in_features))
                )
                for i in range(self._num_layers)
            ]
        )
        self.bias: torch.nn.Module = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.nn.init.zeros_(torch.empty(in_features, 1)))
                for i in range(self._num_layers)
            ]
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x_0 = input.unsqueeze(2)  # (B, N, 1)
        x_l = x_0

        for layer in range(self._num_layers):
            # pyre-ignore[29]: `Union[torch.Tensor, torch.nn.Module]` is not a function.
            xl_w = torch.matmul(self.kernels[layer], x_l)  # (B, N, 1)
            # pyre-ignore[29]: `Union[torch.Tensor, torch.nn.Module]` is not a function.
            x_l = x_0 * (xl_w + self.bias[layer]) + x_l  # (B, N, 1)

        return torch.squeeze(x_l, dim=2)


class LowRankCrossNet(torch.nn.Module):
    r"""
     Low Rank Cross net is a high-efficient cross net. Instead of using full rank cross matrix (NxN)
     at each layer, it will use two kernels :math:`W (N * r)` and :math:`V (r * N)`, where `r << N`, to simplify the matrix
     multiplication.

     On each layer l, the tensor is transformed into:

     .. math::    x_{l+1} = x_0 * (W_l \dot (V_l \dot x_l) + b_l) + x_l

    where :math:`W_l` is either a vector, :math:`*` means element-wise multiplication, and :math:`\dot` means matrix multiplication.

     Note that, rank `r` should be chosen smartly. Usually, we should expect `r < N/2` to have computation saving; we should
     expect :math:`r ~= N/4` to perserve the accuracy of full rank cross net.

     Constructor Args:
         * in_features (int): the dimension of the input.
         * num_layers (int): the number of layers in the module.
         * low_rank (int): the rank setup of the cross matrix (default = 0). Value must be always >= 0

     Call Args:
         * input (torch.Tensor): tensor with shape [batch_size, in_features]

     Returns:
         * output (torch.Tensor): tensor with shape [batch_size, in_features]

     Example:
         >>> batch_size = 3
         >>> num_layers = 2
         >>> in_features = 10
         >>> input = torch.randn(batch_size, in_features)
         >>> dcn = LowRankCrossNet(num_layers=num_layers, low_rank=3)
         >>> output = dcn(input)
    """

    def __init__(
        self,
        in_features: int,
        num_layers: int,
        low_rank: int = 1,
    ) -> None:
        super().__init__()
        assert low_rank >= 1, "Low rank must be larger or equal to 1"

        self._num_layers = num_layers
        self._low_rank = low_rank
        self.W_kernels: torch.nn.Module = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.nn.init.xavier_normal_(
                        torch.empty(in_features, self._low_rank)
                    )
                )
                for i in range(self._num_layers)
            ]
        )
        self.V_kernels: torch.nn.Module = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.nn.init.xavier_normal_(
                        torch.empty(self._low_rank, in_features)
                    )
                )
                for i in range(self._num_layers)
            ]
        )
        self.bias: torch.nn.Module = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.nn.init.zeros_(torch.empty(in_features, 1)))
                for i in range(self._num_layers)
            ]
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x_0 = input.unsqueeze(2)  # (B, N, 1)
        x_l = x_0

        for layer in range(self._num_layers):
            xl_w = torch.matmul(
                # pyre-ignore[29]: `Union[torch.Tensor, torch.nn.Module]` is not a
                #  function.
                self.W_kernels[layer],
                # pyre-ignore[29]: `Union[torch.Tensor, torch.nn.Module]` is not a
                #  function.
                torch.matmul(self.V_kernels[layer], x_l),
            )  # (B, N, 1)
            # pyre-ignore[29]: `Union[torch.Tensor, torch.nn.Module]` is not a function.
            x_l = x_0 * (xl_w + self.bias[layer]) + x_l  # (B, N, 1)

        return torch.squeeze(x_l, dim=2)  # (B, N)


class VectorCrossNet(torch.nn.Module):
    r"""
    Vector Cross Network can be refered as DCN-V1 (https://arxiv.org/pdf/1708.05123.pdf).

    It is also a specialized low rank cross net, where rank=1. In this version, on each layer, instead
    of keeping two kernels W and V, we only keep one vector kernel W (Nx1). So, we will use dot
    operation to compute the "crossing" effect of features; thus, we can save two matrix multiplications
    to further reduce computational cost and cut the number of learnable parameter number.

    On each layer l, the tensor is transformed into

    .. math::    x_{l+1} = x_0 * (W_l . x_l + b_l) + x_l

    where :math:`W_l` is either a vector, :math:`*` means element-wise multiplication; :math:`.` means dot operations.

    Constructor Args:
        * in_features (int): the dimension of the input.
        * num_layers (int): the number of layers in the module.

    Call Args:
        * input (torch.Tensor): tensor with shape [batch_size, in_features]

    Returns:
        * output (torch.Tensor): tensor with shape [batch_size, in_features]

    Example:
        >>> batch_size = 3
        >>> num_layers = 2
        >>> in_features = 10
        >>> input = torch.randn(batch_size, in_features)
        >>> dcn = VectorCrossNet(num_layers=num_layers)
        >>> output = dcn(input)
    """

    def __init__(
        self,
        in_features: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self._num_layers = num_layers
        self.kernels: torch.nn.Module = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.nn.init.xavier_normal_(torch.empty(in_features, 1))
                )
                for i in range(self._num_layers)
            ]
        )
        self.bias: torch.nn.Module = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.nn.init.zeros_(torch.empty(in_features, 1)))
                for i in range(self._num_layers)
            ]
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x_0 = input.unsqueeze(2)  # (B, N, 1)
        x_l = x_0

        for layer in range(self._num_layers):
            xl_w = torch.tensordot(
                x_l,
                # pyre-ignore[29]: `Union[torch.Tensor, torch.nn.Module]` is not a
                #  function.
                self.kernels[layer],
                dims=([1], [0]),
            )  # (B, 1, 1)
            # pyre-ignore[29]: `Union[torch.Tensor, torch.nn.Module]` is not a function.
            x_l = torch.matmul(x_0, xl_w) + self.bias[layer] + x_l  # (B, N, 1)

        return torch.squeeze(x_l, dim=2)  # (B, N)


class LowRankMixtureCrossNet(torch.nn.Module):
    r"""
    LowRankMixtureCrossNet is a DCN V2 implementation from the paper: https://arxiv.org/pdf/2008.13535.pdf

    LowRankMixtureCrossNet defines the learnable crossing parameter per layer as low-rank matrix :math:`(N*r)` together
    with mixture of expert. Compared to LowRankCrossNet, instead of relying on one single expert to learn
    feature crosses, this module leverages such :math:`K` experts; each learning feature interactions in a
    different subspaces, and adaptively combine the learned crosses using a gating mechanism that depends
    on input :math:`x`..

    On each layer l, the tensor is transformed into:

    .. math::    x_{l+1} = MoE(expert_i foreach i in K experts) + x_l

    and each :math:`expert_i` is defined as:

    .. math::    expert_i = x_0 * (U_l_i \dot g(C_l_i \dot g(V_l_i \dot x_l)) + b_l)

    where :math:`U_l_i (N, r)`, :math:`C_l_i (r, r)` and :math:`V_l_i (r, N)` are low-rank matrix, :math:`*` means element-wise multiplication,
    :math:`x` means matrix multiplication, and :math:`g()` is the non-linear activation function.

    One optimization is when num_expert is 1, the gate evaluation and MOE will be skipped for computation saving.

    Constructor Args:
        * in_features (int): the dimension of the input.
        * num_layers (int): the number of layers in the module.
        * low_rank (int): the rank setup of the cross matrix (default = 0). Value must be always >= 0
        * activation (Union[torch.nn.Module, Callable[[torch.Tensor], torch.Tensor]]): the non-linear activation function, used in defining experts. Default is relu.

    Call Args:
        * input (torch.Tensor): tensor with shape [batch_size, in_features]

    Returns:
        * output (torch.Tensor): tensor with shape [batch_size, in_features]

    Example:
        >>> batch_size = 3
        >>> num_layers = 2
        >>> in_features = 10
        >>> input = torch.randn(batch_size, in_features)
        >>> dcn = LowRankCrossNet(num_layers=num_layers, num_experts=5, low_rank=3)
        >>> output = dcn(input)
    """

    def __init__(
        self,
        in_features: int,
        num_layers: int,
        num_experts: int = 1,
        low_rank: int = 1,
        activation: Union[
            torch.nn.Module,
            Callable[[torch.Tensor], torch.Tensor],
        ] = torch.relu,
    ) -> None:
        super().__init__()
        assert num_experts >= 1, "num_experts must be larger or equal to 1"
        assert low_rank >= 1, "Low rank must be larger or equal to 1"

        self._num_layers = num_layers
        self._num_experts = num_experts
        self._low_rank = low_rank
        self._in_features = in_features
        self.U_kernels: torch.nn.Module = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.nn.init.xavier_normal_(
                        torch.empty(
                            self._num_experts, self._in_features, self._low_rank
                        )
                    )
                )
                for i in range(self._num_layers)
            ]
        )
        self.V_kernels: torch.nn.Module = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.nn.init.xavier_normal_(
                        torch.empty(
                            self._num_experts, self._low_rank, self._in_features
                        )
                    )
                )
                for i in range(self._num_layers)
            ]
        )
        self.bias: torch.nn.Module = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.nn.init.zeros_(torch.empty(self._in_features, 1))
                )
                for i in range(self._num_layers)
            ]
        )
        self.gates: Optional[torch.nn.Module] = (
            torch.nn.ModuleList(
                [
                    torch.nn.Linear(self._in_features, 1, bias=False)
                    for i in range(self._num_experts)
                ]
            )
            if self._num_experts > 1
            else None
        )

        self._activation = activation
        self.C_kernels: torch.nn.Module = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.nn.init.xavier_normal_(
                        torch.empty(self._num_experts, self._low_rank, self._low_rank)
                    )
                )
                for i in range(self._num_layers)
            ]
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x_0 = input.unsqueeze(2)  # (B, N, 1)
        x_l = x_0

        for layer in range(self._num_layers):
            # set up gating:
            if self._num_experts > 1:
                gating = []
                for i in range(self._num_experts):
                    # pyre-ignore[16]: `Optional` has no attribute `__getitem__`.
                    gating.append(self.gates[i](x_l.squeeze(2)))
                gating = torch.stack(gating, 1)  # (B, K, 1)

            # set up experts
            experts = []
            for i in range(self._num_experts):
                expert = torch.matmul(
                    # pyre-ignore[29]
                    self.V_kernels[layer][i],
                    x_l,
                )  # (B, r, 1)
                expert = torch.matmul(
                    # pyre-ignore[29]
                    self.C_kernels[layer][i],
                    self._activation(expert),
                )  # (B, r, 1)
                expert = torch.matmul(
                    # pyre-ignore[29]
                    self.U_kernels[layer][i],
                    self._activation(expert),
                )  # (B, N, 1)
                # pyre-ignore[29]
                expert = x_0 * (expert + self.bias[layer])  # (B, N, 1)
                experts.append(expert.squeeze(2))  # (B, N)
            experts = torch.stack(experts, 2)  # (B, N, K)

            if self._num_experts > 1:
                # MOE update
                moe = torch.matmul(
                    experts,
                    # pyre-ignore[61]: `gating` may not be initialized here.
                    torch.nn.functional.softmax(gating, 1),
                )  # (B, N, 1)
                x_l = moe + x_l  # (B, N, 1)
            else:
                x_l = experts + x_l  # (B, N, 1)

        return torch.squeeze(x_l, dim=2)  # (B, N)
