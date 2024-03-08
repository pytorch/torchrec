#!/usr/bin/env python3

# pyre-strict

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3
# @nolint

import torch


class Simple(torch.nn.Module):
    def __init__(self, N: int, M: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(N, M))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.weight + input
        return output

    def set_weight(self, weight: torch.Tensor) -> None:
        self.weight[:] = torch.nn.Parameter(weight)


class Nested(torch.nn.Module):
    def __init__(self, N: int, M: int) -> None:
        super().__init__()
        self.simple = Simple(N, M)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.simple(input)
