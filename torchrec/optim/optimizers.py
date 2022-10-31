#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

from typing import Iterable

import torch

from torch.optim.optimizer import Optimizer


class SGD(Optimizer):
    r"""
    Placeholder for SGD. This optimizer will not functionally run.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        # pyre-ignore
        **kwargs,
    ) -> None:
        self._params = params
        # pyre-ignore
        self._kwargs = kwargs

    @torch.no_grad()
    # pyre-ignore
    def step(self, closure=None) -> torch.Tensor:
        raise NotImplementedError


class LarsSGD(Optimizer):
    r"""
    Placeholder for LARS_SGD. This optimizer will not functionally run.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        # pyre-ignore
        **kwargs,
    ) -> None:
        self._params = params
        # pyre-ignore
        self._kwargs = kwargs

    @torch.no_grad()
    # pyre-ignore
    def step(self, closure=None) -> torch.Tensor:
        raise NotImplementedError


class LAMB(Optimizer):
    r"""
    Placeholder for LAMB. This optimizer will not functionally run.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        # pyre-ignore
        **kwargs,
    ) -> None:
        self._params = params
        # pyre-ignore
        self._kwargs = kwargs

    @torch.no_grad()
    # pyre-ignore
    def step(self, closure=None) -> torch.Tensor:
        raise NotImplementedError


class PartialRowWiseLAMB(Optimizer):
    r"""
    Placeholder for PartialRowWiseLAMB. This optimizer will not functionally run.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        # pyre-ignore
        **kwargs,
    ) -> None:
        self._params = params
        # pyre-ignore
        self._kwargs = kwargs

    @torch.no_grad()
    # pyre-ignore
    def step(self, closure=None) -> torch.Tensor:
        raise NotImplementedError


class Adam(Optimizer):
    r"""
    Placeholder for Adam. This optimizer will not functionally run.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        # pyre-ignore
        **kwargs,
    ) -> None:
        self._params = params
        # pyre-ignore
        self._kwargs = kwargs

    @torch.no_grad()
    # pyre-ignore
    def step(self, closure=None) -> torch.Tensor:
        raise NotImplementedError


class PartialRowWiseAdam(Optimizer):
    r"""
    Placeholder for PartialRowWiseAdam. This optimizer will not functionally run.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        # pyre-ignore
        **kwargs,
    ) -> None:
        self._params = params
        # pyre-ignore
        self._kwargs = kwargs

    @torch.no_grad()
    # pyre-ignore
    def step(self, closure=None) -> torch.Tensor:
        raise NotImplementedError


class Adagrad(Optimizer):
    r"""
    Placeholder for Adagrad. This optimizer will not functionally run.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        # pyre-ignore
        **kwargs,
    ) -> None:
        self._params = params
        # pyre-ignore
        self._kwargs = kwargs

    @torch.no_grad()
    # pyre-ignore
    def step(self, closure=None) -> torch.Tensor:
        raise NotImplementedError
