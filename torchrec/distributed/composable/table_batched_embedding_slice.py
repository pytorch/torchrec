#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch

from torch import nn


class TableBatchedEmbeddingSlice(nn.Parameter):
    """
    Parameter to represent a slice of a table batched embedding. The slice will be
    a view of the TBE of shape (num_embeddings, embedding_dim) and contain consistent .grad
    """

    __slots__ = [
        "_original_tensor",
        "_start_offset",
        "_end_offset",
        "_num_embeddings",
        "_embedding_dim",
    ]

    def __init__(
        self,
        original_tensor: torch.Tensor,
        start_offset: int,
        end_offset: int,
        num_embeddings: int,
        embedding_dim: int,
    ) -> None:
        self._original_tensor: torch.Tensor = original_tensor
        self._start_offset: int = start_offset
        self._end_offset: int = end_offset
        self._num_embeddings: int = num_embeddings
        self._embedding_dim: int = embedding_dim
        self._init_grad: Optional[torch.Tensor] = None
        if original_tensor.requires_grad:
            self.retain_grad()

    def __new__(
        cls,
        original_tensor: torch.Tensor,
        start_offset: int,
        end_offset: int,
        num_embeddings: int,
        embedding_dim: int,
    ) -> "TableBatchedEmbeddingSlice":
        _slice = original_tensor[start_offset:end_offset].view(
            num_embeddings, embedding_dim
        )
        return _slice.as_subclass(cls)

    @property
    def grad(self) -> Optional[torch.Tensor]:
        if self._original_tensor.grad is None:
            return self._init_grad
        return self._original_tensor.grad[self._start_offset : self._end_offset].view(
            self._num_embeddings, self._embedding_dim
        )

    @grad.setter
    def grad(self, set_grad: torch.Tensor) -> None:
        self._init_grad = set_grad
        if set_grad is None:
            self._original_tensor.grad = None
        elif self._original_tensor.grad is not None:
            self._original_tensor.grad[self._start_offset : self._end_offset].copy_(
                set_grad.view(-1)
            )

    @property
    def grad_fn(self) -> None:
        # set as leaf node
        return None
