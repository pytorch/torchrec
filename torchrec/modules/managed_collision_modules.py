#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

import abc
from typing import Optional

import torch

from torch import nn
from torchrec.sparse.jagged_tensor import JaggedTensor


class ManagedCollisionModule(nn.Module):
    """
    Abstract base class for feature processor.

    Args:
        max_output_id: Number of physical slots for ids in down stream embedding bag
        max_input_id: Number of logical ids to manage

    Example::
        jt = JaggedTensor(...)
        mcm = ManagedCollisionModule(...)
        mcm_jt = mcm(fp)
    """

    def __init__(self, max_output_id: int, max_input_id: int = 2**40) -> None:
        # slots is the number of rows to map from global id to
        # for example, if we want to manage 1000 ids to 10 slots
        super().__init__()
        self._max_input_id: int = max_input_id
        self._max_output_id = max_output_id

    @abc.abstractmethod
    def forward(
        self,
        features: JaggedTensor,
    ) -> JaggedTensor:
        """
        Args:
        features (JaggedTensor]): feature representation

        Returns:
            JaggedTensor: modified JT
        """
        pass

    @abc.abstractmethod
    def evict(self) -> Optional[torch.Tensor]:
        """
        Returns None if no eviction should be done tihs iteration. Otherwise, return ids of slots to reset.
        On eviction, this module should reset its state for those slots, with the assumptionn that the downstream module
        will handle this properly.
        """
        pass

    @abc.abstractmethod
    def rebuild_with_max_output_id(
        self, max_output_id: int, device: torch.device
    ) -> "ManagedCollisionModule":
        """
        Used for creating local MC modules for RW sharding, hack for now
        """
        pass


class TrivialManagedCollisionModule(ManagedCollisionModule):
    def __init__(
        self, max_output_id: int, device: torch.device, max_input_id: int = 2**64
    ) -> None:
        super().__init__(max_output_id, max_input_id)
        self.register_buffer(
            "count",
            torch.zeros(
                (max_output_id,),
                device=device,
            ),
        )

    def forward(
        self,
        features: JaggedTensor,
    ) -> JaggedTensor:
        values = features.values() % self._max_output_id
        self.count[values] += 1
        return JaggedTensor(
            values=values,
            lengths=features.lengths(),
            offsets=features.offsets(),
            weights=features.weights_or_none(),
        )

    def evict(self) -> Optional[torch.Tensor]:
        return None

    def rebuild_with_max_output_id(
        self, max_output_id: int, device: Optional[torch.device] = None
    ) -> "TrivialManagedCollisionModule":
        return type(self)(
            max_output_id=max_output_id,
            device=device or self._device,
            max_input_id=self._max_input_id,
        )
