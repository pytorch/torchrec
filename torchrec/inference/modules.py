#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
from typing import Dict, Optional, Any

import torch
import torch.nn as nn
import torch.quantization as quant
import torchrec as trec
import torchrec.quant as trec_quant


def quantize_embeddings(
    module: nn.Module, dtype: torch.dtype, inplace: bool
) -> nn.Module:
    qconfig = quant.QConfig(
        activation=quant.PlaceholderObserver,
        weight=quant.PlaceholderObserver.with_args(dtype=dtype),
    )
    return quant.quantize_dynamic(
        module,
        qconfig_spec={
            trec.EmbeddingBagCollection: qconfig,
        },
        mapping={
            trec.EmbeddingBagCollection: trec_quant.EmbeddingBagCollection,
        },
        inplace=inplace,
    )


class PredictFactory(abc.ABC):
    """
    Creates a model (with already learned weights) to be used inference time.
    """

    @abc.abstractmethod
    def create_predict_module(self) -> nn.Module:
        """
        Returns already sharded model with allocated weights.
        state_dict() must match TransformModule.transform_state_dict().
        It assumes that torch.distributed.init_process_group was already called
        and will shard model according to torch.distributed.get_world_size().
        """
        pass

    @abc.abstractmethod
    def batching_metadata(self) -> Dict[str, str]:
        """
        Returns a dict from input name to feature type. This infomation is used for batching.
        """
        pass


class PredictModule(nn.Module):
    """
    Interface for modules to work in a torch.deploy based backend. Users should
    override predict_forward to convert batch input format to module input format.

    Call Args:
        batch: a dict of input tensors

    Returns:
        output: a dict of output tensors

    Constructor Args:
        module: the actual predict module
        device: the primary device for this module that will be used in forward calls.

    Example:
        >>> module = PredictModule(torch.device("cuda", torch.cuda.current_device()))
    """

    def __init__(
        self, module: nn.Module, device: Optional[torch.device] = None
    ) -> None:
        super().__init__()
        self._module: nn.Module = module
        self._device: torch.device = (
            torch.device("cuda", torch.cuda.current_device())
            if device is None
            else device
        )

        self._module.eval()

    @property
    def predict_module(
        self,
    ) -> nn.Module:
        return self._module

    @abc.abstractmethod
    def predict_forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        pass

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        with torch.cuda.device(self._device), torch.inference_mode():
            return self.predict_forward(batch)

    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        return self._module.state_dict(destination, prefix, keep_vars)


class MultistreamPredictModule(PredictModule):
    """
    Interface derived from PredictModule that supports using different CUDA streams in forward calls.
    """

    def __init__(
        self, module: nn.Module, device: Optional[torch.device] = None
    ) -> None:
        super().__init__(module, device)
        self._stream: Optional[torch.cuda.streams.Stream] = None

    @abc.abstractmethod
    def predict_forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        pass

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self._stream is None:
            # Lazily initialize stream to make sure it's created in the correct device.
            self._stream = torch.cuda.Stream(device=self._device)

        with torch.cuda.stream(self._stream):
            return super().forward(batch)
