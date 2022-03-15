#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
from typing import List, Type, Dict, Optional, Any

import torch
import torch.nn as nn
import torch.quantization as quant
import torchrec as trec
import torchrec.quant as trec_quant


def quantize_embeddings(
    module: nn.Module,
    dtype: torch.dtype,
    inplace: bool,
    additional_qconfig_spec_keys: Optional[List[Type[nn.Module]]] = None,
    additional_mapping: Optional[Dict[Type[nn.Module], Type[nn.Module]]] = None,
) -> nn.Module:
    qconfig = quant.QConfig(
        activation=quant.PlaceholderObserver,
        weight=quant.PlaceholderObserver.with_args(dtype=dtype),
    )
    qconfig_spec: Dict[Type[nn.Module], quant.QConfig] = {
        trec.EmbeddingBagCollection: qconfig,
    }
    mapping: Dict[Type[nn.Module], Type[nn.Module]] = {
        trec.EmbeddingBagCollection: trec_quant.EmbeddingBagCollection,
    }
    if additional_qconfig_spec_keys is not None:
        for t in additional_qconfig_spec_keys:
            qconfig_spec[t] = qconfig
    if additional_mapping is not None:
        mapping.update(additional_mapping)
    return quant.quantize_dynamic(
        module,
        qconfig_spec=qconfig_spec,
        mapping=mapping,
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

    @abc.abstractmethod
    def result_metadata(self) -> str:
        """
        Returns a string which represents the result type. This information is used for result split.
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

    Args:
        module: the actual predict module
        device: the primary device for this module that will be used in forward calls.

    Example::

        module = PredictModule(torch.device("cuda", torch.cuda.current_device()))
    """

    def __init__(
        self,
        module: nn.Module,
    ) -> None:
        super().__init__()
        self._module: nn.Module = module
        # lazy init device from thread inited device guard
        self._device: Optional[torch.device] = None
        self._module.eval()

    @property
    def predict_module(
        self,
    ) -> nn.Module:
        return self._module

    @abc.abstractmethod
    # pyre-fixme[3]
    def predict_forward(self, batch: Dict[str, torch.Tensor]) -> Any:
        pass

    # pyre-fixme[3]
    def forward(self, batch: Dict[str, torch.Tensor]) -> Any:
        if self._device is None:
            self._device = torch.device("cuda", torch.cuda.current_device())
        with torch.cuda.device(self._device), torch.inference_mode():
            return self.predict_forward(batch)

    # pyre-fixme[14]: `state_dict` overrides method defined in `Module` inconsistently.
    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        return self._module.state_dict(destination, prefix, keep_vars)
