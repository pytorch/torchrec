#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.quantization as quant
import torchrec as trec
import torchrec.quant as trec_quant
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollectionInterface,
    EmbeddingCollectionInterface,
)


def quantize_feature(
    module: torch.nn.Module, inputs: Tuple[torch.Tensor, ...]
) -> Tuple[torch.Tensor, ...]:
    return tuple(
        [
            input.half()
            if isinstance(input, torch.Tensor)
            and input.dtype in [torch.float32, torch.float64]
            else input
            for input in inputs
        ]
    )


def quantize_embeddings(
    module: nn.Module,
    dtype: torch.dtype,
    inplace: bool,
    additional_qconfig_spec_keys: Optional[List[Type[nn.Module]]] = None,
    additional_mapping: Optional[Dict[Type[nn.Module], Type[nn.Module]]] = None,
    output_dtype: torch.dtype = torch.float,
) -> nn.Module:
    qconfig = quant.QConfig(
        activation=quant.PlaceholderObserver.with_args(dtype=output_dtype),
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


@dataclass
class QualNameMetadata:
    need_preproc: bool


@dataclass
class BatchingMetadata:
    """
    Metadata class for batching, this should be kept in sync with the C++ definition.
    """

    type: str
    # cpu or cuda
    device: str
    # list of tensor suffixes to deserialize to pinned memory (e.g. "lengths")
    # use "" (empty string) to pin without suffix
    pinned: List[str]


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
    def batching_metadata(self) -> Dict[str, BatchingMetadata]:
        """
        Returns a dict from input name to BatchingMetadata. This infomation is used for batching for input requests.
        """
        pass

    def batching_metadata_json(self) -> str:
        """
        Serialize the batching metadata to JSON, for ease of parsing with torch::deploy environments.
        """
        return json.dumps(
            {key: asdict(value) for key, value in self.batching_metadata().items()}
        )

    @abc.abstractmethod
    def result_metadata(self) -> str:
        """
        Returns a string which represents the result type. This information is used for result split.
        """
        pass

    @abc.abstractmethod
    def run_weights_independent_tranformations(
        self, predict_module: torch.nn.Module
    ) -> torch.nn.Module:
        """
        Run transformations that don't rely on weights of the predict module. e.g. fx tracing, model
        split etc.
        """
        pass

    @abc.abstractmethod
    def run_weights_dependent_transformations(
        self, predict_module: torch.nn.Module
    ) -> torch.nn.Module:
        """
        Run transformations that depends on weights of the predict module. e.g. lowering to a backend.
        """
        pass

    def qualname_metadata(self) -> Dict[str, QualNameMetadata]:
        """
        Returns a dict from qualname (method name) to QualNameMetadata. This is additional information for execution of specific methods of the model.
        """
        return {}

    def qualname_metadata_json(self) -> str:
        """
        Serialize the qualname metadata to JSON, for ease of parsing with torch::deploy environments.
        """
        return json.dumps(
            {key: asdict(value) for key, value in self.qualname_metadata().items()}
        )

    def model_inputs_data(self) -> Dict[str, Any]:
        """
        Returns a dict of various data for benchmarking input generation.
        """
        return {}


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
        # pyre-fixme[19]: Expected 0 positional arguments.
        return self._module.state_dict(destination, prefix, keep_vars)


def quantize_dense(
    predict_module: PredictModule,
    dtype: torch.dtype,
    additional_embedding_module_type: List[Type[nn.Module]] = [],
) -> nn.Module:
    module = predict_module.predict_module
    reassign = {}

    for name, mod in module.named_children():
        # both fused modules and observed custom modules are
        # swapped as one unit
        if not (
            isinstance(mod, EmbeddingBagCollectionInterface)
            or isinstance(mod, EmbeddingCollectionInterface)
            or any([type(mod) is clazz for clazz in additional_embedding_module_type])
        ):
            if dtype == torch.half:
                new_mod = mod.half()
                new_mod.register_forward_pre_hook(quantize_feature)
                reassign[name] = new_mod
            else:
                raise NotImplementedError(
                    "only fp16 is supported for non-embedding module lowering"
                )
    for key, value in reassign.items():
        module._modules[key] = value
    return predict_module
