#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Type

import torch

from torch import nn
from torchrec.ir.schema import EBCMetadata, EmbeddingBagConfigMetadata

from torchrec.ir.types import SerializerInterface
from torchrec.modules.embedding_configs import DataType, EmbeddingBagConfig, PoolingType
from torchrec.modules.embedding_modules import EmbeddingBagCollection

logger: logging.Logger = logging.getLogger(__name__)


def embedding_bag_config_to_metadata(
    table_config: EmbeddingBagConfig,
) -> EmbeddingBagConfigMetadata:
    return EmbeddingBagConfigMetadata(
        num_embeddings=table_config.num_embeddings,
        embedding_dim=table_config.embedding_dim,
        name=table_config.name,
        data_type=table_config.data_type.value,
        feature_names=table_config.feature_names,
        weight_init_max=table_config.weight_init_max,
        weight_init_min=table_config.weight_init_min,
        need_pos=table_config.need_pos,
        pooling=table_config.pooling.value,
    )


def embedding_metadata_to_config(
    table_config: EmbeddingBagConfigMetadata,
) -> EmbeddingBagConfig:
    return EmbeddingBagConfig(
        num_embeddings=table_config.num_embeddings,
        embedding_dim=table_config.embedding_dim,
        name=table_config.name,
        data_type=DataType(table_config.data_type),
        feature_names=table_config.feature_names,
        weight_init_max=table_config.weight_init_max,
        weight_init_min=table_config.weight_init_min,
        need_pos=table_config.need_pos,
        pooling=PoolingType(table_config.pooling),
    )


def get_deserialized_device(
    config_device: Optional[str], device: Optional[torch.device]
) -> Optional[torch.device]:
    if config_device:
        original_device = torch.device(config_device)
        if device is None:
            device = original_device
        elif original_device.type != device.type:
            logger.warning(
                f"deserialized device={device} overrides the original device={original_device}"
            )
    return device


class JsonSerializer(SerializerInterface):
    """
    Serializer for torch.export IR using json.
    """

    module_to_serializer_cls: Dict[str, Type["JsonSerializer"]] = {}
    _module_cls: Optional[Type[nn.Module]] = None
    _children: Optional[List[str]] = None

    @classmethod
    def children(cls, module: nn.Module) -> List[str]:
        return [] if not cls._children else cls._children

    @classmethod
    def serialize_to_dict(cls, module: nn.Module) -> Dict[str, Any]:
        raise NotImplementedError()

    @classmethod
    def deserialize_from_dict(
        cls,
        metadata_dict: Dict[str, Any],
        device: Optional[torch.device] = None,
        unflatten_ep: Optional[nn.Module] = None,
    ) -> nn.Module:
        raise NotImplementedError()

    @classmethod
    def serialize(
        cls,
        module: nn.Module,
    ) -> Tuple[torch.Tensor, List[str]]:
        typename = type(module).__name__
        serializer = cls.module_to_serializer_cls.get(typename)
        if serializer is None:
            raise ValueError(
                f"Expected typename to be one of {list(cls.module_to_serializer_cls.keys())}, got {typename}"
            )
        assert issubclass(serializer, JsonSerializer)
        assert serializer._module_cls is not None
        if not isinstance(module, serializer._module_cls):
            raise ValueError(
                f"Expected module to be of type {serializer._module_cls.__name__}, "
                f"got {type(module)}"
            )
        metadata_dict = serializer.serialize_to_dict(module)
        raw_dict = {"typename": typename, "metadata_dict": metadata_dict}
        ir_metadata_tensor = torch.frombuffer(
            json.dumps(raw_dict).encode(), dtype=torch.uint8
        )
        return ir_metadata_tensor, serializer.children(module)

    @classmethod
    def deserialize(
        cls,
        ir_metadata: torch.Tensor,
        device: Optional[torch.device] = None,
        unflatten_ep: Optional[nn.Module] = None,
    ) -> nn.Module:
        raw_bytes = ir_metadata.numpy().tobytes()
        raw_dict = json.loads(raw_bytes.decode())
        typename = raw_dict["typename"]
        if typename not in cls.module_to_serializer_cls:
            raise ValueError(
                f"Expected typename to be one of {list(cls.module_to_serializer_cls.keys())}, got {typename}"
            )
        serializer = cls.module_to_serializer_cls[typename]
        assert issubclass(serializer, JsonSerializer)
        module = serializer.deserialize_from_dict(
            raw_dict["metadata_dict"], device, unflatten_ep
        )

        if serializer._module_cls is None:
            raise ValueError(
                "Must assign a nn.Module to class static variable _module_cls"
            )
        if not isinstance(module, serializer._module_cls):
            raise ValueError(
                f"Expected module to be of type {serializer._module_cls.__name__}, got {type(module)}"
            )
        return module


class EBCJsonSerializer(JsonSerializer):
    _module_cls = EmbeddingBagCollection

    @classmethod
    def serialize_to_dict(
        cls,
        module: nn.Module,
    ) -> Dict[str, Any]:
        ebc_metadata = EBCMetadata(
            tables=[
                embedding_bag_config_to_metadata(table_config)
                for table_config in module.embedding_bag_configs()
            ],
            is_weighted=module.is_weighted(),
            device=str(module.device),
        )
        ebc_metadata_dict = ebc_metadata.__dict__
        ebc_metadata_dict["tables"] = [
            table_config.__dict__ for table_config in ebc_metadata_dict["tables"]
        ]
        return ebc_metadata_dict

    @classmethod
    def deserialize_from_dict(
        cls,
        metadata_dict: Dict[str, Any],
        device: Optional[torch.device] = None,
        unflatten_ep: Optional[nn.Module] = None,
    ) -> nn.Module:
        tables = [
            EmbeddingBagConfigMetadata(**table_config)
            for table_config in metadata_dict["tables"]
        ]

        device = get_deserialized_device(metadata_dict.get("device"), device)
        return EmbeddingBagCollection(
            tables=[
                embedding_metadata_to_config(table_config) for table_config in tables
            ],
            is_weighted=metadata_dict["is_weighted"],
            device=device,
        )


JsonSerializer.module_to_serializer_cls["EmbeddingBagCollection"] = EBCJsonSerializer
