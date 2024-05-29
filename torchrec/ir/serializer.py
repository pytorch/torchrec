#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import json
import logging
from typing import Dict, Optional, Type

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


class EBCJsonSerializer(SerializerInterface):
    """
    Serializer for torch.export IR using thrift.
    """

    @classmethod
    def serialize(
        cls,
        module: nn.Module,
    ) -> torch.Tensor:
        if not isinstance(module, EmbeddingBagCollection):
            raise ValueError(
                f"Expected module to be of type EmbeddingBagCollection, got {type(module)}"
            )

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

        return torch.frombuffer(
            json.dumps(ebc_metadata_dict).encode(), dtype=torch.uint8
        )

    @classmethod
    def deserialize(
        cls, input: torch.Tensor, typename: str, device: Optional[torch.device] = None
    ) -> nn.Module:
        if typename != "EmbeddingBagCollection":
            raise ValueError(
                f"Expected typename to be EmbeddingBagCollection, got {typename}"
            )

        raw_bytes = input.numpy().tobytes()
        ebc_metadata_dict = json.loads(raw_bytes.decode())
        tables = [
            EmbeddingBagConfigMetadata(**table_config)
            for table_config in ebc_metadata_dict["tables"]
        ]

        device = get_deserialized_device(ebc_metadata_dict.get("device"), device)
        return EmbeddingBagCollection(
            tables=[
                embedding_metadata_to_config(table_config) for table_config in tables
            ],
            is_weighted=ebc_metadata_dict["is_weighted"],
            device=device,
        )


class JsonSerializer(SerializerInterface):
    """
    Serializer for torch.export IR using thrift.
    """

    module_to_serializer_cls: Dict[str, Type[SerializerInterface]] = {
        "EmbeddingBagCollection": EBCJsonSerializer,
    }

    @classmethod
    def serialize(
        cls,
        module: nn.Module,
    ) -> torch.Tensor:
        typename = type(module).__name__
        if typename not in cls.module_to_serializer_cls:
            raise ValueError(
                f"Expected typename to be one of {list(cls.module_to_serializer_cls.keys())}, got {typename}"
            )

        return cls.module_to_serializer_cls[typename].serialize(module)

    @classmethod
    def deserialize(
        cls, input: torch.Tensor, typename: str, device: Optional[torch.device] = None
    ) -> nn.Module:
        if typename not in cls.module_to_serializer_cls:
            raise ValueError(
                f"Expected typename to be one of {list(cls.module_to_serializer_cls.keys())}, got {typename}"
            )

        return cls.module_to_serializer_cls[typename].deserialize(
            input, typename, device
        )
