#!/usr/bin/env python3

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Union, Iterator, Tuple, cast

import torch
import torch.nn as nn
from fbgemm_gpu.split_embedding_configs import SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops import (
    PoolingMode,
    IntNBitTableBatchedEmbeddingBagsCodegen,
    EmbeddingLocation,
)
from torch import Tensor
from torchrec.distributed.embedding_lookup import (
    GroupedEmbeddingBag,
)
from torchrec.distributed.embedding_types import (
    EmbeddingTableConfig,
)
from torchrec.distributed.types import ShardedTensor, ShardedTensorMetadata
from torchrec.modules.embedding_configs import (
    EmbeddingBagConfig,
    PoolingType,
    DataType,
    DATA_TYPE_NUM_BITS,
)
from torchrec.modules.embedding_modules import EmbeddingBagCollectionInterface
from torchrec.sparse.jagged_tensor import (
    KeyedJaggedTensor,
    KeyedTensor,
)

torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")


class EmbeddingBagCollection(EmbeddingBagCollectionInterface):
    def __init__(
        self,
        table_name_to_quantized_weights: Dict[str, Union[Tensor, ShardedTensor]],
        embedding_configs: List[EmbeddingBagConfig],
        data_type: DataType,
        state_dict_prefix: str = "",
    ) -> None:
        @dataclass
        class QuantEmbeddingConfig:
            embedding_tables: List[EmbeddingTableConfig]
            data_type: DataType
            pooling: PoolingType

            def feature_names(self) -> List[str]:
                feature_names = []
                for table in self.embedding_tables:
                    feature_names.extend(table.feature_names)
                return feature_names

            def embedding_dims(self) -> List[int]:
                embedding_dims = []
                for table in self.embedding_tables:
                    embedding_dims.extend([table.embedding_dim] * table.num_features())
                return embedding_dims

        def to_pooling_mode(pooling_type: PoolingType) -> PoolingMode:
            if pooling_type == PoolingType.SUM:
                return PoolingMode.SUM
            else:
                assert pooling_type == PoolingType.MEAN
                return PoolingMode.MEAN

        def to_sparse_type(data_type: DataType) -> SparseType:
            if data_type == DataType.FP16:
                return SparseType.FP16
            elif data_type == DataType.INT8:
                return SparseType.INT8
            elif data_type == DataType.INT4:
                return SparseType.INT4
            elif data_type == DataType.INT2:
                return SparseType.INT2
            else:
                raise ValueError(f"Invalid DataType {data_type}")

        super().__init__()

        quantized_embedding_configs = []
        for embedding_config in embedding_configs:
            quantized_embedding_configs.append(
                QuantEmbeddingConfig(
                    embedding_tables=[
                        EmbeddingTableConfig(
                            num_embeddings=embedding_config.num_embeddings,
                            embedding_dim=embedding_config.embedding_dim,
                            name=embedding_config.name,
                            data_type=data_type,
                            feature_names=embedding_config.feature_names,
                        )
                    ],
                    data_type=data_type,
                    pooling=embedding_config.pooling,
                )
            )

        self._emb_modules: nn.ModuleList[nn.Module] = nn.ModuleList()
        self._input_emb_configs = embedding_configs
        self._state_dict_prefix = state_dict_prefix
        self._emb_configs: List[QuantEmbeddingConfig] = quantized_embedding_configs
        self._emb_names: List[str] = []
        self._lengths_per_emb: List[int] = []
        self._emb_sharding_metadata: Dict[str, ShardedTensorMetadata] = {}
        for emb_config in self._emb_configs:
            # TODO: support BatchedEmbeddingBag.
            weights_list = []
            for embedding_table in emb_config.embedding_tables:
                tensor = table_name_to_quantized_weights[embedding_table.name]
                # pyre-ignore [16]
                if tensor.is_meta:
                    continue

                quantized_weights = (
                    torch.ops.fbgemm.FloatToFusedNBitRowwiseQuantizedSBHalf(
                        tensor, DATA_TYPE_NUM_BITS[emb_config.data_type]
                    )
                )

                # weight and 4 byte scale shift (2xfp16)
                weights_list.append(
                    (quantized_weights[:, :-4], quantized_weights[:, -4:])
                )

            emb_module = IntNBitTableBatchedEmbeddingBagsCodegen(
                embedding_specs=[
                    (
                        "",
                        table_config.num_embeddings,
                        table_config.embedding_dim,
                        to_sparse_type(emb_config.data_type),
                        EmbeddingLocation.DEVICE,
                    )
                    for table_config in emb_config.embedding_tables
                ],
                pooling_mode=to_pooling_mode(emb_config.pooling),
                # TODO: pass in weights here
                weight_lists=weights_list,
            )

            self._emb_modules.append(emb_module)
            self._emb_names.extend(emb_config.feature_names())
            self._lengths_per_emb.extend(emb_config.embedding_dims())

    def forward(
        self,
        features: KeyedJaggedTensor,
    ) -> KeyedTensor:
        pooled_embeddings: List[Tensor] = []

        for emb_config, emb_module in zip(self._emb_configs, self._emb_modules):
            for feature_name in emb_config.feature_names():
                values = features[feature_name].values()
                offsets = features[feature_name].offsets()
                weights = features[feature_name].weights_or_none()
                pooled_embeddings.append(
                    emb_module(
                        indices=values.int(),
                        offsets=offsets.int(),
                        per_sample_weights=weights,
                    ).float()
                )

        return KeyedTensor(
            keys=self._emb_names,
            values=torch.cat(pooled_embeddings, dim=1),
            length_per_key=self._lengths_per_emb,
        )

    @property
    def embedding_bag_configs(self) -> List[EmbeddingBagConfig]:
        return self._input_emb_configs

    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        if destination is None:
            destination = OrderedDict()
            # pyre-ignore [16]
            destination._metadata = OrderedDict()
        for emb_config, emb_module in zip(
            self._emb_configs,
            self._emb_modules,
        ):
            for emb_table_config, split in zip(
                emb_config.embedding_tables, emb_module.split_embedding_weights()
            ):
                first, second = split
                if first.is_meta:
                    assert second.is_meta
                    tensor = torch.empty((first.size() + second.size()), device="meta")
                else:
                    tensor = torch.cat([second, first], dim=1)
                destination[
                    prefix + f"{self._state_dict_prefix}{emb_table_config.name}.weight"
                ] = tensor
        return destination

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        state_dict = self.state_dict(prefix=prefix, keep_vars=True)
        for key, value in state_dict.items():
            yield key, cast(
                nn.Parameter,
                value,
            )

    def _get_name(self) -> str:
        return "QuantizedEmbeddingBagCollection"

    @classmethod
    def from_float(cls, module: EmbeddingBagCollectionInterface) -> nn.Module:
        assert hasattr(
            module, "qconfig"
        ), "EmbeddingBagCollectionInterface input float module must have qconfig defined"

        def _to_data_type(dtype: torch.dtype) -> DataType:
            if dtype == torch.quint8 or dtype == torch.qint8:
                return DataType.INT8
            elif dtype == torch.quint4 or dtype == torch.qint4:
                return DataType.INT4
            elif dtype == torch.quint2 or dtype == torch.qint2:
                return DataType.INT2
            else:
                raise Exception(f"Invalid data type {dtype}")

        # pyre-ignore [16]
        data_type = _to_data_type(module.qconfig.weight().dtype)

        table_name_to_quantized_weights: Dict[str, Union[Tensor, ShardedTensor]] = {}
        for key, tensor in module.state_dict().items():
            # Extract table name from state dict key.
            # e.g. ebc.embedding_bags.t1.weight
            splits = key.split(".")
            assert splits[-1] == "weight"
            table_name = splits[-2]
            table_name_to_quantized_weights[table_name] = tensor

        return cls(
            table_name_to_quantized_weights,
            module.embedding_bag_configs,
            data_type,
            "" if isinstance(module, GroupedEmbeddingBag) else "embedding_bags.",
        )
