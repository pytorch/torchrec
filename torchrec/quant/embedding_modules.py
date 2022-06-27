#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from collections import OrderedDict
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
from fbgemm_gpu.split_table_batched_embeddings_ops import (
    EmbeddingLocation,
    IntNBitTableBatchedEmbeddingBagsCodegen,
    PoolingMode,
)
from torch import Tensor
from torchrec.modules.embedding_configs import (
    DATA_TYPE_NUM_BITS,
    data_type_to_sparse_type,
    DataType,
    dtype_to_data_type,
    EmbeddingBagConfig,
    EmbeddingConfig,
    pooling_type_to_pooling_mode,
)
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection as OriginalEmbeddingBagCollection,
    EmbeddingBagCollectionInterface,
    EmbeddingCollection as OriginalEmbeddingCollection,
    EmbeddingCollectionInterface,
    get_embedding_names_by_table,
)
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor, KeyedTensor


try:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
except OSError:
    pass

# OSS
try:
    import fbgemm_gpu  # @manual  # noqa
except ImportError:
    pass


def quantize_state_dict(
    module: nn.Module,
    table_name_to_quantized_weights: Dict[str, Tuple[Tensor, Tensor]],
    data_type: DataType,
) -> torch.device:
    device = torch.device("cpu")
    for key, tensor in module.state_dict().items():
        # Extract table name from state dict key.
        # e.g. ebc.embedding_bags.t1.weight
        splits = key.split(".")
        assert splits[-1] == "weight"
        table_name = splits[-2]
        device = tensor.device
        num_bits = DATA_TYPE_NUM_BITS[data_type]
        if tensor.is_meta:
            quant_weight = torch.empty(
                (tensor.shape[0], (tensor.shape[1] * num_bits) // 8),
                device="meta",
                # pyre-fixme[16]: Item `Tensor` of `Union[Tensor, Module]` has
                #  no attribute `weight`.
                dtype=module.qconfig.weight().dtype,
            )
            if (
                data_type == DataType.INT8
                or data_type == DataType.INT4
                or data_type == DataType.INT2
            ):
                scale_shift = torch.empty(
                    (tensor.shape[0], 4),
                    device="meta",
                    # pyre-fixme[16]: Item `Tensor` of `Union[Tensor, Module]` has
                    #  no attribute `weight`.
                    dtype=module.qconfig.weight().dtype,
                )
            else:
                scale_shift = None
        else:
            if tensor.dtype == torch.float or tensor.dtype == torch.float16:
                if tensor.dtype == torch.float16 and data_type == DataType.FP16:
                    quant_res = tensor.view(torch.uint8)
                else:
                    quant_res = (
                        torch.ops.fbgemm.FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf(
                            tensor, num_bits
                        )
                    )
            else:
                raise Exception("Unsupported dtype: {tensor.dtype}")
            if (
                data_type == DataType.INT8
                or data_type == DataType.INT4
                or data_type == DataType.INT2
            ):
                quant_weight, scale_shift = (
                    quant_res[:, :-4],
                    quant_res[:, -4:],
                )
            else:
                quant_weight, scale_shift = quant_res, None
        table_name_to_quantized_weights[table_name] = (quant_weight, scale_shift)
    return device


class EmbeddingBagCollection(EmbeddingBagCollectionInterface):
    """
    EmbeddingBagCollection represents a collection of pooled embeddings (EmbeddingBags).
    This EmbeddingBagCollection is quantized for lower precision. It relies on fbgemm quantized ops

    It processes sparse data in the form of KeyedJaggedTensor
    with values of the form [F X B X L]
    F: features (keys)
    B: batch size
    L: Length of sparse features (jagged)

    and outputs a KeyedTensor with values of the form [B * (F * D)]
    where
    F: features (keys)
    D: each feature's (key's) embedding dimension
    B: batch size

    Args:
        table_name_to_quantized_weights (Dict[str, Tuple[Tensor, Tensor]]): map of tables to quantized weights
        embedding_configs (List[EmbeddingBagConfig]): list of embedding tables
        is_weighted: (bool): whether input KeyedJaggedTensor is weighted
        device: (Optional[torch.device]): default compute device

    Call Args:
        features: KeyedJaggedTensor,

    Returns:
        KeyedTensor

    Example::

        table_0 = EmbeddingBagConfig(
            name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1"]
        )
        table_1 = EmbeddingBagConfig(
            name="t2", embedding_dim=4, num_embeddings=10, feature_names=["f2"]
        )
        ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])

        #        0       1        2  <-- batch
        # "f1"   [0,1] None    [2]
        # "f2"   [3]    [4]    [5,6,7]
        #  ^
        # feature
        features = KeyedJaggedTensor(
            keys=["f1", "f2"],
            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
            offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8]),
        )

        ebc.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.PlaceholderObserver.with_args(
                dtype=torch.qint8
            ),
            weight=torch.quantization.PlaceholderObserver.with_args(dtype=torch.qint8),
        )

        qebc = QuantEmbeddingBagCollection.from_float(ebc)
        quantized_embeddings = qebc(features)
    """

    def __init__(
        self,
        table_name_to_quantized_weights: Dict[str, Tuple[Tensor, Tensor]],
        embedding_configs: List[EmbeddingBagConfig],
        is_weighted: bool,
        device: torch.device,
        output_dtype: torch.dtype = torch.float,
    ) -> None:
        super().__init__()
        self._is_weighted = is_weighted
        self._embedding_bag_configs: List[EmbeddingBagConfig] = embedding_configs
        self.embedding_bags: nn.ModuleList = nn.ModuleList()
        self._lengths_per_embedding: List[int] = []
        self._output_dtype = output_dtype
        table_names = set()
        for emb_config in self._embedding_bag_configs:
            if emb_config.name in table_names:
                raise ValueError(f"Duplicate table name {emb_config.name}")
            table_names.add(emb_config.name)
            emb_module = IntNBitTableBatchedEmbeddingBagsCodegen(
                embedding_specs=[
                    (
                        "",
                        emb_config.num_embeddings,
                        emb_config.embedding_dim,
                        data_type_to_sparse_type(emb_config.data_type),
                        EmbeddingLocation.HOST
                        if device.type == "cpu"
                        else EmbeddingLocation.DEVICE,
                    )
                ],
                pooling_mode=pooling_type_to_pooling_mode(emb_config.pooling),
                weight_lists=[table_name_to_quantized_weights[emb_config.name]],
                device=device,
                output_dtype=data_type_to_sparse_type(dtype_to_data_type(output_dtype)),
                row_alignment=16,
            )
            self.embedding_bags.append(emb_module)
            if not emb_config.feature_names:
                emb_config.feature_names = [emb_config.name]
            self._lengths_per_embedding.extend(
                len(emb_config.feature_names) * [emb_config.embedding_dim]
            )

        self._embedding_names: List[str] = [
            embedding
            for embeddings in get_embedding_names_by_table(embedding_configs)
            for embedding in embeddings
        ]

    def forward(
        self,
        features: KeyedJaggedTensor,
    ) -> KeyedTensor:
        pooled_embeddings: List[Tensor] = []
        length_per_key: List[int] = []
        feature_dict = features.to_dict()
        for emb_config, emb_module in zip(
            self._embedding_bag_configs, self.embedding_bags
        ):
            for feature_name in emb_config.feature_names:
                f = feature_dict[feature_name]
                values = f.values()
                offsets = f.offsets()
                pooled_embeddings.append(
                    emb_module(
                        indices=values.int(),
                        offsets=offsets.int(),
                        per_sample_weights=f.weights() if self._is_weighted else None,
                    )
                )

                length_per_key.append(emb_config.embedding_dim)

        return KeyedTensor(
            keys=self._embedding_names,
            values=torch.cat(pooled_embeddings, dim=1),
            length_per_key=self._lengths_per_embedding,
        )

    # pyre-fixme[14]: `state_dict` overrides method defined in `Module` inconsistently.
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
            self._embedding_bag_configs,
            self.embedding_bags,
        ):
            (weight, _) = emb_module.split_embedding_weights(split_scale_shifts=False)[
                0
            ]
            destination[prefix + f"embedding_bags.{emb_config.name}.weight"] = weight
        return destination

    def named_buffers(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        state_dict = self.state_dict(prefix=prefix, keep_vars=True)
        for key, value in state_dict.items():
            yield key, value

    def _get_name(self) -> str:
        return "QuantizedEmbeddingBagCollection"

    @classmethod
    def from_float(
        cls, module: OriginalEmbeddingBagCollection
    ) -> "EmbeddingBagCollection":
        assert hasattr(
            module, "qconfig"
        ), "EmbeddingBagCollection input float module must have qconfig defined"

        # pyre-ignore [16]
        data_type = dtype_to_data_type(module.qconfig.weight().dtype)
        embedding_bag_configs = copy.deepcopy(module.embedding_bag_configs())
        for config in embedding_bag_configs:
            config.data_type = data_type

        table_name_to_quantized_weights: Dict[str, Tuple[Tensor, Tensor]] = {}
        device = quantize_state_dict(module, table_name_to_quantized_weights, data_type)
        return cls(
            table_name_to_quantized_weights,
            embedding_bag_configs,
            module.is_weighted(),
            device=device,
            # pyre-ignore [16]
            output_dtype=module.qconfig.activation().dtype,
        )

    def embedding_bag_configs(
        self,
    ) -> List[EmbeddingBagConfig]:
        return self._embedding_bag_configs

    def is_weighted(self) -> bool:
        return self._is_weighted

    def output_dtype(self) -> torch.dtype:
        return self._output_dtype


class EmbeddingCollection(EmbeddingCollectionInterface):
    """
    EmbeddingCollection represents a collection of non-pooled embeddings.

    It processes sparse data in the form of `KeyedJaggedTensor` of the form [F X B X L]
    where:

    * F: features (keys)
    * B: batch size
    * L: length of sparse features (variable)

    and outputs `Dict[feature (key), JaggedTensor]`.
    Each `JaggedTensor` contains values of the form (B * L) X D
    where:

    * B: batch size
    * L: length of sparse features (jagged)
    * D: each feature's (key's) embedding dimension and lengths are of the form L

    Args:
        tables (List[EmbeddingConfig]): list of embedding tables.
        device (Optional[torch.device]): default compute device.
        need_indices (bool): if we need to pass indices to the final lookup result dict

    Example::

        e1_config = EmbeddingConfig(
            name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1"]
        )
        e2_config = EmbeddingConfig(
            name="t2", embedding_dim=3, num_embeddings=10, feature_names=["f2"]
        )

        ec = EmbeddingCollection(tables=[e1_config, e2_config])

        #     0       1        2  <-- batch
        # 0   [0,1] None    [2]
        # 1   [3]    [4]    [5,6,7]
        # ^
        # feature

        features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f2"],
            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
            offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8]),
        )
        feature_embeddings = ec(features)
        print(feature_embeddings['f2'].values())
        tensor([[-0.2050,  0.5478,  0.6054],
        [ 0.7352,  0.3210, -3.0399],
        [ 0.1279, -0.1756, -0.4130],
        [ 0.7519, -0.4341, -0.0499],
        [ 0.9329, -1.0697, -0.8095]], grad_fn=<EmbeddingBackward>)
    """

    def __init__(  # noqa C901
        self,
        table_name_to_quantized_weights: Dict[str, Tuple[Tensor, Tensor]],
        tables: List[EmbeddingConfig],
        device: torch.device,
        need_indices: bool = False,
        output_dtype: torch.dtype = torch.float,
    ) -> None:
        super().__init__()
        self.embeddings: nn.ModuleList = nn.ModuleList()
        self._embedding_configs = tables
        self._embedding_dim: int = -1
        self._need_indices: bool = need_indices
        self._output_dtype = output_dtype
        table_names = set()
        for config in tables:
            if config.name in table_names:
                raise ValueError(f"Duplicate table name {config.name}")
            table_names.add(config.name)
            self._embedding_dim = (
                config.embedding_dim if self._embedding_dim < 0 else self._embedding_dim
            )
            if self._embedding_dim != config.embedding_dim:
                raise ValueError(
                    "All tables in a EmbeddingCollection are required to have same embedding dimension."
                )
            self.embeddings.append(
                IntNBitTableBatchedEmbeddingBagsCodegen(
                    embedding_specs=[
                        (
                            "",
                            config.num_embeddings,
                            config.embedding_dim,
                            data_type_to_sparse_type(config.data_type),
                            EmbeddingLocation.HOST
                            if device.type == "cpu"
                            else EmbeddingLocation.DEVICE,
                        )
                    ],
                    pooling_mode=PoolingMode.NONE,
                    weight_lists=[table_name_to_quantized_weights[config.name]],
                    device=device,
                    output_dtype=data_type_to_sparse_type(
                        dtype_to_data_type(output_dtype)
                    ),
                    row_alignment=16,
                )
            )
            if not config.feature_names:
                config.feature_names = [config.name]

        self._embedding_names_by_table: List[List[str]] = get_embedding_names_by_table(
            tables
        )

    def forward(
        self,
        features: KeyedJaggedTensor,
    ) -> Dict[str, JaggedTensor]:
        """
        Args:
            features (KeyedJaggedTensor): KJT of form [F X B X L].

        Returns:
            Dict[str, JaggedTensor]
        """

        feature_embeddings: Dict[str, JaggedTensor] = {}
        jt_dict: Dict[str, JaggedTensor] = features.to_dict()
        for config, embedding_names, emb_module in zip(
            self._embedding_configs,
            self._embedding_names_by_table,
            self.embeddings,
        ):
            for feature_name, embedding_name in zip(
                config.feature_names, embedding_names
            ):
                f = jt_dict[feature_name]
                values = f.values()
                offsets = f.offsets()
                lookup = emb_module(
                    indices=values.int(),
                    offsets=offsets.int(),
                )
                feature_embeddings[embedding_name] = JaggedTensor(
                    values=lookup,
                    lengths=f.lengths(),
                    weights=f.values() if self.need_indices else None,
                )
        return feature_embeddings

    # pyre-fixme[14]: `state_dict` overrides method defined in `Module` inconsistently.
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
            self._embedding_configs,
            self.embeddings,
        ):
            (weight, _) = emb_module.split_embedding_weights(split_scale_shifts=False)[
                0
            ]
            destination[prefix + f"embeddings.{emb_config.name}.weight"] = weight
        return destination

    @classmethod
    def from_float(cls, module: OriginalEmbeddingCollection) -> "EmbeddingCollection":
        assert hasattr(
            module, "qconfig"
        ), "EmbeddingCollection input float module must have qconfig defined"

        # pyre-ignore [16]
        data_type = dtype_to_data_type(module.qconfig.weight().dtype)
        tables = copy.deepcopy(module.embedding_configs())
        for config in tables:
            config.data_type = data_type

        table_name_to_quantized_weights: Dict[str, Tuple[Tensor, Tensor]] = {}
        device = quantize_state_dict(module, table_name_to_quantized_weights, data_type)

        return cls(
            table_name_to_quantized_weights,
            tables,
            device=device,
            need_indices=module.need_indices(),
        )

    def named_buffers(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        state_dict = self.state_dict(prefix=prefix, keep_vars=True)
        for key, value in state_dict.items():
            yield key, value

    def _get_name(self) -> str:
        return "QuantizedEmbeddingCollection"

    def need_indices(self) -> bool:
        return self._need_indices

    def embedding_dim(self) -> int:
        return self._embedding_dim

    def embedding_configs(self) -> List[EmbeddingConfig]:
        return self._embedding_configs

    def embedding_names_by_table(self) -> List[List[str]]:
        return self._embedding_names_by_table

    def output_dtype(self) -> torch.dtype:
        return self._output_dtype
