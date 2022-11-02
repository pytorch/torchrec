#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import itertools
from collections import defaultdict, OrderedDict
from typing import Any, Dict, List, Optional, Tuple

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
    PoolingType,
)
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection as OriginalEmbeddingBagCollection,
    EmbeddingBagCollectionInterface,
    EmbeddingCollection as OriginalEmbeddingCollection,
    EmbeddingCollectionInterface,
    get_embedding_names_by_table,
)
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor, KeyedTensor
from torchrec.types import ModuleNoCopyMixin

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
                dtype=torch.uint8,
            )
            if (
                data_type == DataType.INT8
                or data_type == DataType.INT4
                or data_type == DataType.INT2
            ):
                scale_shift = torch.empty(
                    (tensor.shape[0], 4),
                    device="meta",
                    dtype=torch.uint8,
                )
            else:
                scale_shift = None
        else:
            if tensor.dtype == torch.float or tensor.dtype == torch.float16:
                if data_type == DataType.FP16:
                    if tensor.dtype == torch.float:
                        tensor = tensor.half()
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


class EmbeddingBagCollection(EmbeddingBagCollectionInterface, ModuleNoCopyMixin):
    """
    EmbeddingBagCollection represents a collection of pooled embeddings (EmbeddingBags).
    This EmbeddingBagCollection is quantized for lower precision. It relies on fbgemm quantized ops and provides
    table batching.

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
        tables: List[EmbeddingBagConfig],
        is_weighted: bool,
        device: torch.device,
        output_dtype: torch.dtype = torch.float,
        table_name_to_quantized_weights: Optional[
            Dict[str, Tuple[Tensor, Tensor]]
        ] = None,
    ) -> None:
        super().__init__()
        self._is_weighted = is_weighted
        self._embedding_bag_configs: List[EmbeddingBagConfig] = tables
        self._key_to_tables: Dict[
            Tuple[PoolingType, DataType], List[EmbeddingBagConfig]
        ] = defaultdict(list)
        self._length_per_key: List[int] = []
        # Registering in a List instead of ModuleList because we want don't want them to be auto-registered.
        # Their states will be modified via self.embedding_bags
        self._emb_modules: List[nn.Module] = []
        self._output_dtype = output_dtype
        self._device: torch.device = device
        self._table_name_to_quantized_weights: Optional[
            Dict[str, Tuple[Tensor, Tensor]]
        ] = None

        table_names = set()
        for table in self._embedding_bag_configs:
            if table.name in table_names:
                raise ValueError(f"Duplicate table name {table.name}")
            table_names.add(table.name)
            self._length_per_key.extend(
                [table.embedding_dim] * len(table.feature_names)
            )
            key = (table.pooling, table.data_type)
            self._key_to_tables[key].append(table)

        self._sum_length_per_key: int = sum(self._length_per_key)

        location = (
            EmbeddingLocation.HOST if device.type == "cpu" else EmbeddingLocation.DEVICE
        )

        for key, emb_configs in self._key_to_tables.items():
            (pooling, data_type) = key
            embedding_specs = []
            weight_lists: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = (
                [] if table_name_to_quantized_weights else None
            )
            feature_table_map: List[int] = []

            for idx, table in enumerate(emb_configs):
                embedding_specs.append(
                    (
                        table.name,
                        table.num_embeddings,
                        table.embedding_dim,
                        data_type_to_sparse_type(data_type),
                        location,
                    )
                )
                if table_name_to_quantized_weights:
                    # pyre-ignore
                    weight_lists.append(table_name_to_quantized_weights[table.name])
                feature_table_map.extend([idx] * table.num_features())

            emb_module = IntNBitTableBatchedEmbeddingBagsCodegen(
                embedding_specs=embedding_specs,
                pooling_mode=pooling_type_to_pooling_mode(pooling),
                weight_lists=weight_lists,
                device=device,
                output_dtype=data_type_to_sparse_type(dtype_to_data_type(output_dtype)),
                row_alignment=16,
                feature_table_map=feature_table_map,
            )
            if device != torch.device("meta") and weight_lists is None:
                emb_module.initialize_weights()
            self._emb_modules.append(emb_module)

        self._embedding_names: List[str] = list(
            itertools.chain(*get_embedding_names_by_table(self._embedding_bag_configs))
        )
        # We map over the parameters from FBGEMM backed kernels to the canonical nn.EmbeddingBag
        # representation. This provides consistency between this class and the EmbeddingBagCollection
        # nn.Module API calls (state_dict, named_modules, etc)
        self.embedding_bags: nn.ModuleDict = nn.ModuleDict()
        for (_key, tables), emb_module in zip(
            self._key_to_tables.items(), self._emb_modules
        ):
            for embedding_config, (weight, _) in zip(
                tables, emb_module.split_embedding_weights(split_scale_shifts=False)
            ):
                self.embedding_bags[embedding_config.name] = torch.nn.Module()
                # register as a buffer so it's exposed in state_dict.
                # however, since this is only needed for inference, we do not need to expose it as part of parameters.
                # Additionally, we cannot expose uint8 weights as parameters due to autograd restrictions.
                self.embedding_bags[embedding_config.name].register_buffer(
                    "weight", weight
                )

    def forward(
        self,
        features: KeyedJaggedTensor,
    ) -> KeyedTensor:
        """
        Args:
            features (KeyedJaggedTensor): KJT of form [F X B X L].

        Returns:
            KeyedTensor
        """

        feature_dict = features.to_dict()
        embeddings = []

        # TODO ideally we can accept KJTs with any feature order. However, this will require an order check + permute, which will break torch.script.
        # Once torchsccript is no longer a requirement, we should revisit this.

        for emb_op, (_key, tables) in zip(
            self._emb_modules, self._key_to_tables.items()
        ):
            indices = []
            lengths = []
            offsets = []
            weights = []

            for table in tables:
                for feature in table.feature_names:
                    f = feature_dict[feature]
                    indices.append(f.values())
                    lengths.append(f.lengths())
                    if self._is_weighted:
                        weights.append(f.weights())

            indices = torch.cat(indices)
            lengths = torch.cat(lengths)

            offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
            if self._is_weighted:
                weights = torch.cat(weights)

            embeddings.append(
                emb_op(
                    indices=indices.int(),
                    offsets=offsets.int(),
                    per_sample_weights=weights if self._is_weighted else None,
                )
            )

        embeddings = torch.stack(embeddings).reshape(-1, self._sum_length_per_key)

        return KeyedTensor(
            keys=self._embedding_names,
            values=embeddings,
            length_per_key=self._length_per_key,
        )

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
            embedding_bag_configs,
            module.is_weighted(),
            device=device,
            # pyre-ignore [16]
            output_dtype=module.qconfig.activation().dtype,
            table_name_to_quantized_weights=table_name_to_quantized_weights,
        )

    def embedding_bag_configs(
        self,
    ) -> List[EmbeddingBagConfig]:
        return self._embedding_bag_configs

    def is_weighted(self) -> bool:
        return self._is_weighted

    def output_dtype(self) -> torch.dtype:
        return self._output_dtype

    @property
    def device(self) -> torch.device:
        return self._device


class EmbeddingCollection(EmbeddingCollectionInterface, ModuleNoCopyMixin):
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
        tables: List[EmbeddingConfig],
        device: torch.device,
        need_indices: bool = False,
        output_dtype: torch.dtype = torch.float,
        table_name_to_quantized_weights: Optional[
            Dict[str, Tuple[Tensor, Tensor]]
        ] = None,
    ) -> None:
        super().__init__()
        self.embeddings: nn.ModuleList = nn.ModuleList()
        self._embedding_configs = tables
        self._embedding_dim: int = -1
        self._need_indices: bool = need_indices
        self._output_dtype = output_dtype
        self._device = device

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
            weight_lists: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = (
                [] if table_name_to_quantized_weights else None
            )
            if table_name_to_quantized_weights:
                # pyre-ignore
                weight_lists.append(table_name_to_quantized_weights[config.name])
            emb_module = IntNBitTableBatchedEmbeddingBagsCodegen(
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
                weight_lists=weight_lists,
                device=device,
                output_dtype=data_type_to_sparse_type(dtype_to_data_type(output_dtype)),
                row_alignment=16,
            )
            if device != torch.device("meta") and weight_lists is None:
                emb_module.initialize_weights()

            self.embeddings.append(emb_module)

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
            tables,
            device=device,
            need_indices=module.need_indices(),
            table_name_to_quantized_weights=table_name_to_quantized_weights,
        )

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

    @property
    def device(self) -> torch.device:
        return self._device
