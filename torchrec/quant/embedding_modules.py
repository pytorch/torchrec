#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from collections import OrderedDict
from typing import Dict, Any, Optional, List, Iterator, Tuple

import torch
import torch.nn as nn
from fbgemm_gpu.split_table_batched_embeddings_ops import (
    IntNBitTableBatchedEmbeddingBagsCodegen,
    EmbeddingLocation,
)
from torch import Tensor
from torchrec.modules.embedding_configs import (
    EmbeddingBagConfig,
    DataType,
    DATA_TYPE_NUM_BITS,
    data_type_to_sparse_type,
    dtype_to_data_type,
    pooling_type_to_pooling_mode,
)
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection as OriginalEmbeddingBagCollection,
)
from torchrec.modules.embedding_modules import EmbeddingBagCollectionInterface
from torchrec.sparse.jagged_tensor import (
    KeyedJaggedTensor,
    KeyedTensor,
)

try:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
except OSError:
    pass

# OSS
try:
    import fbgemm_gpu  # @manual # noqa
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
            scale_shift = torch.empty(
                (tensor.shape[0], 4),
                device="meta",
                # pyre-fixme[16]: Item `Tensor` of `Union[Tensor, Module]` has
                #  no attribute `weight`.
                dtype=module.qconfig.weight().dtype,
            )
        else:
            quant_res = torch.ops.fbgemm.FloatToFusedNBitRowwiseQuantizedSBHalf(
                tensor, num_bits
            )
            quant_weight, scale_shift = (
                quant_res[:, :-4],
                quant_res[:, -4:],
            )
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

    Constructor Args:
        table_name_to_quantized_weights (Dict[str, Tuple[Tensor, Tensor]]): map of tables to quantized weights
        embedding_configs (List[EmbeddingBagConfig]): list of embedding tables
        is_weighted: (bool): whether input KeyedJaggedTensor is weighted
        device: (Optional[torch.device]): default compute device

    Call Args:
        features: KeyedJaggedTensor,

    Returns:
        KeyedTensor

    Example:
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
    ) -> None:

        super().__init__()

        self._is_weighted = is_weighted
        self._embedding_bag_configs: List[EmbeddingBagConfig] = embedding_configs
        self.embedding_bags: nn.ModuleList = nn.ModuleList()
        self._embedding_names: List[str] = []
        self._lengths_per_embedding: List[int] = []
        shared_feature: Dict[str, bool] = {}
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
            )
            self.embedding_bags.append(emb_module)
            if not emb_config.feature_names:
                emb_config.feature_names = [emb_config.name]
            for feature_name in emb_config.feature_names:
                if feature_name not in shared_feature:
                    shared_feature[feature_name] = False
                else:
                    shared_feature[feature_name] = True
                self._lengths_per_embedding.append(emb_config.embedding_dim)

        for emb_config in self._embedding_bag_configs:
            for feature_name in emb_config.feature_names:
                if shared_feature[feature_name]:
                    self._embedding_names.append(feature_name + "@" + emb_config.name)
                else:
                    self._embedding_names.append(feature_name)

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
                    ).float()
                )

                length_per_key.append(emb_config.embedding_dim)

        return KeyedTensor(
            keys=self._embedding_names,
            values=torch.cat(pooled_embeddings, dim=1),
            length_per_key=self._lengths_per_embedding,
        )

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
        embedding_bag_configs = copy.deepcopy(module.embedding_bag_configs)
        for config in embedding_bag_configs:
            config.data_type = data_type

        table_name_to_quantized_weights: Dict[str, Tuple[Tensor, Tensor]] = {}
        device = quantize_state_dict(module, table_name_to_quantized_weights, data_type)

        return cls(
            table_name_to_quantized_weights,
            embedding_bag_configs,
            module.is_weighted,
            device=device,
        )

    @property
    def embedding_bag_configs(
        self,
    ) -> List[EmbeddingBagConfig]:
        return self._embedding_bag_configs

    @property
    def is_weighted(self) -> bool:
        return self._is_weighted
