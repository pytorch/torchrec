#!/usr/bin/env python3

from typing import List, Dict, Optional

import torch
import torch.nn as nn
from torchrec.modules.embedding_configs import (
    DataType,
    EmbeddingConfig,
    EmbeddingBagConfig,
    PoolingType,
)
from torchrec.sparse.jagged_tensor import (
    KeyedJaggedTensor,
    JaggedTensor,
    KeyedTensor,
)


def _to_mode(pooling: PoolingType) -> str:
    if pooling == PoolingType.SUM:
        return "sum"
    elif pooling == PoolingType.MEAN:
        return "mean"
    else:
        raise ValueError(f"Unsupported pooling {pooling}")


class EmbeddingBagCollection(nn.Module):
    """
    EmbeddingBagCollection represents a collection of pooled embeddings (EmbeddingBags)
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
        tables (List[EmbeddingBagConfig]): list of embedding tables
        is_weighted: (bool): whether input KeyedJaggedTensor is weighted
        device: (Optional[torch.device]): default compute device

    Call Args:
        features: KeyedJaggedTensor,
        weighted_features: KeyedJaggedTensor,

    Returns:
        KeyedTensor

    Example:
        table_0 = EmbeddingBagConfig(
            name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1"]
        )
        table_1 = EmbeddingBagConfig(
            name="t2", embedding_dim=4, num_embeddings=10, feature_names=["f2"]
        )

        ebc = EmbeddingBagCollection(tables=[table_0, table_1])

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

        pooled_embeddings = ebc(features)
        print(pooled_embeddings.values())
        tensor([[-0.6149,  0.0000, -0.3176],
        [-0.8876,  0.0000, -1.5606],
        [ 1.6805,  0.0000,  0.6810],
        [-1.4206, -1.0409,  0.2249],
        [ 0.1823, -0.4697,  1.3823],
        [-0.2767, -0.9965, -0.1797],
        [ 0.8864,  0.1315, -2.0724]], grad_fn=<TransposeBackward0>)
        print(pooled_embeddings.keys())
        ['f1', 'f2']
        print(pooled_embeddings.offset_per_key())
        tensor([0, 3, 7])
    """

    def __init__(
        self,
        tables: List[EmbeddingBagConfig],
        is_weighted: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torchrec.modules.{self.__class__.__name__}")
        self.is_weighted = is_weighted
        # pyre-ignore[11]
        self.embedding_bags: nn.ModuleDict = nn.ModuleDict()
        self.embedding_bag_configs = tables
        self._embedding_names: List[str] = []
        self._lengths_per_embedding: List[int] = []
        table_names = set()
        shared_feature: Dict[str, bool] = {}
        for embedding_config in tables:
            if embedding_config.name in table_names:
                raise ValueError(f"Duplicate table name {embedding_config.name}")
            table_names.add(embedding_config.name)
            dtype = (
                torch.float32
                if embedding_config.data_type == DataType.FP32
                else torch.float16
            )
            self.embedding_bags[embedding_config.name] = nn.EmbeddingBag(
                num_embeddings=embedding_config.num_embeddings,
                embedding_dim=embedding_config.embedding_dim,
                mode=_to_mode(embedding_config.pooling),
                device=device,
                include_last_offset=True,
                dtype=dtype,
            )
            if not embedding_config.feature_names:
                embedding_config.feature_names = [embedding_config.name]
            for feature_name in embedding_config.feature_names:
                if feature_name not in shared_feature:
                    shared_feature[feature_name] = False
                else:
                    shared_feature[feature_name] = True
                self._lengths_per_embedding.append(embedding_config.embedding_dim)

        for embedding_config in tables:
            for feature_name in embedding_config.feature_names:
                if shared_feature[feature_name]:
                    self._embedding_names.append(
                        feature_name + "@" + embedding_config.name
                    )
                else:
                    self._embedding_names.append(feature_name)

    def forward(self, features: KeyedJaggedTensor) -> KeyedTensor:
        pooled_embeddings: List[torch.Tensor] = []

        for embedding_config, embedding_bag in zip(
            self.embedding_bag_configs, self.embedding_bags.values()
        ):
            for feature_name in embedding_config.feature_names:
                f = features[feature_name]
                if self.is_weighted:
                    res = embedding_bag(
                        input=f.values(),
                        offsets=f.offsets(),
                        per_sample_weights=f.weights(),
                    )
                else:
                    res = embedding_bag(
                        input=f.values(),
                        offsets=f.offsets(),
                    )
                pooled_embeddings.append(res)
        data = torch.cat(pooled_embeddings, dim=1)
        return KeyedTensor(
            keys=self._embedding_names,
            values=data,
            length_per_key=self._lengths_per_embedding,
        )


class EmbeddingCollection(nn.Module):
    """
    EmbeddingCollection represents a collection of non-pooled embeddings
    It processes sparse data in the form of KeyedJaggedTensor
    of the form [F X B X L]
    F: features (keys)
    B: batch size
    L: Length of sparse features (variable)

    and outputs Dict[feature (key), JaggedTensor].
    Each JaggedTensor contains values of the form (B * L) X D
    where
    B: batch size
    L: Length of sparse features (jagged)
    D: each feature's (key's) embedding dimension
    and lengths are of the form L

    Constructor Args:
        tables (List[EmbeddingBagConfig]): list of embedding tables
        device: (Optional[torch.device]): default compute device

    Call Args:
        features: KeyedJaggedTensor,

    Returns:
        Dict[str, JaggedTensor]

    Example:
        >>> e1_config = EmbeddingConfig(
            name="t1", embedding_dim=2, num_embeddings=10, feature_names=["f1"]
        )
        e2_config = EmbeddingConfig(
            name="t2", embedding_dim=3, num_embeddings=10, feature_names=["f2"]
        )
        ec_config = EmbeddingCollectionConfig(tables=[e1_config, e2_config])

        ec = EmbeddingCollection(config=ec_config)

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

    def __init__(
        self,
        tables: List[EmbeddingConfig],
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torchrec.modules.{self.__class__.__name__}")
        self.embeddings: nn.ModuleDict = nn.ModuleDict()
        self.embedding_configs = tables
        self._embedding_names: List[str] = []
        table_names = set()
        shared_feature: Dict[str, bool] = {}
        for embedding_config in tables:
            if embedding_config.name in table_names:
                raise ValueError(f"Duplicate table name {embedding_config.name}")
            table_names.add(embedding_config.name)
            self.embeddings[embedding_config.name] = nn.Embedding(
                num_embeddings=embedding_config.num_embeddings,
                embedding_dim=embedding_config.embedding_dim,
                device=device,
            )
            if not embedding_config.feature_names:
                embedding_config.feature_names = [embedding_config.name]
            for feature_name in embedding_config.feature_names:
                if feature_name not in shared_feature:
                    shared_feature[feature_name] = False
                else:
                    shared_feature[feature_name] = True

        for embedding_config in tables:
            for feature_name in embedding_config.feature_names:
                if shared_feature[feature_name]:
                    self._embedding_names.append(
                        feature_name + "@" + embedding_config.name
                    )
                else:
                    self._embedding_names.append(feature_name)

    def forward(self, features: KeyedJaggedTensor) -> Dict[str, JaggedTensor]:
        feature_embeddings: Dict[str, JaggedTensor] = {}
        idx = 0
        for embedding_config, embedding in zip(
            self.embedding_configs, self.embeddings.values()
        ):
            for feature_name in embedding_config.feature_names:
                f = features[feature_name]
                lookup = embedding(
                    input=f.values(),
                )
                feature_embeddings[self._embedding_names[idx]] = JaggedTensor(
                    values=lookup,
                    offsets=f.offsets(),
                    lengths=f.lengths(),
                )
                idx += 1
        return feature_embeddings
