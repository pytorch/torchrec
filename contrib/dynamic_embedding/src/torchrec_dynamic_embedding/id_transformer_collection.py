from typing import List, Union

import torch
from torchrec import EmbeddingBagConfig, EmbeddingConfig, KeyedJaggedTensor

from .id_transformer import IDTransformer


__all__ = ["IDTransformerCollection"]


class IDTransformerCollection:
    def __init__(
        self,
        tables: Union[List[EmbeddingBagConfig], List[EmbeddingConfig]],
        eviction_config=None,
        transform_config=None,
    ):
        self._configs = tables

        self._transformers = []
        table_names = set()
        feature_names = set()
        for config in tables:
            if config.name in table_names:
                raise ValueError(f"Duplicate table name {config.name}")
            if not config.feature_names:
                config.feature_names = [config.name]
            table_names.add(config.name)
            for feature_name in config.feature_names:
                if feature_name in feature_names:
                    raise ValueError(f"Shared feature not allowed yet.")
                feature_names
            self._transformers.append(
                IDTransformer(
                    num_embedding=config.num_embeddings,
                    eviction_config=eviction_config,
                    transform_config=transform_config,
                )
            )
        self._feature_names: List[List[str]] = [
            config.feature_names for config in tables
        ]

    def transform(self, global_features: KeyedJaggedTensor) -> KeyedJaggedTensor:
        global_values = global_features.values()
        cache_values = torch.empty_like(global_values)

        global_feature_indices = {
            feature_name: i for i, feature_name in enumerate(global_features.keys())
        }
        offset_per_key = global_features.offset_per_key()

        for i, transformer in enumerate(self._transformers):
            feature_names = self._feature_names[i]
            feature_indices = [
                global_feature_indices[feature_name] for feature_name in feature_names
            ]
            global_ids = torch.cat(
                [
                    global_values[offset_per_key[idx] : offset_per_key[idx + 1]]
                    for idx in feature_indices
                ]
            )
            cache_ids = torch.empty_like(global_ids)
            # TODO(zilinzhu) Do fetch and evict.
            num_transformed, fetch_ids = transformer.transform(global_ids, cache_ids)

            offset = 0
            for idx in feature_indices:
                length = offset_per_key[idx + 1] - offset_per_key[idx]
                cache_values[offset_per_key[idx] : offset_per_key[idx + 1]] = cache_ids[
                    offset : offset + length
                ]
                offset += length
        cache_values = KeyedJaggedTensor(
            keys=global_features.keys(),
            values=cache_values,
            lengths=global_features.lengths(),
            weights=global_features.weights_or_none(),
        )
        return cache_values
