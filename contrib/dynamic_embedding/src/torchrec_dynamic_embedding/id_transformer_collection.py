from typing import List, Tuple, Union

import torch
from torchrec import EmbeddingBagConfig, EmbeddingConfig, KeyedJaggedTensor

from .id_transformer import IDTransformer, TensorList
from .ps import PSCollection


__all__ = []


class IDTransformerCollection:
    def __init__(
        self,
        tables: Union[List[EmbeddingConfig], List[EmbeddingBagConfig]],
        eviction_config=None,
        transform_config=None,
        ps_collection: PSCollection = None,
    ):
        """
        IDTransformerCollection could transform the input of a `Embedding(Bag)Collection`.
        It contains the `IDTransformer` of tables in the
        `Embedding(Bag)Collection`.

        Args:
            tables: list of `Embedding(Bag)Config` or `EmbeddingBagConfig` one passed to
                `Embedding(Bag)Collection`.
            eviction_config: config of the eviction strategy for IDTransformers.
            transformer_config: config of the transform strategy for IDTransformers.
            ps_collection: `PSCollection` of the collection, if `None`, won't do eviction or fetch.
                By default, IDTransformerCollection will evict half the ids when full.
        """
        self._configs = tables
        self._ps_collection = ps_collection

        self._transformers = []
        self._table_names = []
        feature_names = set()
        for config in tables:
            if config.name in self._table_names:
                raise ValueError(f"Duplicate table name {config.name}")
            if not config.feature_names:
                config.feature_names = [config.name]
            self._table_names.append(config.name)
            for feature_name in config.feature_names:
                if feature_name in feature_names:
                    raise ValueError(f"Shared feature not allowed yet.")
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
        self._ever_evicted = False
        self._time = 0

    def transform(
        self, global_features: KeyedJaggedTensor
    ) -> Tuple[KeyedJaggedTensor, List[torch.classes.tde.FetchHandle]]:
        """
        Transform global kjts into local kjts.

        Return:
            KeyedJaggedTensor: the transformed kjt.
            List[torch.classes.tde.FetchHandle]: list of fetch handles to wait.
        """
        global_values = global_features.values()
        cache_values = torch.empty_like(global_values)

        global_feature_indices = {
            feature_name: i for i, feature_name in enumerate(global_features.keys())
        }
        offset_per_key = global_features.offset_per_key()

        fetch_handles = []
        for i, transformer in enumerate(self._transformers):
            feature_names = self._feature_names[i]
            feature_indices = [
                global_feature_indices[feature_name] for feature_name in feature_names
            ]
            global_ids = [
                global_values[offset_per_key[idx] : offset_per_key[idx + 1]]
                for idx in feature_indices
            ]
            cache_ids = [
                cache_values[offset_per_key[idx] : offset_per_key[idx + 1]]
                for idx in feature_indices
            ]

            result = transformer.transform(
                TensorList(global_ids), TensorList(cache_ids), self._time
            )
            if self._ps_collection is not None:
                table_name = self._table_names[i]
                ps = self._ps_collection[table_name]
                if result.ids_to_fetch.numel() > 0:
                    handle = ps.fetch(
                        result.ids_to_fetch,
                        self._time,
                        self._ever_evicted,
                        self._configs[i].get_weight_init_min(),
                        self._configs[i].get_weight_init_max(),
                    )
                    fetch_handles.append(handle)
                if not result.success:
                    # TODO(zilinzhu): make this configurable
                    ids_to_evict = transformer.evict(transformer._num_embedding // 2)
                    ps.evict(ids_to_evict)
                    self._ever_evicted = True

                    # retry after eviction.
                    result = transformer.transform(
                        TensorList(global_ids), TensorList(cache_ids), self._time
                    )
                    if not result.success:
                        raise RuntimeError(
                            "Failed to transform global ids after eviction. "
                            f"Maybe the num_embedding of table {table_name} is too small?"
                        )
                    if result.ids_to_fetch is not None:
                        fetch_handles.append(
                            ps.fetch(
                                result.ids_to_fetch,
                                self._time,
                                self._ever_evicted,
                                self._configs[i].get_weight_init_min(),
                                self._configs[i].get_weight_init_max(),
                            )
                        )

        cache_values = KeyedJaggedTensor(
            keys=global_features.keys(),
            values=cache_values,
            lengths=global_features.lengths(),
            weights=global_features.weights_or_none(),
        )
        self._time += 1
        return cache_values, fetch_handles

    def save(self):
        if self._ps_collection is None:
            return
        for i, transformer in enumerate(self._transformers):
            table_name = self._table_names[i]
            ids = transformer.save()
            self._ps_collection[table_name].evict(ids)
