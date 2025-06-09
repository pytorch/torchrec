# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
import sys
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)
from torchrec.modules.hash_mc_evictions import (
    HashZchEvictionConfig,
    HashZchEvictionPolicyName,
)
from torchrec.modules.hash_mc_modules import HashZchManagedCollisionModule
from torchrec.modules.mc_embedding_modules import (
    ManagedCollisionEmbeddingBagCollection,
    ManagedCollisionEmbeddingCollection,
)
from torchrec.modules.mc_modules import (
    DistanceLFU_EvictionPolicy,
    ManagedCollisionCollection,
    MCHManagedCollisionModule,
)
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor


class McEmbeddingCollectionAdapter(nn.Module):
    """
    Managed Collision Embedding Collection Adapter
    The adapter to convert exiting EmbeddingCollection to Managed Collision Embedding Collection module
    The adapter will use the original EmbeddingCollection table but will pass input
    """

    def __init__(
        self,
        embedding_collection: EmbeddingCollection,
        input_hash_size: int,
        device: torch.device,
        eviction_interval: int = 2,
        allow_in_place_embed_weight_update: bool = False,
    ) -> None:
        """
        INIT_DOC_STRING
        """
        super().__init__()
        # build dictionary for {table_name: table_config}
        mc_modules = {}
        for table_name, table_config in embedding_collection.embedding_configs():
            mc_modules[table_name] = MCHManagedCollisionModule(
                zch_size=table_config.num_embeddings,
                device=device,
                input_hash_size=input_hash_size,
                eviction_interval=eviction_interval,
                eviction_policy=DistanceLFU_EvictionPolicy(),
            )
        self.mc_embedding_collection = ManagedCollisionEmbeddingCollection(
            embedding_collection=embedding_collection,
            managed_collision_collection=ManagedCollisionCollection(
                managed_collision_modules=mc_modules,
                embedding_configs=embedding_collection.embedding_configs(),
            ),
            allow_in_place_embed_weight_update=allow_in_place_embed_weight_update,
            return_remapped_features=True,  # not return remapped features
        )
        self.remapped_ids = None  # to store remapped ids

    def forward(self, input: KeyedJaggedTensor) -> Dict[str, JaggedTensor]:
        """
        Args:
            input (KeyedJaggedTensor): KJT of form [F X B X L].
        Returns:
            Dict[str, JaggedTensor]: dictionary of {'feature_name': JaggedTensor}
        """
        mc_ec_out, remapped_ids = self.mc_embedding_collection(input)
        self.remapped_ids = remapped_ids
        return mc_ec_out[0]


class McEmbeddingBagCollectionAdapter(nn.Module):
    """
    Managed Collision Embedding Collection Adapter
    The adapter to convert exiting EmbeddingCollection to Managed Collision Embedding Collection module
    The adapter will use the original EmbeddingCollection table but will pass input
    """

    def __init__(
        self,
        tables: List[EmbeddingConfig],
        input_hash_size: int,
        device: torch.device,
        eviction_interval: int = 2,
        allow_in_place_embed_weight_update: bool = False,
        use_mpzch: bool = False,
    ) -> None:
        """
        INIT_DOC_STRING
        """
        # super().__init__(tables=tables, device=device)
        super().__init__()
        # create ebc from table configs
        ebc = EmbeddingBagCollection(tables=tables, device=torch.device("meta"))
        # build dictionary for {table_name: table_config}
        mc_modules = {}

        for table_config in ebc.embedding_bag_configs():
            table_name = table_config.name
            if use_mpzch:
                mc_modules[table_name] = HashZchManagedCollisionModule(  # MPZCH
                    is_inference=False,
                    zch_size=(table_config.num_embeddings),
                    input_hash_size=input_hash_size,
                    device=device,
                    total_num_buckets=4,
                    eviction_policy_name=HashZchEvictionPolicyName.SINGLE_TTL_EVICTION,
                    eviction_config=HashZchEvictionConfig(
                        features=table_config.feature_names,
                        single_ttl=eviction_interval,
                    ),
                )
            else:
                mc_modules[table_name] = MCHManagedCollisionModule(  # sort ZCH
                    zch_size=table_config.num_embeddings,
                    device=device,
                    input_hash_size=input_hash_size,
                    eviction_interval=eviction_interval,
                    eviction_policy=DistanceLFU_EvictionPolicy(),
                )

        # self._managed_collision_collection = ManagedCollisionCollection(
        #     managed_collision_modules=mc_modules,
        #     embedding_configs=self.embedding_bag_configs(),
        # )
        # if use_zch:
        # super().__init__(
        #     embedding_bag_collection=embedding_bag_collection,
        #     managed_collision_collection=ManagedCollisionCollection(
        #         managed_collision_modules=mc_modules,
        #         embedding_configs=embedding_bag_collection.embedding_bag_configs(),
        #     ),
        #     allow_in_place_embed_weight_update=allow_in_place_embed_weight_update,
        #     return_remapped_features=False,  # not return remapped features
        # )
        self.mc_embedding_bag_collection = (
            ManagedCollisionEmbeddingBagCollection(  # ZCH or not
                embedding_bag_collection=ebc,
                managed_collision_collection=ManagedCollisionCollection(
                    managed_collision_modules=mc_modules,
                    embedding_configs=ebc.embedding_bag_configs(),
                ),
                allow_in_place_embed_weight_update=allow_in_place_embed_weight_update,
                return_remapped_features=True,  # not return remapped features
            )
        )
        # else:
        #     self.mc_embedding_bag_collection = embedding_bag_collection  # MPZCH

        self.per_table_remapped_id_list = (
            []
        )  # list of dictionary {feature_name: remapped JT}

        self.num_queries = 0  # record total number of queries received

    def forward(self, input_kjt: KeyedJaggedTensor) -> Dict[str, JaggedTensor]:
        """
        Args:
            input (KeyedJaggedTensor): KJT of form [F X B X L].
        Returns:
            Dict[str, JaggedTensor]: dictionary of {'feature_name': JaggedTensor}
        """
        # input_kjt = self._managed_collision_collection(input_kjt)
        # mc_ebc_out = super().forward(input_kjt)
        mc_ebc_out, per_table_remapped_id = self.mc_embedding_bag_collection(input_kjt)
        self.per_table_remapped_id_list.append(per_table_remapped_id)
        return mc_ebc_out

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        # only return the parameters of the original EmbeddingBagCollection, not _managed_collision_collection modules
        return self.mc_embedding_bag_collection._embedding_module.parameters(
            recurse=recurse
        )

    def embedding_bag_configs(self) -> List[EmbeddingConfig]:
        """
        Returns:
            Dict[str, EmbeddingConfig]: dictionary of {'feature_name': EmbeddingConfig}
        """
        return (
            self.mc_embedding_bag_collection._embedding_module.embedding_bag_configs()
        )

    def record_remapped_ids(self, remapped_ids: Dict[str, JaggedTensor]) -> None:
        """
        Args:
            remapped_ids (Dict[str, JaggedTensor]): dictionary of {'feature_name': JaggedTensor}
        """
        self.remapped_ids = remapped_ids

    # def embedding_bag_configs(self) -> List[EmbeddingConfig]:
    #     """
    #     Returns:
    #         Dict[str, EmbeddingConfig]: dictionary of {'feature_name': EmbeddingConfig}
    #     """
    #     # return (
    #     #     self.mc_embedding_bag_collection._embedding_module.embedding_bag_configs()
    #     # )
    #     return self._embedding_module.embedding_bag_configs()
