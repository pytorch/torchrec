# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
import sys
from typing import Dict, Iterator, List, Optional, Union

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torchrec.modules.embedding_configs import EmbeddingBagConfig, EmbeddingConfig
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
    ManagedCollisionModule,
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
        tables: List[EmbeddingConfig],
        input_hash_size: int,
        device: torch.device,
        world_size: int,
        eviction_interval: int = 1,
        allow_in_place_embed_weight_update: bool = False,
        zch_method: str = "",  # method for managing collisions, one of ["", "mpzch", "sort_zch"]
        mpzch_num_buckets: Optional[int] = 80,
        mpzch_max_probe: Optional[
            int
        ] = 100,  # max_probe for HashZchManagedCollisionModule
    ) -> None:
        """
        INIT_DOC_STRING
        """
        super().__init__()
        # create ec from table configs
        ec = EmbeddingCollection(tables=tables, device=torch.device("meta"))
        # build dictionary for {table_name: table_config}
        mc_modules = {}
        for table_config in ec.embedding_configs():
            table_name = table_config.name
            if zch_method == "mpzch":
                # if use MPZCH, create a HashZchManagedCollisionModule
                mc_modules[table_name] = HashZchManagedCollisionModule(  # MPZCH
                    is_inference=False,
                    zch_size=(table_config.num_embeddings),
                    input_hash_size=input_hash_size,
                    device=device,
                    total_num_buckets=(
                        mpzch_num_buckets if mpzch_num_buckets else world_size
                    ),  # total_num_buckets if not passed, use world_size, WORLD_SIZE should be a factor of total_num_buckets
                    eviction_policy_name=HashZchEvictionPolicyName.SINGLE_TTL_EVICTION,  # defaultly using single ttl eviction policy
                    eviction_config=HashZchEvictionConfig(
                        features=table_config.feature_names,
                        single_ttl=eviction_interval,
                    ),
                )
            elif (
                zch_method == "sort_zch"
            ):  # if not use MPZCH, create a MCHManagedCollisionModule using the sort ZCH algorithm
                mc_modules[table_name] = MCHManagedCollisionModule(  # sort ZCH
                    zch_size=table_config.num_embeddings,
                    device=device,
                    input_hash_size=input_hash_size,
                    eviction_interval=eviction_interval,
                    eviction_policy=DistanceLFU_EvictionPolicy(),
                )  # NOTE: the benchmark for sort ZCH is not implemented yet
            else:  # if not use MPZCH, create a MCHManagedCollisionModule using the sort ZCH
                raise NotImplementedError(
                    f"zc method {zch_method} is not supported yet"
                )
        # create the mcebc module with the mc modules and the original ebc
        self.mc_embedding_collection = (
            ManagedCollisionEmbeddingCollection(  # ZCH or not
                embedding_collection=ec,
                managed_collision_collection=ManagedCollisionCollection(
                    managed_collision_modules=mc_modules,
                    embedding_configs=ec.embedding_configs(),
                ),
                allow_in_place_embed_weight_update=allow_in_place_embed_weight_update,
                return_remapped_features=False,  # not return remapped features
            )
        )
        self.remapped_ids: Optional[Dict[str, torch.Tensor]] = (
            None  # to store remapped ids
        )

    def forward(self, input: KeyedJaggedTensor) -> Dict[str, JaggedTensor]:
        """
        Args:
            input (KeyedJaggedTensor): KJT of form [F X B X L].
        Returns:
            Dict[str, JaggedTensor]: dictionary of {'feature_name': JaggedTensor}
        """
        mc_ec_out, remapped_ids = self.mc_embedding_collection(input)
        self.remapped_ids = remapped_ids
        return mc_ec_out

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        # only return the parameters of the original EmbeddingBagCollection, not _managed_collision_collection modules
        return self.mc_embedding_collection._embedding_module.parameters(
            recurse=recurse
        )

    def embedding_bag_configs(self) -> List[EmbeddingConfig]:
        """
        Returns:
            Dict[str, EmbeddingConfig]: dictionary of {'feature_name': EmbeddingConfig}
        """
        # pyre-ignore [29] # NOTE: the function "embedding_configs" returns the _embedding_module attribute of the EmbeddingCollection
        return self.mc_embedding_collection._embedding_module.embedding_configs()


class McEmbeddingBagCollectionAdapter(nn.Module):
    """
    Managed Collision Embedding Collection Adapter
    The adapter to convert exiting EmbeddingCollection to Managed Collision Embedding Collection module
    The adapter will use the original EmbeddingCollection table but will pass input
    """

    def __init__(
        self,
        tables: List[EmbeddingBagConfig],
        input_hash_size: int,
        device: torch.device,
        world_size: int,
        eviction_interval: int = 1,
        allow_in_place_embed_weight_update: bool = False,
        use_mpzch: bool = False,
        mpzch_num_buckets: Optional[int] = None,
        mpzch_max_probe: Optional[int] = None,
    ) -> None:
        """
        Initialize an EmbeddingBagCollectionAdapter.
        Parameters:
            tables (List[EmbeddingBagConfig]): List of EmbeddingBagConfig. Should be the same as the original EmbeddingBagCollection.
            input_hash_size (int): the upper bound of input feature values
            device (torch.device): the device to use
            world_size (int): the world size
            eviction_interval (int): the eviction interval, default to 1 hour
            allow_in_place_embed_weight_update (bool): whether to allow in-place embedding weight update
            use_mpzch (bool): whether to use MPZCH or not # TODO: change this to a str to support different zch
            mpzch_num_buckets (Optional[int]): the number of buckets for MPZCH # TODO: change this to a config dict to support different zch configs
            mpzch_max_probe (Optional[int]): the maximum probe range starting from the input feature value's hash index
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
                # if use MPZCH, create a HashZchManagedCollisionModule
                mc_modules[table_name] = HashZchManagedCollisionModule(  # MPZCH
                    is_inference=False,
                    zch_size=(table_config.num_embeddings),
                    input_hash_size=input_hash_size,
                    device=device,
                    total_num_buckets=(
                        mpzch_num_buckets if mpzch_num_buckets else world_size
                    ),  # total_num_buckets if not passed, use world_size, WORLD_SIZE should be a factor of total_num_buckets
                    eviction_policy_name=HashZchEvictionPolicyName.SINGLE_TTL_EVICTION,  # defaultly using single ttl eviction policy
                    eviction_config=HashZchEvictionConfig(
                        features=table_config.feature_names,
                        single_ttl=eviction_interval,
                    ),
                    max_probe=(
                        mpzch_max_probe
                        if mpzch_max_probe is not None
                        and mpzch_max_probe
                        < (table_config.num_embeddings // world_size)
                        else table_config.num_embeddings // world_size
                    ),  # max_probe for HashZchManagedCollisionModule
                )
            else:  # if not use MPZCH, create a MCHManagedCollisionModule using the sort ZCH algorithm
                mc_modules[table_name] = MCHManagedCollisionModule(  # sort ZCH
                    zch_size=table_config.num_embeddings,
                    device=device,
                    input_hash_size=input_hash_size,
                    eviction_interval=eviction_interval,
                    eviction_policy=DistanceLFU_EvictionPolicy(),
                )  # NOTE: the benchmark for sort ZCH is not implemented yet
            # TODO: add the pure hash module here

        # create the mcebc module with the mc modules and the original ebc
        self.mc_embedding_bag_collection = (
            ManagedCollisionEmbeddingBagCollection(  # ZCH or not
                embedding_bag_collection=ebc,
                managed_collision_collection=ManagedCollisionCollection(
                    managed_collision_modules=mc_modules,
                    embedding_configs=ebc.embedding_bag_configs(),
                ),
                allow_in_place_embed_weight_update=allow_in_place_embed_weight_update,
                return_remapped_features=False,  # not return remapped features
            )
        )

        self.remapped_ids: Optional[Dict[str, torch.Tensor]] = (
            None  # to store remapped ids
        )

    def forward(self, input_kjt: KeyedJaggedTensor) -> Dict[str, JaggedTensor]:
        """
        Args:
            input (KeyedJaggedTensor): KJT of form [F X B X L].
        Returns:
            Dict[str, JaggedTensor]: dictionary of {'feature_name': JaggedTensor}
        """
        mc_ebc_out, remapped_ids = self.mc_embedding_bag_collection(input_kjt)
        self.remapped_ids = remapped_ids
        return mc_ebc_out

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        # only return the parameters of the original EmbeddingBagCollection, not _managed_collision_collection modules
        return self.mc_embedding_bag_collection._embedding_module.parameters(
            recurse=recurse
        )

    def embedding_bag_configs(self) -> List[EmbeddingBagConfig]:
        """
        Returns:
            Dict[str, EmbeddingConfig]: dictionary of {'feature_name': EmbeddingConfig}
        """
        return (
            # pyre-ignore [29] # NOTE: the function "embedding_configs" returns the _embedding_module attribute of the EmbeddingCollection
            self.mc_embedding_bag_collection._embedding_module.embedding_bag_configs()
        )
