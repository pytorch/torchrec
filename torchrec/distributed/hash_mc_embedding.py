#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import logging as logger
from collections import defaultdict
from typing import Dict, List

import torch
from torchrec.distributed.quant_state import WeightSpec
from torchrec.distributed.types import ShardingType
from torchrec.modules.hash_mc_modules import HashZchManagedCollisionModule


def sharded_zchs_buffers_spec(
    sharded_model: torch.nn.Module,
) -> Dict[str, WeightSpec]:
    # OUTPUT:
    # Example:
    # "main_module.module.ec_in_task_arch_hash._decoupled_embedding_collection._mcec_lookup.0.0._mcc_remapper.zchs.viewer_rid_duplicate._hash_zch_identities", [0, 0], [500, 1])
    # "main_module.module.ec_in_task_arch_hash._decoupled_embedding_collection._mcec_lookup.0.1._mcc_remapper.zchs.viewer_rid_duplicate._hash_zch_identities", [500, 0], [1000, 1])

    # 'main_module.module.ec_in_task_arch_hash._decoupled_embedding_collection._mcec_lookup.0.0._mcc_remapper.zchs.viewer_rid_duplicate._hash_zch_identities': WeightSpec(fqn='main_module.module.ec_in_task_arch_hash._ d_embedding_collection._managed_collision_collection.viewer_rid_duplicate._hash_zch_identities'
    def _get_table_names(
        sharded_module: torch.nn.Module,
    ) -> List[str]:
        table_names: List[str] = []
        for _, module in sharded_module.named_modules():
            type_name: str = type(module).__name__
            if "ShardedMCCRemapper" in type_name:
                for table_name in module._tables:
                    if table_name not in table_names:
                        table_names.append(table_name)
        return table_names

    def _get_unsharded_fqn_identities(
        sharded_module: torch.nn.Module,
        fqn: str,
        table_name: str,
    ) -> str:
        for module_fqn, module in sharded_module.named_modules():
            type_name: str = type(module).__name__
            if "ManagedCollisionCollection" in type_name:
                if table_name in module._table_to_features:
                    return f"{fqn}.{module_fqn}._managed_collision_modules.{table_name}.{HashZchManagedCollisionModule.IDENTITY_BUFFER}"
        logger.info(f"did not find table {table_name} in module {fqn}")
        return ""

    ret: Dict[str, WeightSpec] = defaultdict()
    for module_fqn, module in sharded_model.named_modules():
        type_name: str = type(module).__name__
        if "ShardedQuantManagedCollisionEmbeddingCollection" in type_name:
            sharding_type = ShardingType.ROW_WISE.value
            table_name_to_unsharded_fqn_identities: Dict[str, str] = {}
            for subfqn, submodule in module.named_modules():
                type_name: str = type(submodule).__name__
                if "ShardedMCCRemapper" in type_name:
                    for table_name in submodule.zchs.keys():
                        # identities tensor has only one column
                        shard_offsets: List[int] = [
                            submodule._shard_metadata[table_name][0],
                            0,
                        ]
                        shard_sizes: List[int] = [
                            submodule._shard_metadata[table_name][1],
                            1,
                        ]
                        if table_name not in table_name_to_unsharded_fqn_identities:
                            table_name_to_unsharded_fqn_identities[table_name] = (
                                _get_unsharded_fqn_identities(
                                    module, module_fqn, table_name
                                )
                            )
                        unsharded_fqn_identities: str = (
                            table_name_to_unsharded_fqn_identities[table_name]
                        )
                        # subfqn contains the index of sharding, so no need to add it specifically here
                        sharded_fqn_identities: str = (
                            f"{module_fqn}.{subfqn}.zchs.{table_name}.{HashZchManagedCollisionModule.IDENTITY_BUFFER}"
                        )
                        ret[sharded_fqn_identities] = WeightSpec(
                            fqn=unsharded_fqn_identities,
                            shard_offsets=shard_offsets,
                            shard_sizes=shard_sizes,
                            sharding_type=sharding_type,
                        )
    return ret
