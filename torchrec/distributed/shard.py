#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.distributed as dist
from torch import nn
from torchrec.distributed.comm import get_local_size
from torchrec.distributed.model_parallel import get_default_sharders
from torchrec.distributed.planner import EmbeddingShardingPlanner
from torchrec.distributed.planner.types import Topology
from torchrec.distributed.types import ModuleSharder, ShardingEnv, ShardingPlan


def _join_module_path(path: str, name: str) -> str:
    return (path + "." + name) if path else name


def shard(
    module: nn.Module,
    env: Optional[ShardingEnv] = None,
    device: Optional[torch.device] = None,
    plan: Optional[ShardingPlan] = None,
    sharders: Optional[List[ModuleSharder[torch.nn.Module]]] = None,
) -> Tuple[nn.Module, List[str]]:
    """
    Replaces all sub_modules that are embedding modules with their sharded variants. This embedding_module -> sharded_embedding_module mapping
    is derived from the passed in sharders.

    This will leave the other parts of the model unaffected.

    It returns the module (with replacements), as well as parameter names of the modules that were swapped out.

    Args:
        module (nn.Module): module to wrap.
        env (Optional[ShardingEnv]): sharding environment that has the process group.
        device (Optional[torch.device]): compute device, defaults to cpu.
        plan (Optional[ShardingPlan]): plan to use when sharding, defaults to
            `EmbeddingShardingPlanner.collective_plan()`.
        sharders (Optional[List[ModuleSharder[nn.Module]]]): `ModuleSharders` available
            to shard with, defaults to `EmbeddingBagCollectionSharder()`.

    Example::

        @torch.no_grad()
        def init_weights(m):
            if isinstance(m, nn.Linear):
                m.weight.fill_(1.0)
            elif isinstance(m, EmbeddingBagCollection):
                for param in m.parameters():
                    init.kaiming_normal_(param)

        m = MyModel(device='meta')
        m = shard(m)
        assert isinstance(m.embedding_bag_collection, ShardedEmbeddingBagCollection)
    """

    if sharders is None:
        sharders = get_default_sharders()

    if env is None:
        pg = dist.GroupMember.WORLD
        assert pg is not None, "Process group is not initialized"
        env = ShardingEnv.from_process_group(pg)

    if device is None:
        device = torch.device("cpu")

    sharder_map: Dict[Type[nn.Module], ModuleSharder[nn.Module]] = {
        sharder.module_type: sharder for sharder in sharders
    }

    if plan is None:
        planner = EmbeddingShardingPlanner(
            topology=Topology(
                local_world_size=get_local_size(env.world_size),
                world_size=env.world_size,
                compute_device=device.type,
            )
        )
        pg = env.process_group
        if pg is not None:
            plan = planner.collective_plan(module, sharders, pg)
        else:
            plan = planner.plan(module, sharders)

    sharded_param_names: List[str] = []

    if type(module) in sharder_map:
        # If the top level module is itself a shardable module, return the sharded variant.
        # Note, we cannot do an inplace replacement in this case.
        sharded_params = plan.get_plan_for_module("")
        if sharded_params is not None:
            sharded_module = sharder_map[type(module)].shard(
                module, sharded_params, env, device
            )
            sharded_param_names.extend([name for name, _ in module.named_parameters()])
            return sharded_module, sharded_param_names

    def _replace(_model: nn.Module, path: str = "") -> None:
        for child_name, child in _model.named_children():
            child_path = _join_module_path(path, child_name)
            if type(child) in sharder_map:
                assert plan is not None
                sharded_params = plan.get_plan_for_module(child_path)
                if sharded_params is not None:
                    sharded_module = sharder_map[type(child)].shard(
                        child, sharded_params, env, device
                    )
                    _model.register_module(
                        child_name,
                        sharded_module,
                    )

                    sharded_param_names.extend(
                        [
                            _join_module_path(child_path, name)
                            for name, _ in child.named_parameters()
                        ]
                    )
            else:
                _replace(child, child_path)

    _replace(module)

    return module, sharded_param_names
