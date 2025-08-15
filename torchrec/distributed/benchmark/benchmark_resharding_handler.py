#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import logging
import random
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torchrec.distributed.embeddingbag import EmbeddingBagCollection

from torchrec.distributed.sharding.dynamic_sharding import output_sharding_plan_delta

from torchrec.distributed.sharding_plan import (
    column_wise,
    construct_module_sharding_plan,
    table_wise,
)

from torchrec.distributed.test_utils.test_sharding import generate_rank_placements
from torchrec.distributed.types import EmbeddingModuleShardingPlan

logger: logging.Logger = logging.getLogger(__name__)


class ReshardingHandler:
    """
    Handles the resharding of a training module by generating and applying different sharding plans.
    """

    def __init__(self, train_module: nn.Module, num_plans: int) -> None:
        """
        Initializes the ReshardingHandler with a training module and the number of sharding plans to generate.

        Args:
            train_module (nn.Module): The training module to be resharded.
            num_plans (int): The number of sharding plans to generate.
        """
        self._train_module = train_module
        if not hasattr(train_module, "_module"):
            raise RuntimeError("Incorrect train module")

        if not hasattr(train_module._module, "plan"):
            raise RuntimeError("sharding plan cannot be found")

        # Pyre-ignore
        plan = train_module._module.plan.plan
        key = "main_module.sparse_arch.embedding_bag_collection"
        module = (
            # Pyre-ignore
            train_module._module.module.main_module.sparse_arch.embedding_bag_collection
        )
        self._resharding_plans: List[EmbeddingModuleShardingPlan] = []
        world_size = dist.get_world_size()

        if key in plan:
            ebc = plan[key]
            num_tables = len(ebc)
            ranks_per_tables = [1 for _ in range(num_tables)]
            ranks_per_tables_for_CW = []
            for index, table_config in enumerate(module._embedding_bag_configs):
                # CW sharding
                valid_candidates = [
                    i
                    for i in range(1, world_size + 1)
                    if table_config.embedding_dim % i == 0
                ]
                rng = random.Random(index)
                ranks_per_tables_for_CW.append(rng.choice(valid_candidates))

            for i in range(num_plans):
                new_ranks = generate_rank_placements(
                    world_size, num_tables, ranks_per_tables, i
                )
                new_ranks_cw = generate_rank_placements(
                    world_size, num_tables, ranks_per_tables_for_CW, i
                )
                new_per_param_sharding = {}
                for i, (talbe_id, param) in enumerate(ebc.items()):
                    if param.sharding_type == "column_wise":
                        cw_gen = column_wise(
                            ranks=new_ranks_cw[i],
                            compute_kernel=param.compute_kernel,
                        )
                        new_per_param_sharding[talbe_id] = cw_gen
                    else:
                        tw_gen = table_wise(
                            rank=new_ranks[i][0],
                            compute_kernel=param.compute_kernel,
                        )
                        new_per_param_sharding[talbe_id] = tw_gen

                lightweight_ebc = EmbeddingBagCollection(
                    tables=module._embedding_bag_configs,
                    device=torch.device(
                        "meta"
                    ),  # Use meta device to avoid actual memory allocation
                )

                meta_device = torch.device("meta")
                new_plan = construct_module_sharding_plan(
                    lightweight_ebc,
                    per_param_sharding=new_per_param_sharding,
                    local_size=world_size,
                    world_size=world_size,
                    # Pyre-ignore
                    device_type=meta_device,
                )
                self._resharding_plans.append(new_plan)
        else:
            raise RuntimeError(f"Plan does not have key: {key}")

    def step(self, batch_no: int) -> float:
        """
        Executes a step in the training process by selecting and applying a sharding plan.

        Args:
            batch_no (int): The current batch number.

        Returns:
            float: The data volume of the sharding plan delta.
        """
        # Pyre-ignore
        plan = self._train_module._module.plan.plan
        key = "main_module.sparse_arch.embedding_bag_collection"

        # Use the current step as a seed to ensure all ranks get the same random number
        # but it changes on each call

        rng = random.Random(batch_no)
        index = rng.randint(0, len(self._resharding_plans) - 1)
        logger.info(f"Selected resharding plan index {index} for step {batch_no}")
        # Get the selected plan
        selected_plan = self._resharding_plans[index]

        # Fix the device mismatch by updating the placement device in the sharding spec
        # This is necessary because the plan was created with meta device but needs to be applied on CUDA
        for _, param_sharding in selected_plan.items():
            sharding_spec = param_sharding.sharding_spec
            if sharding_spec is not None:
                # pyre-ignore
                for shard in sharding_spec.shards:
                    placement = shard.placement
                    rank: Optional[int] = placement.rank()
                    assert rank is not None
                    current_device = (
                        torch.cuda.current_device()
                        if rank == torch.distributed.get_rank()
                        else rank % torch.cuda.device_count()
                    )
                    shard.placement = torch.distributed._remote_device(
                        f"rank:{rank}/cuda:{current_device}"
                    )

        data_volume, delta_plan = output_sharding_plan_delta(
            plan[key], selected_plan, True
        )
        # Pyre-ignore
        self._train_module.module.reshard(
            sharded_module_fqn="main_module.sparse_arch.embedding_bag_collection",
            changed_shard_to_params=delta_plan,
        )
        return data_volume
