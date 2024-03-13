#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

import unittest
from typing import cast

import torch

from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.infer_utils import get_tbe_specs_from_sqebc
from torchrec.distributed.quant_embeddingbag import ShardedQuantEmbeddingBagCollection
from torchrec.distributed.shard import _shard_modules
from torchrec.distributed.sharding_plan import (
    construct_module_sharding_plan,
    table_wise,
)
from torchrec.distributed.tests.test_quant_model_parallel import (
    _quantize,
    TestQuantEBCSharder,
)
from torchrec.distributed.types import ShardingEnv, ShardingPlan, ShardingType
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection


class UtilsTest(unittest.TestCase):
    # pyre-ignore
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
    )
    def test_get_tbe_specs_from_sqebc(self) -> None:
        device = torch.device("cuda:0")

        num_features = 3

        tables = [
            EmbeddingBagConfig(
                num_embeddings=(i + 1) * 20,
                embedding_dim=(i + 1) * 10,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(num_features)
        ]

        model = torch.nn.Sequential(
            EmbeddingBagCollection(
                tables=tables,
                device=device,
            )
        )
        model.training = False

        quant_model = _quantize(
            model,
            inplace=True,
            output_type=torch.float,
            quant_state_dict_split_scale_bias=True,
        )

        sharder = TestQuantEBCSharder(
            sharding_type=ShardingType.TABLE_WISE.value,
            kernel_type=EmbeddingComputeKernel.QUANT.value,
            shardable_params=[f"table_{i}" for i in range(num_features)],
        )

        module_plan = construct_module_sharding_plan(
            quant_model[0],
            per_param_sharding={
                "table_0": table_wise(rank=1),
                "table_1": table_wise(rank=0),
                "table_2": table_wise(rank=0),
            },
            # pyre-ignore
            sharder=sharder,
            local_size=2,
            world_size=2,
        )

        plan = ShardingPlan(plan={"": module_plan})

        sharded_model = _shard_modules(
            module=quant_model[0],
            # pyre-ignore
            sharders=[sharder],
            device=device,
            plan=plan,
            env=ShardingEnv.from_local(world_size=2, rank=0),
        )

        specs = get_tbe_specs_from_sqebc(
            cast(ShardedQuantEmbeddingBagCollection, sharded_model)
        )

        expected_specs = [
            ("table_1", 40, 20, "int8", "EmbeddingLocation.DEVICE"),
            ("table_2", 60, 30, "int8", "EmbeddingLocation.DEVICE"),
            ("table_0", 20, 10, "int8", "EmbeddingLocation.DEVICE"),
        ]

        self.assertEqual(specs, expected_specs)
