#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import cast

import torch
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.modules.managed_collision_modules import (
    DistanceLFU_EvictionPolicy,
    ManagedCollisionModule,
    MCHManagedCollisionModule,
    TrivialManagedCollisionModule,
)
from torchrec.modules.mc_embedding_modules import ManagedCollisionEmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class TriviallyManagedCollisionEmbeddingBagCollectionTest(unittest.TestCase):
    def test_trivial_managed_ebc(self) -> None:
        device = torch.device("cpu")
        #     0       1        2  <-- batch
        # 0   [0,1] None    [2]
        # 1   [3]    [4]    [5,6,7]
        # ^
        # feature

        features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f2"],
            values=torch.tensor([100, 101, 102, 103, 104, 105, 106, 107]),
            offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8]),
        ).to(device)

        ebc = EmbeddingBagCollection(
            tables=[
                EmbeddingBagConfig(
                    name="t1", embedding_dim=8, num_embeddings=16, feature_names=["f1"]
                ),
                EmbeddingBagConfig(
                    name="t2", embedding_dim=8, num_embeddings=16, feature_names=["f2"]
                ),
            ],
            device=device,
        )
        mc_modules = {
            "t1": cast(
                ManagedCollisionModule,
                TrivialManagedCollisionModule(
                    max_output_id=16, max_input_id=32, device=device
                ),
            ),
            "t2": cast(
                ManagedCollisionModule,
                TrivialManagedCollisionModule(
                    max_output_id=16, max_input_id=32, device=device
                ),
            ),
        }

        mc_ebc = ManagedCollisionEmbeddingBagCollection(ebc, mc_modules)

        pooled_embeddings = mc_ebc(features)

        self.assertEqual(pooled_embeddings.keys(), ["f1", "f2"])
        self.assertEqual(pooled_embeddings.values().size(), (3, 16))
        self.assertEqual(pooled_embeddings.offset_per_key(), [0, 8, 16])

        print("state dict", mc_ebc.state_dict())


class MCHManagedCollisionEmbeddingBagCollectionTest(unittest.TestCase):
    def test_zch_managed_ebc(self) -> None:
        device = torch.device("cpu")
        table_size = 100
        zch_size = 50

        ebc = EmbeddingBagCollection(
            tables=[
                EmbeddingBagConfig(
                    name="t1", embedding_dim=8, num_embeddings=100, feature_names=["f1"]
                ),
            ],
            device=device,
        )
        mc_modules = {
            "t1": cast(
                ManagedCollisionModule,
                MCHManagedCollisionModule(
                    max_output_id=table_size,
                    device=device,
                    is_train=True,
                    max_history_size=zch_size,
                    zch_size=zch_size,
                    hash_func=lambda input_ids, hash_size: torch.remainder(
                        input_ids, hash_size
                    ),
                    eviction_policy=DistanceLFU_EvictionPolicy(),
                ),
            ),
        }
        mc_ebc = ManagedCollisionEmbeddingBagCollection(
            ebc,
            mc_modules,
            return_remapped_features=True,
        )

        ################ 1 ################
        # values in first id set will be tracked but won't trigger zch update
        # output values should therefore be in hashed range
        update_one_size = zch_size // 2
        update_one = KeyedJaggedTensor.from_lengths_sync(
            keys=["f1"],
            values=torch.arange(
                start=0, end=update_one_size, step=1, dtype=torch.int64
            ),
            lengths=torch.ones((update_one_size,), dtype=torch.int64),
            weights=None,
        )
        _, remapped_jt1 = mc_ebc.forward(
            update_one,
            mc_kwargs={
                "t1": {
                    "force_update": False,
                }
            },
        )
        remapped_jt = remapped_jt1
        assert remapped_jt is not None
        assert torch.all(
            # pyre-ignore[6]
            remapped_jt["f1"].values()
            >= zch_size
        ), "no remapped ids should be in zch_range"

        ################ 2 ################
        # second id set is same as first. as max_coalesce_history_size == zch_size
        # this will trigger a coalesce and the output values should be in zch_range
        _, remapped_jt2 = mc_ebc.forward(
            update_one,
            mc_kwargs={
                "t1": {
                    "force_update": False,
                }
            },
        )
        remapped_jt = remapped_jt2
        assert remapped_jt is not None
        assert torch.all(
            # pyre-ignore[6]
            remapped_jt["f1"].values()
            < zch_size
        ), "all remapped ids should be in zch_range"

        ################ 3 ################
        # third id set will fill remainder of unused zch_range.
        # by setting count_multiplier to 5, coalesce will be triggered
        # output values should be in zch_range.
        update_two_size = zch_size - update_one_size
        update_two = KeyedJaggedTensor.from_lengths_sync(
            keys=["f1"],
            values=torch.arange(
                start=update_one_size,
                end=update_one_size + update_two_size,
                step=1,
                dtype=torch.int64,
            ),
            lengths=torch.ones((update_two_size,), dtype=torch.int64),
            weights=None,
        )
        _, remapped_jt3 = mc_ebc.forward(
            update_two,
            mc_kwargs={"t1": {"force_update": False, "count_multiplier": 5}},
        )
        remapped_jt = remapped_jt3
        assert remapped_jt is not None
        assert torch.all(
            # pyre-ignore[6]
            remapped_jt["f1"].values()
            < zch_size
        ), "all remapped ids should be in zch_range"

        ################ 4 ################
        # fourth id set is same size as first and will overwrite it via eviction.
        # output values should be in zch_range _and_ same as remapped update_one ids
        update_three_size = update_one_size
        update_three = KeyedJaggedTensor.from_lengths_sync(
            keys=["f1"],
            values=torch.arange(
                start=update_one_size + update_two_size,
                end=update_one_size + update_two_size + update_three_size,
                step=1,
                dtype=torch.int64,
            ),
            lengths=torch.ones((update_three_size,), dtype=torch.int64),
            weights=None,
        )
        _, remapped_jt4 = mc_ebc.forward(
            update_three,
            mc_kwargs={"t1": {"force_update": True, "count_multiplier": 3}},
        )
        remapped_jt = remapped_jt4
        assert remapped_jt is not None
        assert torch.all(
            # pyre-ignore[6]
            remapped_jt4["f1"].values().sort(stable=True)[0]
            # pyre-ignore[6]
            == remapped_jt2["f1"].values().sort(stable=True)[0]
        ), "all remapped ids should match evicted update_one"

        ################ 5 ################
        # fifth id set is not part of current mapped zch
        # and will not trigger a zch_update.
        # output values should be in hashed range.
        update_four_size = zch_size - 1
        update_four = KeyedJaggedTensor.from_lengths_sync(
            keys=["f1"],
            values=torch.arange(
                start=update_one_size + update_two_size + update_three_size,
                end=update_one_size
                + update_two_size
                + update_three_size
                + update_four_size,
                step=1,
                dtype=torch.int64,
            ),
            lengths=torch.ones((update_four_size,), dtype=torch.int64),
            weights=None,
        )

        _, remapped_jt5 = mc_ebc.forward(update_four)
        remapped_jt = remapped_jt5
        assert remapped_jt is not None
        assert torch.all(
            # pyre-ignore[6]
            remapped_jt["f1"].values()
            >= zch_size
        ), "no remapped ids should be in zch_range"
