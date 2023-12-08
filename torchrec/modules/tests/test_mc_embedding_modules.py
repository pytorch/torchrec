#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import cast, List, Optional

import torch
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.modules.mc_embedding_modules import ManagedCollisionEmbeddingBagCollection
from torchrec.modules.mc_modules import (
    DistanceLFU_EvictionPolicy,
    ManagedCollisionCollection,
    ManagedCollisionModule,
    MCHManagedCollisionModule,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class Tracer(torch.fx.Tracer):
    _leaf_module_names: List[str]

    def __init__(self, leaf_module_names: Optional[List[str]] = None) -> None:
        super().__init__()
        self._leaf_module_names = leaf_module_names or []

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        if (
            type(m).__name__ in self._leaf_module_names
            or module_qualified_name in self._leaf_module_names
        ):
            return True
        return super().is_leaf_module(m, module_qualified_name)


class MCHManagedCollisionEmbeddingBagCollectionTest(unittest.TestCase):
    def test_zch_ebc_train(self) -> None:
        device = torch.device("cpu")
        zch_size = 20
        update_interval = 2
        update_size = 10

        embedding_configs = [
            EmbeddingBagConfig(
                name="t1",
                embedding_dim=8,
                num_embeddings=zch_size,
                feature_names=["f1", "f2"],
            ),
        ]
        ebc = EmbeddingBagCollection(
            tables=embedding_configs,
            device=device,
        )
        mc_modules = {
            "t1": cast(
                ManagedCollisionModule,
                MCHManagedCollisionModule(
                    zch_size=zch_size,
                    device=device,
                    eviction_interval=update_interval,
                    eviction_policy=DistanceLFU_EvictionPolicy(),
                ),
            ),
        }
        mcc = ManagedCollisionCollection(
            managed_collision_modules=mc_modules,
            # pyre-ignore[6]
            embedding_configs=embedding_configs,
        )
        mc_ebc = ManagedCollisionEmbeddingBagCollection(
            ebc,
            mcc,
            return_remapped_features=True,
        )

        update_one = KeyedJaggedTensor.from_lengths_sync(
            keys=["f1", "f2"],
            values=torch.concat(
                [
                    torch.arange(1000, 1000 + update_size, dtype=torch.int64),
                    torch.arange(
                        1000 + update_size,
                        1000 + 2 * update_size,
                        dtype=torch.int64,
                    ),
                ]
            ),
            lengths=torch.ones((2 * update_size,), dtype=torch.int64),
            weights=None,
        )
        _, remapped_kjt1 = mc_ebc.forward(update_one)
        _, remapped_kjt2 = mc_ebc.forward(update_one)
        assert torch.all(
            # pyre-ignore[16]
            remapped_kjt1["f1"].values()
            == zch_size - 1
        ), "all remapped ids should be mapped to end of range"
        assert torch.all(
            remapped_kjt1["f2"].values() == zch_size - 1
        ), "all remapped ids should be mapped to end of range"

        assert torch.all(
            remapped_kjt2["f1"].values() == torch.arange(0, 10, dtype=torch.int64)
        )
        assert torch.all(
            remapped_kjt2["f2"].values()
            == torch.cat(
                [
                    torch.arange(10, 19, dtype=torch.int64),
                    torch.tensor([zch_size - 1], dtype=torch.int64),  # empty value
                ]
            )
        )
        update_two = KeyedJaggedTensor.from_lengths_sync(
            keys=["f1", "f2"],
            values=torch.concat(
                [
                    torch.arange(2000, 2000 + update_size, dtype=torch.int64),
                    torch.arange(
                        1000 + update_size,
                        1000 + 2 * update_size,
                        dtype=torch.int64,
                    ),
                ]
            ),
            lengths=torch.ones((2 * update_size,), dtype=torch.int64),
            weights=None,
        )
        _, remapped_kjt3 = mc_ebc.forward(update_two)
        _, remapped_kjt4 = mc_ebc.forward(update_two)

        assert torch.all(
            remapped_kjt3["f1"].values() == zch_size - 1
        ), "all remapped ids should be mapped to end of range"

        assert torch.all(remapped_kjt3["f2"].values() == remapped_kjt2["f2"].values())

        assert torch.all(
            remapped_kjt4["f1"].values()
            == torch.cat(
                [
                    torch.arange(1, 10, dtype=torch.int64),
                    torch.tensor([zch_size - 1], dtype=torch.int64),  # empty value
                ]
            )
        )
        assert torch.all(
            remapped_kjt4["f2"].values()
            == torch.cat(
                [
                    torch.arange(10, 19, dtype=torch.int64),
                    torch.tensor([0], dtype=torch.int64),  # assigned first open slot
                ]
            )
        )

    def test_zch_ebc_eval(self) -> None:
        device = torch.device("cpu")
        zch_size = 20
        update_interval = 2
        update_size = 10

        embedding_configs = [
            EmbeddingBagConfig(
                name="t1",
                embedding_dim=8,
                num_embeddings=zch_size,
                feature_names=["f1", "f2"],
            ),
        ]
        ebc = EmbeddingBagCollection(
            tables=embedding_configs,
            device=device,
        )
        mc_modules = {
            "t1": cast(
                ManagedCollisionModule,
                MCHManagedCollisionModule(
                    zch_size=zch_size,
                    device=device,
                    eviction_interval=update_interval,
                    eviction_policy=DistanceLFU_EvictionPolicy(),
                ),
            ),
        }
        mcc = ManagedCollisionCollection(
            managed_collision_modules=mc_modules,
            # pyre-ignore[6]
            embedding_configs=embedding_configs,
        )
        mc_ebc = ManagedCollisionEmbeddingBagCollection(
            ebc,
            mcc,
            return_remapped_features=True,
        )

        update_one = KeyedJaggedTensor.from_lengths_sync(
            keys=["f1", "f2"],
            values=torch.concat(
                [
                    torch.arange(1000, 1000 + update_size, dtype=torch.int64),
                    torch.arange(
                        1000 + update_size,
                        1000 + 2 * update_size,
                        dtype=torch.int64,
                    ),
                ]
            ),
            lengths=torch.ones((2 * update_size,), dtype=torch.int64),
            weights=None,
        )
        _, remapped_kjt1 = mc_ebc.forward(update_one)
        _, remapped_kjt2 = mc_ebc.forward(update_one)

        assert torch.all(
            # pyre-ignore[16]
            remapped_kjt1["f1"].values()
            == zch_size - 1
        ), "all remapped ids should be mapped to end of range"
        assert torch.all(
            remapped_kjt1["f2"].values() == zch_size - 1
        ), "all remapped ids should be mapped to end of range"

        assert torch.all(
            remapped_kjt2["f1"].values() == torch.arange(0, 10, dtype=torch.int64)
        )
        assert torch.all(
            remapped_kjt2["f2"].values()
            == torch.cat(
                [
                    torch.arange(10, 19, dtype=torch.int64),
                    torch.tensor([zch_size - 1], dtype=torch.int64),  # empty value
                ]
            )
        )

        # Trigger eval mode, zch should not update
        mc_ebc.eval()

        update_two = KeyedJaggedTensor.from_lengths_sync(
            keys=["f1", "f2"],
            values=torch.concat(
                [
                    torch.arange(2000, 2000 + update_size, dtype=torch.int64),
                    torch.arange(
                        1000 + update_size,
                        1000 + 2 * update_size,
                        dtype=torch.int64,
                    ),
                ]
            ),
            lengths=torch.ones((2 * update_size,), dtype=torch.int64),
            weights=None,
        )
        _, remapped_kjt3 = mc_ebc.forward(update_two)
        _, remapped_kjt4 = mc_ebc.forward(update_two)

        assert torch.all(
            remapped_kjt3["f1"].values() == zch_size - 1
        ), "all remapped ids should be mapped to end of range"

        assert torch.all(remapped_kjt3["f2"].values() == remapped_kjt2["f2"].values())

        assert torch.all(
            remapped_kjt4["f1"].values() == zch_size - 1
        ), "all remapped ids should be mapped to end of range"

        assert torch.all(remapped_kjt4["f2"].values() == remapped_kjt2["f2"].values())

    def test_mc_collection_traceable(self) -> None:
        device = torch.device("cpu")
        zch_size = 20
        update_interval = 2

        embedding_configs = [
            EmbeddingBagConfig(
                name="t1",
                embedding_dim=8,
                num_embeddings=zch_size,
                feature_names=["f1", "f2"],
            ),
        ]
        mc_modules = {
            "t1": cast(
                ManagedCollisionModule,
                MCHManagedCollisionModule(
                    zch_size=zch_size,
                    device=device,
                    input_hash_size=2 * zch_size,
                    eviction_interval=update_interval,
                    eviction_policy=DistanceLFU_EvictionPolicy(),
                ),
            ),
        }
        mcc = ManagedCollisionCollection(
            managed_collision_modules=mc_modules,
            # pyre-ignore[6]
            embedding_configs=embedding_configs,
        )

        mcc = ManagedCollisionCollection(
            managed_collision_modules=mc_modules,
            # pyre-ignore[6]
            embedding_configs=embedding_configs,
        )
        trec_tracer = Tracer(["ComputeJTDictToKJT"])
        graph: torch.fx.Graph = trec_tracer.trace(mcc)
        gm: torch.fx.GraphModule = torch.fx.GraphModule(mcc, graph)
        gm.print_readable()

        # TODO: since this is unsharded module, also check torch.jit.script

    def test_mch_ebc(self) -> None:
        device = torch.device("cpu")
        zch_size = 10
        mch_size = 10
        update_interval = 2
        update_size = 10

        embedding_configs = [
            EmbeddingBagConfig(
                name="t1",
                embedding_dim=8,
                num_embeddings=zch_size + mch_size,
                feature_names=["f1", "f2"],
            ),
        ]
        ebc = EmbeddingBagCollection(
            tables=embedding_configs,
            device=device,
        )

        def preprocess_func(id: torch.Tensor, hash_size: int) -> torch.Tensor:
            return id % hash_size

        mc_modules = {
            "t1": cast(
                ManagedCollisionModule,
                MCHManagedCollisionModule(
                    zch_size=zch_size,
                    mch_size=mch_size,
                    device=device,
                    eviction_interval=update_interval,
                    eviction_policy=DistanceLFU_EvictionPolicy(),
                    mch_hash_func=preprocess_func,
                ),
            ),
        }
        mcc = ManagedCollisionCollection(
            managed_collision_modules=mc_modules,
            # pyre-ignore[6]
            embedding_configs=embedding_configs,
        )
        mc_ebc = ManagedCollisionEmbeddingBagCollection(
            ebc,
            mcc,
            return_remapped_features=True,
        )

        update_one = KeyedJaggedTensor.from_lengths_sync(
            keys=["f1", "f2"],
            values=torch.concat(
                [
                    torch.arange(1000, 1000 + update_size, dtype=torch.int64),
                    torch.arange(
                        1000 + update_size,
                        1000 + 2 * update_size,
                        dtype=torch.int64,
                    ),
                ]
            ),
            lengths=torch.ones((2 * update_size,), dtype=torch.int64),
            weights=None,
        )
        _, remapped_kjt1 = mc_ebc.forward(update_one)
        _, remapped_kjt2 = mc_ebc.forward(update_one)

        assert torch.all(
            # pyre-ignore[16]
            remapped_kjt1["f1"].values()
            == torch.arange(zch_size, zch_size + mch_size, dtype=torch.int64)
        ), "all remapped ids are in mch section"
        assert torch.all(
            remapped_kjt1["f2"].values()
            == torch.arange(zch_size, zch_size + mch_size, dtype=torch.int64)
        ), "all remapped ids are in mch section"

        assert torch.all(
            remapped_kjt2["f1"].values()
            == torch.cat(
                [
                    torch.arange(0, 9, dtype=torch.int64),
                    torch.tensor([19], dtype=torch.int64),  # % MCH for last value
                ]
            )
        )

        assert torch.all(
            remapped_kjt2["f2"].values()
            == torch.arange(zch_size, zch_size + mch_size, dtype=torch.int64)
        ), "all remapped ids are in mch section"

        update_two = KeyedJaggedTensor.from_lengths_sync(
            keys=["f1", "f2"],
            values=torch.concat(
                [
                    torch.arange(2000, 2000 + update_size, dtype=torch.int64),
                    torch.arange(
                        1000 + update_size,
                        1000 + 2 * update_size,
                        dtype=torch.int64,
                    ),
                ]
            ),
            lengths=torch.ones((2 * update_size,), dtype=torch.int64),
            weights=None,
        )

        _, remapped_kjt3 = mc_ebc.forward(update_two)
        _, remapped_kjt4 = mc_ebc.forward(update_two)

        assert torch.all(
            remapped_kjt3["f1"].values()
            == torch.arange(zch_size, zch_size + mch_size, dtype=torch.int64)
        ), "all remapped ids are in mch section"

        assert torch.all(
            remapped_kjt3["f2"].values()
            == torch.arange(zch_size, zch_size + mch_size, dtype=torch.int64)
        ), "all remapped ids are in mch section"

        assert torch.all(
            remapped_kjt4["f1"].values()
            == torch.arange(zch_size, zch_size + mch_size, dtype=torch.int64)
        ), "all remapped ids are in mch section"

        assert torch.all(
            remapped_kjt4["f2"].values()
            == torch.cat(
                [
                    torch.arange(0, 9, dtype=torch.int64),
                    torch.tensor([19], dtype=torch.int64),  # assigned first open slot
                ]
            )
        )
