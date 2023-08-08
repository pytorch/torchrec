#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from functools import partial

import torch
import torch.fx
from torchrec.fx import symbolic_trace
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.modules.multiplexing_embedding_modules import (
    MultiplexingEmbeddingBagCollection,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


class FlavorAEmbeddingBagCollection(EmbeddingBagCollection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class FlavorBEmbeddingBagCollection(EmbeddingBagCollection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class MultiplexingEmbeddingBagCollectionTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.eb1_conf: EmbeddingBagConfig = EmbeddingBagConfig(
            name="t1",
            embedding_dim=3,
            num_embeddings=15,
            feature_names=["f1"],
            init_fn=partial(torch.nn.init.normal_, mean=0.0, std=1.5),
        )
        self.eb2_conf: EmbeddingBagConfig = EmbeddingBagConfig(
            name="t2",
            embedding_dim=4,
            num_embeddings=15,
            feature_names=["f1", "f2"],
            init_fn=partial(torch.nn.init.normal_, mean=7.1, std=1.9),
        )
        self.eb3_conf: EmbeddingBagConfig = EmbeddingBagConfig(
            name="t3",
            embedding_dim=2,
            num_embeddings=15,
            feature_names=["f2", "f3"],
            init_fn=partial(torch.nn.init.normal_, mean=5, std=0.3),
        )

    def instantiate_ebc(self, is_weighted: bool) -> MultiplexingEmbeddingBagCollection:
        return MultiplexingEmbeddingBagCollection(
            tables=[self.eb1_conf, self.eb2_conf, self.eb3_conf],
            regroup_functor=lambda conf: "multi-feature"
            if len(conf.feature_names) > 1
            else "single-feature",
            ebc_init_functor={
                "single-feature": lambda confs: FlavorAEmbeddingBagCollection(
                    tables=confs, is_weighted=is_weighted
                ),
                "*": lambda confs: FlavorBEmbeddingBagCollection(
                    tables=confs, is_weighted=is_weighted
                ),
            },
            is_weighted=is_weighted,
        )

    def prepare_kjt(self, require_weights=False) -> KeyedJaggedTensor:
        #     0       1        2  <-- batch
        # 0   [0,1] None    [2]
        # 1   [3]    [4]    [5,6,7]
        # 2   None  [8,9]   [10]
        # ^
        # feature
        return KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f2", "f3"],
            values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8, 8, 10, 11]),
            weights=torch.tensor(
                [0.1, 0.2, 0.4, 0.6, 1.2, 0.3, 0.6, 2.7, 0.7, 1.1, 1.3]
            )
            if require_weights
            else None,
        )

    def test_class_type(self) -> None:
        ebc = self.instantiate_ebc(False)
        self.assertIsInstance(ebc.ebcs["multi-feature"], FlavorBEmbeddingBagCollection)
        self.assertIsInstance(ebc.ebcs["single-feature"], FlavorAEmbeddingBagCollection)
        self.assertCountEqual(
            list(ebc.ebcs.keys()), ["multi-feature", "single-feature"]
        )

    def test_duplicate_config_name_fails(self) -> None:
        self.eb3_conf.name = "t1"
        with self.assertRaises(ValueError):
            self.instantiate_ebc(False)

    def test_inconsistent_weight(self) -> None:
        with self.assertRaises(ValueError):
            MultiplexingEmbeddingBagCollection(
                tables=[self.eb1_conf, self.eb2_conf, self.eb3_conf],
                regroup_functor=lambda conf: "multi-feature"
                if len(conf.feature_names) > 1
                else "single-feature",
                ebc_init_functor={
                    "single-feature": lambda confs: FlavorAEmbeddingBagCollection(
                        tables=confs, is_weighted=False
                    ),
                    "*": lambda confs: FlavorBEmbeddingBagCollection(
                        tables=confs, is_weighted=True
                    ),
                },
                is_weighted=False,
            ).is_weighted()

    def test_unweighted(self) -> None:
        ebc = self.instantiate_ebc(False)

        self.assertFalse(ebc.is_weighted())
        pooled_embeddings = ebc(self.prepare_kjt()).to_dict()
        self.assertEqual(
            list(pooled_embeddings.keys()), ["f1@t1", "f1@t2", "f2@t2", "f2@t3", "f3"]
        )
        self.assertEqual(list(pooled_embeddings["f1@t1"].size()), [3, 3])
        self.assertEqual(list(pooled_embeddings["f1@t2"].size()), [3, 4])
        self.assertEqual(list(pooled_embeddings["f2@t2"].size()), [3, 4])
        self.assertEqual(list(pooled_embeddings["f2@t3"].size()), [3, 2])
        self.assertEqual(list(pooled_embeddings["f3"].size()), [3, 2])

        # Make sure the result is consistent with per-EBC result
        submodule_pooled_embs = ebc.ebcs["multi-feature"](self.prepare_kjt()).to_dict()
        self.assertEqual(
            pooled_embeddings["f1@t2"].tolist(), submodule_pooled_embs["f1"].tolist()
        )
        self.assertEqual(
            pooled_embeddings["f2@t2"].tolist(), submodule_pooled_embs["f2@t2"].tolist()
        )
        self.assertEqual(
            pooled_embeddings["f2@t3"].tolist(), submodule_pooled_embs["f2@t3"].tolist()
        )
        self.assertEqual(
            pooled_embeddings["f3"].tolist(), submodule_pooled_embs["f3"].tolist()
        )

        submodule_pooled_embs = ebc.ebcs["single-feature"](self.prepare_kjt()).to_dict()
        self.assertEqual(
            pooled_embeddings["f1@t1"].tolist(), submodule_pooled_embs["f1"].tolist()
        )

    def test_weighted(self) -> None:
        ebc = self.instantiate_ebc(True)
        self.assertTrue(ebc.is_weighted())

        kjt: KeyedJaggedTensor = self.prepare_kjt(require_weights=True)
        pooled_embeddings = ebc(kjt).to_dict()
        self.assertEqual(
            list(pooled_embeddings.keys()), ["f1@t1", "f1@t2", "f2@t2", "f2@t3", "f3"]
        )
        self.assertEqual(list(pooled_embeddings["f1@t1"].size()), [3, 3])
        self.assertEqual(list(pooled_embeddings["f1@t2"].size()), [3, 4])
        self.assertEqual(list(pooled_embeddings["f2@t2"].size()), [3, 4])
        self.assertEqual(list(pooled_embeddings["f2@t3"].size()), [3, 2])
        self.assertEqual(list(pooled_embeddings["f3"].size()), [3, 2])

        # Make sure the result is consistent with per-EBC result
        submodule_pooled_embs = ebc.ebcs["multi-feature"](kjt).to_dict()
        self.assertEqual(
            pooled_embeddings["f1@t2"].tolist(), submodule_pooled_embs["f1"].tolist()
        )
        self.assertEqual(
            pooled_embeddings["f2@t2"].tolist(), submodule_pooled_embs["f2@t2"].tolist()
        )
        self.assertEqual(
            pooled_embeddings["f2@t3"].tolist(), submodule_pooled_embs["f2@t3"].tolist()
        )
        self.assertEqual(
            pooled_embeddings["f3"].tolist(), submodule_pooled_embs["f3"].tolist()
        )

        submodule_pooled_embs = ebc.ebcs["single-feature"](kjt).to_dict()
        self.assertEqual(
            pooled_embeddings["f1@t1"].tolist(), submodule_pooled_embs["f1"].tolist()
        )

    def test_fx(self) -> None:
        ebc = self.instantiate_ebc(False)
        gm = symbolic_trace(ebc)
        torch.jit.script(gm)

        pooled_embeddings = gm(self.prepare_kjt()).to_dict()
        self.assertEqual(
            list(pooled_embeddings.keys()), ["f1@t1", "f1@t2", "f2@t2", "f2@t3", "f3"]
        )
        self.assertEqual(list(pooled_embeddings["f1@t1"].size()), [3, 3])
        self.assertEqual(list(pooled_embeddings["f1@t2"].size()), [3, 4])
        self.assertEqual(list(pooled_embeddings["f2@t2"].size()), [3, 4])
        self.assertEqual(list(pooled_embeddings["f2@t3"].size()), [3, 2])
        self.assertEqual(list(pooled_embeddings["f3"].size()), [3, 2])

    def test_scripting(self) -> None:
        ebc = self.instantiate_ebc(False)
        torch.jit.script(ebc)

    def test_emb_configs(self) -> None:
        ebc = self.instantiate_ebc(False)
        self.assertEqual(
            ebc.embedding_bag_configs(), [self.eb1_conf, self.eb2_conf, self.eb3_conf]
        )
        self.assertEqual(
            ebc.ebcs["multi-feature"].embedding_bag_configs(),
            [self.eb2_conf, self.eb3_conf],
        )
        self.assertEqual(
            ebc.ebcs["single-feature"].embedding_bag_configs(), [self.eb1_conf]
        )
