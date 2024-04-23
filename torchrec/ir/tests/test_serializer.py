#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

#!/usr/bin/env python3

import unittest

import torch
from torch import nn
from torchrec.ir.serializer import JsonSerializer

from torchrec.ir.utils import deserialize_embedding_modules, serialize_embedding_modules

from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor


class TestJsonSerializer(unittest.TestCase):
    def generate_model(self) -> nn.Module:
        class Model(nn.Module):
            def __init__(self, ebc):
                super().__init__()
                self.sparse_arch = ebc

            def forward(
                self,
                features: KeyedJaggedTensor,
            ) -> KeyedTensor:
                return self.sparse_arch(features)

        tb1_config = EmbeddingBagConfig(
            name="t1",
            embedding_dim=3,
            num_embeddings=10,
            feature_names=["f1"],
        )
        tb2_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=4,
            num_embeddings=10,
            feature_names=["f2"],
        )

        ebc = EmbeddingBagCollection(
            tables=[tb1_config, tb2_config],
            is_weighted=False,
        )

        model = Model(ebc)

        return model

    def test_serialize_deserialize_ebc(self) -> None:
        model = self.generate_model()
        id_list_features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f2"],
            values=torch.tensor([0, 1, 2, 3]),
            offsets=torch.tensor([0, 2, 2, 3, 4]),
        )

        eager_kt = model(id_list_features)

        # Serialize PEA
        model, sparse_fqns = serialize_embedding_modules(model, JsonSerializer)
        ep = torch.export.export(
            model,
            (id_list_features,),
            {},
            strict=False,
            # Allows KJT to not be unflattened and run a forward on unflattened EP
            preserve_module_call_signature=(tuple(sparse_fqns)),
        )

        # Run forward on ExportedProgram
        ep_output = ep.module()(id_list_features)

        self.assertTrue(isinstance(ep_output, KeyedTensor))
        self.assertEqual(eager_kt.keys(), ep_output.keys())
        self.assertEqual(eager_kt.values().shape, ep_output.values().shape)

        # Deserialize EBC
        deserialized_model = deserialize_embedding_modules(ep, JsonSerializer)

        self.assertTrue(
            isinstance(deserialized_model.sparse_arch, EmbeddingBagCollection)
        )

        for deserialized_config, org_config in zip(
            deserialized_model.sparse_arch.embedding_bag_configs(),
            model.sparse_arch.embedding_bag_configs(),
        ):
            self.assertEqual(deserialized_config.name, org_config.name)
            self.assertEqual(
                deserialized_config.embedding_dim, org_config.embedding_dim
            )
            self.assertEqual(
                deserialized_config.num_embeddings, org_config.num_embeddings
            )
            self.assertEqual(
                deserialized_config.feature_names, org_config.feature_names
            )

        # Run forward on deserialized model
        deserialized_kt = deserialized_model(id_list_features)

        self.assertEqual(eager_kt.keys(), deserialized_kt.keys())
        self.assertEqual(eager_kt.values().shape, deserialized_kt.values().shape)
