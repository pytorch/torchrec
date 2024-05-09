#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

#!/usr/bin/env python3

import copy
import unittest

import torch
from torch import nn
from torchrec.ir.serializer import JsonSerializer

from torchrec.ir.utils import deserialize_embedding_modules, serialize_embedding_modules
from torchrec.modules import utils as module_utils

from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.modules.utils import (
    operator_registry_state,
    register_custom_op,
    register_custom_ops_for_nodes,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor


class TestJsonSerializer(unittest.TestCase):
    def generate_model(self) -> nn.Module:
        class Model(nn.Module):
            def __init__(self, ebc):
                super().__init__()
                self.ebc1 = ebc
                self.ebc2 = copy.deepcopy(ebc)
                self.ebc3 = copy.deepcopy(ebc)
                self.ebc4 = copy.deepcopy(ebc)
                self.ebc5 = copy.deepcopy(ebc)

            def forward(
                self,
                features: KeyedJaggedTensor,
            ) -> torch.Tensor:
                kt1 = self.ebc1(features)
                kt2 = self.ebc2(features)
                kt3 = self.ebc3(features)
                kt4 = self.ebc4(features)
                kt5 = self.ebc5(features)

                return (
                    kt1.values()
                    + kt2.values()
                    + kt3.values()
                    + kt4.values()
                    + kt5.values()
                )

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

        eager_out = model(id_list_features)

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

        total_dim = sum(model.ebc1._lengths_per_embedding)
        with operator_registry_state.op_registry_lock:
            # Run forward on ExportedProgram
            ep_output = ep.module()(id_list_features)

            self.assertEqual(eager_out.shape, ep_output.shape)

            # Only 1 custom op registered, as dimensions of ebc are same
            self.assertEqual(len(operator_registry_state.op_registry_schema), 1)

            # Check if custom op is registered with the correct name
            # EmbeddingBagCollection type and total dim
            self.assertTrue(
                f"EmbeddingBagCollection_{total_dim}"
                in operator_registry_state.op_registry_schema
            )

            # Reset the op registry
            operator_registry_state.op_registry_schema = {}

            # Reset lib
            module_utils.lib = torch.library.Library("custom", "FRAGMENT")

        # Ensure custom op is reregistered
        register_custom_ops_for_nodes(list(ep.graph_module.graph.nodes))

        with operator_registry_state.op_registry_lock:
            self.assertTrue(
                f"EmbeddingBagCollection_{total_dim}"
                in operator_registry_state.op_registry_schema
            )

        ep.module()(id_list_features)
        # Deserialize EBC
        deserialized_model = deserialize_embedding_modules(ep, JsonSerializer)

        for i in range(5):
            ebc_name = f"ebc{i + 1}"
            self.assertTrue(
                isinstance(
                    getattr(deserialized_model, ebc_name), EmbeddingBagCollection
                )
            )

            for deserialized_config, org_config in zip(
                getattr(deserialized_model, ebc_name).embedding_bag_configs(),
                getattr(model, ebc_name).embedding_bag_configs(),
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
        deserialized_out = deserialized_model(id_list_features)

        self.assertEqual(eager_out.shape, deserialized_out.shape)
