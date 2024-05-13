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
from typing import List

import torch
from torch import nn
from torchrec.ir.serializer import JsonSerializer

from torchrec.ir.utils import deserialize_embedding_modules, serialize_embedding_modules

from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.modules.feature_processor_ import PositionWeightedModuleCollection
from torchrec.modules.fp_embedding_modules import FeatureProcessedEmbeddingBagCollection
from torchrec.modules.utils import operator_registry_state
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor


class TestJsonSerializer(unittest.TestCase):
    def generate_model(self) -> nn.Module:
        class Model(nn.Module):
            def __init__(self, ebc, fpebc):
                super().__init__()
                self.ebc1 = ebc
                self.ebc2 = copy.deepcopy(ebc)
                self.ebc3 = copy.deepcopy(ebc)
                self.ebc4 = copy.deepcopy(ebc)
                self.ebc5 = copy.deepcopy(ebc)
                self.fpebc = fpebc

            def forward(
                self,
                features: KeyedJaggedTensor,
            ) -> List[torch.Tensor]:
                kt1 = self.ebc1(features)
                kt2 = self.ebc2(features)
                kt3 = self.ebc3(features)
                kt4 = self.ebc4(features)
                kt5 = self.ebc5(features)

                fpebc_res = self.fpebc(features)
                ebc_kt_vals = [kt.values() for kt in [kt1, kt2, kt3, kt4, kt5]]
                sparse_arch_vals = sum(ebc_kt_vals)
                sparse_arch_res = KeyedTensor(
                    keys=kt1.keys(),
                    values=sparse_arch_vals,
                    length_per_key=kt1.length_per_key(),
                )

                return KeyedTensor.regroup(
                    [sparse_arch_res, fpebc_res], [["f1"], ["f2", "f3"]]
                )

        tb1_config = EmbeddingBagConfig(
            name="t1",
            embedding_dim=4,
            num_embeddings=10,
            feature_names=["f1"],
        )
        tb2_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=4,
            num_embeddings=10,
            feature_names=["f2"],
        )
        tb3_config = EmbeddingBagConfig(
            name="t3",
            embedding_dim=4,
            num_embeddings=10,
            feature_names=["f3"],
        )

        ebc = EmbeddingBagCollection(
            tables=[tb1_config, tb2_config, tb3_config],
            is_weighted=False,
        )
        max_feature_lengths = {"f1": 100, "f2": 100}

        fpebc = FeatureProcessedEmbeddingBagCollection(
            EmbeddingBagCollection(
                tables=[tb1_config, tb2_config],
                is_weighted=True,
            ),
            PositionWeightedModuleCollection(
                max_feature_lengths=max_feature_lengths,
            ),
        )

        model = Model(ebc, fpebc)

        return model

    def test_serialize_deserialize_ebc(self) -> None:
        model = self.generate_model()
        id_list_features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f2", "f3"],
            values=torch.tensor([0, 1, 2, 3, 2, 3]),
            offsets=torch.tensor([0, 2, 2, 3, 4, 5, 6]),
        )

        eager_out = model(id_list_features)

        # Serialize EBC
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

        for i, tensor in enumerate(ep_output):
            self.assertEqual(eager_out[i].shape, tensor.shape)

        # Only 1 custom op registered, as dimensions of ebc are same
        self.assertEqual(len(operator_registry_state.op_registry_schema), 2)

        total_dim_ebc = sum(model.ebc1._lengths_per_embedding)
        total_dim_fpebc = sum(
            model.fpebc._embedding_bag_collection._lengths_per_embedding
        )
        # Check if custom op is registered with the correct name
        # EmbeddingBagCollection type and total dim
        self.assertTrue(
            f"EmbeddingBagCollection_{total_dim_ebc}"
            in operator_registry_state.op_registry_schema
        )
        self.assertTrue(
            f"EmbeddingBagCollection_{total_dim_fpebc}"
            in operator_registry_state.op_registry_schema
        )

        # Can rerun ep forward
        ep.module()(id_list_features)
        # Deserialize EBC
        deserialized_model = deserialize_embedding_modules(ep, JsonSerializer)

        for i in range(5):
            ebc_name = f"ebc{i + 1}"
            assert isinstance(
                getattr(deserialized_model, ebc_name), EmbeddingBagCollection
            )

            for deserialized_config, org_config in zip(
                getattr(deserialized_model, ebc_name).embedding_bag_configs(),
                getattr(model, ebc_name).embedding_bag_configs(),
            ):
                assert deserialized_config.name == org_config.name
                assert deserialized_config.embedding_dim == org_config.embedding_dim
                assert deserialized_config.num_embeddings, org_config.num_embeddings
                assert deserialized_config.feature_names, org_config.feature_names

        deserialized_model.load_state_dict(model.state_dict())
        # Run forward on deserialized model
        deserialized_out = deserialized_model(id_list_features)

        for i, tensor in enumerate(deserialized_out):
            assert eager_out[i].shape == tensor.shape
            assert torch.allclose(eager_out[i], tensor)
