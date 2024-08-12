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
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch import nn
from torchrec.ir.serializer import JsonSerializer

from torchrec.ir.utils import (
    decapsulate_ir_modules,
    encapsulate_ir_modules,
    mark_dynamic_kjt,
)

from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.modules.feature_processor_ import (
    PositionWeightedModule,
    PositionWeightedModuleCollection,
)
from torchrec.modules.fp_embedding_modules import FeatureProcessedEmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor


class CompoundModule(nn.Module):
    def __init__(
        self,
        ebc: EmbeddingBagCollection,
        comp: Optional["CompoundModule"] = None,
        mlist: List[Union[EmbeddingBagCollection, "CompoundModule"]] = [],
    ) -> None:
        super().__init__()
        self.ebc = ebc
        self.comp = comp
        self.list = nn.ModuleList(mlist)

    def forward(self, features: KeyedJaggedTensor) -> List[torch.Tensor]:
        res = self.comp(features) if self.comp else []
        res.append(self.ebc(features).values())
        for m in self.list:
            if isinstance(m, CompoundModule):
                res.extend(m(features))
            else:
                res.append(m(features).values())
        return res


class CompoundModuleSerializer(JsonSerializer):
    _module_cls = CompoundModule

    @classmethod
    def children(cls, module: nn.Module) -> List[str]:
        children = ["ebc", "list"]
        if module.comp is not None:
            children += ["comp"]
        return children

    @classmethod
    def serialize_to_dict(
        cls,
        module: nn.Module,
    ) -> Dict[str, Any]:
        return {}

    @classmethod
    def deserialize_from_dict(
        cls,
        metadata_dict: Dict[str, Any],
        device: Optional[torch.device] = None,
        unflatten_ep: Optional[nn.Module] = None,
    ) -> nn.Module:
        assert unflatten_ep is not None
        ebc = unflatten_ep.ebc
        comp = getattr(unflatten_ep, "comp", None)
        i = 0
        mlist = []
        while hasattr(unflatten_ep.list, str(i)):
            mlist.append(getattr(unflatten_ep.list, str(i)))
            i += 1
        return CompoundModule(ebc, comp, mlist)


class TestJsonSerializer(unittest.TestCase):
    # in the model we have 5 duplicated EBCs, 1 fpEBC with fpCollection, and 1 fpEBC with fpDict
    def generate_model(self) -> nn.Module:
        class Model(nn.Module):
            def __init__(self, ebc, fpebc1, fpebc2):
                super().__init__()
                self.ebc1 = ebc
                self.ebc2 = copy.deepcopy(ebc)
                self.ebc3 = copy.deepcopy(ebc)
                self.ebc4 = copy.deepcopy(ebc)
                self.ebc5 = copy.deepcopy(ebc)
                self.fpebc1 = fpebc1
                self.fpebc2 = fpebc2

            def forward(
                self,
                features: KeyedJaggedTensor,
            ) -> List[torch.Tensor]:
                kt1 = self.ebc1(features)
                kt2 = self.ebc2(features)
                kt3 = self.ebc3(features)
                kt4 = self.ebc4(features)
                kt5 = self.ebc5(features)

                fpebc1_res = self.fpebc1(features)
                fpebc2_res = self.fpebc2(features)
                res: List[torch.Tensor] = []
                for kt in [kt1, kt2, kt3, kt4, kt5, fpebc1_res, fpebc2_res]:
                    res.extend(KeyedTensor.regroup([kt], [[key] for key in kt.keys()]))
                return res

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
        tb3_config = EmbeddingBagConfig(
            name="t3",
            embedding_dim=5,
            num_embeddings=10,
            feature_names=["f3"],
        )

        ebc = EmbeddingBagCollection(
            tables=[tb1_config, tb2_config, tb3_config],
            is_weighted=False,
        )
        max_feature_lengths = {"f1": 100, "f2": 100}

        fpebc1 = FeatureProcessedEmbeddingBagCollection(
            EmbeddingBagCollection(
                tables=[tb1_config, tb2_config],
                is_weighted=True,
            ),
            PositionWeightedModuleCollection(
                max_feature_lengths=max_feature_lengths,
            ),
        )
        fpebc2 = FeatureProcessedEmbeddingBagCollection(
            EmbeddingBagCollection(
                tables=[tb1_config, tb3_config],
                is_weighted=True,
            ),
            {
                "f1": PositionWeightedModule(max_feature_length=10),
                "f3": PositionWeightedModule(max_feature_length=20),
            },
        )

        model = Model(ebc, fpebc1, fpebc2)

        return model

    # def test_serialize_deserialize_ebc(self) -> None:
    #     model = self.generate_model()
    #     id_list_features = KeyedJaggedTensor.from_offsets_sync(
    #         keys=["f1", "f2", "f3"],
    #         values=torch.tensor([0, 1, 2, 3, 2, 3]),
    #         offsets=torch.tensor([0, 2, 2, 3, 4, 5, 6]),
    #     )

    #     eager_out = model(id_list_features)

    #     # Serialize EBC
    #     model, sparse_fqns = encapsulate_ir_modules(model, JsonSerializer)
    #     ep = torch.export.export(
    #         model,
    #         (id_list_features,),
    #         {},
    #         strict=False,
    #         # Allows KJT to not be unflattened and run a forward on unflattened EP
    #         preserve_module_call_signature=(tuple(sparse_fqns)),
    #     )

    #     # Run forward on ExportedProgram
    #     ep_output = ep.module()(id_list_features)

    #     for i, tensor in enumerate(ep_output):
    #         self.assertEqual(eager_out[i].shape, tensor.shape)

    #     # Deserialize EBC
    #     unflatten_ep = torch.export.unflatten(ep)
    #     deserialized_model = decapsulate_ir_modules(unflatten_ep, JsonSerializer)

    #     # check EBC config
    #     for i in range(5):
    #         ebc_name = f"ebc{i + 1}"
    #         self.assertIsInstance(
    #             getattr(deserialized_model, ebc_name), EmbeddingBagCollection
    #         )

    #         for deserialized, orginal in zip(
    #             getattr(deserialized_model, ebc_name).embedding_bag_configs(),
    #             getattr(model, ebc_name).embedding_bag_configs(),
    #         ):
    #             self.assertEqual(deserialized.name, orginal.name)
    #             self.assertEqual(deserialized.embedding_dim, orginal.embedding_dim)
    #             self.assertEqual(deserialized.num_embeddings, orginal.num_embeddings)
    #             self.assertEqual(deserialized.feature_names, orginal.feature_names)

    #     # check FPEBC config
    #     for i in range(2):
    #         fpebc_name = f"fpebc{i + 1}"
    #         assert isinstance(
    #             getattr(deserialized_model, fpebc_name),
    #             FeatureProcessedEmbeddingBagCollection,
    #         )

    #         for deserialized, orginal in zip(
    #             getattr(
    #                 deserialized_model, fpebc_name
    #             )._embedding_bag_collection.embedding_bag_configs(),
    #             getattr(
    #                 model, fpebc_name
    #             )._embedding_bag_collection.embedding_bag_configs(),
    #         ):
    #             self.assertEqual(deserialized.name, orginal.name)
    #             self.assertEqual(deserialized.embedding_dim, orginal.embedding_dim)
    #             self.assertEqual(deserialized.num_embeddings, orginal.num_embeddings)
    #             self.assertEqual(deserialized.feature_names, orginal.feature_names)

    #     # Run forward on deserialized model and compare the output
    #     deserialized_model.load_state_dict(model.state_dict())
    #     deserialized_out = deserialized_model(id_list_features)

    #     self.assertEqual(len(deserialized_out), len(eager_out))
    #     for deserialized, orginal in zip(deserialized_out, eager_out):
    #         self.assertEqual(deserialized.shape, orginal.shape)
    #         self.assertTrue(torch.allclose(deserialized, orginal))

    # def test_dynamic_shape_ebc(self) -> None:
    #     model = self.generate_model()
    #     feature1 = KeyedJaggedTensor.from_offsets_sync(
    #         keys=["f1", "f2", "f3"],
    #         values=torch.tensor([0, 1, 2, 3, 2, 3]),
    #         offsets=torch.tensor([0, 2, 2, 3, 4, 5, 6]),
    #     )

    #     feature2 = KeyedJaggedTensor.from_offsets_sync(
    #         keys=["f1", "f2", "f3"],
    #         values=torch.tensor([0, 1, 2, 3, 2, 3, 4]),
    #         offsets=torch.tensor([0, 2, 2, 3, 4, 5, 7]),
    #     )
    #     eager_out = model(feature2)

    #     # Serialize EBC
    #     collection = mark_dynamic_kjt(feature1)
    #     model, sparse_fqns = encapsulate_ir_modules(model, JsonSerializer)
    #     ep = torch.export.export(
    #         model,
    #         (feature1,),
    #         {},
    #         dynamic_shapes=collection.dynamic_shapes(model, (feature1,)),
    #         strict=False,
    #         # Allows KJT to not be unflattened and run a forward on unflattened EP
    #         preserve_module_call_signature=tuple(sparse_fqns),
    #     )

    #     # Run forward on ExportedProgram
    #     ep_output = ep.module()(feature2)

    #     # other asserts
    #     for i, tensor in enumerate(ep_output):
    #         self.assertEqual(eager_out[i].shape, tensor.shape)

    #     # Deserialize EBC
    #     unflatten_ep = torch.export.unflatten(ep)
    #     deserialized_model = decapsulate_ir_modules(unflatten_ep, JsonSerializer)
    #     deserialized_model.load_state_dict(model.state_dict())

    #     # Run forward on deserialized model
    #     deserialized_out = deserialized_model(feature2)

    #     for i, tensor in enumerate(deserialized_out):
    #         self.assertEqual(eager_out[i].shape, tensor.shape)
    #         assert torch.allclose(eager_out[i], tensor)

    # def test_ir_custom_op_device(self) -> None:
    #     model = self.generate_model()
    #     model.fpebc1 = copy.deepcopy(model.ebc1)
    #     model.fpebc2 = copy.deepcopy(model.ebc1)
    #     feature1 = KeyedJaggedTensor.from_offsets_sync(
    #         keys=["f1", "f2", "f3"],
    #         values=torch.tensor([0, 1, 2, 3, 2, 3]),
    #         offsets=torch.tensor([0, 2, 2, 3, 4, 5, 6]),
    #     )

    #     model, sparse_fqns = encapsulate_ir_modules(model, JsonSerializer)
    #     for device in ["cpu", "cuda", "meta"]:
    #         if device == "cuda" and not torch.cuda.is_available():
    #             continue
    #         device = torch.device(device)
    #         outputs = model.to(device)(feature1.to(device))
    #         for output in outputs:
    #             self.assertEqual(output.device.type, device.type)

    # def test_deserialized_device(self) -> None:
    #     model = self.generate_model()
    #     id_list_features = KeyedJaggedTensor.from_offsets_sync(
    #         keys=["f1", "f2", "f3"],
    #         values=torch.tensor([0, 1, 2, 3, 2, 3]),
    #         offsets=torch.tensor([0, 2, 2, 3, 4, 5, 6]),
    #     )

    #     # Serialize EBC
    #     model, sparse_fqns = encapsulate_ir_modules(model, JsonSerializer)
    #     ep = torch.export.export(
    #         model,
    #         (id_list_features,),
    #         {},
    #         strict=False,
    #         # Allows KJT to not be unflattened and run a forward on unflattened EP
    #         preserve_module_call_signature=(tuple(sparse_fqns)),
    #     )

    #     # Deserialize EBC on different devices (<cpu>, <cuda>, <meta>)
    #     for device in ["cpu", "cuda", "meta"]:
    #         if device == "cuda" and not torch.cuda.is_available():
    #             continue
    #         device = torch.device(device)
    #         unflatten_ep = torch.export.unflatten(ep)
    #         deserialized_model = decapsulate_ir_modules(
    #             unflatten_ep, JsonSerializer, device
    #         )
    #         for name, m in deserialized_model.named_modules():
    #             if hasattr(m, "device"):
    #                 assert m.device.type == device.type, f"{name} should be on {device}"
    #         for name, param in deserialized_model.named_parameters():
    #             # TODO: we don't support FPEBC yet, so we skip the FPEBC params
    #             if "_feature_processors" in name:
    #                 continue
    #             assert param.device.type == device.type, f"{name} should be on {device}"

    # def test_compound_module(self) -> None:
    #     tb1_config = EmbeddingBagConfig(
    #         name="t1",
    #         embedding_dim=4,
    #         num_embeddings=10,
    #         feature_names=["f1"],
    #     )
    #     tb2_config = EmbeddingBagConfig(
    #         name="t2",
    #         embedding_dim=4,
    #         num_embeddings=10,
    #         feature_names=["f2"],
    #     )
    #     tb3_config = EmbeddingBagConfig(
    #         name="t3",
    #         embedding_dim=4,
    #         num_embeddings=10,
    #         feature_names=["f3"],
    #     )
    #     ebc: Callable[[], EmbeddingBagCollection] = lambda: EmbeddingBagCollection(
    #         tables=[tb1_config, tb2_config, tb3_config],
    #         is_weighted=False,
    #     )

    #     class MyModel(nn.Module):
    #         def __init__(self, comp: CompoundModule) -> None:
    #             super().__init__()
    #             self.comp = comp

    #         def forward(self, features: KeyedJaggedTensor) -> List[torch.Tensor]:
    #             return self.comp(features)

    #     model = MyModel(
    #         CompoundModule(
    #             ebc=ebc(),
    #             comp=CompoundModule(ebc(), CompoundModule(ebc(), mlist=[ebc(), ebc()])),
    #             mlist=[ebc(), CompoundModule(ebc(), CompoundModule(ebc()))],
    #         )
    #     )
    #     id_list_features = KeyedJaggedTensor.from_offsets_sync(
    #         keys=["f1", "f2", "f3"],
    #         values=torch.tensor([0, 1, 2, 3, 2, 3]),
    #         offsets=torch.tensor([0, 2, 2, 3, 4, 5, 6]),
    #     )

    #     eager_out = model(id_list_features)

    #     JsonSerializer.module_to_serializer_cls["CompoundModule"] = (
    #         CompoundModuleSerializer
    #     )
    #     # Serialize
    #     model, sparse_fqns = encapsulate_ir_modules(model, JsonSerializer)
    #     ep = torch.export.export(
    #         model,
    #         (id_list_features,),
    #         {},
    #         strict=False,
    #         # Allows KJT to not be unflattened and run a forward on unflattened EP
    #         preserve_module_call_signature=(tuple(sparse_fqns)),
    #     )

    #     ep_output = ep.module()(id_list_features)
    #     self.assertEqual(len(ep_output), len(eager_out))
    #     for x, y in zip(ep_output, eager_out):
    #         self.assertEqual(x.shape, y.shape)

    #     # Deserialize
    #     unflatten_ep = torch.export.unflatten(ep)
    #     deserialized_model = decapsulate_ir_modules(unflatten_ep, JsonSerializer)
    #     # Check if Compound Module is deserialized correctly
    #     self.assertIsInstance(deserialized_model.comp, CompoundModule)
    #     self.assertIsInstance(deserialized_model.comp.comp, CompoundModule)
    #     self.assertIsInstance(deserialized_model.comp.comp.comp, CompoundModule)
    #     self.assertIsInstance(deserialized_model.comp.list[1], CompoundModule)
    #     self.assertIsInstance(deserialized_model.comp.list[1].comp, CompoundModule)

    #     deserialized_model.load_state_dict(model.state_dict())
    #     # Run forward on deserialized model
    #     deserialized_out = deserialized_model(id_list_features)
    #     self.assertEqual(len(deserialized_out), len(eager_out))
    #     for x, y in zip(deserialized_out, eager_out):
    #         self.assertTrue(torch.allclose(x, y))
