#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Any, Dict, List, Tuple

import hypothesis.strategies as st
import torch
from hypothesis import given, settings, Verbosity
from torchrec.modules.embedding_configs import (
    DataType,
    EmbeddingBagConfig,
    EmbeddingConfig,
)
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)
from torchrec.quant.embedding_modules import (
    DEFAULT_ROW_ALIGNMENT,
    EmbeddingBagCollection as QuantEmbeddingBagCollection,
    EmbeddingCollection as QuantEmbeddingCollection,
    quant_prep_enable_quant_state_dict_split_scale_bias,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor


class EmbeddingBagCollectionTest(unittest.TestCase):
    def _asserting_same_embeddings(
        self,
        pooled_embeddings_1: KeyedTensor,
        pooled_embeddings_2: KeyedTensor,
        atol: float = 1e-08,
    ) -> None:

        self.assertEqual(pooled_embeddings_1.keys(), pooled_embeddings_2.keys())
        for key in pooled_embeddings_1.keys():
            self.assertEqual(
                pooled_embeddings_1[key].shape, pooled_embeddings_2[key].shape
            )
            self.assertTrue(
                torch.allclose(
                    pooled_embeddings_1[key].cpu().float(),
                    pooled_embeddings_2[key].cpu().float(),
                    atol=atol,
                )
            )

    def _test_ebc(
        self,
        tables: List[EmbeddingBagConfig],
        features: KeyedJaggedTensor,
        quant_type: torch.dtype = torch.qint8,
        output_type: torch.dtype = torch.float,
        quant_state_dict_split_scale_bias: bool = False,
    ) -> None:
        ebc = EmbeddingBagCollection(tables=tables)
        if quant_state_dict_split_scale_bias:
            quant_prep_enable_quant_state_dict_split_scale_bias(ebc)

        embeddings = ebc(features)

        # test forward
        # pyre-ignore [16]
        ebc.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.PlaceholderObserver.with_args(
                dtype=output_type
            ),
            weight=torch.quantization.PlaceholderObserver.with_args(dtype=quant_type),
        )

        qebc = QuantEmbeddingBagCollection.from_float(ebc)
        quantized_embeddings = qebc(features)

        self.assertEqual(quantized_embeddings.values().dtype, output_type)

        self._asserting_same_embeddings(embeddings, quantized_embeddings, atol=1.0)

        # test state dict
        state_dict = ebc.state_dict()
        quantized_state_dict = qebc.state_dict()
        self.assertTrue(
            set(state_dict.keys()).issubset(set(quantized_state_dict.keys()))
        )

    # pyre-fixme[56]
    @given(
        data_type=st.sampled_from(
            [
                DataType.FP32,
                DataType.FP16,
            ]
        ),
        quant_type=st.sampled_from(
            [
                torch.half,
                torch.qint8,
            ]
        ),
        output_type=st.sampled_from(
            [
                torch.half,
                torch.float,
            ]
        ),
        permute_order=st.booleans(),
        quant_state_dict_split_scale_bias=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=8, deadline=None)
    def test_ebc(
        self,
        data_type: DataType,
        quant_type: torch.dtype,
        output_type: torch.dtype,
        permute_order: bool,
        quant_state_dict_split_scale_bias: bool,
    ) -> None:
        eb1_config = EmbeddingBagConfig(
            name="t1",
            embedding_dim=16,
            num_embeddings=10,
            feature_names=["f1"],
            data_type=data_type,
        )
        eb2_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=16,
            num_embeddings=10,
            feature_names=["f2"],
            data_type=data_type,
        )
        features = (
            KeyedJaggedTensor(
                keys=["f1", "f2"],
                values=torch.as_tensor([0, 1]),
                lengths=torch.as_tensor([1, 1]),
            )
            if not permute_order
            else KeyedJaggedTensor(
                keys=["f2", "f1"],
                values=torch.as_tensor([1, 0]),
                lengths=torch.as_tensor([1, 1]),
            )
        )
        self._test_ebc(
            [eb1_config, eb2_config],
            features,
            quant_type,
            output_type,
            quant_state_dict_split_scale_bias,
        )

    def test_shared_tables(self) -> None:
        eb_config = EmbeddingBagConfig(
            name="t1", embedding_dim=16, num_embeddings=10, feature_names=["f1", "f2"]
        )
        features = KeyedJaggedTensor(
            keys=["f1", "f2"],
            values=torch.as_tensor([0, 1]),
            lengths=torch.as_tensor([1, 1]),
        )
        self._test_ebc([eb_config], features)

    def test_shared_features(self) -> None:
        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=16, num_embeddings=10, feature_names=["f1"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2", embedding_dim=16, num_embeddings=10, feature_names=["f1"]
        )
        features = KeyedJaggedTensor(
            keys=["f1"],
            values=torch.as_tensor([0, 1]),
            lengths=torch.as_tensor([1, 1]),
        )
        self._test_ebc([eb1_config, eb2_config], features)

    def test_multiple_features(self) -> None:
        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=16, num_embeddings=10, feature_names=["f1", "f2"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2", embedding_dim=16, num_embeddings=10, feature_names=["f3"]
        )
        features = KeyedJaggedTensor(
            keys=["f1", "f2", "f3"],
            values=torch.as_tensor([0, 1, 2]),
            lengths=torch.as_tensor([1, 1, 1]),
        )
        self._test_ebc([eb1_config, eb2_config], features)

    # pyre-ignore
    @given(
        data_type=st.sampled_from(
            [
                DataType.FP32,
                DataType.FP16,
            ]
        ),
        quant_type=st.sampled_from(
            [
                torch.half,
                torch.qint8,
            ]
        ),
        output_type=st.sampled_from(
            [
                torch.half,
                torch.float,
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=8, deadline=None)
    def test_save_load_state_dict(
        self,
        data_type: DataType,
        quant_type: torch.dtype,
        output_type: torch.dtype,
    ) -> None:
        eb1_config = EmbeddingBagConfig(
            name="t1",
            embedding_dim=16,
            num_embeddings=10,
            feature_names=["f1"],
            data_type=data_type,
        )
        eb2_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=16,
            num_embeddings=10,
            feature_names=["f1"],
            data_type=data_type,
        )
        tables = [eb1_config, eb2_config]

        ebc = EmbeddingBagCollection(tables=tables)

        # test forward
        # pyre-ignore [16]
        ebc.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.PlaceholderObserver.with_args(
                dtype=output_type
            ),
            weight=torch.quantization.PlaceholderObserver.with_args(dtype=quant_type),
        )

        qebc = QuantEmbeddingBagCollection.from_float(ebc)
        qebc_state_dict = qebc.state_dict()

        ebc_2 = EmbeddingBagCollection(tables=tables)
        ebc_2.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.PlaceholderObserver.with_args(
                dtype=output_type
            ),
            weight=torch.quantization.PlaceholderObserver.with_args(dtype=quant_type),
        )

        qebc_2 = QuantEmbeddingBagCollection.from_float(ebc_2)

        qebc_2.load_state_dict(qebc_state_dict)
        qebc_2_state_dict = qebc_2.state_dict()

        for key in qebc_state_dict:
            torch.testing.assert_close(qebc_state_dict[key], qebc_2_state_dict[key])

        features = KeyedJaggedTensor(
            keys=["f1", "f2"],
            values=torch.as_tensor([0, 1]),
            lengths=torch.as_tensor([1, 1]),
        )

        embeddings = qebc(features)
        embeddings_2 = qebc_2(features)
        self._asserting_same_embeddings(embeddings, embeddings_2)

    # pyre-ignore
    @given(
        data_type=st.sampled_from(
            [
                DataType.FP32,
                DataType.FP16,
            ]
        ),
        quant_type=st.sampled_from(
            [
                torch.half,
                torch.qint8,
            ]
        ),
        output_type=st.sampled_from(
            [
                torch.half,
                torch.float,
            ]
        ),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=8, deadline=None)
    def test_composability(
        self,
        data_type: DataType,
        quant_type: torch.dtype,
        output_type: torch.dtype,
    ) -> None:
        class TestModel(torch.nn.Module):
            def __init__(self, ebc: EmbeddingBagCollection) -> None:
                super().__init__()
                self.ebc = ebc
                self.over_arch = torch.nn.Linear(
                    16,
                    1,
                )

            def forward(self, kjt: KeyedJaggedTensor) -> torch.Tensor:
                ebc_output = self.ebc.forward(kjt).to_dict()
                sparse_features = []
                for key in kjt.keys():
                    sparse_features.append(ebc_output[key])
                sparse_features = torch.cat(sparse_features, dim=0)
                return self.over_arch(sparse_features)

        eb1_config = EmbeddingBagConfig(
            name="t1",
            embedding_dim=16,
            num_embeddings=10,
            feature_names=["f1"],
            data_type=data_type,
        )
        eb2_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=16,
            num_embeddings=10,
            feature_names=["f1"],
            data_type=data_type,
        )
        tables = [eb1_config, eb2_config]

        ebc = EmbeddingBagCollection(tables=tables)
        # test forward
        # pyre-ignore [16]
        ebc.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.PlaceholderObserver.with_args(
                dtype=output_type
            ),
            weight=torch.quantization.PlaceholderObserver.with_args(dtype=quant_type),
        )

        test_model = TestModel(ebc)

        before_quant_state_dict = test_model.state_dict()
        test_model.ebc = QuantEmbeddingBagCollection.from_float(ebc)

        state_dict = test_model.state_dict()
        self.assertTrue(
            set(before_quant_state_dict.keys()).issubset(set(state_dict.keys()))
        )
        test_model.load_state_dict(state_dict)

    def test_trace_and_script(self) -> None:
        data_type = DataType.FP16
        quant_type = torch.half
        output_type = torch.half

        eb1_config = EmbeddingBagConfig(
            name="t1",
            embedding_dim=16,
            num_embeddings=10,
            feature_names=["f1"],
            data_type=data_type,
        )
        eb2_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=16,
            num_embeddings=10,
            feature_names=["f1"],
            data_type=data_type,
        )

        ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])
        # pyre-ignore
        ebc.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.PlaceholderObserver.with_args(
                dtype=output_type
            ),
            weight=torch.quantization.PlaceholderObserver.with_args(dtype=quant_type),
        )

        qebc = QuantEmbeddingBagCollection.from_float(ebc)

        from torchrec.fx import symbolic_trace

        gm = symbolic_trace(qebc)

        features = KeyedJaggedTensor(
            keys=["f1", "f2"],
            values=torch.as_tensor([0, 1]),
            lengths=torch.as_tensor([1, 1]),
        )

        original_out = qebc(features)
        traced_out = gm(features)

        scripted_module = torch.jit.script(gm)
        scripted_out = scripted_module(features)

        self.assertEqual(original_out.keys(), traced_out.keys())
        torch.testing.assert_close(original_out.values(), traced_out.values())
        self.assertEqual(original_out.offset_per_key(), traced_out.offset_per_key())

        self.assertEqual(original_out.keys(), scripted_out.keys())
        torch.testing.assert_close(original_out.values(), scripted_out.values())
        self.assertEqual(original_out.offset_per_key(), scripted_out.offset_per_key())


class EmbeddingCollectionTest(unittest.TestCase):
    def _test_ec(
        self,
        tables: List[EmbeddingConfig],
        features: KeyedJaggedTensor,
        quant_type: torch.dtype = torch.qint8,
        output_type: torch.dtype = torch.float,
        quant_state_dict_split_scale_bias: bool = False,
    ) -> None:
        ec = EmbeddingCollection(tables=tables)
        if quant_state_dict_split_scale_bias:
            quant_prep_enable_quant_state_dict_split_scale_bias(ec)

        embeddings = ec(features)

        # test forward
        # pyre-ignore [16]
        ec.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.PlaceholderObserver.with_args(
                dtype=output_type
            ),
            weight=torch.quantization.PlaceholderObserver.with_args(dtype=quant_type),
        )

        qec = QuantEmbeddingCollection.from_float(ec)
        quantized_embeddings = qec(features)
        self.assertEqual(
            list(quantized_embeddings.values())[0].values().dtype, output_type
        )

        self.assertEqual(embeddings.keys(), quantized_embeddings.keys())
        for key in embeddings.keys():
            self.assertEqual(
                embeddings[key].values().size(),
                quantized_embeddings[key].values().size(),
            )
            self.assertTrue(
                torch.allclose(
                    embeddings[key].values().cpu().float(),
                    quantized_embeddings[key].values().cpu().float(),
                    atol=1,
                )
            )

        # test state dict
        state_dict = ec.state_dict()
        quantized_state_dict = ec.state_dict()
        self.assertEqual(state_dict.keys(), quantized_state_dict.keys())

    # pyre-fixme[56]
    @given(
        data_type=st.sampled_from(
            [
                DataType.FP32,
                DataType.INT8,
            ]
        ),
        quant_type=st.sampled_from(
            [
                torch.half,
                torch.qint8,
            ]
        ),
        output_type=st.sampled_from(
            [
                torch.half,
                torch.float,
            ]
        ),
        quant_state_dict_split_scale_bias=st.booleans(),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=2, deadline=None)
    def test_ec(
        self,
        data_type: DataType,
        quant_type: torch.dtype,
        output_type: torch.dtype,
        quant_state_dict_split_scale_bias: bool,
    ) -> None:
        eb1_config = EmbeddingConfig(
            name="t1",
            embedding_dim=16,
            num_embeddings=10,
            feature_names=["f1"],
            data_type=data_type,
        )
        eb2_config = EmbeddingConfig(
            name="t2",
            embedding_dim=16,
            num_embeddings=10,
            feature_names=["f2"],
            data_type=data_type,
        )
        features = KeyedJaggedTensor(
            keys=["f1", "f2"],
            values=torch.as_tensor([0, 1]),
            lengths=torch.as_tensor([1, 1]),
        )
        self._test_ec(
            tables=[eb1_config, eb2_config],
            features=features,
            quant_state_dict_split_scale_bias=quant_state_dict_split_scale_bias,
        )

    def test_shared_tables(self) -> None:
        eb_config = EmbeddingConfig(
            name="t1", embedding_dim=16, num_embeddings=10, feature_names=["f1", "f2"]
        )
        features = KeyedJaggedTensor(
            keys=["f1", "f2"],
            values=torch.as_tensor([0, 1]),
            lengths=torch.as_tensor([1, 1]),
        )
        self._test_ec([eb_config], features)

    def test_shared_features(self) -> None:
        eb1_config = EmbeddingConfig(
            name="t1", embedding_dim=16, num_embeddings=10, feature_names=["f1"]
        )
        eb2_config = EmbeddingConfig(
            name="t2", embedding_dim=16, num_embeddings=10, feature_names=["f1"]
        )
        features = KeyedJaggedTensor(
            keys=["f1"],
            values=torch.as_tensor([0, 1]),
            lengths=torch.as_tensor([1, 1]),
        )
        self._test_ec([eb1_config, eb2_config], features)


def round_up(a: int, b: int) -> int:
    return int((a + b - 1) // b) * b


class QuantEmbeddingModulesStateDictTest(unittest.TestCase):
    def _test_state_dict(
        self,
        # pyre-ignore
        clz,
    ) -> None:
        # For QEC all table embedding_dim must be equal
        t1_dim: int = 3 if clz == QuantEmbeddingBagCollection else 4
        t2_dim: int = 4
        kwargs: Dict[str, Any] = (
            {"is_weighted": False} if clz == QuantEmbeddingBagCollection else {}
        )
        prefix = (
            "embedding_bags" if clz == QuantEmbeddingBagCollection else "embeddings"
        )

        table_0 = EmbeddingBagConfig(
            name="t1", embedding_dim=t1_dim, num_embeddings=10, feature_names=["f1"]
        )
        # Testing behavior of default dtype
        self.assertEqual(table_0.data_type, DataType.FP32)
        table_1 = EmbeddingBagConfig(
            name="t2",
            embedding_dim=t2_dim,
            num_embeddings=10,
            feature_names=["f2"],
            data_type=DataType.INT8,
        )

        def _assert_state_dict(
            qebc: QuantEmbeddingBagCollection,
            state_dict_tensors: List[Tuple[str, torch.device, torch.dtype, List[int]]],
        ) -> None:
            state_dict = qebc.state_dict()
            assert len(state_dict) == len(state_dict_tensors)
            for key, device, dtype, shape in state_dict_tensors:
                t = state_dict[key]
                self.assertEqual(t.device, device)
                self.assertEqual(t.dtype, dtype)
                self.assertEqual(t.shape, torch.Size(shape))

        cuda0 = torch.device("cuda:0")
        meta = torch.device("meta")
        u8 = torch.uint8
        f16 = torch.float16
        f32b = 4

        _assert_state_dict(
            clz(
                tables=[table_0, table_1],
                device=cuda0,
                quant_state_dict_split_scale_bias=True,
                **kwargs,
            ),
            [
                (f"{prefix}.t1.weight", cuda0, u8, [10, f32b * t1_dim]),
                (f"{prefix}.t2.weight", cuda0, u8, [10, t2_dim]),
                (f"{prefix}.t2.weight_qscale", cuda0, f16, [10, 1]),
                (f"{prefix}.t2.weight_qbias", cuda0, f16, [10, 1]),
            ],
        )
        _assert_state_dict(
            clz(
                tables=[table_0, table_1],
                device=meta,
                quant_state_dict_split_scale_bias=True,
                **kwargs,
            ),
            [
                (f"{prefix}.t1.weight", meta, u8, [10, f32b * t1_dim]),
                (f"{prefix}.t2.weight", meta, u8, [10, t2_dim]),
                (f"{prefix}.t2.weight_qscale", meta, f16, [10, 1]),
                (f"{prefix}.t2.weight_qbias", meta, f16, [10, 1]),
            ],
        )
        _assert_state_dict(
            clz(
                tables=[table_0, table_1],
                device=cuda0,
                quant_state_dict_split_scale_bias=False,
                **kwargs,
            ),
            [
                (
                    f"{prefix}.t1.weight",
                    cuda0,
                    u8,
                    [10, round_up(f32b * t1_dim, DEFAULT_ROW_ALIGNMENT)],
                ),
                (
                    f"{prefix}.t2.weight",
                    cuda0,
                    u8,
                    [10, round_up(f32b * t2_dim, DEFAULT_ROW_ALIGNMENT)],
                ),
            ],
        )
        _assert_state_dict(
            clz(
                tables=[table_0, table_1],
                device=meta,
                quant_state_dict_split_scale_bias=False,
                **kwargs,
            ),
            [
                (
                    f"{prefix}.t1.weight",
                    meta,
                    u8,
                    [10, round_up(f32b * t1_dim, DEFAULT_ROW_ALIGNMENT)],
                ),
                (
                    f"{prefix}.t2.weight",
                    meta,
                    u8,
                    [10, round_up(f32b * t2_dim, DEFAULT_ROW_ALIGNMENT)],
                ),
            ],
        )

    def test_state_dict_qebc(self) -> None:
        self._test_state_dict(QuantEmbeddingBagCollection)

    def test_state_dict_qec(self) -> None:
        self._test_state_dict(QuantEmbeddingCollection)
