#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

import itertools
import sys
import unittest
from enum import auto, Enum
from typing import Any, Dict, List, Tuple

import torch
from hypothesis import given, settings, strategies as st
from torchrec.distributed.test_utils.infer_utils import (
    KJTInputExportDynamicShapeWrapper,
    KJTInputExportWrapperWithStrides,
    TestQuantFPEBCSharder,
)
from torchrec.pt2.utils import kjt_for_pt2_tracing
from torchrec.sparse.jagged_tensor import KeyedTensor

try:
    # pyre-ignore
    from caffe2.test.inductor.test_aot_inductor import AOTIRunnerUtil
except (unittest.SkipTest, ImportError):
    if __name__ == "__main__":
        sys.exit(0)


from fbgemm_gpu import sparse_ops  # noqa: F401, E402
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.fused_params import FUSED_PARAM_BOUNDS_CHECK_MODE
from torchrec.distributed.shard import _shard_modules
from torchrec.distributed.test_utils.infer_utils import (
    assert_close,
    create_test_model_ebc_only,
    dynamo_skipfiles_allow,
    KJTInputExportWrapper,
    prep_inputs,
    replace_registered_tbes_with_mock_tbes,
    replace_sharded_quant_modules_tbes_with_mock_tbes,
    TestQuantEBCSharder,
)
from torchrec.distributed.types import BoundsCheckMode, ShardingEnv, ShardingType
from torchrec.sparse.jagged_tensor import (
    ComputeKJTToJTDict,
    JaggedTensor,
    KeyedJaggedTensor,
)


def make_kjt(values: List[int], lengths: List[int]) -> KeyedJaggedTensor:
    values_tensor = torch.tensor(values, dtype=torch.int32)
    lengths_tensor = torch.tensor(lengths, dtype=torch.int32)
    weights_tensor = torch.randn(len(values), dtype=torch.float32)
    torch._check(torch.sum(lengths_tensor).item() == values_tensor.size(0))
    kjt = KeyedJaggedTensor(
        keys=[f"key{i}" for i in range(len(lengths))],
        values=values_tensor,
        lengths=lengths_tensor,
        weights=weights_tensor,
    )
    return kjt


def kjt_module_kjt_inputs_with_strides(kjt: KeyedJaggedTensor) -> Tuple:
    return (
        kjt._values,
        kjt._lengths,
        kjt._stride_per_key_per_rank,
    )


def _sharded_quant_ebc_model(
    local_device: str = "cuda",
    compute_device: str = "cuda",
    feature_processor: bool = False,
) -> Tuple[torch.nn.Module, List[KeyedJaggedTensor]]:
    num_embeddings = 256
    emb_dim = 12
    world_size = 2
    batch_size = 4

    local_device = torch.device(local_device)
    mi = create_test_model_ebc_only(
        num_embeddings,
        emb_dim,
        world_size,
        batch_size,
        num_features=2,
        num_weighted_features=1,
        dense_device=local_device,
        sparse_device=local_device,
        quant_state_dict_split_scale_bias=True,
        compute_device=compute_device,
        feature_processor=feature_processor,
    )
    input_kjts = [
        inp.to(local_device).idlist_features
        for inp in prep_inputs(mi, world_size, batch_size, long_indices=False)
    ]

    sharding_type: ShardingType = ShardingType.TABLE_WISE

    fused_params = {
        FUSED_PARAM_BOUNDS_CHECK_MODE: BoundsCheckMode.NONE,
    }
    if feature_processor:
        sharder = TestQuantFPEBCSharder(
            sharding_type=sharding_type.value,
            kernel_type=EmbeddingComputeKernel.QUANT.value,
            shardable_params=[table.name for table in mi.tables],
            fused_params=fused_params,
        )
    else:
        sharder = TestQuantEBCSharder(
            sharding_type=sharding_type.value,
            kernel_type=EmbeddingComputeKernel.QUANT.value,
            shardable_params=[table.name for table in mi.tables],
            fused_params=fused_params,
        )
    # pyre-ignore
    plan = mi.planner.plan(
        mi.quant_model,
        [sharder],
    )

    sharded_model = _shard_modules(
        module=mi.quant_model,
        # pyre-ignore
        sharders=[sharder],
        # Always shard on meta
        device=torch.device("meta"),
        plan=plan,
        # pyre-ignore
        env=ShardingEnv.from_local(world_size=mi.topology.world_size, rank=0),
    )

    model: torch.nn.Module = KJTInputExportWrapper(sharded_model, input_kjts[0].keys())
    return model, input_kjts


class _TestType(Enum):
    EXPORT = auto()
    DYNAMO_COMPILE = auto()


class TestPt2(unittest.TestCase):
    def setUp(self):
        super().setUp()

        @torch._library.register_fake_class("fbgemm::AtomicCounter")
        class FakeAtomicCounter:
            def __init__(self, counter_):
                self.counter_ = counter_

            @classmethod
            def __obj_unflatten__(cls, flat_obj):
                return cls(**dict(flat_obj))

            def increment(self) -> int:
                self.counter_ += 1
                return self.counter_

            def decrement(self) -> int:
                self.counter_ -= 1
                return self.counter_

            def reset(self):
                self.counter_ = 0

            def get(self) -> int:
                return self.counter_

            def set(self, val):
                self.counter_ = val

        @torch._library.register_fake_class("fbgemm::TensorQueue")
        class FakeTensorQueue:
            def __init__(self, queue, init_tensor):
                self.queue = queue
                self.init_tensor = init_tensor

            @classmethod
            def __obj_unflatten__(cls, flattened_ctx):
                return cls(**dict(flattened_ctx))

            def push(self, x):
                self.queue.append(x)

            def pop(self):
                if len(self.queue) == 0:
                    return self.init_tensor
                return self.queue.pop(0)

            def top(self):
                if len(self.queue) == 0:
                    return self.init_tensor
                return self.queue[0]

            def size(self):
                return len(self.queue)

    def tearDown(self):
        torch._library.fake_class_registry.deregister_fake_class(
            "fbgemm::AtomicCounter"
        )
        torch._library.fake_class_registry.deregister_fake_class("fbgemm::TensorQueue")
        super().tearDown()

    def _test_kjt_input_module(
        self,
        kjt_input_module: torch.nn.Module,
        kjt: KeyedJaggedTensor,
        inputs: Tuple[Any],
        test_dynamo: bool = True,
        test_aot_inductor: bool = True,
        test_pt2_ir_export: bool = False,
    ) -> None:
        with dynamo_skipfiles_allow("torchrec"):
            EM: torch.nn.Module = KJTInputExportWrapper(kjt_input_module, kjt.keys())
            em_inputs = (kjt.values(), kjt.lengths(), kjt.weights_or_none(), *inputs)
            eager_output = EM(*em_inputs)
            if test_dynamo:
                x = torch._dynamo.export(EM, same_signature=True)(*em_inputs)

                export_gm = x.graph_module
                export_gm_output = export_gm(*em_inputs)

                assert_close(eager_output, export_gm_output)

            if test_aot_inductor:
                # pyre-ignore
                so_path: str = AOTIRunnerUtil.compile(
                    EM,
                    inputs,
                )
                device = "cuda"
                # pyre-ignore
                aot_inductor_module = AOTIRunnerUtil.load(device, so_path)
                aot_actual_output = aot_inductor_module(*em_inputs)
                assert_close(eager_output, aot_actual_output)

            if test_pt2_ir_export:
                symint_wrapper = KJTInputExportDynamicShapeWrapper(EM)

                # KJTInputExportDynamicShapeWrapper represents sizes of values/weights
                # from first element of values/weights respectively (simulate symint)
                # Need to set as size in order to run a proper forward
                em_inputs[0][0] = kjt.values().size(0)
                em_inputs[2][0] = kjt.weights().size(0)
                eager_output = symint_wrapper(*em_inputs)
                pt2_ir = torch.export.export(
                    symint_wrapper, em_inputs, {}, strict=False
                )

                pt2_ir_output = pt2_ir.module()(*em_inputs)
                assert_close(eager_output, pt2_ir_output)

    # Separate test for Dynamo, as it fallbacks on VB path.
    # Torchrec has lazy init modules, depending on the first input => we need to run eager with tracing inputs.
    # But other test cases do not need to go VB.
    def _test_kjt_input_module_dynamo_compile(
        self,
        kjt_input_module: torch.nn.Module,
        kjt_keys: List[str],
        # pyre-ignore
        inputs,
        backend: str = "eager",
    ) -> None:
        with dynamo_skipfiles_allow("torchrec"):
            EM: torch.nn.Module = KJTInputExportWrapperWithStrides(
                kjt_input_module, kjt_keys
            )
            eager_output = EM(*inputs)
            torch._dynamo.config.capture_scalar_outputs = True
            torch._dynamo.config.capture_dynamic_output_shape_ops = True

            dynamo_eager_out = torch.compile(EM, backend=backend, fullgraph=True)(
                *inputs
            )
            assert_close(eager_output, dynamo_eager_out)

    @given(
        test_type_backend=st.sampled_from(
            [(_TestType.EXPORT, ""), (_TestType.DYNAMO_COMPILE, "aot_eager")]
        )
    )
    @settings(deadline=None)
    def test_kjt_split(self, test_type_backend: Tuple[_TestType, str]) -> None:
        test_type, backend = test_type_backend

        class M(torch.nn.Module):
            def forward(self, kjt: KeyedJaggedTensor):
                return kjt.split([1, 2, 1])

        kjt: KeyedJaggedTensor = make_kjt([2, 3, 4, 5, 6], [1, 2, 1, 1])
        if test_type == _TestType.EXPORT:
            self._test_kjt_input_module(
                M(),
                kjt,
                (),
                test_aot_inductor=False,
                test_dynamo=False,
                test_pt2_ir_export=True,
            )
        elif test_type == _TestType.DYNAMO_COMPILE:
            self._test_kjt_input_module_dynamo_compile(
                M(),
                kjt.keys(),
                kjt_module_kjt_inputs_with_strides(kjt_for_pt2_tracing(kjt)),
                backend=backend,
            )

    @given(
        test_type_backend=st.sampled_from(
            [(_TestType.EXPORT, ""), (_TestType.DYNAMO_COMPILE, "aot_eager")]
        )
    )
    @settings(deadline=None)
    def test_kjt_permute(self, test_type_backend: Tuple[_TestType, str]) -> None:
        test_type, backend = test_type_backend

        class M(torch.nn.Module):
            def forward(self, kjt: KeyedJaggedTensor, indices: List[int]):
                return kjt.permute(indices)

        kjt: KeyedJaggedTensor = make_kjt([2, 3, 4, 5, 6], [1, 2, 1, 1])
        indices: List[int] = [1, 0, 3, 2]

        if test_type == _TestType.EXPORT:
            self._test_kjt_input_module(
                M(),
                kjt,
                (indices,),
                test_aot_inductor=False,
                test_pt2_ir_export=True,
            )
        elif test_type == _TestType.DYNAMO_COMPILE:

            def inputs_fn(kjt):
                return *kjt_module_kjt_inputs_with_strides(kjt), indices

            self._test_kjt_input_module_dynamo_compile(
                M(),
                kjt.keys(),
                inputs_fn(kjt_for_pt2_tracing(kjt)),
                backend=backend,
            )

    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    def test_kt_regroup_as_dict(
        self,
    ) -> None:

        class M(torch.nn.Module):
            def forward(self, inputs: List[KeyedTensor]) -> Dict[str, torch.Tensor]:
                groups = [["dense_0", "sparse_1", "dense_2"], ["dense_1", "sparse_0"]]
                keys = ["group_0", "group_1"]
                return KeyedTensor.regroup_as_dict(inputs, groups, keys)

        m = M()

        key_dim = 1
        tensor_list_1 = [torch.randn(2, 3) for i in range(3)]
        keys_1 = ["dense_0", "dense_1", "dense_2"]
        kt_1 = KeyedTensor.from_tensor_list(keys_1, tensor_list_1, key_dim)
        tensor_list_2 = [torch.randn(2, 3) for i in range(2)]
        keys_2 = ["sparse_0", "sparse_1"]
        kt_2 = KeyedTensor.from_tensor_list(keys_2, tensor_list_2, key_dim)
        inputs = [kt_1, kt_2]

        for t in itertools.chain(tensor_list_1, tensor_list_2):
            torch._dynamo.decorators.mark_dynamic(t, 0)
            torch._dynamo.decorators.mark_dynamic(t, 1)

        eager_output = m(inputs)
        with dynamo_skipfiles_allow("torchrec"):
            torch_compile_backend = "eager"

            torch._dynamo.config.capture_scalar_outputs = True
            torch._dynamo.config.capture_dynamic_output_shape_ops = True
            opt_fn = torch.compile(
                m, backend=torch_compile_backend, fullgraph=True, dynamic=True
            )
            compile_output = opt_fn(inputs)
            torch.testing.assert_close(eager_output, compile_output)

    def test_kjt_length_per_key(self) -> None:
        class M(torch.nn.Module):
            def forward(self, kjt: KeyedJaggedTensor):
                return kjt.length_per_key()

        kjt: KeyedJaggedTensor = make_kjt([2, 3, 4, 5, 6], [1, 2, 1, 1])

        self._test_kjt_input_module(
            M(),
            kjt,
            (),
            test_aot_inductor=False,
            test_pt2_ir_export=True,
        )

    def test_kjt_offset_per_key(self) -> None:
        class M(torch.nn.Module):
            def forward(self, kjt: KeyedJaggedTensor):
                return kjt.offset_per_key()

        kjt: KeyedJaggedTensor = make_kjt([2, 3, 4, 5, 6], [1, 2, 1, 1])

        self._test_kjt_input_module(
            M(),
            kjt,
            (),
            test_aot_inductor=False,
            test_pt2_ir_export=True,
        )

    @given(
        test_type_backend=st.sampled_from(
            [(_TestType.EXPORT, ""), (_TestType.DYNAMO_COMPILE, "aot_eager")]
        )
    )
    @settings(deadline=None)
    def test_kjt__getitem__(self, test_type_backend: Tuple[_TestType, str]) -> None:
        test_type, backend = test_type_backend

        class M(torch.nn.Module):
            def forward(self, kjt: KeyedJaggedTensor):
                out0 = kjt["key0"]
                out1 = kjt["key1"]

                return out0, out1

        # First element represents symint for values and weights shape
        kjt: KeyedJaggedTensor = make_kjt([2, 3, 4, 5, 6], [1, 2, 1, 1])

        if test_type == _TestType.EXPORT:
            self._test_kjt_input_module(
                M(),
                kjt,
                (),
                test_dynamo=False,
                test_aot_inductor=False,
                test_pt2_ir_export=True,
            )
        elif test_type == _TestType.DYNAMO_COMPILE:
            self._test_kjt_input_module_dynamo_compile(
                M(),
                kjt.keys(),
                kjt_module_kjt_inputs_with_strides(kjt_for_pt2_tracing(kjt)),
                backend=backend,
            )

    def test_kjt_to_dict_with_strides_dynamo(self) -> None:
        class M(torch.nn.Module):
            def forward(self, kjt: KeyedJaggedTensor) -> Dict[str, JaggedTensor]:
                return kjt.to_dict()

        kjt: KeyedJaggedTensor = make_kjt([2, 3, 4, 5, 6], [1, 2, 1, 1])

        self._test_kjt_input_module_dynamo_compile(
            M(),
            kjt.keys(),
            kjt_module_kjt_inputs_with_strides(kjt_for_pt2_tracing(kjt)),
        )

    # pyre-ignores
    @unittest.skipIf(
        True or torch.cuda.device_count() <= 1,
        "Test fails all the time, skip it for now\n Not enough GPUs available",
    )
    def test_sharded_quant_ebc_dynamo_export_aot_inductor(self) -> None:
        sharded_model, input_kjts = _sharded_quant_ebc_model()
        kjt = input_kjts[0]
        sharded_model(kjt.values(), kjt.lengths())

        model: torch.nn.Module = sharded_model
        model.training = False
        replace_registered_tbes_with_mock_tbes(model)
        replace_sharded_quant_modules_tbes_with_mock_tbes(model)

        example_inputs = (kjt.values(), kjt.lengths())

        # pyre-ignore
        def kjt_to_inputs(kjt):
            return (kjt.values(), kjt.lengths())

        expected_outputs = [model(*kjt_to_inputs(kjt)) for kjt in input_kjts[1:]]

        device: str = "cuda"

        with dynamo_skipfiles_allow("torchrec"):
            tracing_values = kjt.values()
            tracing_lengths = kjt.lengths()
            torch._dynamo.mark_dynamic(tracing_values, 0)
            dynamo_gm, guard = torch._dynamo.export(model, same_signature=False)(
                tracing_values, tracing_lengths
            )
            dynamo_gm.print_readable()
            dynamo_actual_outputs = [  # noqa
                dynamo_gm(*kjt_to_inputs(kjt)) for kjt in input_kjts[1:]
            ]
            # TODO(ivankobzarev): Why dynamo outputs are different than expected, but aot outputs are correct.
            # assert_close(expected_outputs, dynamo_actual_outputs)

            # pyre-ignore
            so_path: str = AOTIRunnerUtil.compile(
                model,
                example_inputs,
            )
            # pyre-ignore
            aot_inductor_module = AOTIRunnerUtil.load(device, so_path)
            aot_inductor_module(*example_inputs)

            aot_actual_outputs = [
                aot_inductor_module(*kjt_to_inputs(kjt)) for kjt in input_kjts[1:]
            ]
            assert_close(expected_outputs, aot_actual_outputs)

    def test_sharded_quant_ebc_non_strict_export(self) -> None:
        sharded_model, input_kjts = _sharded_quant_ebc_model(
            local_device="cpu", compute_device="cpu"
        )
        kjt = input_kjts[0]
        kjt = kjt.to("meta")
        sharded_model(kjt.values(), kjt.lengths())

        from torch.export import _trace

        ep = _trace._export(
            sharded_model,
            (
                kjt.values(),
                kjt.lengths(),
            ),
            {},
            strict=False,
            pre_dispatch=True,
        )

        ep.module()(kjt.values(), kjt.lengths())

        # PT2 IR autofunctionalizes mutation funcs (bounds_check_indices)
        # ensure such node isn't present, as it causes issues with IR
        for n in ep.graph_module.graph.nodes:
            self.assertFalse("auto_functionalized" in str(n.name))

        # TODO: Fix Unflatten
        # torch.export.unflatten(ep)

    # pyre-ignore
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
    )
    def test_sharded_quant_fpebc_non_strict_export(self) -> None:
        sharded_model, input_kjts = _sharded_quant_ebc_model(
            local_device="cpu", compute_device="cpu", feature_processor=True
        )
        kjt = input_kjts[0]
        kjt = kjt.to("meta")
        # Move FP parameters
        sharded_model.to("meta")

        sharded_model(kjt.values(), kjt.lengths())

        from torch.export import _trace

        ep = _trace._export(
            sharded_model,
            (
                kjt.values(),
                kjt.lengths(),
            ),
            {},
            strict=False,
            pre_dispatch=True,
        )
        ep.module()(kjt.values(), kjt.lengths())

        # PT2 IR autofunctionalizes mutation funcs (bounds_check_indices)
        # ensure such node isn't present, as it causes issues with IR
        for n in ep.graph_module.graph.nodes:
            self.assertFalse("auto_functionalized" in str(n.name))

        torch.export.unflatten(ep)

        ep(kjt.values(), kjt.lengths())

    def test_maybe_compute_kjt_to_jt_dict(self) -> None:
        kjt: KeyedJaggedTensor = make_kjt([2, 3, 4, 5, 6], [1, 2, 1, 1])
        self._test_kjt_input_module(
            ComputeKJTToJTDict(),
            kjt,
            (),
            # TODO: turn on AOT Inductor test once the support is ready
            test_aot_inductor=False,
        )
