#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

import sys
import unittest
from typing import List, Tuple

import torch

try:
    # pyre-ignore
    from caffe2.test.inductor.test_aot_inductor import AOTIRunnerUtil
except (unittest.SkipTest, ImportError):
    if __name__ == "__main__":
        sys.exit(0)


from fbgemm_gpu import sparse_ops  # noqa: F401, E402
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
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
from torchrec.distributed.types import ShardingEnv, ShardingType
from torchrec.sparse.jagged_tensor import ComputeKJTToJTDict, KeyedJaggedTensor


def make_kjt(values: List[int], lengths: List[int]) -> KeyedJaggedTensor:
    values_tensor = torch.tensor(values, dtype=torch.int32)
    lengths_tensor = torch.tensor(lengths, dtype=torch.int32)
    torch._check(torch.sum(lengths_tensor).item() == values_tensor.size(0))
    kjt = KeyedJaggedTensor(
        keys=[f"key{i}" for i in range(len(lengths))],
        values=values_tensor,
        lengths=lengths_tensor,
    )
    return kjt


def _sharded_quant_ebc_model() -> Tuple[torch.nn.Module, List[KeyedJaggedTensor]]:
    num_embeddings = 256
    emb_dim = 12
    world_size = 2
    batch_size = 4
    local_device = torch.device("cuda:0")
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
    )
    input_kjts = [
        inp.to(local_device).idlist_features
        for inp in prep_inputs(mi, world_size, batch_size, long_indices=False)
    ]
    model: torch.nn.Module = KJTInputExportWrapper(mi.quant_model, input_kjts[0].keys())

    sharding_type: ShardingType = ShardingType.TABLE_WISE
    sharder = TestQuantEBCSharder(
        sharding_type=sharding_type.value,
        kernel_type=EmbeddingComputeKernel.QUANT.value,
        shardable_params=[table.name for table in mi.tables],
    )
    # pyre-ignore
    plan = mi.planner.plan(
        model,
        [sharder],
    )
    sharded_model = _shard_modules(
        module=model,
        # pyre-ignore
        sharders=[sharder],
        device=local_device,
        plan=plan,
        # pyre-ignore
        env=ShardingEnv.from_local(world_size=mi.topology.world_size, rank=0),
    )
    return sharded_model, input_kjts


class TestPt2(unittest.TestCase):
    def _test_kjt_input_module(
        self,
        kjt_input_module: torch.nn.Module,
        kjt_keys: List[str],
        # pyre-ignore
        inputs,
        test_dynamo: bool = True,
        test_aot_inductor: bool = True,
        test_pt2_ir_export: bool = False,
    ) -> None:
        with dynamo_skipfiles_allow("torchrec"):
            EM: torch.nn.Module = KJTInputExportWrapper(kjt_input_module, kjt_keys)
            eager_output = EM(*inputs)
            if test_dynamo:
                x = torch._dynamo.export(EM, same_signature=True)(*inputs)

                export_gm = x.graph_module
                export_gm_output = export_gm(*inputs)

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
                aot_actual_output = aot_inductor_module(*inputs)
                assert_close(eager_output, aot_actual_output)

            if test_pt2_ir_export:
                pt2_ir = torch.export.export(EM, inputs, {}, strict=False)
                pt2_ir_output = pt2_ir(*inputs)
                assert_close(eager_output, pt2_ir_output)

    def test_kjt_split(self) -> None:
        class M(torch.nn.Module):
            def forward(self, kjt: KeyedJaggedTensor, segments: List[int]):
                return kjt.split(segments)

        kjt: KeyedJaggedTensor = make_kjt([2, 3, 4, 5, 6], [1, 2, 1, 1])
        segments: List[int] = [1, 2, 1]
        self._test_kjt_input_module(
            M(),
            kjt.keys(),
            (kjt._values, kjt._lengths, segments),
            test_aot_inductor=False,
        )

    def test_kjt_permute(self) -> None:
        class M(torch.nn.Module):
            def forward(self, kjt: KeyedJaggedTensor, indices: List[int]):
                return kjt.permute(indices)

        kjt: KeyedJaggedTensor = make_kjt([2, 3, 4, 5, 6], [1, 2, 1, 1])
        indices: List[int] = [1, 0, 3, 2]
        self._test_kjt_input_module(
            M(),
            kjt.keys(),
            (kjt._values, kjt._lengths, indices),
            test_aot_inductor=False,
        )

    def test_kjt_length_per_key(self) -> None:
        class M(torch.nn.Module):
            def forward(self, kjt: KeyedJaggedTensor):
                return kjt.length_per_key()

        kjt: KeyedJaggedTensor = make_kjt([2, 3, 4, 5, 6], [1, 2, 1, 1])

        self._test_kjt_input_module(
            M(),
            kjt.keys(),
            (kjt._values, kjt._lengths),
            test_aot_inductor=False,
            test_pt2_ir_export=True,
        )

    # pyre-ignore
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs available",
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

    def test_maybe_compute_kjt_to_jt_dict(self) -> None:
        kjt: KeyedJaggedTensor = make_kjt([2, 3, 4, 5, 6], [1, 2, 1, 1])
        self._test_kjt_input_module(
            ComputeKJTToJTDict(),
            kjt.keys(),
            (kjt._values, kjt._lengths),
            # TODO: turn on AOT Inductor test once the support is ready
            test_aot_inductor=False,
        )

    def test_tensor_tolist(self) -> None:
        class M(torch.nn.Module):
            def forward(self, kjt: KeyedJaggedTensor):
                return kjt.values().tolist()

        kjt: KeyedJaggedTensor = make_kjt([2, 3, 4, 5, 6], [1, 2, 1, 1])
        self._test_kjt_input_module(
            M(),
            kjt.keys(),
            (kjt._values, kjt._lengths),
            test_dynamo=False,
            test_aot_inductor=False,
            test_pt2_ir_export=True,
        )
