#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import copy
import multiprocessing
import unittest
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
from hypothesis import settings
from libfb.py.pyre import none_throws
from torch import nn

from torchrec import EmbeddingConfig, inference as trec_infer, KeyedJaggedTensor
from torchrec.distributed.global_settings import set_propogate_device
from torchrec.distributed.mapped_embedding import MappedEmbeddingCollectionSharder
from torchrec.distributed.quant_mapped_embedding import (
    QuantMappedEmbeddingCollectionSharder,
)
from torchrec.distributed.shard import _shard_modules
from torchrec.distributed.sharding_plan import construct_module_sharding_plan, row_wise

from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.types import ModuleSharder, ShardingEnv, ShardingPlan
from torchrec.inference.modules import (
    DEFAULT_FUSED_PARAMS,
    trim_torch_package_prefix_from_typename,
)
from torchrec.modules.embedding_modules import EmbeddingCollection
from torchrec.modules.mapped_embedding_module import MappedEmbeddingCollection
from torchrec.quant.embedding_modules import (
    EmbeddingCollection as QuantEmbeddingCollection,
    QuantMappedEmbeddingCollection,
)

QUANTIZATION_MAPPING: Dict[str, Type[torch.nn.Module]] = {
    trim_torch_package_prefix_from_typename(
        torch.typename(EmbeddingCollection)
    ): QuantEmbeddingCollection,
    trim_torch_package_prefix_from_typename(
        torch.typename(MappedEmbeddingCollection)
    ): QuantMappedEmbeddingCollection,
}


class SparseArch(nn.Module):
    def __init__(
        self,
        tables: List[EmbeddingConfig],
        device: torch.device,
        buckets: int,
        input_hash_size: int = 4000,
        is_inference: bool = False,
    ) -> None:
        super().__init__()

        self._ec: MappedEmbeddingCollection = MappedEmbeddingCollection(
            tables=tables,
            device=device,
        )

    def forward(self, kjt: KeyedJaggedTensor) -> Tuple[torch.Tensor, torch.Tensor]:

        ec_out = self._ec(kjt)
        pred: torch.Tensor = torch.cat(
            [ec_out[key].values() for key in ["feature_0", "feature_1"]],
            dim=0,
        )
        loss = pred.mean()
        return loss, pred


class TestQuantMappedEmbedding(MultiProcessTestBase):
    # pyre-ignore[56]: Pyre was not able to infer the type of argument
    @unittest.skipIf(
        torch.cuda.device_count() <= 1,
        "Not enough GPUs, this test requires at least two GPUs",
    )
    @settings(deadline=None)
    def test_quant_sharding_mapped_ec(self) -> None:

        WORLD_SIZE = 2

        embedding_config = [
            EmbeddingConfig(
                name="table_0",
                feature_names=["feature_0"],
                embedding_dim=8,
                num_embeddings=16,
            ),
            EmbeddingConfig(
                name="table_1",
                feature_names=["feature_1"],
                embedding_dim=8,
                num_embeddings=32,
            ),
        ]

        train_input_per_rank = [
            KeyedJaggedTensor.from_lengths_sync(
                keys=["feature_0", "feature_1"],
                values=torch.cat([torch.randint(16, (8,)), torch.randint(32, (16,))]),
                lengths=torch.LongTensor([1] * 8 + [2] * 8),
                weights=None,
            ),
            KeyedJaggedTensor.from_lengths_sync(
                keys=["feature_0", "feature_1"],
                values=torch.cat([torch.randint(16, (8,)), torch.randint(32, (16,))]),
                lengths=torch.LongTensor([1] * 8 + [2] * 8),
                weights=None,
            ),
        ]

        infer_input = KeyedJaggedTensor.from_lengths_sync(
            keys=["feature_0", "feature_1"],
            values=torch.randint(
                32,
                (8,),
            ),
            lengths=torch.LongTensor([1, 1, 1, 1, 1, 1, 1, 1]),
            weights=None,
        )

        train_info = multiprocessing.Manager().dict()

        # Train Model with ZCH on GPU
        import fbvscode

        fbvscode.attach_debugger()
        self._run_multi_process_test(
            callable=_train_model,
            world_size=WORLD_SIZE,
            tables=embedding_config,
            num_buckets=2,
            kjt_input_per_rank=train_input_per_rank,
            sharder=MappedEmbeddingCollectionSharder(),
            return_dict=train_info,
            backend="nccl",
            infer_input=infer_input,
        )
        print(f"train_info: {train_info}")
        # Load Train Model State Dict into Inference Model
        inference_model = SparseArch(
            tables=embedding_config,
            device=torch.device("cpu"),
            input_hash_size=0,
            buckets=4,
        )

        merged_state_dict = {
            "_ec.embeddings.table_0.weight": torch.cat(
                [value for key, value in train_info.items() if "table_0" in key],
                dim=0,
            ),
            "_ec.embeddings.table_1.weight": torch.cat(
                [value for key, value in train_info.items() if "table_1" in key],
                dim=0,
            ),
        }

        inference_model.load_state_dict(merged_state_dict)

        # Get Train Model Output
        # train_output = inference_model(infer_input)

        # Quantize Inference Model
        quant_inference_model = trec_infer.modules.quantize_inference_model(
            inference_model, QUANTIZATION_MAPPING, None, torch.quint4x2
        )

        # Get Quantized Inference Model Output
        _, quant_output = quant_inference_model(infer_input)

        # Verify Quantized Inference Model Output is close to Train Model Output
        # TODO: [Kaus] Check why this fails
        # self.assertTrue(
        #     torch.allclose(
        #         train_info["train_output_0"],
        #         quant_output,
        #         atol=1e-02,
        #     )
        # )

        # Shard Quantized Inference Model
        sharder = QuantMappedEmbeddingCollectionSharder(
            fused_params=DEFAULT_FUSED_PARAMS
        )
        module_sharding_plan = construct_module_sharding_plan(
            quant_inference_model._ec,  # pyre-ignore
            per_param_sharding={"table_0": row_wise(), "table_1": row_wise()},
            local_size=2,
            world_size=WORLD_SIZE,
            device_type="cpu",
            sharder=sharder,  # pyre-ignore
        )
        set_propogate_device(True)

        sharded_quant_inference_model = _shard_modules(
            module=copy.deepcopy(quant_inference_model),
            plan=ShardingPlan({"_mc_ec": module_sharding_plan}),
            env=ShardingEnv.from_local(
                WORLD_SIZE,
                0,
            ),
            sharders=[sharder],  # pyre-ignore
            device=torch.device("cpu"),
        )

        _, sharded_quant_output = sharded_quant_inference_model(infer_input)
        self.assertTrue(
            torch.allclose(
                sharded_quant_output,
                quant_output,
                atol=0,
            )
        )


def _train_model(
    tables: List[EmbeddingConfig],
    num_buckets: int,
    rank: int,
    world_size: int,
    kjt_input_per_rank: List[KeyedJaggedTensor],
    sharder: ModuleSharder[nn.Module],
    backend: str,
    return_dict: Dict[str, Any],
    infer_input: KeyedJaggedTensor,
    local_size: Optional[int] = None,
) -> None:
    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        import fbvscode

        fbvscode.attach_debugger()
        kjt_input = kjt_input_per_rank[rank].to(ctx.device)

        train_model = SparseArch(
            tables=tables,
            device=torch.device("cuda"),
            input_hash_size=0,
            buckets=num_buckets,
        )
        train_sharding_plan = construct_module_sharding_plan(
            train_model._ec,
            per_param_sharding={"table_0": row_wise(), "table_1": row_wise()},
            local_size=local_size,
            world_size=world_size,
            device_type="cuda",
            sharder=sharder,
        )
        print(f"train_sharding_plan: {train_sharding_plan}")
        sharded_train_model = _shard_modules(
            module=copy.deepcopy(train_model),
            plan=ShardingPlan({"_ec": train_sharding_plan}),
            env=ShardingEnv.from_process_group(none_throws(ctx.pg)),
            sharders=[sharder],
            device=ctx.device,
        )
        optim = torch.optim.SGD(sharded_train_model.parameters(), lr=0.1)
        # train
        optim.zero_grad()
        sharded_train_model.train(True)
        loss, output = sharded_train_model(kjt_input.to(ctx.device))
        loss.backward()
        optim.step()

        # infer
        with torch.no_grad():
            sharded_train_model.train(False)
            _, infer_output = sharded_train_model(infer_input.to(ctx.device))

        return_dict[f"train_output_{rank}"] = infer_output.cpu()

        for (
            key,
            value,
            # pyre-ignore
        ) in sharded_train_model._ec.embeddings.state_dict().items():
            return_dict[f"ec_{key}_{rank}"] = value.cpu()
