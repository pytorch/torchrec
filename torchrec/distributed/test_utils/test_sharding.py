#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import Any, cast, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from torchrec.distributed.embedding_types import EmbeddingTableConfig
from torchrec.distributed.fbgemm_qcomm_codec import (
    CommType,
    get_qcomm_codecs_registry,
    QCommsConfig,
)
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner import (
    EmbeddingShardingPlanner,
    ParameterConstraints,
    Topology,
)
from torchrec.distributed.test_utils.multi_process import MultiProcessContext
from torchrec.distributed.test_utils.test_model import (
    ModelInput,
    TestEBCSharder,
    TestEBSharder,
    TestETCSharder,
    TestETSharder,
    TestSparseNNBase,
)
from torchrec.distributed.types import (
    ModuleSharder,
    ShardedTensor,
    ShardingEnv,
    ShardingPlan,
    ShardingType,
)
from torchrec.modules.embedding_configs import BaseEmbeddingConfig, EmbeddingBagConfig
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from typing_extensions import Protocol


class SharderType(Enum):
    EMBEDDING_BAG = "embedding_bag"
    EMBEDDING_BAG_COLLECTION = "embedding_bag_collection"
    EMBEDDING_TOWER = "embedding_tower"
    EMBEDDING_TOWER_COLLECTION = "embedding_tower_collection"


def create_test_sharder(
    sharder_type: str,
    sharding_type: str,
    kernel_type: str,
    fused_params: Optional[Dict[str, Any]] = None,
    qcomms_config: Optional[QCommsConfig] = None,
    device: Optional[torch.device] = None,
    variable_batch_size: bool = False,
) -> Union[TestEBSharder, TestEBCSharder, TestETSharder, TestETCSharder]:
    if fused_params is None:
        fused_params = {}
    qcomm_codecs_registry = {}
    if qcomms_config is not None:
        qcomm_codecs_registry = get_qcomm_codecs_registry(qcomms_config, device=device)
    if "learning_rate" not in fused_params:
        fused_params["learning_rate"] = 0.1
    if sharder_type == SharderType.EMBEDDING_BAG.value:
        return TestEBSharder(
            sharding_type, kernel_type, fused_params, qcomm_codecs_registry
        )
    elif sharder_type == SharderType.EMBEDDING_BAG_COLLECTION.value:
        return TestEBCSharder(
            sharding_type,
            kernel_type,
            fused_params,
            qcomm_codecs_registry,
            variable_batch_size,
        )
    elif sharder_type == SharderType.EMBEDDING_TOWER.value:
        return TestETSharder(
            sharding_type, kernel_type, fused_params, qcomm_codecs_registry
        )
    elif sharder_type == SharderType.EMBEDDING_TOWER_COLLECTION.value:
        return TestETCSharder(
            sharding_type, kernel_type, fused_params, qcomm_codecs_registry
        )
    else:
        raise ValueError(f"Sharder not supported {sharder_type}")


class ModelInputCallable(Protocol):
    def __call__(
        self,
        batch_size: int,
        world_size: int,
        num_float_features: int,
        tables: Union[List[EmbeddingTableConfig], List[EmbeddingBagConfig]],
        weighted_tables: Union[List[EmbeddingTableConfig], List[EmbeddingBagConfig]],
        pooling_avg: int = 10,
        dedup_tables: Optional[
            Union[List[EmbeddingTableConfig], List[EmbeddingBagConfig]]
        ] = None,
        variable_batch_size: bool = False,
    ) -> Tuple["ModelInput", List["ModelInput"]]:
        ...


def generate_inputs(
    world_size: int,
    tables: List[EmbeddingTableConfig],
    dedup_tables: List[EmbeddingTableConfig],
    generate: ModelInputCallable = ModelInput.generate,
    weighted_tables: Optional[List[EmbeddingTableConfig]] = None,
    batch_size: int = 4,
    num_float_features: int = 16,
    variable_batch_size: bool = False,
) -> Tuple[ModelInput, List[ModelInput]]:
    return generate(
        batch_size=batch_size,
        world_size=world_size,
        num_float_features=num_float_features,
        tables=tables,
        dedup_tables=dedup_tables,
        weighted_tables=weighted_tables or [],
        variable_batch_size=variable_batch_size,
    )


def gen_model_and_input(
    model_class: TestSparseNNBase,
    tables: List[EmbeddingTableConfig],
    embedding_groups: Dict[str, List[str]],
    world_size: int,
    generate: ModelInputCallable = ModelInput.generate,
    weighted_tables: Optional[List[EmbeddingTableConfig]] = None,
    num_float_features: int = 16,
    dense_device: Optional[torch.device] = None,
    sparse_device: Optional[torch.device] = None,
    dedup_feature_names: Optional[List[str]] = None,
    dedup_tables: Optional[List[EmbeddingTableConfig]] = None,
    variable_batch_size: bool = False,
    batch_size: int = 4,
    feature_processor_modules: Optional[Dict[str, torch.nn.Module]] = None,
) -> Tuple[nn.Module, List[Tuple[ModelInput, List[ModelInput]]]]:
    torch.manual_seed(0)
    if dedup_feature_names:
        model = model_class(
            tables=cast(
                List[BaseEmbeddingConfig],
                tables + dedup_tables if dedup_tables else tables,
            ),
            num_float_features=num_float_features,
            dedup_feature_names=dedup_feature_names,
            weighted_tables=cast(List[BaseEmbeddingConfig], weighted_tables),
            embedding_groups=embedding_groups,
            dense_device=dense_device,
            sparse_device=sparse_device,
            feature_processor_modules=feature_processor_modules,
        )
    else:
        model = model_class(
            tables=cast(
                List[BaseEmbeddingConfig],
                tables,
            ),
            num_float_features=num_float_features,
            weighted_tables=cast(List[BaseEmbeddingConfig], weighted_tables),
            embedding_groups=embedding_groups,
            dense_device=dense_device,
            sparse_device=sparse_device,
            feature_processor_modules=feature_processor_modules,
        )
    inputs = [
        generate_inputs(
            world_size=world_size,
            tables=tables,
            # pyre-ignore [6]
            dedup_tables=dedup_tables,
            generate=generate,
            weighted_tables=weighted_tables,
            num_float_features=num_float_features,
            variable_batch_size=variable_batch_size,
            batch_size=batch_size,
        )
    ]
    return (model, inputs)


def copy_state_dict(
    loc: Dict[str, Union[torch.Tensor, ShardedTensor]],
    glob: Dict[str, torch.Tensor],
    exclude_predfix: Optional[str] = None,
) -> None:
    for name, tensor in loc.items():
        if exclude_predfix is not None and name.startswith(exclude_predfix):
            continue
        else:
            assert name in glob, name
        global_tensor = glob[name]
        if isinstance(global_tensor, ShardedTensor):
            global_tensor = global_tensor.local_shards()[0].tensor
        if isinstance(tensor, ShardedTensor):
            for local_shard in tensor.local_shards():
                assert global_tensor.ndim == local_shard.tensor.ndim
                shard_meta = local_shard.metadata
                t = global_tensor.detach()
                if t.ndim == 1:
                    t = t[
                        shard_meta.shard_offsets[0] : shard_meta.shard_offsets[0]
                        + local_shard.tensor.shape[0]
                    ]
                elif t.ndim == 2:
                    t = t[
                        shard_meta.shard_offsets[0] : shard_meta.shard_offsets[0]
                        + local_shard.tensor.shape[0],
                        shard_meta.shard_offsets[1] : shard_meta.shard_offsets[1]
                        + local_shard.tensor.shape[1],
                    ]
                else:
                    raise ValueError("Tensors with ndim > 2 are not supported")
                local_shard.tensor.copy_(t)
        else:
            tensor.copy_(global_tensor)


def sharding_single_rank_test(
    rank: int,
    world_size: int,
    model_class: TestSparseNNBase,
    embedding_groups: Dict[str, List[str]],
    tables: List[EmbeddingTableConfig],
    sharders: List[ModuleSharder[nn.Module]],
    backend: str,
    optim: EmbOptimType,
    weighted_tables: Optional[List[EmbeddingTableConfig]] = None,
    constraints: Optional[Dict[str, ParameterConstraints]] = None,
    local_size: Optional[int] = None,
    qcomms_config: Optional[QCommsConfig] = None,
    apply_optimizer_in_backward_config: Optional[
        Dict[str, Tuple[Type[torch.optim.Optimizer], Dict[str, Any]]]
    ] = None,
    variable_batch_size: bool = False,
    batch_size: int = 4,
    feature_processor_modules: Optional[Dict[str, torch.nn.Module]] = None,
) -> None:

    with MultiProcessContext(rank, world_size, backend, local_size) as ctx:
        # Generate model & inputs.
        (global_model, inputs) = gen_model_and_input(
            model_class=model_class,
            tables=tables,
            weighted_tables=weighted_tables,
            embedding_groups=embedding_groups,
            world_size=world_size,
            num_float_features=16,
            variable_batch_size=variable_batch_size,
            batch_size=batch_size,
            feature_processor_modules=feature_processor_modules,
        )
        global_model = global_model.to(ctx.device)
        global_input = inputs[0][0].to(ctx.device)
        local_input = inputs[0][1][rank].to(ctx.device)

        # Shard model.
        local_model = model_class(
            tables=cast(List[BaseEmbeddingConfig], tables),
            weighted_tables=cast(List[BaseEmbeddingConfig], weighted_tables),
            embedding_groups=embedding_groups,
            dense_device=ctx.device,
            sparse_device=torch.device("meta"),
            num_float_features=16,
            feature_processor_modules=feature_processor_modules,
        )

        global_model_named_params_as_dict = dict(global_model.named_parameters())
        local_model_named_params_as_dict = dict(local_model.named_parameters())

        if apply_optimizer_in_backward_config is not None:
            for apply_optim_name, (
                optimizer_type,
                optimizer_kwargs,
            ) in apply_optimizer_in_backward_config.items():
                for name, param in global_model_named_params_as_dict.items():
                    if name not in apply_optim_name:
                        continue
                    assert name in local_model_named_params_as_dict
                    local_param = local_model_named_params_as_dict[name]
                    apply_optimizer_in_backward(
                        optimizer_type, [param], optimizer_kwargs
                    )
                    apply_optimizer_in_backward(
                        optimizer_type, [local_param], optimizer_kwargs
                    )

        planner = EmbeddingShardingPlanner(
            topology=Topology(
                world_size, ctx.device.type, local_world_size=ctx.local_size
            ),
            constraints=constraints,
        )
        plan: ShardingPlan = planner.collective_plan(local_model, sharders, ctx.pg)
        """
        Simulating multiple nodes on a single node. However, metadata information and
        tensor placement must still be consistent. Here we overwrite this to do so.

        NOTE:
            inter/intra process groups should still behave as expected.

        TODO: may need to add some checks that only does this if we're running on a
        single GPU (which should be most cases).
        """

        for group in plan.plan:
            for _, parameter_sharding in plan.plan[group].items():
                if (
                    parameter_sharding.sharding_type
                    in {
                        ShardingType.TABLE_ROW_WISE.value,
                        ShardingType.TABLE_COLUMN_WISE.value,
                    }
                    and ctx.device.type != "cpu"
                ):
                    sharding_spec = parameter_sharding.sharding_spec
                    if sharding_spec is not None:
                        # pyre-ignore
                        for shard in sharding_spec.shards:
                            placement = shard.placement
                            rank: Optional[int] = placement.rank()
                            assert rank is not None
                            shard.placement = torch.distributed._remote_device(
                                f"rank:{rank}/cuda:{rank}"
                            )

        local_model = DistributedModelParallel(
            local_model,
            env=ShardingEnv.from_process_group(ctx.pg),
            plan=plan,
            sharders=sharders,
            device=ctx.device,
        )

        dense_optim = KeyedOptimizerWrapper(
            dict(local_model.named_parameters()),
            lambda params: torch.optim.SGD(params, lr=0.1),
        )
        local_opt = CombinedOptimizer([local_model.fused_optimizer, dense_optim])

        # Load model state from the global model.
        copy_state_dict(local_model.state_dict(), global_model.state_dict())

        # Run a single training step of the sharded model.
        local_pred = gen_full_pred_after_one_step(local_model, local_opt, local_input)

        all_local_pred = []
        for _ in range(world_size):
            all_local_pred.append(torch.empty_like(local_pred))
        dist.all_gather(all_local_pred, local_pred, group=ctx.pg)

        # Run second training step of the unsharded model.
        assert optim == EmbOptimType.EXACT_SGD
        global_opt = torch.optim.SGD(global_model.parameters(), lr=0.1)

        global_pred = gen_full_pred_after_one_step(
            global_model, global_opt, global_input
        )

        # Compare predictions of sharded vs unsharded models.
        if qcomms_config is None:
            torch.testing.assert_close(global_pred, torch.cat(all_local_pred))
        else:
            # With quantized comms, we can relax constraints a bit
            rtol = 0.003
            if CommType.FP8 in [
                qcomms_config.forward_precision,
                qcomms_config.backward_precision,
            ]:
                rtol = 0.05
            atol = global_pred.max().item() * rtol
            torch.testing.assert_close(
                global_pred, torch.cat(all_local_pred), rtol=rtol, atol=atol
            )


def gen_full_pred_after_one_step(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    input: ModelInput,
) -> torch.Tensor:
    # Run a single training step of the global model.
    opt.zero_grad()
    model.train(True)
    loss, _ = model(input)
    loss.backward()
    opt.step()

    # Run a forward pass of the global model.
    with torch.no_grad():
        model.train(False)
        full_pred = model(input)
        return full_pred
