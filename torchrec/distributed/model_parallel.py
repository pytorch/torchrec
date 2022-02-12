#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
from collections import OrderedDict
from typing import Dict, Any, Optional, cast, List, Tuple, Iterator

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel
from torch.nn.modules.module import _IncompatibleKeys
from torch.nn.parallel import DistributedDataParallel
from torchrec.distributed.embeddingbag import (
    EmbeddingBagCollectionSharder,
)
from torchrec.distributed.planner import (
    EmbeddingShardingPlanner,
    sharder_name,
    Topology,
)
from torchrec.distributed.quant_embeddingbag import QuantEmbeddingBagCollectionSharder
from torchrec.distributed.types import (
    ShardingPlan,
    ModuleSharder,
    ShardedModule,
    ShardingEnv,
)
from torchrec.distributed.utils import append_prefix
from torchrec.distributed.utils import filter_state_dict
from torchrec.optim.fused import FusedOptimizerModule
from torchrec.optim.keyed import KeyedOptimizer, CombinedOptimizer


def get_default_sharders() -> List[ModuleSharder[nn.Module]]:
    return [
        cast(ModuleSharder[nn.Module], EmbeddingBagCollectionSharder()),
        cast(ModuleSharder[nn.Module], QuantEmbeddingBagCollectionSharder()),
    ]


class DataParallelWrapper(abc.ABC):
    """
    Interface implemented by custom data parallel wrappers.
    """

    @abc.abstractmethod
    def wrap(
        self,
        dmp: "DistributedModelParallel",
        env: ShardingEnv,
        device: torch.device,
    ) -> None:
        pass


class DefaultDataParallelWrapper(DataParallelWrapper):
    """
    Default data parallel wrapper, which applies data parallel for all
    unsharded modules.
    """

    def wrap(
        self,
        dmp: "DistributedModelParallel",
        env: ShardingEnv,
        device: torch.device,
    ) -> None:
        pg = env.process_group
        if pg is None:
            raise RuntimeError("Can only init DDP for ProcessGroup-based ShardingEnv")
        sharded_parameter_names = set(
            DistributedModelParallel._sharded_parameter_names(dmp.module)
        )
        DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(
            module=dmp.module,
            params_and_buffers_to_ignore=[
                key
                for key, _ in dmp.named_parameters()
                if key in sharded_parameter_names
            ],
        )
        # initailize DDP
        dmp.module = cast(
            nn.Module,
            DistributedDataParallel(
                module=dmp.module.to(device),
                device_ids=None if device.type == "cpu" else [device],
                process_group=pg,
                gradient_as_bucket_view=True,
                broadcast_buffers=False,
                static_graph=True,
            ),
        )


def _strip_DDP(module: nn.Module) -> nn.Module:
    if isinstance(module, FullyShardedDataParallel) or isinstance(
        module, DistributedDataParallel
    ):
        module = module.module
    return module


class DistributedModelParallel(nn.Module, FusedOptimizerModule):
    """
    Entry point to model parallelism.
    Example:
        >>> @torch.no_grad()
        def init_weights(m):
            if isinstance(m, nn.Linear)
                m.weight.fill_(1.0)
            elif isinstance(m, EmbeddingBagCollection)
                for param in m.parameters():
                    init.kaiming_normal_(param)

        m = MyModel(device='meta')
        m = DistributedModelParallel(m)
        m.apply(init_weights)

    Constructor Args:
        module: module to wrap,
        pg: this processes' process group, defaults to dist.GroupMember.WORLD,
        device: this device, defaults to cpu,
        plan: plan to use when sharding, defaults to EmbeddingShardingPlanner.collective_plan(),
        sharders: ModuleSharders available to shard with, defaults to EmbeddingBagCollectionSharder(),
        init_data_parallel: data-parallel modules can be lazy, i.e. they delay parameter initialization until
        the first forward pass. Pass True if that's a case to delay initialization of data parallel modules.
        Do first forward pass and then call DistributedModelParallel.init_data_parallel().
        init_parameters: initialize parameters for modules still on meta device.
        data_parallel_wrapper: custom wrapper for data parallel modules.

    Call Args:

    Returns:
        None
    """

    SHARE_SHARDED: bool = False
    # pyre-fixme [4]
    SHARED_SHARDED_MODULE: Dict[str, ShardedModule[Any, Any, Any]] = {}

    def __init__(
        self,
        module: nn.Module,
        env: Optional[ShardingEnv] = None,
        device: Optional[torch.device] = None,
        plan: Optional[ShardingPlan] = None,
        sharders: Optional[List[ModuleSharder[nn.Module]]] = None,
        init_data_parallel: bool = True,
        init_parameters: bool = True,
        data_parallel_wrapper: Optional[DataParallelWrapper] = None,
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torchrec.distributed.{self.__class__.__name__}")

        self.module = module
        self.init_parameters = init_parameters

        if env is None:
            pg = dist.GroupMember.WORLD
            assert pg is not None, "Process group is not initialized"
            env = ShardingEnv.from_process_group(pg)
        self._env: ShardingEnv = env

        if device is None:
            device = torch.device("cpu")
        self.device: torch.device = device

        if sharders is None:
            sharders = get_default_sharders()
        self._sharder_map: Dict[str, ModuleSharder[nn.Module]] = {
            sharder_name(sharder.module_type): sharder for sharder in sharders
        }

        if data_parallel_wrapper is None:
            data_parallel_wrapper = DefaultDataParallelWrapper()
        self._data_parallel_wrapper: DataParallelWrapper = data_parallel_wrapper

        # 2. Call ShardingPlanner.collective_plan passing all found modules and corresponding sharders.
        if plan is None:
            planner = EmbeddingShardingPlanner(
                topology=Topology(
                    world_size=self._env.world_size, compute_device=self.device.type
                )
            )
            pg = self._env.process_group
            if pg is not None:
                plan = planner.collective_plan(module, sharders, pg)
            else:
                plan = planner.plan(module, sharders)
        self._plan: ShardingPlan = plan

        # 3. Replace modules w/ sharded versions,
        # and then wrap w/ DistributedDataParallel.
        fused_optims = []
        self._init_dmp(
            fused_optims=fused_optims,
        )
        if init_data_parallel:
            self.init_data_parallel()
        self._optim = CombinedOptimizer(fused_optims)

    @property
    def dmp_module(self) -> nn.Module:
        """
        Property to directly access sharded module, which
        may or may not yet be wrapped in DDP
        """
        return (
            self.module.module
            if isinstance(self.module, DistributedDataParallel)
            or isinstance(self.module, FullyShardedDataParallel)
            else self.module
        )

    # pyre-ignore [2, 3]
    def forward(self, *args, **kwargs) -> Any:
        return self.module(*args, **kwargs)

    def init_data_parallel(self) -> None:
        """
        See init_data_parallel c-tor argument for usage.
        It's safe to call this method multiple times.
        """
        if not isinstance(self.module, DistributedDataParallel) and not isinstance(
            self.module, FullyShardedDataParallel
        ):
            # Allocate any 'meta' tensors
            if self.init_parameters:
                self._init_parameters(self.module)
            self._data_parallel_wrapper.wrap(self, self._env, self.device)

    def _init_dmp(
        self,
        fused_optims: List[Tuple[str, KeyedOptimizer]],
    ) -> None:
        self._shard_modules_impl(
            self.module,
            "",
            fused_optims,
        )

    def _shard_modules_impl(
        self,
        module: nn.Module,
        path: str,
        fused_optims: List[Tuple[str, KeyedOptimizer]],
    ) -> None:
        sharded_children = set()
        for name, child in module.named_children():
            curr_path = path + name
            sharded_params = self._plan.get_plan_for_module(curr_path)
            sharder_key = sharder_name(type(child))
            if sharded_params:
                if DistributedModelParallel.SHARE_SHARDED:
                    if name in DistributedModelParallel.SHARED_SHARDED_MODULE:
                        sharded_child = DistributedModelParallel.SHARED_SHARDED_MODULE[
                            name
                        ]
                    else:
                        # Shard module device-agnostic
                        # This is the multi-threading programming model case
                        sharded_child = self._sharder_map[sharder_key].shard(
                            child,
                            sharded_params,
                            self._env,
                            self.device,
                        )
                        DistributedModelParallel.SHARED_SHARDED_MODULE[
                            name
                        ] = sharded_child
                else:
                    # Shard module
                    sharded_child = self._sharder_map[sharder_key].shard(
                        child,
                        sharded_params,
                        self._env,
                        self.device,
                    )
                setattr(module, name, sharded_child)
                if isinstance(sharded_child, FusedOptimizerModule):
                    fused_optims.append((curr_path, sharded_child.fused_optimizer))
                sharded_children.add(name)
            else:
                self._shard_modules_impl(
                    child,
                    curr_path + ".",
                    fused_optims,
                )

    def _init_parameters(self, module: nn.Module) -> None:
        @torch.no_grad()
        def init_parameters(module: nn.Module) -> None:
            # Allocate parameters and buffers if over 'meta' device.
            has_meta_param = False
            for name, param in module._parameters.items():
                if isinstance(param, torch.Tensor) and param.device.type == "meta":
                    module._parameters[name] = nn.Parameter(
                        torch.empty_like(param, device=self.device),
                        requires_grad=param.requires_grad,
                    )
                    has_meta_param = True
            for name, buffer in module._buffers.items():
                if isinstance(buffer, torch.Tensor) and buffer.device.type == "meta":
                    module._buffers[name] = torch.empty_like(buffer, device=self.device)

            # Init parameters if at least one parameter is over 'meta' device.
            if has_meta_param and hasattr(module, "reset_parameters"):
                # pyre-ignore [29]
                module.reset_parameters()

        module.apply(init_parameters)

    def sparse_grad_parameter_names(
        self, destination: Optional[List[str]] = None, prefix: str = ""
    ) -> List[str]:
        destination = [] if destination is None else destination
        return self._sparse_grad_parameter_names(self.dmp_module, destination, prefix)

    def _sparse_grad_parameter_names(
        self, module: nn.Module, destination: List[str], prefix: str = ""
    ) -> List[str]:
        module = _strip_DDP(module)
        if isinstance(module, ShardedModule):
            module.sparse_grad_parameter_names(destination, prefix)
        elif isinstance(module, nn.Embedding):
            if module.sparse:
                destination.append(append_prefix(prefix, "weight"))
        elif isinstance(module, nn.EmbeddingBag):
            if module.sparse:
                destination.append(append_prefix(prefix, "weight"))
        else:
            for name, child in module.named_children():
                self._sparse_grad_parameter_names(
                    child, destination, append_prefix(prefix, name)
                )
        return destination

    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        if destination is None:
            destination = OrderedDict()
            # pyre-ignore [16]
            destination._metadata = OrderedDict()

        return self._state_dict(self.dmp_module, destination, prefix, keep_vars)

    def _state_dict(
        self,
        module: nn.Module,
        destination: Dict[str, Any],
        prefix: str,
        keep_vars: bool,
    ) -> Dict[str, Any]:
        module = _strip_DDP(module)
        if isinstance(module, ShardedModule):
            module.state_dict(destination, prefix, keep_vars)
        else:
            module._save_to_state_dict(destination, prefix, keep_vars)
            for name, child in module.named_children():
                self._state_dict(child, destination, prefix + name + ".", keep_vars)
        return destination

    def load_state_dict(
        self,
        state_dict: "OrderedDict[str, torch.Tensor]",
        prefix: str = "",
        strict: bool = True,
    ) -> _IncompatibleKeys:
        return self._load_state_dict(self.dmp_module, state_dict, prefix, strict)

    def _load_state_dict(
        self,
        module: nn.Module,
        state_dict: "OrderedDict[str, torch.Tensor]",
        prefix: str = "",
        strict: bool = True,
    ) -> _IncompatibleKeys:
        missing_keys = []
        unexpected_keys = []
        module = _strip_DDP(module)
        if isinstance(module, ShardedModule):
            return module.load_state_dict(state_dict, strict=strict)
        else:
            module._load_from_state_dict(
                state_dict, prefix, {}, strict, missing_keys, unexpected_keys, []
            )
            for name, child in module.named_children():
                m_keys, u_keys = self._load_state_dict(
                    child,
                    filter_state_dict(state_dict, prefix + name),
                    "",
                    strict,
                )
                missing_keys.extend(m_keys)
                unexpected_keys.extend(u_keys)
        return _IncompatibleKeys(
            missing_keys=missing_keys, unexpected_keys=unexpected_keys
        )

    def _named_parameters(
        self, module: nn.Module, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        module = _strip_DDP(module)
        if isinstance(module, ShardedModule):
            yield from module.named_parameters(prefix, recurse)
        else:
            yield from module.named_parameters(prefix, recurse=False)
            for name, child in module.named_children():
                yield from self._named_parameters(
                    child, append_prefix(prefix, name), recurse
                )

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        yield from self._named_parameters(self.dmp_module, prefix, recurse)

    @staticmethod
    def _sharded_parameter_names(module: nn.Module, prefix: str = "") -> Iterator[str]:
        module = _strip_DDP(module)
        if isinstance(module, ShardedModule):
            yield from module.sharded_parameter_names(prefix)
        else:
            for name, child in module.named_children():
                yield from DistributedModelParallel._sharded_parameter_names(
                    child, append_prefix(prefix, name)
                )

    def _named_buffers(
        self, module: nn.Module, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        module = _strip_DDP(module)
        if isinstance(module, ShardedModule):
            yield from module.named_buffers(prefix, recurse)
        else:
            yield from module.named_buffers(prefix, recurse=False)
            for name, child in module.named_children():
                yield from self._named_buffers(
                    child, append_prefix(prefix, name), recurse
                )

    def named_buffers(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        yield from self._named_buffers(self.dmp_module, prefix, recurse)

    @property
    def fused_optimizer(self) -> KeyedOptimizer:
        return self._optim

    @property
    def plan(self) -> ShardingPlan:
        return self._plan

    @staticmethod
    def _reset_parameters(module: nn.Module) -> None:
        for _, m in module.named_modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
