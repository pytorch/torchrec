#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
from collections import OrderedDict
from typing import Any, cast, Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.nn.modules.module import _IncompatibleKeys
from torch.nn.parallel import DistributedDataParallel
from torchrec.distributed.comm import get_local_size
from torchrec.distributed.planner import (
    EmbeddingShardingPlanner,
    sharder_name,
    Topology,
)
from torchrec.distributed.sharding_plan import get_default_sharders
from torchrec.distributed.types import (
    ModuleSharder,
    ShardedModule,
    ShardedTensor,
    ShardingEnv,
    ShardingPlan,
)
from torchrec.distributed.utils import (
    add_prefix_to_state_dict,
    append_prefix,
    copy_to_device,
    filter_state_dict,
)
from torchrec.optim.fused import FusedOptimizerModule
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizer


_DDP_STATE_DICT_PREFIX = "module."


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
    Default data parallel wrapper, which applies data parallel to all unsharded modules.
    """

    def __init__(
        self,
        bucket_cap_mb: int = 25,
    ) -> None:
        self._bucket_cap_mb: int = bucket_cap_mb

    def wrap(
        self,
        dmp: "DistributedModelParallel",
        env: ShardingEnv,
        device: torch.device,
    ) -> None:
        if isinstance(dmp._dmp_wrapped_module, DistributedDataParallel) or isinstance(
            dmp._dmp_wrapped_module, FullyShardedDataParallel
        ):
            return
        pg = env.process_group
        if pg is None:
            raise RuntimeError("Can only init DDP for ProcessGroup-based ShardingEnv")
        sharded_parameter_names = set(
            DistributedModelParallel._sharded_parameter_names(dmp._dmp_wrapped_module)
        )
        all_parameter_names = set(dict(dmp.named_parameters()).keys())
        if len(all_parameter_names - sharded_parameter_names) == 0:
            return
        DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(
            module=dmp._dmp_wrapped_module,
            params_and_buffers_to_ignore=sharded_parameter_names,
        )
        # initialize DDP
        dmp._dmp_wrapped_module = cast(
            nn.Module,
            DistributedDataParallel(
                module=dmp._dmp_wrapped_module.to(device),
                device_ids=None if device.type == "cpu" else [device],
                process_group=pg,
                gradient_as_bucket_view=True,
                broadcast_buffers=False,
                static_graph=True,
                bucket_cap_mb=self._bucket_cap_mb,
            ),
        )


def get_unwrapped_module(module: nn.Module) -> nn.Module:
    """
    Unwraps module wrapped by DMP, DDP, or FSDP.
    """
    while (
        isinstance(module, DistributedModelParallel)
        or isinstance(module, DistributedDataParallel)
        or isinstance(module, FullyShardedDataParallel)
    ):
        if isinstance(module, DistributedModelParallel):
            module = module._dmp_wrapped_module
        elif isinstance(module, FullyShardedDataParallel):
            module = module._fsdp_wrapped_module
        else:
            module = module.module
    return module


def get_module(module: nn.Module) -> nn.Module:
    """
    Unwraps DMP module.

    Does not unwrap data parallel wrappers (i.e. DDP/FSDP), so overriding
    implementations by the wrappers can be used.
    """
    while isinstance(module, DistributedModelParallel):
        module = module._dmp_wrapped_module
    return module


class DistributedModelParallel(nn.Module, FusedOptimizerModule):
    """
    Entry point to model parallelism.

    Args:
        module (nn.Module): module to wrap.
        env (Optional[ShardingEnv]): sharding environment that has the process group.
        device (Optional[torch.device]): compute device, defaults to cpu.
        plan (Optional[ShardingPlan]): plan to use when sharding, defaults to
            `EmbeddingShardingPlanner.collective_plan()`.
        sharders (Optional[List[ModuleSharder[nn.Module]]]): `ModuleSharders` available
            to shard with, defaults to `EmbeddingBagCollectionSharder()`.
        init_data_parallel (bool): data-parallel modules can be lazy, i.e. they delay
            parameter initialization until the first forward pass. Pass `True` to delay
            initialization of data parallel modules. Do first forward pass and then call
            DistributedModelParallel.init_data_parallel().
        init_parameters (bool): initialize parameters for modules still on meta device.
        data_parallel_wrapper (Optional[DataParallelWrapper]): custom wrapper for data
            parallel modules.

    Example::

        @torch.no_grad()
        def init_weights(m):
            if isinstance(m, nn.Linear):
                m.weight.fill_(1.0)
            elif isinstance(m, EmbeddingBagCollection):
                for param in m.parameters():
                    init.kaiming_normal_(param)

        m = MyModel(device='meta')
        m = DistributedModelParallel(m)
        m.apply(init_weights)
    """

    def __init__(
        self,
        module: nn.Module,
        env: Optional[ShardingEnv] = None,
        device: Optional[torch.device] = None,
        plan: Optional[ShardingPlan] = None,
        sharders: Optional[List[ModuleSharder[torch.nn.Module]]] = None,
        init_data_parallel: bool = True,
        init_parameters: bool = True,
        data_parallel_wrapper: Optional[DataParallelWrapper] = None,
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torchrec.distributed.{self.__class__.__name__}")

        self.init_parameters = init_parameters

        self._ddp_wrapped: bool = False

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

        if plan is None:
            planner = EmbeddingShardingPlanner(
                topology=Topology(
                    local_world_size=get_local_size(self._env.world_size),
                    world_size=self._env.world_size,
                    compute_device=self.device.type,
                )
            )
            pg = self._env.process_group
            if pg is not None:
                plan = planner.collective_plan(module, sharders, pg)
            else:
                plan = planner.plan(module, sharders)
        self._plan: ShardingPlan = plan

        self._dmp_wrapped_module: nn.Module = self._init_dmp(module)
        self._optim: CombinedOptimizer = self._init_optim(self._dmp_wrapped_module)

        if init_parameters:
            self._init_parameters(self.module)

        if init_data_parallel:
            self.init_data_parallel()

    @property
    def module(self) -> nn.Module:
        """
        Property to directly access sharded module, which will not be wrapped in DDP,
        FSDP, DMP, or any other parallelism wrappers.
        """
        return get_unwrapped_module(self)

    @module.setter
    def module(self, value: nn.Module) -> None:
        if isinstance(self.module, DistributedDataParallel) or isinstance(
            self.module, FullyShardedDataParallel
        ):
            raise RuntimeError(
                "module can't be set after calling init_data_parallel(...)"
            )
        else:
            self._dmp_wrapped_module = value

    # pyre-ignore [2, 3]
    def forward(self, *args, **kwargs) -> Any:
        return self._dmp_wrapped_module(*args, **kwargs)

    def init_data_parallel(self) -> None:
        """
        See init_data_parallel c-tor argument for usage.
        It's safe to call this method multiple times.
        """
        if not self._ddp_wrapped:
            # Allocate any 'meta' tensors
            if self.init_parameters:
                self._init_parameters(self._dmp_wrapped_module)
            self._data_parallel_wrapper.wrap(self, self._env, self.device)
            self._ddp_wrapped = True

    def copy(
        self,
        device: torch.device,
    ) -> "DistributedModelParallel":
        """
        Recursively copy submodules to new device by calling per-module customized copy
        process, since some modules needs to use the original references (like
        `ShardedModule` for inference).
        """
        assert isinstance(device, torch.device)
        copy_dmp = copy_to_device(self, self.device, device)
        return cast(DistributedModelParallel, copy_dmp)

    def _init_dmp(self, module: nn.Module) -> nn.Module:
        return self._shard_modules_impl(module)

    def _init_optim(self, module: nn.Module) -> CombinedOptimizer:
        # pyre-ignore [6]
        return CombinedOptimizer(self._fused_optim_impl(module, []))

    def _fused_optim_impl(
        self,
        module: nn.Module,
        fused_optims: List[Tuple[str, KeyedOptimizer]],
        path: str = "",
    ) -> List[Tuple[str, KeyedOptimizer]]:
        if isinstance(module, FusedOptimizerModule):
            fused_optims.append((path, module.fused_optimizer))
            return fused_optims

        for name, child in module.named_children():
            self._fused_optim_impl(
                child,
                fused_optims,
                path + "." + name if path else name,
            )
        return fused_optims

    def _shard_modules_impl(
        self,
        module: nn.Module,
        path: str = "",
    ) -> nn.Module:

        # pre-sharded module
        if isinstance(module, ShardedModule):
            return module

        # shardable module
        module_sharding_plan = self._plan.get_plan_for_module(path)
        if module_sharding_plan:
            sharder_key = sharder_name(type(module))
            module = self._sharder_map[sharder_key].shard(
                module,
                module_sharding_plan,
                self._env,
                self.device,
            )
            return module

        for name, child in module.named_children():
            child = self._shard_modules_impl(
                child,
                path + "." + name if path else name,
            )
            setattr(module, name, child)

        return module

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
        return self._sparse_grad_parameter_names(self.module, destination, prefix)

    def _sparse_grad_parameter_names(
        self, module: nn.Module, destination: List[str], prefix: str = ""
    ) -> List[str]:
        module = get_unwrapped_module(module)
        if isinstance(module, ShardedModule):
            pass
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

    # pyre-ignore [14]
    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        state_dict = get_module(self).state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
            state_dict, prefix + _DDP_STATE_DICT_PREFIX
        )
        add_prefix_to_state_dict(state_dict, prefix)
        return state_dict

    # pyre-fixme[14]: `load_state_dict` overrides method defined in `Module`
    #  inconsistently.
    def load_state_dict(
        self,
        state_dict: "OrderedDict[str, torch.Tensor]",
        prefix: str = "",
        strict: bool = True,
    ) -> _IncompatibleKeys:
        return self._load_state_dict(self, state_dict, prefix, strict)

    def _load_state_dict(
        self,
        module: nn.Module,
        state_dict: "OrderedDict[str, torch.Tensor]",
        prefix: str = "",
        strict: bool = True,
    ) -> _IncompatibleKeys:
        missing_keys = []
        unexpected_keys = []
        module = get_module(module)
        if isinstance(module, DistributedDataParallel):
            torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
                state_dict, prefix
            )
            add_prefix_to_state_dict(state_dict, prefix + _DDP_STATE_DICT_PREFIX)
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
        self,
        module: nn.Module,
        prefix: str = "",
        recurse: bool = True,
        strip_ddp: bool = True,
    ) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        if strip_ddp:
            module = get_unwrapped_module(module)
        if isinstance(module, ShardedModule):
            yield from module.named_parameters(prefix, recurse)
        else:
            yield from module.named_parameters(prefix, recurse=False)
            for name, child in module.named_children():
                yield from self._named_parameters(
                    child,
                    append_prefix(prefix, name),
                    recurse,
                    strip_ddp,
                )

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        gen = self._named_parameters(
            self.module,
            prefix,
            recurse,
        )
        memo = set()
        for key, param in gen:
            if param in memo:
                continue
            if remove_duplicate:
                memo.add(param)
            yield key, param

    def bare_named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
    ) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        gen = self._named_parameters(
            self.module,
            prefix,
            recurse,
        )
        memo = set()
        for key, param in gen:
            if param in memo:
                continue
            memo.add(param)
            yield key, param

    @staticmethod
    def _sharded_parameter_names(module: nn.Module, prefix: str = "") -> Iterator[str]:
        module = get_unwrapped_module(module)
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
        module = get_unwrapped_module(module)
        if isinstance(module, ShardedModule):
            yield from module.named_buffers(prefix, recurse)
        else:
            yield from module.named_buffers(prefix, recurse=False)
            for name, child in module.named_children():
                yield from self._named_buffers(
                    child, append_prefix(prefix, name), recurse
                )

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        gen = self._named_buffers(self.module, prefix, recurse)
        memo = set()
        for key, param in gen:
            if param in memo:
                continue
            if remove_duplicate:
                memo.add(param)
            yield key, param

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
