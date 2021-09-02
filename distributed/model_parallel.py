#!/usr/bin/env python3

from collections import OrderedDict
from typing import Dict, Any, Optional, cast, List, Tuple, Iterator

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import parallel
from torchrec.distributed.collective_utils import (
    invoke_on_rank_and_broadcast_result,
)
from torchrec.distributed.embedding import EmbeddingBagCollectionSharder
from torchrec.distributed.planner import EmbeddingShardingPlanner, sharder_name
from torchrec.distributed.types import (
    ShardingPlan,
    ModuleSharder,
    ShardedModule,
    ShardedTensor,
)
from torchrec.optim.fused import FusedOptimizerModule
from torchrec.optim.keyed import KeyedOptimizer, CombinedOptimizer


default_sharders: List[ModuleSharder[nn.Module]] = [
    EmbeddingBagCollectionSharder(),
]


class DistributedModelParallel(nn.Module, FusedOptimizerModule):
    """
    Entry point to model parallelism.
    Example:
        @torch.no_grad()
        def init_weights(m):
            if isinstance(m, nn.Linear)
                m.weight.fill_(1.0)
            elif isinstance(m, EmbeddingBagCollection)
                for param in m.parameters():
                    init.kaiming_normal_(param)

        m = MyModel(device='meta')
        m = DistributedModelParallel(m)
        m.apply(init_weights)

    Call Args:
        module: module to shard
        plan: sharding plan to use
        sharders: list of supported sharders per module type
        init_data_parallel: data-parallel modules can be lazy,
            i.e. they delay parameter initialization until the first forward pass.
            Pass True if that's a case to delay initialization of data parallel modules.
            Do first forward pass and then call DistributedModelParallel.init_data_parallel().

    Returns:
        None
    """

    def __init__(
        self,
        module: nn.Module,
        pg: Optional[dist.ProcessGroup] = None,
        device: Optional[torch.device] = None,
        plan: Optional[ShardingPlan] = None,
        sharders: List[ModuleSharder[nn.Module]] = default_sharders,
        init_data_parallel: bool = True,
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torchrec.distributed.{self.__class__.__name__}")

        self.module = module

        if pg is None:
            pg = dist.GroupMember.WORLD
            assert pg is not None, "Process group is not initialized"
        self._pg: dist.ProcessGroup = pg

        if device is None:
            device = torch.device("cpu")

        self.device: torch.device = device

        self._sharder_map: Dict[str, ModuleSharder[nn.Module]] = {
            sharder_name(sharder.module_type): sharder for sharder in sharders
        }

        # 2. Call ShardingPlanner.plan passing all found modules and corresponding sharders.
        if plan is None:
            planner = EmbeddingShardingPlanner(self._pg, self.device)
            plan = invoke_on_rank_and_broadcast_result(
                pg,
                0,
                planner.plan,
                module,
                sharders,
            )

        self._plan: ShardingPlan = plan

        # 3. Replace modules w/ the result of ModuleSharder.shard call,
        # replace rest w/ DistributedDataParallel wrappers.
        fused_optims = []
        self._shard_modules(
            init_data_parallel=init_data_parallel,
            init_model_parallel=True,
            fused_optims=fused_optims,
        )
        self._optim = CombinedOptimizer(fused_optims)

    # pyre-ignore [2, 3]
    def forward(self, *args, **kwargs) -> Any:
        return self.module(*args, **kwargs)

    def init_data_parallel(self) -> None:
        """
        See init_data_parallel c-tor argument for usage.
        It's safe to call this method multiple times.
        """
        self._shard_modules(
            init_data_parallel=True, init_model_parallel=False, fused_optims=[]
        )

    def _shard_modules(
        self,
        init_data_parallel: bool,
        init_model_parallel: bool,
        fused_optims: List[Tuple[str, KeyedOptimizer]],
    ) -> None:
        sharded = self._shard_modules_impl(
            self.module,
            "",
            init_data_parallel,
            init_model_parallel,
            fused_optims,
        )
        if init_data_parallel and not sharded:
            self.module = self._init_ddp(self.module)

    def _shard_modules_impl(
        self,
        module: nn.Module,
        path: str,
        init_data_parallel: bool,
        init_model_parallel: bool,
        fused_optims: List[Tuple[str, KeyedOptimizer]],
    ) -> bool:
        sharded_children = set()
        for name, child in module.named_children():
            curr_path = path + name
            sharded_params = self._plan.get_plan_for_module(curr_path)
            if sharded_params:
                if init_model_parallel:
                    # Shard module
                    sharder_key = sharder_name(type(child))
                    sharded_child = self._sharder_map[sharder_key].shard(
                        child,
                        sharded_params,
                        self._pg,
                        self.device,
                    )
                    setattr(module, name, sharded_child)

                    # Collect fused optimizer.
                    if isinstance(sharded_child, FusedOptimizerModule):
                        fused_optims.append((curr_path, sharded_child.fused_optimizer))
                sharded_children.add(name)
            else:
                child_sharded = self._shard_modules_impl(
                    child,
                    curr_path + ".",
                    init_data_parallel,
                    init_model_parallel,
                    fused_optims,
                )
                if child_sharded:
                    sharded_children.add(name)

        if init_data_parallel and len(sharded_children) > 0:
            for name, child in module.named_children():
                if name not in sharded_children:
                    dp_child = self._init_ddp(child)
                    setattr(module, name, dp_child)

        return len(sharded_children) > 0

    def _init_ddp(self, module: nn.Module) -> nn.Module:
        if isinstance(module, parallel.DistributedDataParallel):
            return module
        self._init_parameters(module)
        module = module.to(self.device)
        return cast(
            nn.Module,
            parallel.DistributedDataParallel(
                module,
                device_ids=None if self.device.type == "cpu" else [self.device],
                process_group=self._pg,
                gradient_as_bucket_view=True,
                broadcast_buffers=False,
            ),
        )

    def _init_parameters(self, module: nn.Module) -> None:
        @torch.no_grad()
        def init_parameters(module: nn.Module) -> None:
            # Allocate parameters and buffers if over 'meta' device.
            has_meta_param = False
            # pyre-ignore [16]
            for name, param in module._parameters.items():
                if isinstance(param, torch.Tensor) and param.device.type == "meta":
                    # pyre-ignore [29]
                    module._parameters[name] = nn.Parameter(
                        torch.empty_like(param, device=self.device),
                        requires_grad=param.requires_grad,
                    )
                    has_meta_param = True
            for name, buffer in module._buffers.items():
                if isinstance(buffer, torch.Tensor) and buffer.device.type == "meta":
                    # pyre-ignore [29]
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
        if isinstance(module, parallel.DistributedDataParallel):
            self._sparse_grad_parameter_names(module.module, destination, prefix)
        elif isinstance(module, ShardedModule):
            module.sparse_grad_parameter_names(destination, prefix)
        elif isinstance(module, nn.Embedding):
            if module.sparse:
                destination.append(prefix + "weight")
        elif isinstance(module, nn.EmbeddingBag):
            if module.sparse:
                destination.append(prefix + "weight")
        else:
            for name, child in module.named_children():
                self._sparse_grad_parameter_names(
                    child, destination, prefix + name + "."
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

        return self._state_dict(self.module, destination, prefix, keep_vars)

    def _state_dict(
        self,
        module: nn.Module,
        destination: Dict[str, Any],
        prefix: str,
        keep_vars: bool,
    ) -> Dict[str, Any]:
        if isinstance(module, parallel.DistributedDataParallel):
            module.module.state_dict(destination, prefix, keep_vars)
        elif isinstance(module, ShardedModule):
            module.state_dict(destination, prefix, keep_vars)
        else:
            for name, child in module.named_children():
                self._state_dict(child, destination, prefix + name + ".", keep_vars)
        return destination

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        state_dict = self.state_dict(prefix=prefix, keep_vars=True)
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                if value.requires_grad:
                    yield key, cast(
                        nn.Parameter,
                        value,
                    )
            elif isinstance(value, ShardedTensor):
                if value.local_shard.requires_grad:
                    yield key, cast(
                        nn.Parameter,
                        value,
                    )

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
                # pyre-ignore [29]
                m.reset_parameters()
