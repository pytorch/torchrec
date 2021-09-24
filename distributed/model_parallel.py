#!/usr/bin/env python3

from collections import OrderedDict
from typing import Dict, Any, Optional, cast, List, Tuple, Iterator

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed._sharded_tensor import ShardedTensor
from torch.nn.modules.module import _IncompatibleKeys
from torch.nn.parallel import DistributedDataParallel
from torchrec.distributed.embedding import (
    EmbeddingBagCollectionSharder,
    filter_state_dict,
)
from torchrec.distributed.planner import EmbeddingShardingPlanner, sharder_name
from torchrec.distributed.types import (
    ShardingPlan,
    ModuleSharder,
    ShardedModule,
)
from torchrec.distributed.utils import append_prefix
from torchrec.optim.fused import FusedOptimizerModule
from torchrec.optim.keyed import KeyedOptimizer, CombinedOptimizer


default_sharders: List[ModuleSharder[nn.Module]] = [
    EmbeddingBagCollectionSharder(),
]


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
        init_data_parallel: data-parallel modules can be lazy, i.e. they delay parameter initialization until the first forward pass. Pass True if that's a case to delay initialization of data parallel modules. Do first forward pass and then call DistributedModelParallel.init_data_parallel().

    Call Args:

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

        # 2. Call ShardingPlanner.collective_plan passing all found modules and corresponding sharders.
        if plan is None:
            plan = EmbeddingShardingPlanner(self._pg, self.device).collective_plan(
                module, sharders
            )
        self._plan: ShardingPlan = plan

        # 3. Replace modules w/ sharded versions,
        # and then wrap w/ DistributedDataParallel.
        fused_optims = []
        self._init_dmp(
            fused_optims=fused_optims,
        )
        if init_data_parallel:
            self._init_ddp()
        self._optim = CombinedOptimizer(fused_optims)

    @property
    def dmp_module(self) -> nn.Module:
        """
        Property to directly access sharded module, which
        may or may not yet be wrapped in DDP
        """
        # pyre-ignore [7]
        return (
            self.module.module
            if isinstance(self.module, DistributedDataParallel)
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
        if not isinstance(self.module, DistributedDataParallel):
            self._init_ddp()

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
            if sharded_params:
                # Shard module
                sharder_key = sharder_name(type(child))
                sharded_child = self._sharder_map[sharder_key].shard(
                    child,
                    sharded_params,
                    self._pg,
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

    def _init_ddp(self) -> None:
        DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(
            module=self.module,
            params_and_buffers_to_ignore=[
                key
                for key, value in self.module.state_dict().items()
                if isinstance(value, ShardedTensor)
            ],
        )
        # Allocate any 'meta' tensors
        self._init_parameters(self.module)
        # initailize DDP
        self.module = cast(
            nn.Module,
            DistributedDataParallel(
                module=self.module.to(self.device),
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
        return self._sparse_grad_parameter_names(self.dmp_module, destination, prefix)

    def _sparse_grad_parameter_names(
        self, module: nn.Module, destination: List[str], prefix: str = ""
    ) -> List[str]:
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
        if isinstance(module, ShardedModule):
            module.state_dict(destination, prefix, keep_vars)
        else:
            # pyre-ignore [29]
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
        module.load_state_dict(state_dict, strict=strict)
        missing_keys = []
        unexpected_keys = []
        if isinstance(module, ShardedModule):
            return module.load_state_dict(state_dict, strict=strict)
        else:
            # pyre-ignore [29]
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

    def _named_buffers(
        self, module: nn.Module, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
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
                # pyre-ignore [29]
                m.reset_parameters()
